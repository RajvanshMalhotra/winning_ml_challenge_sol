"""Unseen-country stand-in ("proxy") built from real test S1 records of the new country. Source 1 is deduplicated,
so every S1 record is a distinct business and the labels come for free.
  measure  : noise rates, match counts and distractor mix from train true pairs -> <run_dir>/proxy_stats.json
  build    : <data_dir>/<name>/<name>_source{1,2,3}.tsv + <name>_ground_truth.tsv (same layout as dataset/train)
  score    : macro F0.5 of a matching_results.tsv on the proxy, bootstrap CI, paired difference vs --baseline
  calibrate: implied leaderboard score of the new country per past submission vs its proxy score
The noise generator is deliberately independent of ber.noise / ber.lexicons: a proxy made with the pipeline's own
lexicons would only measure how well the pipeline undoes its own augmentation."""
import argparse
import csv
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from anyascii import anyascii
from rapidfuzz.distance import Levenshtein

from ber.config import run_dir
from ber.evaluate import f05
from ber.io import _read_tsv, read_family, read_truth

NAME_KEYS = ["null", "case", "accent", "amp", "abbrev", "legal", "drop", "add", "reorder", "typo", "domain"]
ADDR_KEYS = ["null", "null_token", "case", "accent", "abbrev", "comp_drop", "comp_add", "comp_reorder", "house",
             "postal_drop", "typo"]

_PUNCT = re.compile(r"[^\w\s]")
_NUM = re.compile(r"\b\d+[a-z]?\b", re.I)
_POSTAL = re.compile(r"\b\d{5,6}\b")
_DOMAIN = re.compile(r"^#\S+$|^\S+\.(com|net|org|in|co|fr|biz|info)$", re.I)
_LEGAL = {"ltd", "limited", "pvt", "private", "inc", "incorporated", "llc", "corp", "corporation", "co", "company",
          "sarl", "sas", "sa", "eurl", "sasu", "sci", "snc", "llp", "plc", "gmbh"}


# ---------------------------------------------------------------- measure
def _fold(s: str) -> str:
    return anyascii(unicodedata.normalize("NFKC", s))


def _toks(s: str) -> list[str]:
    return _PUNCT.sub(" ", _fold(s).lower()).split()


def _is_null(s: str) -> bool:
    return s.strip().upper() in ("", "NULL", "NONE", "NAN")


def _non_latin(s: str) -> bool:
    return any(c.isalpha() and ord(c) > 0x24F for c in s)


def _is_abbrev(short: str, long: str) -> bool:
    """`short` is a first-letter-anchored subsequence of `long` (pvt/private, rd/road, bd/boulevard)."""
    if not short or len(short) >= len(long) or short[0] != long[0]:
        return False
    it = iter(long)
    return all(c in it for c in short)


def _token_edits(ta: list[str], tb: list[str]) -> tuple[bool, bool]:
    only_a, only_b = set(ta) - set(tb), set(tb) - set(ta)
    abbrev = any(_is_abbrev(y, x) or _is_abbrev(x, y) for x in only_a for y in only_b)
    typo = any(min(len(x), len(y)) >= 4 and Levenshtein.distance(x, y) <= 2 and not _is_abbrev(y, x)
               and not _is_abbrev(x, y) for x in only_a for y in only_b)
    return abbrev, typo


def name_indicators(a: str, b: str) -> dict[str, bool]:
    """Which kinds of noise turn S1 name `a` into matched name `b` (not mutually exclusive)."""
    out = dict.fromkeys(NAME_KEYS + ["same", "script"], False)
    if a == b:
        out["same"] = True
        return out
    if _is_null(b):
        out["null"] = True
        return out
    out["script"] = _non_latin(a) != _non_latin(b)
    out["case"] = a.lower() == b.lower() or (a != a.upper() and b == b.upper())
    out["accent"] = not out["script"] and a != _fold(a) and b == _fold(b)
    out["domain"] = bool(_DOMAIN.match(b.strip())) and not _DOMAIN.match(a.strip())
    ta, tb = _toks(a), _toks(b)
    out["amp"] = ("&" in a or "and" in ta or "et" in ta) != ("&" in b or "and" in tb or "et" in tb)
    out["legal"] = {t for t in ta if t in _LEGAL} != {t for t in tb if t in _LEGAL}
    ca, cb = [t for t in ta if t not in _LEGAL], [t for t in tb if t not in _LEGAL]
    if ca != cb:
        out["reorder"] = sorted(ca) == sorted(cb)
        out["drop"] = set(cb) < set(ca)
        out["add"] = set(ca) < set(cb)
        out["abbrev"], out["typo"] = _token_edits(ca, cb)
    return out


def _comps(s: str) -> list[str]:
    return [c.strip().lower() for c in s.split(",") if c.strip()]


def addr_indicators(a: str, b: str) -> dict[str, bool]:
    out = dict.fromkeys(ADDR_KEYS + ["same", "script"], False)
    if a == b:
        out["same"] = True
        return out
    if _is_null(b):
        out["null"] = not _is_null(a)
        out["same"] = _is_null(a)
        return out
    out["script"] = _non_latin(a) != _non_latin(b)
    out["null_token"] = "null" in _toks(b) and "null" not in _toks(a)
    out["case"] = a.lower() == b.lower() or (a != a.upper() and b == b.upper())
    out["accent"] = not out["script"] and a != _fold(a) and b == _fold(b)
    ca, cb = _comps(a), _comps(b)
    out["comp_drop"], out["comp_add"] = len(cb) < len(ca), len(cb) > len(ca)
    ta, tb = [t for t in _toks(a) if t != "null"], [t for t in _toks(b) if t != "null"]
    out["comp_reorder"] = ta != tb and sorted(ta) == sorted(tb)
    na, nb = _NUM.findall(_fold(a).lower()), _NUM.findall(_fold(b).lower())
    na = [n for n in na if not _POSTAL.fullmatch(n)]
    nb = [n for n in nb if not _POSTAL.fullmatch(n)]
    out["house"] = bool(na) and (not nb or na[0] != nb[0])
    out["postal_drop"] = bool(_POSTAL.search(a)) and not _POSTAL.search(b)
    out["abbrev"], out["typo"] = _token_edits(ta, tb)
    return out


def _key(s: str) -> str:
    return " ".join(sorted(_toks(s))) if not _is_null(s) else ""


def measure(raw: pd.DataFrame, truth: dict[str, set[str]], max_pairs: int, seed: int) -> dict:
    """raw: read_family(train) (entity_id, business_name, business_address, country, source)."""
    rng = np.random.default_rng(seed)
    rec = raw.set_index("entity_id")
    s1 = rec[rec.source == 1]
    pairs = pd.DataFrame([(a, b) for a, bs in truth.items() for b in bs], columns=["s1", "m"])
    pairs["country"] = s1.country.reindex(pairs.s1).to_numpy()
    pairs["source"] = rec.source.reindex(pairs.m).to_numpy()
    stats: dict = {"noise": {}, "counts": {}, "unmatched": {}}
    for (country, source), g in pairs.groupby(["country", "source"]):
        g = g.iloc[rng.permutation(len(g))[:max_pairs]]
        an, bn = s1.business_name.reindex(g.s1).to_numpy(), rec.business_name.reindex(g.m).to_numpy()
        aa, ba = s1.business_address.reindex(g.s1).to_numpy(), rec.business_address.reindex(g.m).to_numpy()
        n = pd.DataFrame([name_indicators(x, y) for x, y in zip(an, bn)]).mean()
        ad = pd.DataFrame([addr_indicators(x, y) for x, y in zip(aa, ba)]).mean()
        stats["noise"][f"{country}|{int(source)}"] = {"n_pairs": len(g), "name": n.round(5).to_dict(),
                                                      "addr": ad.round(5).to_dict(),
                                                      "addr_key_equal": float(np.mean([_key(x) == _key(y) for x, y in zip(aa, ba)])),
                                                      "name_key_equal": float(np.mean([_key(x) == _key(y) for x, y in zip(an, bn)]))}
    src = rec.source
    for country, g in s1.groupby("country"):
        n2 = [sum(src.get(m) == 2 for m in truth.get(s, ())) for s in g.index]
        n3 = [sum(src.get(m) == 3 for m in truth.get(s, ())) for s in g.index]
        joint = pd.DataFrame({"n2": n2, "n3": n3}).value_counts().reset_index(name="count")
        stats["counts"][country] = joint.astype(int).values.tolist()
    matched = {m for ms in truth.values() for m in ms}
    for country, g in rec[rec.source > 1].groupby("country"):
        un = g[~g.index.isin(matched)]
        smp = un.iloc[rng.permutation(len(un))[:max_pairs]]
        s1c = s1[s1.country == country]
        addr_keys, name_keys = set(map(_key, s1c.business_address)) - {""}, set(map(_key, s1c.business_name)) - {""}
        same_addr = np.mean([_key(a) in addr_keys for a in smp.business_address]) if len(smp) else 0.0
        same_name = np.mean([_key(a) in name_keys for a in smp.business_name]) if len(smp) else 0.0
        stats["unmatched"][country] = {"n_s1": int(len(s1c)), "n_s23": int(len(g)), "n_unmatched": int(len(un)),
                                       "exact_same_addr": float(same_addr), "exact_same_name": float(same_name)}
    return stats


# ---------------------------------------------------------------- generate
# French-first abbreviation tables, written for the proxy only (not shared with ber.lexicons)
NAME_ABBR = {"societe": "Sté", "société": "Sté", "compagnie": "Cie", "etablissements": "Ets", "établissements": "Ets",
             "saint": "St", "sainte": "Ste", "freres": "Frs", "frères": "Frs", "international": "Intl",
             "services": "Svcs", "entreprise": "Ent", "association": "Asso", "industries": "Ind",
             "distribution": "Distrib", "boulangerie": "Boul", "pharmacie": "Pharma", "restaurant": "Resto",
             "company": "Co", "corporation": "Corp", "limited": "Ltd", "private": "Pvt"}
ADDR_ABBR = {"rue": "R.", "avenue": "Av.", "boulevard": "Bd", "place": "Pl.", "chemin": "Ch.", "route": "Rte",
             "allée": "All.", "allee": "All.", "impasse": "Imp.", "faubourg": "Fbg", "saint": "St", "sainte": "Ste",
             "quartier": "Qt", "résidence": "Rés.", "residence": "Rés.", "bâtiment": "Bât.", "batiment": "Bât.",
             "zone": "Z.", "industrielle": "Ind.", "centre": "Ctre", "cedex": "Cdx", "road": "Rd", "street": "St"}
LEGAL_FORMS = ["SARL", "SAS", "SA", "EURL", "SASU", "SCI", "SNC", "S.A.R.L.", "S.A.S."]
NAME_FILLERS = ["Groupe", "& Fils", "et Cie", "France", "Services", "Conseil", "Distribution"]
ADDR_EXTRAS = ["Bâtiment B", "BP 123", "CEDEX", "Près de la Mairie", "Face à la Poste", "Rez-de-chaussée",
               "Zone Artisanale", "Lieu-dit Le Bourg"]
KEYBOARD = dict(zip("azertyuiopqsdfghjklmwxcvbn", "zertyuiopqsdfghjklmwxcvbna"))
ACCENT_ADD = {"a": "àâä", "e": "éèêë", "i": "îï", "o": "ôö", "u": "ùûü", "c": "ç"}


def _pick(rng, seq):
    return seq[int(rng.integers(len(seq)))]


def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def _case(s, rng):
    return s.upper() if s != s.upper() else s.title()


def _typo(s, rng):
    idx = [i for i, c in enumerate(s) if c.isalpha()]
    if len(idx) < 4:
        return s
    i, k = _pick(rng, idx), int(rng.integers(4))
    if k == 0:
        return s[:i] + s[i + 1:]
    if k == 1 and i + 1 < len(s):
        return s[:i] + s[i + 1] + s[i] + s[i + 2:]
    if k == 2:
        c = KEYBOARD.get(s[i].lower(), s[i])
        return s[:i] + (c.upper() if s[i].isupper() else c) + s[i + 1:]
    return s[:i] + s[i] + s[i:]


def _abbrev(s, rng, table):
    toks = s.split()
    hits = [i for i, t in enumerate(toks) if t.lower().strip(",.") in table]
    if hits:
        i = _pick(rng, hits)
        toks[i] = table[toks[i].lower().strip(",.")] + ("," if toks[i].endswith(",") else "")
        return " ".join(toks)
    long = [i for i, t in enumerate(toks) if len(t) >= 7 and t.isalpha()]
    if not long:
        return s
    i = _pick(rng, long)
    toks[i] = toks[i][: int(rng.integers(3, 5))] + "."
    return " ".join(toks)


def _accent_add(s, rng):
    """Spurious accent on a random vowel / c, as seen in the French test S2/S3 (MÂTERNELLE, Sàrl, çentre)."""
    idx = [i for i, c in enumerate(s) if c.lower() in ACCENT_ADD]
    if not idx:
        return s
    i = _pick(rng, idx)
    c = _pick(rng, ACCENT_ADD[s[i].lower()])
    return s[:i] + (c.upper() if s[i].isupper() else c) + s[i + 1:]


def _bracket(s, rng):
    """Wrap one word in () or [] (Paroissiale (France), Bureau [Sas])."""
    toks = s.split()
    if len(toks) < 2:
        return s
    legal = [i for i, t in enumerate(toks) if t.lower().strip(".,") in _LEGAL]
    i = _pick(rng, legal) if legal else int(rng.integers(1, len(toks)))
    o, c = _pick(rng, ["()", "[]"])
    toks[i] = f"{o}{toks[i]}{c}"
    return " ".join(toks)


def _drop(s, rng):
    toks = s.split()
    if len(toks) < 2:
        return s
    del toks[int(rng.integers(len(toks)))]
    return " ".join(toks)


def _reorder(s, rng):
    toks = s.split()
    if len(toks) < 2:
        return s
    i = int(rng.integers(len(toks) - 1))
    toks[i], toks[i + 1] = toks[i + 1], toks[i]
    return " ".join(toks)


def _legal(s, rng):
    toks = s.split()
    have = [i for i, t in enumerate(toks) if t.strip(".,").replace(".", "").lower() in _LEGAL]
    if have and rng.random() < 0.6:
        del toks[_pick(rng, have)]
        return " ".join(toks) or s
    if have:
        toks[_pick(rng, have)] = _pick(rng, LEGAL_FORMS)
        return " ".join(toks)
    return f"{s} {_pick(rng, LEGAL_FORMS)}"


def _amp(s, rng):
    if re.search(r"\s&\s", s):
        return re.sub(r"\s&\s", _pick(rng, [" et ", " and ", " "]), s, count=1)
    if re.search(r"\s(et|and)\s", s, re.I):
        return re.sub(r"\s(et|and)\s", " & ", s, count=1, flags=re.I)
    return s


def _domain(s, rng):
    body = re.sub(r"[^a-z0-9]", "", _strip_accents(s).lower())
    return f"{body}{_pick(rng, ['.fr', '.com'])}" if body else s


NAME_OPS = {"case": _case, "accent": lambda s, r: _strip_accents(s), "accent_add": _accent_add, "bracket": _bracket,
            "amp": _amp,
            "abbrev": lambda s, r: _abbrev(s, r, NAME_ABBR), "legal": _legal, "drop": _drop,
            "add": lambda s, r: f"{s} {_pick(r, NAME_FILLERS)}" if r.random() < 0.7 else f"{_pick(r, NAME_FILLERS)} {s}",
            "reorder": _reorder, "typo": _typo, "domain": _domain}


def _split(a):
    return [c.strip() for c in a.split(",") if c.strip()]


def _comp_drop(a, rng):
    c = _split(a)
    if len(c) < 2:
        return a
    del c[int(rng.integers(len(c)))]
    return ", ".join(c)


def _comp_add(a, rng):
    c = _split(a)
    c.insert(int(rng.integers(len(c) + 1)), _pick(rng, ADDR_EXTRAS))
    return ", ".join(c)


def _comp_reorder(a, rng):
    c = _split(a)
    if len(c) < 2:
        return _reorder(a, rng)
    i = int(rng.integers(len(c) - 1))
    c[i], c[i + 1] = c[i + 1], c[i]
    return ", ".join(c)


def _house(a, rng):
    nums = [m for m in _NUM.finditer(a) if not _POSTAL.fullmatch(m.group())]
    if not nums:
        return a
    m = nums[0]
    digits = re.match(r"\d+", m.group()).group()
    k = int(rng.integers(4))
    new = ("" if k == 0 else str(max(1, int(digits) + int(rng.choice([-2, -1, 1, 2])))) if k == 1
           else f"{digits} {_pick(rng, ['bis', 'ter', 'B'])}" if k == 2 else f"{digits}-{int(digits) + 2}")
    return re.sub(r"\s{2,}", " ", (a[: m.start()] + new + a[m.end():]).strip(" ,"))


def _postal_drop(a, rng):
    return re.sub(r"\s{2,}", " ", _POSTAL.sub("", a)).strip(" ,") if _POSTAL.search(a) else a


def _null_token(a, rng):
    c = _split(a)
    c.insert(int(rng.integers(len(c) + 1)), "NULL")
    return ", ".join(c)


ADDR_OPS = {"null_token": _null_token, "case": _case, "accent": lambda s, r: _strip_accents(s), "accent_add": _accent_add,
            "abbrev": lambda s, r: _abbrev(s, r, ADDR_ABBR), "comp_drop": _comp_drop, "comp_add": _comp_add,
            "comp_reorder": _comp_reorder, "house": _house, "postal_drop": _postal_drop, "typo": _typo}


def surface_shares(recs: pd.DataFrame) -> dict[int, dict[str, dict[str, float]]]:
    """Per source: share of names / addresses that are all-caps, carry accents, carry a postal code, or are null.
    Needs no labels, so it can be read off the real records of the new country."""
    out = {}
    for src, g in recs.groupby("source"):
        out[int(src)] = {}
        for part, col in (("name", "business_name"), ("addr", "business_address")):
            v = g[col].fillna("")
            live = v[~v.map(_is_null)]
            out[int(src)][part] = {"null": float(v.map(_is_null).mean()),
                                   "upper": float(live.map(lambda x: x == x.upper() and any(c.isalpha() for c in x)).mean()),
                                   "accent": float(live.map(lambda x: x != _strip_accents(x)).mean()),
                                   "postal": float(live.map(lambda x: bool(_POSTAL.search(x))).mean()),
                                   "bracket": float(live.map(lambda x: bool(re.search(r"[(\[]", x))).mean())}
    return out


def calibrate_surface(noise: dict, shares: dict) -> dict:
    """Replace the train rates of the surface ops (case, accents, brackets, postal_drop, null) with the rates that turn the new
    country's real S1 marginals into its real S2/S3 marginals. Train has no accents, so this is the only place the
    accent rates can come from. Structural ops (drops, typos, house numbers, ...) keep the train rates."""
    out = {src: {part: dict(r) for part, r in nz.items()} for src, nz in noise.items()}
    for src in out:
        for part in ("name", "addr"):
            a, b = shares[1][part], shares[src][part]
            clip = lambda x: float(np.clip(x, 0.0, 1.0))
            r = out[src][part]
            r["case"] = clip((b["upper"] - a["upper"]) / max(1e-9, 1 - a["upper"]))
            # accents move both ways in the French data: stripped from addresses, spuriously added to names
            if b["accent"] >= a["accent"]:
                r["accent"], r["accent_add"] = 0.0, clip((b["accent"] - a["accent"]) / max(1e-9, 1 - a["accent"]))
            else:
                r["accent"], r["accent_add"] = clip(1 - b["accent"] / a["accent"]), 0.0
            if part == "name":
                r["bracket"] = clip((b["bracket"] - a["bracket"]) / max(1e-9, 1 - a["bracket"]))
            r["null"] = clip((b["null"] - a["null"]) / max(1e-9, 1 - a["null"]))
            if part == "addr":
                r["postal_drop"] = clip(1 - b["postal"] / a["postal"]) if a["postal"] > 0 else 0.0
    return out


class Noiser:
    def __init__(self, noise: dict, rng: np.random.Generator):
        """noise: {source: {"name": rates, "addr": rates}}, per-op probabilities as measured (incl. `null`)."""
        self.rng, self.plan = rng, {}
        for src, nz in noise.items():
            plan = {}
            for part, ops in (("name", NAME_OPS), ("addr", ADDR_OPS)):
                # each op fires independently at its measured rate (script changes have no counterpart here)
                plan[part] = (float(nz[part].get("null", 0.0)), {k: float(nz[part].get(k, 0.0)) for k in ops})
            self.plan[int(src)] = plan

    def _apply(self, text: str, part: str, src: int) -> str:
        p_null, probs = self.plan[src][part]
        if self.rng.random() < p_null:
            return "" if part == "addr" and self.rng.random() < 0.5 else "NULL"
        ops = NAME_OPS if part == "name" else ADDR_OPS
        keys = [k for k, p in probs.items() if self.rng.random() < p]
        for i in self.rng.permutation(len(keys)):
            new = ops[keys[i]](text, self.rng)
            text = new if new.strip() else text
        return text

    def __call__(self, name: str, addr: str, src: int) -> tuple[str, str]:
        name = self._apply(name, "name", src) if name else name
        addr = self._apply(addr, "addr", src) if addr and not _is_null(addr) else addr
        return name.replace("\t", " ").replace("\n", " "), addr.replace("\t", " ").replace("\n", " ")


def pooled_noise(stats: dict, countries: list[str] | None) -> dict:
    """Pair-weighted mean of the per-(country, source) rates -> {source: {"name": .., "addr": ..}}."""
    out = {}
    for src in (2, 3):
        rows = [(v["n_pairs"], v) for k, v in stats["noise"].items()
                if k.endswith(f"|{src}") and (countries is None or k.split("|")[0] in countries)]
        w = np.array([n for n, _ in rows], dtype=float)
        out[src] = {part: {k: float(np.average([v[part].get(k, 0.0) for _, v in rows], weights=w))
                           for k in rows[0][1][part]} for part in ("name", "addr")}
    return out


def plan_sizes(n_s1: int, pool: int, mean_m: float, mean_k: float, hidden_share: float, min_query_frac: float) -> tuple[int, int]:
    """Choose Q query S1s and H hidden businesses (Q + H = n_s1) so that Q*mean_m matched records plus the unmatched
    records (hidden copies are `hidden_share` of them, mean_k copies each) add up to `pool`."""
    denom = 1 - hidden_share * mean_m / mean_k
    q = (n_s1 - hidden_share * pool / mean_k) / denom if denom > 0 else n_s1
    q = int(np.clip(q, min_query_frac * n_s1, n_s1))
    return q, n_s1 - q


def build(s1: pd.DataFrame, stats: dict, pool: int, rng: np.random.Generator, stats_countries: list[str] | None = None,
          min_query_frac: float = 0.4, max_distractor_share: float = 0.5, surface: dict | None = None) -> tuple[dict[int, pd.DataFrame], pd.DataFrame, dict]:
    """s1: real S1 records of the new country (entity_id, business_name, business_address, country).
    Returns {source: records}, ground truth (source1_entity_id, matched_entity_ids) and a meta dict."""
    countries = stats_countries or sorted(stats["counts"])
    joint = np.array([r for c in countries for r in stats["counts"][c]], dtype=float)  # n2, n3, count
    p = joint[:, 2] / joint[:, 2].sum()
    tot = joint[:, 0] + joint[:, 1]
    mean_m = float((tot * p).sum())
    pk = np.where(tot > 0, p, 0) / p[tot > 0].sum()
    mean_k = float((tot * pk).sum())
    um = [stats["unmatched"][c] for c in countries]
    n_key = [stats["noise"][f"{c}|{s}"] for c in countries for s in (2, 3) if f"{c}|{s}" in stats["noise"]]
    # exact-key hits undercount noisy look-alikes: divide by how often a TRUE match keeps the exact key
    addr_keep = np.mean([v["addr_key_equal"] for v in n_key]) or 1.0
    name_keep = np.mean([v["name_key_equal"] for v in n_key]) or 1.0
    w = np.array([u["n_unmatched"] for u in um], dtype=float)
    share_addr = min(1.0, np.average([u["exact_same_addr"] for u in um], weights=w) / addr_keep)
    share_name = min(1.0, np.average([u["exact_same_name"] for u in um], weights=w) / name_keep)
    if share_addr + share_name > max_distractor_share:
        f = max_distractor_share / (share_addr + share_name)
        share_addr, share_name = share_addr * f, share_name * f
    hidden_share = 1 - share_addr - share_name
    n_q, n_h = plan_sizes(len(s1), pool, mean_m, mean_k, hidden_share, min_query_frac)

    order = rng.permutation(len(s1))
    s1 = s1.iloc[order].reset_index(drop=True)
    names, addrs = s1.business_name.to_numpy(), s1.business_address.to_numpy()
    noise = pooled_noise(stats, countries)
    if surface is not None:
        noise = calibrate_surface(noise, surface)
    noiser = Noiser(noise, rng)
    recs: dict[int, list] = {2: [], 3: []}   # (tmp_id, name, addr, owner or -1, kind)
    for i in range(len(s1)):
        is_q = i < n_q
        j = rng.choice(len(p), p=p if is_q else pk)
        for src, n in ((2, int(joint[j, 0])), (3, int(joint[j, 1]))):
            for _ in range(n):
                nm, ad = noiser(names[i], addrs[i], src)
                recs[src].append((nm, ad, i if is_q else -1, "match" if is_q else "hidden"))
    n_matched = sum(r[3] == "match" for v in recs.values() for r in v)
    n_unmatched = max(0, pool - n_matched)  # hidden copies were sized to fill `hidden_share` of this
    n_addr, n_name = int(share_addr * n_unmatched), int(share_name * n_unmatched)
    p_src2 = (len(recs[2]) + 1) / (len(recs[2]) + len(recs[3]) + 2)

    def distractors(kind: str, n: int) -> None:
        host, donor = rng.integers(len(s1), size=n), rng.integers(len(s1), size=n)
        if kind == "hidden":  # one more noisy copy of a business that is not in S1
            host = rng.integers(n_q, len(s1), size=n) if n_h else host[:0]
        for h, d in zip(host, donor):
            if h == d and kind != "hidden":
                continue
            src = 2 if rng.random() < p_src2 else 3
            nm, ad = {"same_addr": (names[d], addrs[h]), "same_name": (names[h], addrs[d])}.get(kind, (names[h], addrs[h]))
            nm, ad = noiser(nm, ad, src)
            recs[src].append((nm, ad, -1, kind))

    distractors("same_addr", n_addr)
    distractors("same_name", n_name)
    # the real pool can be denser than train's (test has ~5.5-5.8 S2/S3 per S1 vs 4.7 in train, every country):
    # fill what is still missing with the same distractor mix
    topup = max(0, pool - len(recs[2]) - len(recs[3]))
    for kind, share in (("hidden", hidden_share), ("same_addr", share_addr), ("same_name", share_name)):
        distractors(kind, int(round(share * topup)))

    s1_ids = np.array([f"S1-P{k:07d}" for k in range(len(s1))])
    s1_out = pd.DataFrame({"entity_id": s1_ids, "business_name": names, "business_address": addrs,
                           "country": s1.country.to_numpy(), "orig_id": s1.entity_id.to_numpy()})
    s1_out = s1_out.iloc[rng.permutation(len(s1_out))].reset_index(drop=True)  # row order carries no signal
    country = s1.country.iloc[0]
    out, owners = {1: s1_out}, []
    for src in (2, 3):
        df = pd.DataFrame(recs[src], columns=["business_name", "business_address", "owner", "kind"])
        df = df.iloc[rng.permutation(len(df))].reset_index(drop=True)
        df.insert(0, "entity_id", [f"S{src}-P{k:07d}" for k in range(len(df))])
        df["country"] = country
        out[src] = df
        owners.append(df[df.owner >= 0][["entity_id", "owner"]])
    own = pd.concat(owners)
    own["s1"] = s1_ids[own.owner.to_numpy()]
    grouped = own.groupby("s1").entity_id.agg(lambda x: ",".join(sorted(x)))
    q_ids = s1_ids[:n_q]
    gt = pd.DataFrame({"source1_entity_id": q_ids, "matched_entity_ids": grouped.reindex(q_ids).fillna("").to_numpy()})
    kinds = pd.concat([out[2].kind, out[3].kind]).value_counts().to_dict()
    meta = {"n_s1": len(s1), "n_query": n_q, "n_hidden": n_h, "pool_target": pool,
            "pool_built": int(len(out[2]) + len(out[3])), "pool_topup": int(topup), "record_kinds": {k: int(v) for k, v in kinds.items()},
            "mean_matches": mean_m, "singleton_share": float(p[tot == 0].sum()),
            "distractor_shares": {"same_addr": share_addr, "same_name": share_name, "hidden": hidden_share},
            "stats_countries": countries, "noise_rates": noise, "surface_shares": surface}
    return out, gt, meta


# ---------------------------------------------------------------- score / calibrate
def read_lists(path: str | Path, key: str, val: str) -> dict[str, set[str]]:
    df = _read_tsv(Path(path))
    return {k: {x for x in v.split(",") if x} for k, v in zip(df[key], df[val])}


def entity_scores(pred: dict[str, set[str]], truth: dict[str, set[str]]) -> pd.Series:
    return pd.Series({k: f05(pred.get(k, set()), t) for k, t in truth.items()})


def bootstrap_ci(x: np.ndarray, rng: np.random.Generator, n_boot: int = 1000) -> tuple[float, float]:
    means = [x[rng.integers(len(x), size=len(x))].mean() for _ in range(n_boot)]
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def implied_country_score(lb: float, other: dict[str, float], shares: dict[str, float], target: str) -> float:
    """Leaderboard = sum_c share_c * F_c  ->  F_target given the validation scores of the other countries."""
    return (lb - sum(shares[c] * other[c] for c in other)) / shares[target]


def proxy_dir(cfg: dict, name: str) -> Path:
    return Path(cfg["paths"]["data_dir"]) / name


def write_tsv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, sep="\t", index=False, quoting=csv.QUOTE_NONE, escapechar=None)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("action", choices=["measure", "build", "score", "calibrate"])
    parser.add_argument("--name", default="proxy_fr", help="proxy family name (folder under data_dir)")
    parser.add_argument("--country", default="France", help="test country the proxy stands in for")
    parser.add_argument("--seed", type=int, default=None, help="default: config seed; use another seed for a holdout proxy")
    parser.add_argument("--max-pairs", type=int, default=200_000, help="measure: true pairs sampled per country/source")
    parser.add_argument("--pool-size", type=int, default=None, help="build: S2+S3 records; default = real test count")
    parser.add_argument("--stats-countries", nargs="*", default=None, help="build: train countries to take noise from")
    parser.add_argument("--pred", help="score: matching_results.tsv produced on the proxy family")
    parser.add_argument("--baseline", help="score: second matching_results.tsv for a paired difference")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--table", help="calibrate: CSV with submission, lb, oof_<country>..., proxy")


def run(cfg: dict, args: argparse.Namespace) -> None:
    seed = cfg["seed"] if args.seed is None else args.seed
    rng = np.random.default_rng(seed)
    data_dir = cfg["paths"]["data_dir"]
    stats_path = run_dir(cfg) / "proxy_stats.json"
    if args.action == "measure":
        stats = measure(read_family(data_dir, "train"), read_truth(data_dir), args.max_pairs, seed)
        stats_path.write_text(json.dumps(stats, indent=2))
        for k, v in stats["noise"].items():
            print(k, "name same", v["name"]["same"], "| addr same", v["addr"]["same"])
        print(json.dumps(stats["unmatched"], indent=2), f"\n-> {stats_path}")
    elif args.action == "build":
        stats = json.loads(stats_path.read_text())
        test = read_family(data_dir, "test")
        test = test[test.country == args.country]
        if test.empty:
            raise SystemExit(f"no test records with country == {args.country!r}")
        pool = args.pool_size or int((test.source > 1).sum())
        s1 = test[test.source == 1][["entity_id", "business_name", "business_address", "country"]]
        surface = surface_shares(test)
        out, gt, meta = build(s1, stats, pool, rng, args.stats_countries, surface=surface)
        meta["surface_shares_built"] = surface_shares(pd.concat([out[k].assign(source=k) for k in (1, 2, 3)]))
        d = proxy_dir(cfg, args.name)
        d.mkdir(parents=True, exist_ok=True)
        cols = ["entity_id", "business_name", "business_address", "country"]
        for src in (1, 2, 3):
            write_tsv(out[src][cols], d / f"{args.name}_source{src}.tsv")
        write_tsv(gt, d / f"{args.name}_ground_truth.tsv")
        write_tsv(out[1][["entity_id", "orig_id"]], d / f"{args.name}_s1_origin.tsv")
        for src in (2, 3):
            write_tsv(out[src][["entity_id", "kind"]], d / f"{args.name}_source{src}_kind.tsv")
        meta.update(seed=seed, country=args.country)
        (d / f"{args.name}_meta.json").write_text(json.dumps(meta, indent=2))
        print(json.dumps(meta, indent=2), f"\n-> {d}")
    elif args.action == "score":
        d = proxy_dir(cfg, args.name)
        truth = read_lists(d / f"{args.name}_ground_truth.tsv", "source1_entity_id", "matched_entity_ids")
        sc = entity_scores(read_lists(args.pred, "source1_entity_id", "matched_entity_ids"), truth)
        single = np.array([not truth[k] for k in sc.index])
        lo, hi = bootstrap_ci(sc.to_numpy(), rng, args.n_boot)
        print(f"{args.name}: macro F0.5 {sc.mean():.4f} (95% CI {lo:.4f}-{hi:.4f}) | singletons "
              f"{sc[single].mean():.4f} | non-singletons {sc[~single].mean():.4f} | n={len(sc):,}")
        if args.baseline:
            base = entity_scores(read_lists(args.baseline, "source1_entity_id", "matched_entity_ids"), truth)
            diff = (sc - base.reindex(sc.index)).to_numpy()
            lo, hi = bootstrap_ci(diff, rng, args.n_boot)
            print(f"pred - baseline: {diff.mean():+.4f} (95% CI {lo:+.4f} to {hi:+.4f}); "
                  f"better on {(diff > 0).mean():.2%}, worse on {(diff < 0).mean():.2%} of S1s")
    else:
        from scipy.stats import spearmanr
        tab = pd.read_csv(args.table)
        s1 = _read_tsv(Path(data_dir) / "test" / "test_source1.tsv")
        shares = s1.country.value_counts(normalize=True).to_dict()
        others = [c for c in shares if c != args.country]
        missing = [f"oof_{c}" for c in others if f"oof_{c}" not in tab]
        if missing:
            raise SystemExit(f"calibrate needs columns {missing} (validation F0.5 of each other test country)")
        tab["implied"] = [implied_country_score(r.lb, {c: r[f"oof_{c}"] for c in others}, shares, args.country)
                          for _, r in tab.iterrows()]
        tab["gap"] = tab.implied - tab.proxy
        print("test S1 shares:", {k: round(v, 4) for k, v in shares.items()})
        print(tab.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        if len(tab) >= 3:
            slope, icpt = np.polyfit(tab.proxy, tab.implied, 1)
            print(f"\nSpearman(proxy, implied) = {spearmanr(tab.proxy, tab.implied)[0]:.3f} | "
                  f"slope {slope:.3f}, intercept {icpt:+.4f} | gap mean {tab.gap.mean():+.4f} sd {tab.gap.std():.4f}")
        if len(tab) < 5:
            print(f"only {len(tab)} submissions: rank agreement on this few points is weak evidence")
