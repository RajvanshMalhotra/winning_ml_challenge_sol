import re
import unicodedata

import jellyfish
from anyascii import anyascii

from ber.lexicons import Lexicon

NUM_RE = re.compile(r"\d[\d/\-]*[a-z]?")
ADDR_TOKEN_RE = re.compile(r"#|[a-z0-9][a-z0-9/\-]*")
UNIT_WORDS = frozenset({"#", "unit", "suite", "apt", "apartment", "fl", "floor", "box", "lot",
                        "sector", "sec", "pin", "flat", "room", "ste"})
DOMAIN_RE = re.compile(r"#?(www\.)?[a-z0-9\-]+(\.[a-z]{2,4}){0,2}")
MOJIBAKE_RE = re.compile(r"(?<![A-Za-z])[Ââ]|[Ââ](?=\s|$)")
DBA_RE = re.compile(r"\s*\b(?:d\s*/\s*b\s*/\s*a|d\.b\.a\.?|dba|t/a|trading as|a/k/a|aka|enseigne)\b\s*", re.I)
LANDMARK_RE = re.compile(r"\b(?:near|nr|opp|opposite|behind|beside|next to|adjacent to|in front of)\b\.?\s+[^,]+", re.I)
DEFAULT_POSTAL_RE = re.compile(r"\b\d{5,6}\b(?=\s*(?:,|$))")  # countries without a lexicon entry
UNIT_RE = re.compile(r"\b(p\s*o\s*box|po box|box|unit|suite|apt|apartment|flat|room|flr|fl|floor)\s*#?\s*([0-9][a-z0-9/\-]*)")
ORD_FLOOR_RE = re.compile(r"\b(\d+)(?:st|nd|rd|th)\s+(?:fl|flr|floor)\b")
HASH_UNIT_RE = re.compile(r"#\s*([0-9][a-z0-9/\-]*)")
UNIT_KIND = {"unit": "unit", "suite": "unit", "apt": "unit", "apartment": "unit", "flat": "unit", "room": "unit",
             "fl": "floor", "flr": "floor", "floor": "floor", "box": "pobox", "po box": "pobox"}
# bracketed words that are qualifiers, not trade names: "Gmax Automobiles (India) Pvt Ltd"
PAREN_NOISE = frozenset({"india", "usa", "us", "america", "france", "services", "service", "center", "centre", "group",
                         "international", "company", "co", "limited", "ltd", "private", "pvt", "holdings"})
SKELETON_RULES = [("x", "ks"), ("sh", "s"), ("ph", "f"), ("th", "t"), ("dh", "d"), ("bh", "b"), ("kh", "k"),
                  ("gh", "g"), ("aa", "a"), ("ee", "i"), ("oo", "u"), ("ou", "u"), ("w", "v"), ("y", "i"), ("z", "j")]


def fix_mojibake(s: str) -> str:
    """Drop stray \u00c2/\u00e2 left by UTF-8 text decoded as Latin-1 (\u00c2 before a space, or not after a letter)."""
    return re.sub(r"\s+", " ", MOJIBAKE_RE.sub("", s)).strip()


def fold(s: str) -> str:
    return anyascii(unicodedata.normalize("NFKC", fix_mojibake(s))).lower()


def _strip_legal(toks: list[str], lex: Lexicon) -> tuple[list[str], list[str]]:
    core, suffixes, i = [], [], 0
    while i < len(toks):
        for pat, canon in lex.legal_index.get(toks[i], ()):
            if tuple(toks[i:i + len(pat)]) == pat:
                suffixes.append(canon)
                i += len(pat)
                break
        else:
            core.append(toks[i])
            i += 1
    return core, suffixes


def _name_tokens(raw: str, lex: Lexicon) -> list[str]:
    s = fold(raw)
    s = re.sub(r"\bm/s\b", " ", s).replace("&", " and ")
    toks = [t for t in re.sub(r"[^a-z0-9]+", " ", s).split() if t != "null" and t not in lex.fillers]
    return [t for i, t in enumerate(toks) if i == 0 or t != toks[i - 1]]


def normalize_name(raw: str, lex: Lexicon) -> tuple[str, str, str]:
    toks = _name_tokens(raw, lex)
    core, suffixes = _strip_legal(toks, lex)
    core = [t for t in core if t not in lex.stopwords]
    name_norm = " ".join(toks)
    return name_norm, " ".join(core) or name_norm, "+".join(sorted(set(suffixes)))


def _canon_num(tok: str) -> str:
    return tok.replace("-", "").strip("/").lstrip("0") or "0"


_NUM_SIGN = re.compile(r"(?<![A-Za-z])[Nn]\s*[°º]\s*")  # French 'N°8' (anyascii would make it 'ndeg8' and lose the number)


def normalize_address(raw: str, country: str, lex: Lexicon) -> tuple[str, str, str]:
    if country in (lex.country_abbr or {}):
        raw = _NUM_SIGN.sub(" ", raw)
    for native, english in lex.native.items():
        if native in raw:
            raw = raw.replace(native, english)
    states = lex.states.get(country, {})
    comps = [c.strip() for c in fold(raw).split(",")]
    s = " , ".join(states.get(c, c) for c in comps if c and c != "null")
    out, nums, house_no, prev = [], [], "", ""
    for i, t in enumerate(ADDR_TOKEN_RE.findall(s)):
        if NUM_RE.fullmatch(t):
            c = _canon_num(t)
            nums.append(c)
            unit_marker = prev in UNIT_WORDS and not (prev == "#" and i == 1)  # leading '#' = house number
            if not house_no and not unit_marker and prev != "po" and not (c.isdigit() and len(c) == 6):
                house_no = c
            out.append(c)
        elif t != "#":
            out.extend(p for p in t.split("-") if p)
        prev = t
    abbr = (lex.country_abbr or {}).get(country, lex.abbreviations)
    out = [abbr.get(t, t) for t in out]
    out = [t for t in out if t not in lex.addr_fillers]
    return " ".join(out), house_no, " ".join(dict.fromkeys(nums))


def is_domain_name(raw: str) -> bool:
    f = fold(raw).strip()
    return len(f) >= 6 and ("." in f or f.startswith("#")) and DOMAIN_RE.fullmatch(f) is not None


def domain_body(raw: str) -> str:
    f = fold(raw).strip()
    f = re.sub(r"^#|^www\.", "", f)
    f = re.sub(r"(\.[a-z]{2,4})+$", "", f)
    return f.replace("-", "")


def segment(s: str, vocab: set[str], max_len: int = 20) -> list[str]:
    """Min-cost split of s into vocab words; digit runs are free words, unknown chunks cost 1+len."""
    n = len(s)
    best: list[tuple[float, int] | None] = [(0.0, 0)] + [None] * n
    for i in range(1, n + 1):
        for j in range(max(0, i - max_len), i):
            if best[j] is None:
                continue
            w = s[j:i]
            cost = best[j][0] + (1 if (w in vocab or w.isdigit()) else 1 + len(w))
            if best[i] is None or cost < best[i][0]:
                best[i] = (cost, j)
    out, i = [], n
    while i > 0:
        j = best[i][1]
        out.append(s[j:i])
        i = j
    return out[::-1]


def record_text(name: str, addr: str, country: str) -> str:
    return f"name: {name} | address: {addr} | country: {country}"


def trade_name_parts(raw: str, lex: Lexicon) -> str:
    """Core names of the parts of 'X dba Y' / 'X (Y)' names, joined by ' | '; '' when there is only one part."""
    outside = re.sub(r"\([^)]*\)", " ", raw)
    parens = [p for p in re.findall(r"\(([^)]*)\)", raw) if not set(fold(p).split()) <= PAREN_NOISE]
    parts, seen = DBA_RE.split(outside) + parens, []
    for part in parts:
        core, _ = _strip_legal(_name_tokens(part, lex), lex)
        core = " ".join(t for t in core if t not in lex.stopwords)
        if any(c.isalpha() for c in core) and core not in seen:
            seen.append(core)
    return " | ".join(seen) if len(seen) > 1 else ""


def extract_units(folded: str) -> str:
    """Unit / floor / PO box as sorted 'kind:value' tokens, so their presence or absence isn't an address mismatch.
    A leading '#' marks the house number (e.g. '#194, 8th block'), so only later '#'s count as units."""
    folded = folded.strip()
    found = {f"floor:{_canon_num(m.group(1))}" for m in ORD_FLOOR_RE.finditer(folded)}
    for m in UNIT_RE.finditer(folded):
        kind = "pobox" if "box" in m.group(1) else UNIT_KIND[m.group(1)]
        found.add(f"{kind}:{_canon_num(m.group(2))}")
    found |= {f"unit:{_canon_num(m.group(1))}" for m in HASH_UNIT_RE.finditer(folded) if m.start() > 0}
    return " ".join(sorted(found))


def address_extras(raw: str, country: str, lex: Lexicon) -> tuple[str, str, str, str]:
    """(landmark, addr_core, postal, unit). addr_core is addr_norm computed with landmark phrases removed."""
    landmarks = [re.sub(r"[^a-z0-9]+", " ", fold(m)).strip() for m in LANDMARK_RE.findall(raw)]
    addr_core = normalize_address(LANDMARK_RE.sub("", raw), country, lex)[0]
    folded = fold(raw)
    matches = list(lex.postal.get(country, DEFAULT_POSTAL_RE).finditer(folded))
    postal = re.sub(r"\s", "", matches[-1].group()) if matches else ""
    return " ; ".join(landmarks), addr_core, postal, extract_units(folded)


def sorted_name(core: str) -> str:
    return " ".join(sorted(core.split()))


def phonetic_key(core: str) -> str:
    return " ".join(jellyfish.metaphone(t) if t.isalpha() else t for t in core.split())


def skeleton_key(core: str) -> str:
    """Spelling skeleton for transliteration variants: digraph folding, no doubled letters, no inner vowels."""
    out = []
    for t in core.split():
        if not t.isalpha():
            out.append(t)
            continue
        for a, b in SKELETON_RULES:
            t = t.replace(a, b)
        t = re.sub(r"(.)\1+", r"\1", t)
        out.append(t[0] + re.sub(r"[aeiou]", "", t[1:]))
    return " ".join(out)


_STATE_NAMES: dict[str, frozenset[str]] = {}


def extract_state(raw: str, country: str, lex: Lexicon) -> str:
    """Full state name found as an address component (abbreviations and native script mapped); '' if none
    or if the country has no state lexicon (e.g. France). Abbreviations win; otherwise the last full name."""
    abbr = lex.states.get(country)
    if not abbr:
        return ""
    names = _STATE_NAMES.setdefault(country, frozenset(abbr.values()))
    for native, english in lex.native.items():
        if native in raw:
            raw = raw.replace(native, english)
    # An abbreviation component (", TX,") is unambiguous; a full name can also be a city ("Washington, UT"),
    # so among full-name matches take the last one (the state follows the city in these addresses).
    comps = [c.strip() for c in fold(raw).split(",")]
    abbr_hits = [abbr[c] for c in comps if c in abbr]
    if abbr_hits:
        return abbr_hits[-1]
    full_hits = [c for c in comps if c in names]
    if full_hits:
        return full_hits[-1]
    city = (lex.city_state or {}).get(country, {})
    city_hits = [city[c] for c in comps if c in city]
    return city_hits[-1] if city_hits else ""
