"""User's idea on v12-rejected pairs: normalize -> one script (Latin) -> spelling correction against the S1-name vocabulary
-> compare -> accept above a threshold. Evaluated on ALL rejected pairs (true and false), one owner per record, thresholds
chosen on 4 folds and scored on the 5th.   usage: CUDA_VISIBLE_DEVICES= python -u scripts/spellfix_check.py"""
import json
import re
import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from anyascii import anyascii
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein
from rapidfuzz.process import cpdist

from ber.config import load_config, run_dir
from ber.io import read_truth

cfg = load_config("configs/v2.yaml"); RD = run_dir(cfg)
T0 = time.perf_counter(); log = lambda *a: print(f"[{time.perf_counter() - T0:6.0f}s]", *a, flush=True)
W = 24
tau = json.loads((RD / "v12" / "best.json").read_text())["v12"]["tau"]
f = pd.read_parquet(RD / "v12" / "oof_q12.parquet", columns=["s1_id", "cand_id", "label", "fold", "q12"])
truth = read_truth(cfg["paths"]["data_dir"])
s1 = pd.Index(f.s1_id.unique()); nt = np.array([len(truth[s]) for s in s1]); code = s1.get_indexer(f.s1_id)
fold_s1 = f.groupby("s1_id").fold.first().reindex(s1).to_numpy()
lab = f.label.to_numpy().astype(bool); q = f.q12.to_numpy()
acc = np.zeros(len(f), bool); acc[f[q >= tau].groupby("cand_id").q12.idxmax().to_numpy()] = True


def per_s1(pred):
    tp = np.bincount(code, weights=pred & lab, minlength=len(s1)); n = np.bincount(code, weights=pred, minlength=len(s1))
    return np.where(nt == 0, (n == 0) * 1.0, 1.25 * tp / np.maximum(0.25 * nt + n, 1e-9))


base = per_s1(acc)
rej = ~acc & ~f.cand_id.isin(set(f.cand_id[acc])).to_numpy() & (q >= 0.001)
P = f[rej].copy(); idx = np.flatnonzero(rej)
log(f"v12 OOF {base.mean():.5f}; rejected pool (record unassigned, v12 >= 0.001): {len(P):,} pairs, {int(P.label.sum()):,} true")

rec = pd.read_parquet(RD / "records_train.parquet", columns=["entity_id", "source", "country", "name_raw"]).set_index("entity_id")
LEGAL = {"ltd", "limited", "pvt", "private", "llc", "lp", "llp", "inc", "incorporated", "corp", "corporation", "co", "company", "pc",
         "plc", "the", "and", "of", "group", "services", "enterprises"}
LEET = str.maketrans({"0": "o", "1": "l", "3": "e", "5": "s", "8": "b"})


def base_norm(x):  # one script (anyascii: Indic/accents -> Latin), lower, no punctuation/brackets/ID tags, look-alike digits
    x = anyascii(x or "").lower()
    x = re.sub(r"\(?\bid\s*[:#.]?\s*\d+\)?", " ", x)
    toks = [t.translate(LEET) if re.search(r"[a-z]", t) and re.search(r"\d", t) else t for t in re.findall(r"[a-z0-9]+", x)]
    return [t for t in toks if t not in LEGAL]


# spelling vocabulary = tokens of all S1 names (the clean reference source), per country
s1r = rec[rec.source == 1]
vocab = {c: Counter(t for n in g.name_raw for t in base_norm(n)) for c, g in s1r.groupby("country")}
dele = {}
for c, v in vocab.items():
    d = defaultdict(list)
    for w in v:
        if len(w) >= 4:
            for i in range(len(w)):
                d[w[:i] + w[i + 1:]].append(w)
            d[w].append(w)
    dele[c] = d
log(f"vocabulary: {({c: len(v) for c, v in vocab.items()})}")
cache = {}


def correct(tok, c):
    v = vocab[c]
    if len(tok) < 4 or tok.isdigit() or v.get(tok, 0) >= 3:
        return tok
    key = (tok, c)
    if key in cache:
        return cache[key]
    cands = set(dele[c].get(tok, []))
    for i in range(len(tok)):
        cands.update(dele[c].get(tok[:i] + tok[i + 1:], []))
    cands = [w for w in cands if Levenshtein.distance(w, tok) <= (1 if len(tok) < 8 else 2)]
    best = max(cands, key=lambda w: v[w]) if cands else tok
    if best != tok and v.get(tok, 0) >= v[best]:
        best = tok
    cache[key] = best
    return best


ids = pd.Index(pd.unique(np.r_[P.s1_id.to_numpy(), P.cand_id.to_numpy()]))
cty = rec.country.reindex(ids).to_numpy()
raw = [base_norm(n) for n in rec.name_raw.reindex(ids).to_numpy()]
norm = pd.Series([" ".join(t) for t in raw], index=ids)
fixed = pd.Series([" ".join(correct(t, c) for t in toks) for toks, c in zip(raw, cty)], index=ids)
changed = (norm != fixed)
log(f"spelling correction changed {changed.mean():.1%} of the names in the pool")
ex = ids[changed.to_numpy()][:8]
for i in ex:
    print(f"   {rec.name_raw[i]!r} -> {fixed[i]!r}")
a, b = fixed.reindex(P.s1_id).to_numpy(), fixed.reindex(P.cand_id).to_numpy()
P["exact"] = (a == b).astype(float) * 100
P["tset"] = cpdist(a, b, scorer=fuzz.token_set_ratio, workers=W)
P["tsort"] = cpdist(a, b, scorer=fuzz.token_sort_ratio, workers=W)
P["jw"] = 100 * cpdist(a, b, scorer=JaroWinkler.normalized_similarity, workers=W)
P["sim"] = (P.tset + P.tsort + P.jw) / 3
# one owner: keep the record's best S1 by the combined similarity (tie -> v12 score)
P["r"] = P.sim + 0.001 * P.q12
P["top"] = P.r >= P.groupby("cand_id").r.transform("max")
srt = P.sort_values("r", ascending=False)[["cand_id", "r"]]
srt["rk"] = srt.groupby("cand_id").cumcount()
sec = srt[srt.rk == 1].set_index("cand_id").r
P["margin"] = P.r - P.cand_id.map(sec).fillna(0).to_numpy()
log("similarities done")
L = P.label.to_numpy().astype(bool)
print("\nhow often a rejected pair is really a match, by corrected-name similarity (record's best S1 only):")
rows = []
for s_ in [80, 90, 95, 99, 100]:
    for m_ in [0, 5, 20]:
        sel = P.top.to_numpy() & (P.sim.to_numpy() >= s_) & (P.margin.to_numpy() >= m_)
        rows.append({"similarity >=": s_, "margin >=": m_, "pairs": int(sel.sum()), "true": int((sel & L).sum()),
                     "precision": (sel & L).sum() / max(sel.sum(), 1)})
print(pd.DataFrame(rows).round(3).to_string(index=False), flush=True)
grid = {}
for s_ in [85, 90, 95, 99, 100]:
    for m_ in [0, 5, 10, 20, 40]:
        for q_ in [0.001, 0.05, 0.2, 0.4]:
            sel = P.top.to_numpy() & (P.sim.to_numpy() >= s_) & (P.margin.to_numpy() >= m_) & (P.q12.to_numpy() >= q_)
            add = np.zeros(len(f), bool); add[idx[sel]] = True
            grid[(s_, m_, q_)] = (per_s1(acc | add), int(sel.sum()), int((sel & L).sum()))
bestk = max(grid, key=lambda k: grid[k][0].mean())
out, picks = np.zeros(len(s1)), []
for k in range(5):
    tr, te = fold_s1 != k, fold_s1 == k
    b = max(list(grid) + [None], key=lambda kk: (grid[kk][0][tr].mean() if kk else base[tr].mean()))
    picks.append(b); out[te] = (grid[b][0] if b else base)[te]
g = grid[bestk]
print(f"\nbest threshold on all data: similarity>={bestk[0]}, margin>={bestk[1]}, v12>={bestk[2]}: F0.5 {g[0].mean():.5f} "
      f"({g[0].mean() - base.mean():+.5f}), adds {g[1]:,} pairs of which {g[2]:,} true", flush=True)
print(f"thresholds chosen on 4 folds, scored on the 5th: {out.mean():.5f} ({out.mean() - base.mean():+.5f}); picks {picks}", flush=True)
