"""Targeted recovery for v12-rejected pairs whose record has NO address or is written in an Indian script:
count the S1s (all train S1s of the country that have the record as a candidate) whose normalized name matches the
record's name (similarity >= T). count == 1 -> accept that S1; count == 2 -> accept the one with the higher Model A p if the
p-margin >= m; count > 2 -> skip. Evaluated on all such rejected pairs (true and false), one owner per record,
thresholds chosen on 4 folds and scored on the 5th.   usage: CUDA_VISIBLE_DEVICES= python -u scripts/unique_name_check.py"""
import json
import re
import time

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from anyascii import anyascii
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
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
rec = pd.read_parquet(RD / "records_train.parquet", columns=["entity_id", "source", "country", "name_raw", "addr_raw"]).set_index("entity_id")
noaddr = rec.addr_raw.fillna("").str.strip().eq("")
indic = rec.name_raw.fillna("").str.contains("[ऀ-෿]", regex=True)
pool = ~acc & ~f.cand_id.isin(set(f.cand_id[acc])).to_numpy()
grp = np.where(noaddr.reindex(f.cand_id).to_numpy(), "no address", np.where(indic.reindex(f.cand_id).to_numpy(), "Indian script", ""))
pool &= grp != ""
P = f[pool].copy(); P["group"] = grp[pool]; idx = np.flatnonzero(pool)
log(f"v12 OOF {base.mean():.5f}; target pool: {len(P):,} rejected pairs, {int(P.label.sum()):,} true "
    f"({P.groupby('group').label.agg(['size', 'sum']).to_dict('index')})")
# every S1 (all 2.2M train S1s) that has one of these records as a candidate, with Model A p
recs = P.cand_id.unique().tolist()
A = ds.dataset(RD / "train_scores_matcher_v3.parquet").to_table(filter=ds.field("cand_id").isin(recs)).to_pandas()
log(f"all-S1 candidate pairs of these {len(recs):,} records: {len(A):,}")
LEGAL = {"ltd", "limited", "pvt", "private", "llc", "lp", "llp", "inc", "incorporated", "corp", "corporation", "co", "company", "pc",
         "plc", "the"}
LEET = str.maketrans({"0": "o", "1": "l", "3": "e", "5": "s", "8": "b"})


def norm(x):  # one script (Indic/accents -> Latin), lower, no punctuation/brackets/ID tags/legal words, look-alike digits
    x = anyascii(x or "").lower()
    x = re.sub(r"\(?\bid\s*[:#.]?\s*\d+\)?", " ", x)
    toks = [t.translate(LEET) if re.search(r"[a-z]", t) and re.search(r"\d", t) else t for t in re.findall(r"[a-z0-9]+", x)]
    core = [t for t in toks if t not in LEGAL]
    return " ".join(core or toks)


ids = pd.Index(pd.unique(np.r_[A.s1_id.to_numpy(), A.cand_id.to_numpy()]))
nm = pd.Series([norm(x) for x in rec.name_raw.reindex(ids).to_numpy()], index=ids)
a, b = nm.reindex(A.s1_id).to_numpy(), nm.reindex(A.cand_id).to_numpy()
A["sim"] = (cpdist(a, b, scorer=fuzz.token_set_ratio, workers=W) + cpdist(a, b, scorer=fuzz.token_sort_ratio, workers=W)
            + 100 * cpdist(a, b, scorer=JaroWinkler.normalized_similarity, workers=W)) / 3
log("name similarities done")
P = P.merge(A[["s1_id", "cand_id", "sim", "p"]], on=["s1_id", "cand_id"], how="left")
assert len(P) == len(idx)
L = P.label.to_numpy().astype(bool)
res, prec_rows = {}, []
for T in [100, 97, 94, 90, 85]:
    hit = A[A.sim >= T]
    cnt = hit.groupby("cand_id").size()
    srt = hit.sort_values("p", ascending=False)
    srt["rk"] = srt.groupby("cand_id").cumcount()
    top_s1 = srt[srt.rk == 0].set_index("cand_id")
    p2 = srt[srt.rk == 1].set_index("cand_id").p
    n = P.cand_id.map(cnt).fillna(0).to_numpy()
    is_top = (P.s1_id.to_numpy() == P.cand_id.map(top_s1.s1_id).to_numpy())
    margin = P.p.to_numpy() - P.cand_id.map(p2).fillna(0).to_numpy()
    ok = P.sim.to_numpy() >= T
    for g in ["no address", "Indian script"]:
        gm = P.group.to_numpy() == g
        for k in [1, 2]:
            sel = gm & ok & (n == k) & is_top
            prec_rows.append({"group": g, "name sim >=": T, "S1s with that name": k, "pairs": int(sel.sum()), "true": int((sel & L).sum()),
                              "precision": (sel & L).sum() / max(sel.sum(), 1)})
    for use2 in [False, True]:
        for m in [0.0, 0.1, 0.3]:
            for qmin in [0.001, 0.05, 0.2]:
                sel = ok & is_top & (P.q12.to_numpy() >= qmin) & ((n == 1) | (use2 & (n == 2) & (margin >= m)))
                if not use2 and m > 0:
                    continue
                add = np.zeros(len(f), bool); add[idx[sel]] = True  # P rows are in pool order (left merge keeps it)
                res[(T, use2, m, qmin)] = (per_s1(acc | add), int(sel.sum()), int((sel & L).sum()))
pr = pd.DataFrame(prec_rows)
print("\nprecision of the rule (record's best-matching S1, among ALL rejected pairs of the group):")
print(pr.round(3).to_string(index=False), flush=True)
bk = max(res, key=lambda k: res[k][0].mean())
out, picks = np.zeros(len(s1)), []
for k in range(5):
    tr, te = fold_s1 != k, fold_s1 == k
    b_ = max(list(res) + [None], key=lambda kk: res[kk][0][tr].mean() if kk else base[tr].mean())
    picks.append(b_); out[te] = (res[b_][0] if b_ else base)[te]
r = res[bk]
print(f"\nbest on all data: sim>={bk[0]}, count 2 allowed={bk[1]}, margin>={bk[2]}, v12>={bk[3]}: F0.5 {r[0].mean():.5f} "
      f"({r[0].mean() - base.mean():+.5f}); adds {r[1]:,} pairs, {r[2]:,} true", flush=True)
d = out - base
rng = np.random.default_rng(0); bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(500)]
print(f"chosen on 4 folds, scored on the 5th: {out.mean():.5f} ({d.mean():+.5f}, 95% CI {np.percentile(bs, 2.5):+.5f} .. "
      f"{np.percentile(bs, 97.5):+.5f}); picks {picks}", flush=True)
