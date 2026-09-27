"""Validate the STAGE-2 RECOVERY plan on v12 OOF (labels known, same folds).
Pool = pairs v12 rejects (q12 < τ) whose record v12 left unassigned. Measures (1) pool size / true pairs, (2) oracle
ceilings under the plan's own abstain rule, (3) a stage-2 LightGBM with the plan's features, T/M chosen on 4 folds and
scored on the 5th (nested), vs the v12-only baseline.   usage: CUDA_VISIBLE_DEVICES= python -u scripts/stage2_check.py"""
import json
import os
import re
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from anyascii import anyascii
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein
from rapidfuzz.process import cpdist
from sklearn.feature_extraction.text import TfidfVectorizer

from ber.config import load_config, run_dir
from ber.io import read_truth

cfg = load_config("configs/v2.yaml"); RD = run_dir(cfg); OUT = RD / "stage2_check"; OUT.mkdir(exist_ok=True)
T0 = time.perf_counter(); log = lambda *a: print(f"[{time.perf_counter() - T0:6.0f}s]", *a, flush=True)
W = 24
TAU = json.loads((RD / "v12" / "best.json").read_text())["v12"]["tau"]
f = pd.read_parquet(RD / "v12" / "oof_q12.parquet")
truth = read_truth(cfg["paths"]["data_dir"])
s1 = pd.Index(f.s1_id.unique()); nt = np.array([len(truth[s]) for s in s1]); code = s1.get_indexer(f.s1_id)
fold_s1 = f.groupby("s1_id").fold.first().reindex(s1).to_numpy()
lab = f.label.to_numpy().astype(bool); q = f.q12.to_numpy()
own = np.zeros(len(f), bool); own[f[q >= TAU].groupby("cand_id").q12.idxmax().to_numpy()] = True
acc12 = own & (q >= TAU)


def per_s1(pred):
    tp = np.bincount(code, weights=pred & lab, minlength=len(s1)); npred = np.bincount(code, weights=pred, minlength=len(s1))
    return np.where(nt == 0, (npred == 0) * 1.0, 1.25 * tp / np.maximum(0.25 * nt + npred, 1e-9))


def second_max(key: pd.Series, val: pd.Series) -> np.ndarray:
    """per-row: the 2nd-largest value of val within its key group (0 if the group has one row)"""
    d = pd.DataFrame({"k": key.to_numpy(), "v": val.to_numpy(), "i": np.arange(len(key))}).sort_values(["k", "v"], ascending=[True, False])
    first = ~d.k.duplicated().to_numpy()
    nxt = np.r_[d.v.to_numpy()[1:], 0.0]
    same_next = np.r_[d.k.to_numpy()[1:] == d.k.to_numpy()[:-1], False]
    sec_at_first = np.where(first & same_next, nxt, 0.0)
    grp_sec = pd.Series(sec_at_first).groupby(np.cumsum(first)).transform("max").to_numpy()
    out = np.empty(len(key)); out[d.i.to_numpy()] = grp_sec
    return out


base = per_s1(acc12)
assigned = set(f.cand_id[acc12])
pool_m = ~acc12 & ~f.cand_id.isin(assigned).to_numpy()
n_true_all = int(nt.sum()); n_block = n_true_all - int(lab.sum())
log(f"v12 τ={TAU}: OOF F0.5 {base.mean():.5f}; missed in shortlist {int((lab & ~acc12).sum()):,}, blocking {n_block:,}")
log(f"pool (rejected, record unassigned): {int(pool_m.sum()):,} pairs, {int((pool_m & lab).sum()):,} true; "
    f"true misses whose record went to ANOTHER S1 (out of scope): {int((lab & ~acc12 & ~pool_m).sum()):,}")
for lo in [0.001, 0.01, 0.05]:
    m = pool_m & (f.q.to_numpy() >= 0) & (q >= lo)
    log(f"  pool with q12 >= {lo}: {int(m.sum()):,} pairs, {int((m & lab).sum()):,} true")
P = f[pool_m & (q >= 0.001)].copy()
log(f"working pool (q12 >= 0.001): {len(P):,} pairs, {int(P.label.sum()):,} true")

# ---------------- plan features
rec = pd.read_parquet(RD / "records_train.parquet", columns=["entity_id", "source", "country", "name_raw", "addr_raw", "legal_suffix",
                                                              "house_no", "addr_norm"]).set_index("entity_id")
LEGAL = {"ltd", "limited", "pvt", "private", "llc", "lp", "llp", "inc", "incorporated", "corp", "corporation", "partners", "co", "company", "pc", "plc"}
PREFIX = {"the", "dr", "shri", "sri", "shree", "smt", "m", "s", "ms", "mr", "mrs"}
LEET = str.maketrans({"0": "o", "1": "l", "3": "e", "5": "s", "8": "b"})


def norm_name(x):
    x = anyascii(x or "").lower()
    x = re.sub(r"\(?\bid\s*[:#.]?\s*\d+\)?", " ", x)
    x = re.sub(r"(.)\1+", r"\1", x)
    toks = [t.translate(LEET) if re.search(r"[a-z]", t) and re.search(r"\d", t) else t for t in re.findall(r"[a-z0-9]+", x)]
    while toks and toks[0] in PREFIX:
        toks.pop(0)
    toks = [t for i, t in enumerate(toks) if i == 0 or t != toks[i - 1]]
    core = [t for t in toks if t not in LEGAL]
    legal = " ".join(sorted({t for t in toks if t in LEGAL}))
    return " ".join(core or toks), legal


ids = pd.Index(pd.unique(np.r_[P.s1_id.to_numpy(), P.cand_id.to_numpy()]))
nm = [norm_name(x) for x in rec.name_raw.reindex(ids).to_numpy()]
core = pd.Series([a for a, _ in nm], index=ids); legal = pd.Series([b for _, b in nm], index=ids)
s1r = rec[rec.source == 1]
s1core = pd.Series([norm_name(x)[0] for x in s1r.name_raw.to_numpy()], index=s1r.index)
core_cnt = pd.Series(s1core.groupby([s1r.country.to_numpy(), s1core.to_numpy()]).size())
log("names normalized")
ca, cb = core.reindex(P.s1_id).to_numpy(), core.reindex(P.cand_id).to_numpy()
vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), min_df=2, dtype=np.float32).fit(pd.unique(np.r_[ca, cb]))
A, B = vec.transform(ca), vec.transform(cb)
P["name_cos"] = np.asarray(A.multiply(B).sum(axis=1)).ravel()
for k, sc in [("tsort", fuzz.token_sort_ratio), ("tset", fuzz.token_set_ratio), ("partial", fuzz.partial_ratio)]:
    P[k] = cpdist(ca, cb, scorer=sc, workers=W)
idf = {}
for t, c in pd.Series(" ".join(s1core.to_numpy()).split()).value_counts().items():
    idf[t] = np.log(len(s1core) / c)


def idf_cov(a, b):
    ta, tb = a.split(), b.split()
    if not ta or not tb:
        return 0.0, 0.0
    w = [idf.get(t, 12.0) for t in ta]
    hit = [max((fuzz.ratio(t, u) for u in tb), default=0) >= 85 for t in ta]
    rare = max(ta, key=lambda t: idf.get(t, 12.0))
    return sum(x for x, h in zip(w, hit) if h) / sum(w), float(max(fuzz.ratio(rare, u) for u in tb) >= 85)


cov = [idf_cov(a, b) for a, b in zip(ca, cb)]
P["idf_cov"], P["rare_found"] = [c[0] for c in cov], [c[1] for c in cov]
la, lb = legal.reindex(P.s1_id).to_numpy(), legal.reindex(P.cand_id).to_numpy()
P["legal"] = np.where((la == "") | (lb == ""), 0, np.where(la == lb, 1, 2)).astype(np.int8)
raw_b = rec.name_raw.reindex(P.cand_id).fillna("").to_numpy()
init = np.array(["".join(w[0] for w in a.split()) for a in ca])
dom = np.array([re.sub(r"\.(com|in|net|org|co)\b.*", "", x.lower()) if re.search(r"\.(com|in|net|org|co)\b", x.lower()) else "" for x in raw_b])
compact = np.array([a.replace(" ", "") for a in ca])
P["acro_dom"] = np.array([bool(d) and (d == c or d.startswith(i) and len(i) >= 2 or c.endswith(d[-6:])) for d, c, i in zip(dom, compact, init)]
                         ) | np.array([re.fullmatch(r"[A-Z]{2,6}", x.strip()) is not None and x.strip().lower() == i for x, i in zip(raw_b, init)])
aa, ab = rec.addr_norm.reindex(P.s1_id).fillna("").to_numpy(), rec.addr_norm.reindex(P.cand_id).fillna("").to_numpy()
P["has_addr"] = ab != ""
vec2 = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), min_df=2, dtype=np.float32).fit(pd.unique(np.r_[aa, ab]))
P["addr_cos"] = np.where(P.has_addr, np.asarray(vec2.transform(aa).multiply(vec2.transform(ab)).sum(axis=1)).ravel(), np.nan)
ha = np.array([h.lstrip("0") for h in rec.house_no.reindex(P.s1_id).fillna("").to_numpy()])
hb = np.array([h.lstrip("0") for h in rec.house_no.reindex(P.cand_id).fillna("").to_numpy()])
both = (ha != "") & (hb != "")
num = lambda s: int(re.match(r"\d+", s).group()) if re.match(r"\d+", s) else -99
P["house_exact"] = np.where(both, (ha == hb).astype(np.int8), -1)
P["house_near"] = np.where(both, np.array([x != y and (Levenshtein.distance(x, y) == 1 or x.startswith(y) or y.startswith(x) or abs(num(x) - num(y)) <= 2)
                                           for x, y in zip(ha, hb)]).astype(np.int8), -1)
P["nonlatin"] = np.array([bool(re.search(r"[^\x00-\x7f]", x)) and anyascii(x) != x for x in raw_b])
cty = rec.country.reindex(P.s1_id).to_numpy()
P["same_core_n"] = (core_cnt.reindex(pd.MultiIndex.from_arrays([cty, cb])).fillna(0).to_numpy()
                    - (ca == cb)).astype(np.int16)  # other S1s with the record's core name
log("pair features done")
# record-level context over the whole pool (all S1 candidates of the record)
g = P.groupby("cand_id")
P["r_rank"] = g.q12.rank(ascending=False, method="first")
top = g.q12.transform("max"); sec = second_max(P.cand_id, P.q12)
P["r_margin"] = np.where(P.r_rank == 1, P.q12 - sec, P.q12 - top)
P["r_share"] = P.q12 / g.q12.transform("sum").clip(lower=1e-6)
P["r_ncand"] = g.q12.transform("size")
best_cos = g.name_cos.transform("max")
P["n_twins"] = (P.name_cos >= best_cos - 0.02).groupby(P.cand_id).transform("sum")
FEATS = ["q12", "q9", "name_cos", "tsort", "tset", "partial", "idf_cov", "rare_found", "legal", "acro_dom", "addr_cos", "has_addr",
         "house_exact", "house_near", "nonlatin", "same_core_n", "r_rank", "r_margin", "r_share", "r_ncand", "n_twins"]
for c in ["acro_dom", "has_addr", "nonlatin"]:
    P[c] = P[c].astype(np.int8)
log("context done; training stage-2")

# ---------------- stage-2 model, same folds
PARAMS = {"objective": "binary", "metric": "auc", "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 100,
          "feature_fraction": 0.9, "verbose": -1, "num_threads": W}
s2 = np.zeros(len(P), np.float32); imp = []
for k in range(5):
    tr, va = P.fold.to_numpy() != k, P.fold.to_numpy() == k
    m = lgb.train(PARAMS, lgb.Dataset(P.loc[tr, FEATS], P.label[tr]), 3000, valid_sets=[lgb.Dataset(P.loc[va, FEATS], P.label[va])],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    s2[va] = m.predict(P.loc[va, FEATS], num_iteration=m.best_iteration)
    imp.append(pd.Series(m.feature_importance("gain"), index=FEATS))
P["s2"] = s2
from sklearn.metrics import roc_auc_score
log(f"stage-2 OOF AUC {roc_auc_score(P.label, s2):.4f} (q12 alone {roc_auc_score(P.label, P.q12):.4f})")
gi = pd.concat(imp, axis=1).mean(axis=1); gi = (gi / gi.sum() * 100).sort_values(ascending=False).round(1)
log("gain %: " + ", ".join(f"{k} {v}" for k, v in gi.head(12).items()))

# ---------------- decisions: top stage-2 candidate per record, abstain rule, T/M
g = P.groupby("cand_id").s2
P["s2_top"] = P.s2 >= g.transform("max")
sec2 = second_max(P.cand_id, P.s2)
P["s2_margin"] = P.s2 - sec2
abstain = (P.has_addr == 0) & ((P.same_core_n > 0) | (P.n_twins > 1))
idx = P.index.to_numpy()


def score(T, M, use_abstain=True, col="s2"):
    sel = P.s2_top.to_numpy() & (P[col].to_numpy() >= T) & (P.s2_margin.to_numpy() >= M)
    if use_abstain:
        sel &= ~abstain.to_numpy()
    add = np.zeros(len(f), bool); add[idx[sel]] = True
    return per_s1(acc12 | add), int(sel.sum()), int((sel & P.label.to_numpy().astype(bool)).sum())


# oracle ceilings
for nm_, m in [("all true pool pairs", P.label.to_numpy().astype(bool)),
               ("true pool pairs NOT abstained", P.label.to_numpy().astype(bool) & ~abstain.to_numpy())]:
    add = np.zeros(len(f), bool); add[idx[m]] = True
    log(f"ceiling if stage-2 found every {nm_}: {per_s1(acc12 | add).mean():.5f} (+{per_s1(acc12 | add).mean() - base.mean():.5f}), {int(m.sum()):,} pairs")
log(f"abstain rule removes {int((abstain & (P.label == 1)).sum()):,} true and {int((abstain & (P.label == 0)).sum()):,} false pool pairs")
grid = {}
for T in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]:
    for M in [0, 0.05, 0.1, 0.2]:
        grid[(T, M)] = score(T, M)
rows = [{"T": k[0], "M": k[1], "F0.5 (full OOF)": v[0].mean(), "added": v[1], "added true": v[2]} for k, v in grid.items()]
tab = pd.DataFrame(rows).sort_values("F0.5 (full OOF)", ascending=False)
print(tab.head(10).round(5).to_string(index=False), flush=True)
out, picks = np.zeros(len(s1)), []
for k in range(5):
    tr, te = fold_s1 != k, fold_s1 == k
    b = max(grid, key=lambda kk: grid[kk][0][tr].mean()); picks.append(b)
    out[te] = grid[b][0][te]
d = out - base
rng = np.random.default_rng(0); bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(500)]
log(f"v12 baseline {base.mean():.5f} | picked on the same OOF (plan step 6): {tab.iloc[0]['F0.5 (full OOF)']:.5f} | "
    f"nested (T,M chosen on 4 folds): {out.mean():.5f}, gain {d.mean():+.5f} [95% CI {np.percentile(bs, 2.5):+.5f}, {np.percentile(bs, 97.5):+.5f}]; picks {picks}")
na = score(*max(grid, key=lambda kk: grid[kk][0].mean()), use_abstain=False)
log(f"same best (T,M) without the abstain rule: {na[0].mean():.5f}, added {na[1]:,} ({na[2]:,} true)")
P.to_parquet(OUT / "pool.parquet", index=False)
