"""Precision / recall at every pipeline stage (OOF), the F0.5 headroom of every error source, and France vs US/India
diagnostics on test.  usage: python -u scripts/stage_pr.py  -> artifacts/v2/stage_pr/report.md"""
import json
import re

import lightgbm as lgb
import numpy as np
import pandas as pd

from ber.config import load_config, run_dir
from ber.io import read_truth

cfg = load_config("configs/v2.yaml")
RD = run_dir(cfg)
OUT = RD / "stage_pr"
OUT.mkdir(exist_ok=True)
md = []
say = lambda s="": (md.append(s), print(s, flush=True))
BASE = ["p", "rr", "has_rr", "rr_rank", "rr_gap", "p_rank", "p_gap", "n_cands"]


def tbl(df):
    df = df.reset_index() if not isinstance(df.index, pd.RangeIndex) else df
    rows = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    for r in df.itertuples(index=False):
        rows.append("| " + " | ".join(f"{v:.4f}" if isinstance(v, float) else (f"{v:,}" if isinstance(v, (int, np.integer)) else str(v)) for v in r) + " |")
    return "\n".join(rows)


f = pd.read_parquet(RD / "cand_side" / "oof_q.parquet", columns=["s1_id", "cand_id", "label", "fold", "cand_noaddr", "s1_name_cnt"] + BASE + ["q"])
f["q9"] = pd.read_parquet(RD / "cand_side_v9" / "oof_q9.parquet", columns=["q9"]).q9.to_numpy()
f["q10"] = pd.read_parquet(RD / "clean_v10a" / "oof_q10.parquet", columns=["q10"]).q10.to_numpy()
v5 = np.zeros(len(f), np.float32)
for k in range(5):
    m = lgb.Booster(model_file=str(RD / "reranker_v3" / f"stack_fold{k}.txt"))
    va = f.fold.to_numpy() == k
    v5[va] = m.predict(f.loc[va, BASE], num_iteration=m.best_iteration)
f["v5"] = v5
truth = read_truth(cfg["paths"]["data_dir"])
nt = pd.Series({s: len(truth[s]) for s in f.s1_id.unique()})
ctry = pd.read_parquet(RD / "splits.parquet", columns=["s1_id", "country"]).set_index("s1_id").country.reindex(nt.index).to_numpy()
lab = f.label.to_numpy().astype(bool)
n_true_all = int(nt.sum())
code = pd.factorize(f.s1_id)[0]
s1_order = pd.Index(f.s1_id.unique())
assert (s1_order == nt.index).all()


def decide(col, tau):
    s = f.loc[f[col].to_numpy() >= tau, ["cand_id", col]]
    m = np.zeros(len(f), bool)
    m[s.groupby("cand_id")[col].idxmax().to_numpy()] = True
    return m


def per_s1(pred, extra_tp=None):
    tp = np.bincount(code, weights=(pred & lab), minlength=len(nt))
    npred = np.bincount(code, weights=pred, minlength=len(nt))
    if extra_tp is not None:
        tp, npred = tp + extra_tp, npred + extra_tp
    n = nt.to_numpy()
    return np.where(n == 0, (npred == 0) * 1.0, 1.25 * tp / np.maximum(0.25 * n + npred, 1e-9))


def row(name, pred):
    sc = per_s1(pred)
    tp, npred = int((pred & lab).sum()), int(pred.sum())
    r = {"stage": name, "pair precision": tp / max(npred, 1), "pair recall": tp / n_true_all, "macro F0.5": sc.mean(),
         "wrong accepts": npred - tp, "missed": n_true_all - tp}
    for c in ["US", "India"]:
        r[f"F0.5 {c}"] = sc[ctry == c].mean()
    r["singletons"] = sc[nt.to_numpy() == 0].mean()
    return r, sc


say("# Precision and recall at every stage (OOF, 150k B-split S1, 518,414 true pairs)")
say()
rows, scores = [], {}
rows.append({"stage": "blocking (candidates)", "pair precision": lab.mean(), "pair recall": lab.sum() / n_true_all,
             "macro F0.5": float("nan"), "wrong accepts": int((~lab).sum()), "missed": n_true_all - int(lab.sum())})
for name, col, tau in [("Model A (LightGBM)", "p", 0.70), ("+ bge reranker (v5)", "v5", 0.65), ("+ candidate-side (v8)", "q", 0.75),
                       ("+ Model-A competition (v9)", "q9", 0.70), ("+ cleaned names (v10a)", "q10", 0.75)]:
    r, sc = row(name, decide(col, tau))
    rows.append(r)
    scores[name] = sc
say(tbl(pd.DataFrame(rows)))
say()
say("Pair precision = accepted pairs that are true; pair recall = true pairs found (blocking misses count as not found).")
say()

# ------------------------------------------------ headroom of each error source (on v9)
p9 = decide("q9", 0.70)
base = per_s1(p9).mean()
rec = pd.read_parquet(RD / "records_train.parquet", columns=["entity_id", "name_core", "house_no"]).set_index("entity_id")
fn = ~p9 & lab
wa = p9 & ~lab
a_core, c_core = rec.name_core.reindex(f.s1_id[fn]).fillna("").to_numpy(), rec.name_core.reindex(f.cand_id[fn]).fillna("").to_numpy()
noword = np.array([not (set(x.split()) & set(y.split())) for x, y in zip(a_core, c_core)])
na = f.cand_noaddr.to_numpy()[fn] == 1
same = f.s1_name_cnt.to_numpy()[fn] >= 2
fn_idx = np.flatnonzero(fn)
groups = {"no address, name unique (<=1 S1 with that core name)": fn_idx[na & ~same],
          "no address, 2+ S1s share the core name": fn_idx[na & same],
          "names share no word (gibberish name)": fn_idx[~na & noword],
          "other misses (typos, digit typos, script ...)": fn_idx[~na & ~noword]}
heads = []
for g, idx in groups.items():
    add = np.zeros(len(f), bool)
    add[idx] = True
    heads.append({"source": f"find all: {g}", "pairs": len(idx), "F0.5 if fixed": per_s1(p9 | add).mean()})
heads.append({"source": "remove all wrong accepts", "pairs": int(wa.sum()), "F0.5 if fixed": per_s1(p9 & ~wa).mean()})
extra = (nt.to_numpy() - np.bincount(code, weights=lab, minlength=len(nt)))
heads.append({"source": "blocking: every true pair in the shortlist AND found", "pairs": int(extra.sum()),
              "F0.5 if fixed": per_s1(p9, extra_tp=extra).mean()})
h = pd.DataFrame(heads)
h["max gain"] = h["F0.5 if fixed"] - base
say(f"# Where the remaining F0.5 is (v9 OOF {base:.4f}; each row alone)")
say()
say(tbl(h))
say()

# ------------------------------------------------ France on test (no labels)
rtest = pd.read_parquet(RD / "records_test.parquet", columns=["entity_id", "source", "country", "addr_raw"])
cmap = rtest.set_index("entity_id").country
s1 = rtest[rtest.source == 1]
say("# Test: France vs US / India (no labels; v9 test scores)")
say()
say("S1 share by country: " + ", ".join(f"{k} {v:.1%}" for k, v in s1.country.value_counts(normalize=True).items()))
oth = rtest[rtest.source != 1]
say("S2/S3 empty-address share: " + ", ".join(f"{k} {v:.1%}" for k, v in
    oth.addr_raw.fillna("").str.strip().eq("").groupby(oth.country).mean().items()))
q = pd.read_parquet(RD / "cand_side_v9" / "test_q9.parquet")
sc_ = q.p.to_numpy()
top = q.groupby("s1_id", sort=False).p.max()
nc = q.groupby("s1_id", sort=False).size()
hi = pd.Series(sc_ >= 0.70).groupby(q.s1_id.to_numpy()).sum()
mid = pd.Series((sc_ >= 0.2) & (sc_ < 0.70)).groupby(q.s1_id.to_numpy()).sum()
d = pd.DataFrame({"top": top, "n": nc}).join(hi.rename("hi")).join(mid.rename("mid"))
d["country"] = cmap.reindex(d.index).to_numpy()
t = d.groupby("country").agg(candidates=("n", "mean"), accepted=("hi", "mean"), uncertain_pairs=("mid", "mean"),
                             median_top_score=("top", "median"), no_confident_match=("top", lambda x: float((x < 0.7).mean())))
say(tbl(t))
say()
lb = {"v2": (0.9707, 0.962), "v3": (0.9784, 0.973), "v5": (0.9834, 0.979), "v8": (0.9885, 0.98299)}
fr = float((s1.country == "France").mean())
say(f"Implied France score if US/India score on the leaderboard what they score OOF (France = {fr:.1%} of test S1):")
say()
say(tbl(pd.DataFrame([{"version": k, "OOF (US+India)": o, "LB": l, "implied France": (l - (1 - fr) * o) / fr} for k, (o, l) in lb.items()])))
(OUT / "report.md").write_text("\n".join(md) + "\n")
