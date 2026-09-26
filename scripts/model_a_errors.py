"""Where does Model A lose F0.5? Splits the loss into wrong merges, missed-by-model, missed-by-blocking, and
singleton false merges, with examples. usage: model_a_errors.py [tau]"""
import sys

import numpy as np
import pandas as pd

from ber.evaluate import f05
from ber.io import read_truth

TAU = float(sys.argv[1]) if len(sys.argv) > 1 else 0.70
OOF = sys.argv[2] if len(sys.argv) > 2 else "artifacts/v2/matcher/oof.parquet"
o = pd.read_parquet(OOF)
rec = pd.read_parquet("artifacts/v2/records_train.parquet", columns=["entity_id", "country", "name_raw", "addr_raw"]).set_index("entity_id")
truth = read_truth("data/student_resource/dataset")
s = o[o.p >= TAU]
s = s.loc[s.groupby("cand_id").p.idxmax()]
pred = s.groupby("s1_id").cand_id.agg(set).to_dict()
s1s = o.s1_id.unique()
cands = o.groupby("s1_id").cand_id.agg(set).to_dict()
rows = []
for x in s1s:
    t, p, c = truth[x], pred.get(x, set()), cands.get(x, set())
    rows.append((x, rec.country.get(x), len(t), len(p & t), len(p - t), len((t & c) - p), len(t - c), f05(p, t)))
d = pd.DataFrame(rows, columns=["s1", "country", "n_true", "tp", "fp", "fn_model", "fn_blocking", "f05"])
d["loss"] = 1 - d.f05
print(f"tau={TAU}: macro F0.5 {d.f05.mean():.4f} over {len(d):,} S1s | total loss {d.loss.sum():,.0f}")
single = d.n_true == 0
print(f"  singletons: {single.mean():.1%} of S1s, their F0.5 {d.f05[single].mean():.4f}, share of all loss {d.loss[single].sum() / d.loss.sum():.1%}")
nz = d[~single]
print("  non-singletons: entities with >=1 wrong merge", f"{(nz.fp > 0).mean():.1%}", "| with >=1 model miss", f"{(nz.fn_model > 0).mean():.1%}",
      "| with >=1 blocking miss", f"{(nz.fn_blocking > 0).mean():.1%}")
print("  pair totals: TP", int(d.tp.sum()), "FP", int(d.fp.sum()), "FN(model)", int(d.fn_model.sum()), "FN(blocking)", int(d.fn_blocking.sum()))
print("  pair precision", f"{d.tp.sum() / max(d.tp.sum() + d.fp.sum(), 1):.4f}", "| pair recall", f"{d.tp.sum() / max(d.n_true.sum(), 1):.4f}")
print("  by country:", d.groupby("country").f05.mean().round(4).to_dict())
def show(label, mask_pairs, n=6):
    print(f"\n== {label}")
    for r in mask_pairs.sample(min(n, len(mask_pairs)), random_state=0).itertuples():
        print(f"  p={r.p:.2f}  S1: {rec.name_raw[r.s1_id]} | {rec.addr_raw[r.s1_id][:70]}")
        print(f"          C : {rec.name_raw[r.cand_id]} | {rec.addr_raw[r.cand_id][:70]}")
fp = s[~s.label.astype(bool)]
fn = o[o.label.astype(bool) & (o.p < TAU)]
show("wrong merges (predicted, not true) — highest-confidence ones", fp.nlargest(200, "p"))
show("true matches the model rejected (p < tau)", fn)
show("singleton false merges", s[s.s1_id.isin(d.s1[single]) ])

# loss decomposition: how much F0.5 would we gain by fixing each error type alone (per S1, recompute F0.5)?
def f05_from(tp, fp, n_true):
    if n_true == 0:
        return 1.0 if fp == 0 else 0.0
    if tp == 0:
        return 0.0
    p, r = tp / (tp + fp), tp / n_true
    return 1.25 * p * r / (0.25 * p + r)
base = d.f05.mean()
fix_fp = np.mean([f05_from(t, 0, n) for t, n in zip(d.tp, d.n_true)])
fix_model = np.mean([f05_from(t + m, f, n) for t, f, m, n in zip(d.tp, d.fp, d.fn_model, d.n_true)])
fix_block = np.mean([f05_from(t + b, f, n) for t, f, b, n in zip(d.tp, d.fp, d.fn_blocking, d.n_true)])
print(f"\n== potential gains (fix one error type completely): wrong merges +{fix_fp - base:.4f} | "
      f"model-rejected true matches +{fix_model - base:.4f} | blocking misses +{fix_block - base:.4f}")
for c, g in d.groupby("country"):
    print(f"   {c}: F0.5 {g.f05.mean():.4f} | fix FP +{np.mean([f05_from(t, 0, n) for t, n in zip(g.tp, g.n_true)]) - g.f05.mean():.4f}"
          f" | fix model-FN +{np.mean([f05_from(t + m, f, n) for t, f, m, n in zip(g.tp, g.fp, g.fn_model, g.n_true)]) - g.f05.mean():.4f}"
          f" | fix blocking-FN +{np.mean([f05_from(t + b, f, n) for t, f, b, n in zip(g.tp, g.fp, g.fn_blocking, g.n_true)]) - g.f05.mean():.4f}")
