"""Two ideas for v9's no-address misses, tested on OOF before building anything.
A) sister-record signal: does this S1 already hold a confident match from the OTHER source whose name equals / nearly
   equals the no-address record's name? (label separation only, own S1)
B) no-address assignment rule: accept a no-address pair when the S1 is Model A's #1 for the record with margin >= m and
   v9 score >= t2 (on top of the normal cut-off). m, t2 chosen on 4 folds, scored on the 5th (nested CV)."""
import json, time
import numpy as np, pandas as pd
from rapidfuzz import fuzz
from ber.config import load_config, run_dir
from ber.io import read_truth
cfg = load_config("configs/v2.yaml"); RD = run_dir(cfg)
T0 = time.perf_counter(); log = lambda *a: print(f"[{time.perf_counter() - T0:5.0f}s]", *a, flush=True)
f = pd.read_parquet(RD / "cand_side" / "oof_q.parquet", columns=["s1_id", "cand_id", "p", "label", "cand_noaddr", "fold"])
f["q9"] = pd.read_parquet(RD / "cand_side_v9" / "oof_q9.parquet", columns=["q9"]).q9.to_numpy()
v = pd.read_parquet(RD / "cand_side_v9" / "feats_train.parquet", columns=["a_crank", "a_cmargin", "sib_name_max", "sib_crank"])
for c in v: f[c] = v[c].to_numpy()
del v
tau = json.loads((RD / "cand_side_v9" / "best.json").read_text())["tau"]
truth = read_truth(cfg["paths"]["data_dir"])
s1s = f.s1_id.unique(); ntrue = pd.Series({s: len(truth[s]) for s in s1s})
fold_of = f.groupby("s1_id").fold.first().reindex(ntrue.index).to_numpy()
lab = f.label.to_numpy().astype(bool); na = f.cand_noaddr.to_numpy() == 1
log(f"loaded, tau {tau}")

# ---------------- A) sister-record signal
rec = pd.read_parquet(RD / "records_train.parquet", columns=["entity_id", "name_norm"]).set_index("entity_id").name_norm
g = f[na & (f.p.to_numpy() >= 0.01)][["s1_id", "cand_id", "label", "sib_name_max", "sib_crank"]].copy()
conf = f[f.p.to_numpy() >= 0.9][["s1_id", "cand_id"]].rename(columns={"cand_id": "c2"})
x = g.reset_index().merge(conf, on="s1_id")
x = x[(x.cand_id != x.c2) & (x.cand_id.str[:2] != x.c2.str[:2])]      # sister = confident match from the other source
n1, n2 = rec.reindex(x.cand_id).fillna("").to_numpy(), rec.reindex(x.c2).fillna("").to_numpy()
x["exact"] = n1 == n2
x["ratio"] = [fuzz.ratio(a, b) for a, b in zip(n1, n2)]
agg = x.groupby("index").agg(sis_exact=("exact", "max"), sis_ratio=("ratio", "max"))
g = g.join(agg, how="left")
g["sis_exact"] = g.sis_exact.fillna(False).astype(bool); g["has_sister"] = g.sis_ratio.notna()
print("\nA) no-address pairs with p>=0.01 (own S1 only):", len(g), "true share", round(g.label.mean(), 3))
print(g.groupby("label").agg(n=("label", "size"), has_sister=("has_sister", "mean"), sister_exact_name=("sis_exact", "mean"),
                             sister_ratio_med=("sis_ratio", "median"), sib_name_max_med=("sib_name_max", "median"),
                             sib_crank1=("sib_crank", lambda s: (s == 1).mean())).round(3).T.to_string())
print("true-pair rate when sister name is exact:", round(g[g.sis_exact].label.mean(), 3), "| when not:", round(g[~g.sis_exact].label.mean(), 3))
still = g.index.isin(np.flatnonzero(lab & na & (f.q9.to_numpy() < tau)))
print("among v9-missed no-address true pairs: sister exact", round(g[still].sis_exact.mean(), 3), "| among v9 wrong-side (label 0) pairs:", round(g[g.label == 0].sis_exact.mean(), 3))

# ---------------- B) no-address assignment rule (nested CV)
def f05_s1(score):
    s = f.loc[score >= tau, ["cand_id"]].assign(sc=score[score >= tau])
    keep = s.groupby("cand_id").sc.idxmax().to_numpy()
    pred = np.zeros(len(f), bool); pred[keep] = True
    t = pd.DataFrame({"s1_id": f.s1_id, "tp": pred & lab, "np": pred}).groupby("s1_id")[["tp", "np"]].sum().reindex(ntrue.index, fill_value=0)
    nt = ntrue.to_numpy()
    return np.where(nt == 0, (t.np.to_numpy() == 0).astype(float), 1.25 * t.tp.to_numpy() / np.maximum(0.25 * nt + t.np.to_numpy(), 1e-9))
q9 = f.q9.to_numpy(); r1 = f.a_crank.to_numpy() == 1; mg = np.nan_to_num(f.a_cmargin.to_numpy(), nan=1.0)
res = {("none", "-"): f05_s1(q9)}
for m in [-1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
    for t2 in [0.3, 0.4, 0.5, 0.6]:
        rule = na & (q9 >= t2) & ((m < 0) | (r1 & (mg >= m)))
        res[(m, t2)] = f05_s1(np.where(rule, np.maximum(q9, tau), q9))
log(f"{len(res)} variants scored")
tab = pd.DataFrame({k: v.mean() for k, v in res.items()}.items(), columns=["variant", "F0.5"]).sort_values("F0.5", ascending=False)
print("\nB) all-data F0.5 by variant (m=-1 means no margin condition, i.e. a separate cut-off t2 for no-address):")
print(tab.head(8).to_string(index=False))
out = np.zeros(len(ntrue)); picks = []
for k in range(5):
    tr, te = fold_of != k, fold_of == k
    best = max(res, key=lambda kk: res[kk][tr].mean())
    picks.append(best); out[te] = res[best][te]
print(f"\nnested CV: v9 {res[('none', '-')].mean():.5f} -> with rule {out.mean():.5f} (gain {out.mean() - res[('none', '-')].mean():+.5f}); picks per fold {picks}")
