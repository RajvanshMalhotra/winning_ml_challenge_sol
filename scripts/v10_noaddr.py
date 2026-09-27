"""v10 (Qwen replaces bge) OOF: where are the no-address misses, and does a separate cut-off for no-address pairs help?
Saves artifacts/v2/reranker_qwen2/oof_v10.parquet (s1_id, cand_id, label, p, q10, cand_noaddr, ...)."""
import sys
sys.argv = ["x", "stack"]
ns = {"__name__": "x"}
exec(open("scripts/qwen_reranker.py").read().rsplit("\n{", 1)[0], ns)
import numpy as np, pandas as pd, lightgbm as lgb
from ber.io import read_truth
from ber.matcher import decide, evaluate
OUT, RD, cfg = ns["OUT"], ns["RD"], ns["cfg"]
cs = {}
exec(open("scripts/cand_side.py").read().rsplit("\n{", 1)[0], cs)
base = ns["oof_frame"]()
qall = pd.read_parquet(OUT / "oof_qwen.parquet")
r = base.drop(columns=["rr"]).merge(qall.rename(columns={"rr2": "rr"}), on=["s1_id", "cand_id"], how="left")
g = r.groupby("s1_id").rr
r["rr_rank"] = g.rank(ascending=False, method="first").fillna(99).astype(np.int16)
r["rr_gap"] = (g.transform("max") - r.rr).astype(np.float32)
F = cs["BASE"] + cs["NEW"]
q = np.zeros(len(r), np.float32)
for k in range(5):
    m = lgb.Booster(model_file=str(OUT / f"repl_fold{k}.txt")); va = r.fold.to_numpy() == k
    q[va] = m.predict(r.loc[va, F], num_iteration=m.best_iteration)
r["q10"] = q
r[["s1_id", "cand_id", "label", "p", "rr", "q10", "cand_noaddr", "fold", "s1_name_cnt", "d_crank", "n_crank", "name_exact"]].to_parquet(OUT / "oof_v10.parquet", index=False)
truth = read_truth(cfg["paths"]["data_dir"])
sp = pd.read_parquet(RD / "splits.parquet", columns=["s1_id", "country"]).set_index("s1_id")
s1 = sp.loc[r.s1_id.unique()].reset_index()[["s1_id", "country"]]
na = r.cand_noaddr.to_numpy() == 1
def run(tau_na, tau=0.75):
    p = np.where(na, np.clip(r.q10 * (tau / tau_na), 0, 1), r.q10)
    pred = decide(pd.DataFrame({"s1_id": r.s1_id, "cand_id": r.cand_id, "p": p}), tau)
    return pred, evaluate(pred, truth, s1)
pred, e = run(0.75)
print(f"v10 OOF macro F0.5 {e['macro_f05']:.4f}")
y = r.label.astype(bool).to_numpy()
r["pred"] = [c in pred.get(s, set()) for s, c in zip(r.s1_id, r.cand_id)]
t = r[y & na]
print(f"no-address true pairs {len(t):,}: found {t.pred.mean():.1%} | precision of no-address picks {r[na & r.pred].label.mean():.1%}")
miss = t[~t.pred]
zone = pd.cut(miss.p, [-1, 0.01, 0.99, 2], labels=["A<0.01 (never reranked)", "band (reranked)", "A>0.99"])
print("missed no-address by zone:", miss.groupby(zone, observed=False).size().to_dict())
print("missed no-address q10 quantiles:", miss.q10.quantile([.25, .5, .75, .9]).round(3).to_dict())
print("missed no-address: exact same core name", f"{miss.name_exact.mean():.1%}", "| unique S1 name", f"{(miss.s1_name_cnt == 1).mean():.1%}",
      "| d_crank==1", f"{(miss.d_crank == 1).mean():.1%}")
print("\nseparate cut-off for no-address pairs (others stay 0.75):")
for tna in [0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]:
    print(f"  tau_noaddr {tna:.2f} -> macro F0.5 {run(tna)[1]['macro_f05']:.4f}")
# ceiling: if every missed no-address pair below 0.01 were found (reranker never saw them)
low = set(zip(miss[miss.p < 0.01].s1_id, miss[miss.p < 0.01].cand_id))
pred2 = {s: v | {c for c in truth[s] if (s, c) in low} for s, v in pred.items()}
for s in s1.s1_id:
    pred2.setdefault(s, {c for c in truth[s] if (s, c) in low})
print(f"\nceiling if all no-address misses with Model A p<0.01 were found: {evaluate(pred2, truth, s1)['macro_f05']:.4f}")
allna = set(zip(miss.s1_id, miss.cand_id))
pred3 = {s: pred.get(s, set()) | {c for c in truth[s] if (s, c) in allna} for s in s1.s1_id}
print(f"ceiling if ALL no-address misses were found: {evaluate(pred3, truth, s1)['macro_f05']:.4f}")
