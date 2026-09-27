"""OOF check: v12 features + BOTH rerankers (Qwen in rr, bge as an extra column rr_bge + disagreement)."""
import sys
sys.argv = ["v12_combo.py", "train"]
src = open("scripts/v12_combo.py").read().rsplit("\n{\"train\"", 1)[0]
ns = {"__name__": "v12"}
exec(src, ns)
import numpy as np, pandas as pd, lightgbm as lgb
RD, V9, BASE, NEW, PARAMS, log = ns["RD"], ns["V9"], ns["BASE"], ns["NEW"], ns["PARAMS"], ns["log"]
from ber.io import read_truth
f = pd.read_parquet(RD / "cand_side" / "oof_q.parquet")
v = pd.read_parquet(RD / "cand_side_v9" / "feats_train.parquet")
for col in V9:
    f[col] = v[col].to_numpy()
del v
f["rr_bge"] = f.rr.astype(np.float32)
f = ns["qwen_rr"](f, pd.read_parquet(RD / "reranker_qwen2" / "oof_qwen.parquet").rename(columns={"rr2": "rr"}))
f["rr_dis"] = (1 / (1 + np.exp(-f.rr)) - f.rr_bge).astype(np.float32)      # Qwen prob - bge prob
F = BASE + NEW + V9 + ["rr_bge", "rr_dis"]
q = np.zeros(len(f), np.float32)
for k in sorted(f.fold.unique()):
    tr, va = f.fold.to_numpy() != k, f.fold.to_numpy() == k
    m = lgb.train(PARAMS, lgb.Dataset(f.loc[tr, F], f.label[tr]), 3000,
                  valid_sets=[lgb.Dataset(f.loc[va, F], f.label[va])], callbacks=[lgb.early_stopping(100, verbose=False)])
    q[va] = m.predict(f.loc[va, F], num_iteration=m.best_iteration)
    m.save_model(str(ns["OUT"] / f"both_fold{k}.txt"))
    log(f"fold {k} done")
truth = read_truth(ns["cfg"]["paths"]["data_dir"])
sp = pd.read_parquet(RD / "splits.parquet", columns=["s1_id", "country"]).set_index("s1_id")
s1 = sp.loc[f.s1_id.unique()].reset_index()[["s1_id", "country"]]
q12 = pd.read_parquet(ns["OUT"] / "oof_q12.parquet", columns=["q12"]).q12.to_numpy()
grid = ns["ns"]["f05_grid"](f, {"v12": q12, "v12 + bge column": q}, truth, s1)
for n, g in grid.groupby("model"):
    b = g.sort_values("macro_f05").iloc[-1]
    print(f"{n}: best tau {b.tau:.2f} -> {b.macro_f05:.4f} | at 0.75: {g[g.tau == 0.75].macro_f05.iloc[0]:.4f}", flush=True)
