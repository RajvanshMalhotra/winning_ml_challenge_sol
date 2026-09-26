"""Where does v8 (v5 + candidate-side stacker) still lose F0.5? Rebuilds the OOF features, predicts with the saved
fold models, saves artifacts/v2/cand_side/oof_q.parquet and prints the loss split + causes of misses / wrong accepts."""
import sys
sys.argv = ["cand_side.py", "train"]
ns = {"__name__": "cs"}
exec(open("scripts/cand_side.py").read().rsplit("\n{", 1)[0], ns)
import numpy as np, pandas as pd, lightgbm as lgb
from ber.io import read_truth
from ber.matcher import decide
RD, RR, OUT, cfg = ns["RD"], ns["RR"], ns["OUT"], ns["cfg"]
o = pd.read_parquet(RD / ns["A_DIR"] / "oof.parquet")
b = pd.read_parquet(RR / "oof_band_scores.parquet")[["s1_id", "cand_id", "rr"]]
f = ns["stack_features"](o.merge(b, on=["s1_id", "cand_id"], how="left"))
R = ns["Records"]("train")
f = ns["add_new"](f, R, ns["comp_tables"]("train", R))
sp = pd.read_parquet(RD / "splits.parquet", columns=["s1_id", "country", "fold"]).set_index("s1_id")
f["fold"] = sp.fold.reindex(f.s1_id).to_numpy()
F = ns["BASE"] + ns["NEW"]
q = np.zeros(len(f), np.float32)
for k in range(5):
    m = lgb.Booster(model_file=str(OUT / f"cs_fold{k}.txt")); va = f.fold.to_numpy() == k
    q[va] = m.predict(f.loc[va, F], num_iteration=m.best_iteration)
f["q"] = q
f.to_parquet(OUT / "oof_q.parquet", index=False)
pred = decide(f[["s1_id", "cand_id"]].assign(p=f.q), 0.75)
truth = read_truth(cfg["paths"]["data_dir"])
s1s = f.s1_id.unique(); n = len(s1s)
pool = set(zip(f.s1_id, f.cand_id))
def f05(P, T):
    if not T: return 1.0 if not P else 0.0
    tp = len(P & T)
    if tp == 0: return 0.0
    pr, rc = tp / len(P), tp / len(T); return 1.25 * pr * rc / (0.25 * pr + rc)
sc = {s: f05(pred.get(s, set()), truth[s]) for s in s1s}
# what-if: perfect on one error type at a time
single = {s for s in s1s if not truth[s]}
no_fp = {s: f05(pred.get(s, set()) & truth[s], truth[s]) for s in s1s}
no_fn_pool = {s: f05(pred.get(s, set()) | {c for c in truth[s] if (s, c) in pool}, truth[s]) for s in s1s}
no_fn_all = {s: f05(pred.get(s, set()) | truth[s], truth[s]) for s in s1s}
m0 = np.mean(list(sc.values()))
print(f"v8 OOF macro F0.5 {m0:.4f}  (loss {1 - m0:.4f})")
print(f"  if no wrong accepts at all        -> {np.mean(list(no_fp.values())):.4f}")
print(f"  if every shortlisted match found  -> {np.mean(list(no_fn_pool.values())):.4f}")
print(f"  if every true match found (+blocking) -> {np.mean(list(no_fn_all.values())):.4f}")
print(f"  loss from singletons {sum(1 - sc[s] for s in single) / n:.4f} ({sum(sc[s] < 1 for s in single):,} of {len(single):,} singletons wrong)")
f["pred"] = [c in pred.get(s, set()) for s, c in zip(f.s1_id, f.cand_id)]
y = f.label.astype(bool)
fn, fp = f[y & ~f.pred].copy(), f[~y & f.pred].copy()
nb = sum(1 for s in s1s for c in truth[s] if (s, c) not in pool)
print(f"pairs: wrong accepts {len(fp):,} | missed in shortlist {len(fn):,} | missed by blocking {nb:,}")
rec = pd.read_parquet(RD / "records_train.parquet", columns=["entity_id", "name_raw", "addr_raw", "name_core", "addr_norm", "house_no"]).set_index("entity_id")
jac = lambda x, y: len(set(x.split()) & set(y.split())) / max(len(set(x.split()) | set(y.split())), 1)
def cause(d):
    a, c = rec.reindex(d.s1_id), rec.reindex(d.cand_id)
    native = c.name_raw.str.contains("[ऀ-෿]", regex=True).values
    nov = np.array([jac(x, y) for x, y in zip(a.name_core, c.name_core)])
    aov = np.array([jac(x, y) for x, y in zip(a.addr_norm, c.addr_norm)])
    hd = (a.house_no.values != "") & (c.house_no.values != "") & (a.house_no.values != c.house_no.values)
    return np.select([d.cand_noaddr.values == 1, native, nov == 0, hd, aov < 0.2],
                     ["no address", "Indian script", "names share no word", "house no. differs", "address very different"],
                     "similar text")
fn["cause"], fp["cause"] = cause(fn), cause(fp)
fn["zone"] = pd.cut(fn.p, [-1, 0.01, 0.99, 2], labels=["A<0.01 (not reranked)", "band (reranked)", "A>0.99"])
print("\nmissed matches by cause x zone:\n" + pd.crosstab(fn.cause, fn.zone, margins=True).to_string())
fp["single"] = [not truth[s] for s in fp.s1_id]
fp["other_owner"] = [any(c in truth[t] for t in [owner] if t) for c, owner in zip(fp.cand_id, fp.cand_id.map({c: s for s, m in truth.items() for c in m}))]
print("\nwrong accepts by cause (and whether the record truly belongs to another S1):\n" + pd.crosstab(fp.cause, fp.other_owner, margins=True).to_string())
print(f"wrong accepts on singletons: {fp.single.sum():,}")
print("\nfinal score of missed matches:", fn.q.quantile([.25, .5, .75, .9]).round(3).to_dict())
print("missed: S1 true-match count distribution:", pd.Series([len(truth[s]) for s in fn.s1_id]).value_counts().sort_index().head(8).to_dict())
