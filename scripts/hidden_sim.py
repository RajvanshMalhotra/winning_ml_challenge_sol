"""Test-like 'hidden owner' simulation. In test only ~60% of S2/S3 records have their owner in S1 (74% in train), so the
competition features (who else wants this record?) see more uncontested decoys on test than in OOF.
Hide a random HIDE share of train S1s that are NOT in the OOF sample, recompute the v8 + v9 competition features without
them, then (1) score the existing v12 stacker on this test-like frame, (2) retrain the stacker on it.
Writes artifacts/v2/v14/{frame_hidden.parquet, v14_fold*.txt, best.json}.   usage: hidden_sim.py [HIDE=0.2]"""
import json
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

HIDE = float(sys.argv[1]) if len(sys.argv) > 1 else 0.2
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0
sys.argv = ["cand_side_v9.py", "none"]
G = {"__name__": "v9"}
exec(open("scripts/cand_side_v9.py").read().rsplit("\nif sys.argv[1] ==", 1)[0], G)
ns, cfg, RD, log = G["ns"], G["cfg"], G["RD"], G["log"]
BASE, NEW, V9, PARAMS = G["BASE"], G["NEW"], G["V9"], G["PARAMS"]
from ber.io import read_truth
from ber.matcher import decide, evaluate
from ber.normalize import records_path

OUT = RD / ("v14" if SEED == 0 else f"v14_s{SEED}")
OUT.mkdir(exist_ok=True)
rec = pd.read_parquet(records_path(cfg, "train"), columns=["entity_id", "source"])
f = pd.read_parquet(RD / "cand_side" / "oof_q.parquet")
sample = set(f.s1_id.unique())
s1_ids = rec.entity_id[(rec.source == 1).to_numpy()]
pool = s1_ids[~s1_ids.isin(sample)]
hid_ids = pool.sample(frac=HIDE, random_state=SEED)
HID = np.zeros(len(rec), bool)
HID[pd.Index(rec.entity_id).get_indexer(hid_ids)] = True
log(f"hiding {HID.sum():,} of {len(s1_ids):,} train S1s (none of the {len(sample):,} OOF S1s)")

orig_ct = ns["comp_table"]
def ct_h(s, c, sim):
    m = ~HID[s]
    return orig_ct(s[m], c[m], sim[m])
ns["comp_table"] = ct_h
G["comp_table"] = ct_h
orig_ap = G["all_pairs"]
def ap_h(fam, R):
    s, c, p, focus = orig_ap(fam, R)
    keep = ~HID[s]
    assert keep[focus].all()
    idx = np.cumsum(keep) - 1
    return s[keep], c[keep], p[keep], idx[focus]
G["all_pairs"] = ap_h

# v8 candidate-side features without the hidden S1s
R = G["Records"]("train")
src = rec.source.to_numpy()
R.s1cnt = np.bincount(R.nk[(src == 1) & ~HID], minlength=R.nk.max() + 1).astype(np.int32)
f = f.drop(columns=[c for c in NEW if c in f])
f = G["add_new"](f, R, G["comp_tables"]("train", R))
log("v8 features recomputed")
v = G["v9_features"]("train")
assert len(v) == len(f)
for col in V9:
    f[col] = v[col].to_numpy()
del v
log("v9 features recomputed")
q = pd.read_parquet(RD / "reranker_qwen2" / "oof_qwen.parquet").rename(columns={"rr2": "rr"})
f = f.drop(columns=["rr"]).merge(q, on=["s1_id", "cand_id"], how="left")
g = f.groupby("s1_id").rr
f["has_rr"] = f.rr.notna().astype(np.int8)
f["rr_rank"] = g.rank(ascending=False, method="first").fillna(99).astype(np.int16)
f["rr_gap"] = (g.transform("max") - f.rr).astype(np.float32)
FEATS = BASE + NEW + V9
f.to_parquet(OUT / "frame_hidden.parquet", index=False)

truth = read_truth(cfg["paths"]["data_dir"])
sp = pd.read_parquet(RD / "splits.parquet", columns=["s1_id", "country"]).set_index("s1_id")
s1 = sp.loc[f.s1_id.unique()].reset_index()[["s1_id", "country"]]
old = np.zeros(len(f), np.float32)
new = np.zeros(len(f), np.float32)
for k in sorted(f.fold.unique()):
    va, tr = f.fold.to_numpy() == k, f.fold.to_numpy() != k
    m = lgb.Booster(model_file=str(RD / "v12" / f"v12_fold{k}.txt"))
    old[va] = m.predict(f.loc[va, FEATS], num_iteration=m.best_iteration)
    m2 = lgb.train({**PARAMS, "seed": SEED, "bagging_seed": SEED, "feature_fraction_seed": SEED}, lgb.Dataset(f.loc[tr, FEATS], f.label[tr]), 3000,
                   valid_sets=[lgb.Dataset(f.loc[va, FEATS], f.label[va])], callbacks=[lgb.early_stopping(100, verbose=False)])
    new[va] = m2.predict(f.loc[va, FEATS], num_iteration=m2.best_iteration)
    m2.save_model(str(OUT / f"v14_fold{k}.txt"))
    log(f"fold {k} done")
res = {}
for name, p in (("v12 stacker on test-like (hidden) frame", old), ("v14 stacker retrained on hidden frame", new)):
    for tau in (0.7, 0.75, 0.8):
        e = evaluate(decide(pd.DataFrame({"s1_id": f.s1_id, "cand_id": f.cand_id, "p": p}), tau), truth, s1)
        res[f"{name} @ {tau}"] = e.get("macro_f05")
        print(f"{name} tau {tau}: macro F0.5 {e.get('macro_f05'):.5f} (singletons {e.get('singletons'):.4f})", flush=True)
(OUT / "best.json").write_text(json.dumps(res, indent=2))
