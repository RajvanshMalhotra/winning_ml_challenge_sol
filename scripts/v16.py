"""v16 = small validated gains stacked on the test-like (hidden-owner) frames:
  rr      further-trained Qwen3-Reranker-4B (reranker_qwen3)          [v15]
  + rr_q2 previous Qwen (reranker_qwen2) and rr_bge (bge-reranker), and their disagreement with rr
  + stronger LightGBM settings
  + mean over the 3 hidden samples (v14, v14_s1, v14_s2)
usage: v16.py ablate   -> seed-0 test-like frame, one step at a time (5-fold OOF)
       v16.py final    -> train the chosen variant on every hidden frame; writes models + choice.json
       v16.py submit   -> artifacts/submissions/v16"""
import json
import os
import subprocess
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ACTION = sys.argv[1]
sys.argv = ["v12_combo.py", "none"]
G = {"__name__": "v16"}
exec(open("scripts/v12_combo.py").read().rsplit("\n{\"train\"", 1)[0], G)
ns9 = G["ns"]
RD, log, BASE, NEW, V9, PARAMS = G["RD"], G["log"], G["BASE"], G["NEW"], G["V9"], G["PARAMS"]
from ber.io import read_truth
from ber.matcher import decide, evaluate
from ber.normalize import records_path
from ber.predict import write_submission

OUT = RD / "v16"
OUT.mkdir(exist_ok=True)
SEEDS = [s for s in ["v14", "v14_s1", "v14_s2"] if (RD / s / "frame_hidden.parquet").exists()]
EXTRA = ["rr_q2", "rr_bge", "d_q2", "d_bge"]
STRONG = {**PARAMS, "num_leaves": 63, "learning_rate": 0.03, "min_child_samples": 100}
sig = lambda x: 1 / (1 + np.exp(-x))


def add_rr(f: pd.DataFrame, q3: pd.DataFrame, q2: pd.DataFrame, bge: pd.DataFrame) -> pd.DataFrame:
    f = G["qwen_rr"](f, q3)                                              # rr, has_rr, rr_rank, rr_gap from Qwen v13
    f = f.merge(q2.rename(columns={"rr2": "rr_q2"}), on=["s1_id", "cand_id"], how="left")
    f = f.merge(bge.rename(columns={"rr": "rr_bge"}), on=["s1_id", "cand_id"], how="left")
    f["d_q2"] = (sig(f.rr) - sig(f.rr_q2)).astype(np.float32)
    f["d_bge"] = (sig(f.rr) - f.rr_bge).astype(np.float32)
    return f


def oof_inputs():
    q3 = pd.read_parquet(RD / "reranker_qwen3" / "oof_qwen.parquet").rename(columns={"rr2": "rr"})
    q2 = pd.read_parquet(RD / "reranker_qwen2" / "oof_qwen.parquet")
    bge = pd.read_parquet(RD / "reranker_v3" / "oof_band_scores.parquet", columns=["s1_id", "cand_id", "rr"])
    return q3, q2, bge


def fit(f, feats, params, save=None):
    p = np.zeros(len(f), np.float32)
    for k in sorted(f.fold.unique()):
        tr, va = f.fold.to_numpy() != k, f.fold.to_numpy() == k
        m = lgb.train(params, lgb.Dataset(f.loc[tr, feats], f.label[tr]), 5000,
                      valid_sets=[lgb.Dataset(f.loc[va, feats], f.label[va])], callbacks=[lgb.early_stopping(150, verbose=False)])
        p[va] = m.predict(f.loc[va, feats], num_iteration=m.best_iteration)
        if save:
            m.save_model(str(OUT / f"{save}_fold{k}.txt"))
    return p


def score(f, p, truth, s1, taus=(0.7, 0.75, 0.8)):
    return {t: evaluate(decide(pd.DataFrame({"s1_id": f.s1_id, "cand_id": f.cand_id, "p": p}), t), truth, s1).get("macro_f05") for t in taus}


def setup():
    truth = read_truth(G["cfg"]["paths"]["data_dir"])
    sp = pd.read_parquet(RD / "splits.parquet", columns=["s1_id", "country"]).set_index("s1_id")
    return truth, sp


def ablate() -> None:
    truth, sp = setup()
    f = add_rr(pd.read_parquet(RD / "v14" / "frame_hidden.parquet"), *oof_inputs())
    s1 = sp.loc[f.s1_id.unique()].reset_index()[["s1_id", "country"]]
    base = BASE + NEW + V9
    steps = [("v15-like: new Qwen", base, PARAMS), ("+ both rerankers as columns", base + EXTRA, PARAMS),
             ("+ stronger LightGBM", base + EXTRA, STRONG)]
    res = {}
    for name, feats, params in steps:
        r = score(f, fit(f, feats, params), truth, s1)
        res[name] = r
        print(f"{name:32s} " + " | ".join(f"tau {t}: {v:.5f}" for t, v in r.items()), flush=True)
    (OUT / "ablation.json").write_text(json.dumps(res, indent=2))


def final() -> None:
    abl = json.loads((OUT / "ablation.json").read_text())
    best = max(abl, key=lambda n: max(abl[n].values()))
    feats = BASE + NEW + V9 + ([] if best.startswith("v15") else EXTRA)
    params = STRONG if "stronger" in best else PARAMS
    tau = float(max(abl[best], key=lambda t: abl[best][t]))
    truth, sp = setup()
    inp = oof_inputs()
    preds, ref = [], None
    for sd in SEEDS:
        f = add_rr(pd.read_parquet(RD / sd / "frame_hidden.parquet"), *inp)
        preds.append(fit(f, feats, params, save=sd))
        if ref is None:
            ref = f[["s1_id", "cand_id"]]
        log(f"{sd} done")
    s1 = sp.loc[ref.s1_id.unique()].reset_index()[["s1_id", "country"]]
    r = score(ref, np.mean(preds, axis=0), truth, s1)          # note: each seed's OOF on its own frame, averaged
    print(f"chosen: {best} | tau {tau} | mean of {len(SEEDS)} hidden samples: " + " | ".join(f"{t}: {v:.5f}" for t, v in r.items()), flush=True)
    (OUT / "choice.json").write_text(json.dumps({"variant": best, "extra": not best.startswith("v15"), "strong": "stronger" in best,
                                                 "tau": tau, "oof": r}, indent=2))


def submit() -> None:
    ch = json.loads((OUT / "choice.json").read_text())
    feats = BASE + NEW + V9 + (EXTRA if ch["extra"] else [])
    t = pd.read_parquet(ns9["SCORES"]["test"])
    q3 = pd.read_parquet(RD / "reranker_qwen3" / "test_qwen.parquet").rename(columns={"rr2": "rr"})
    f = ns9["stack_features"](t.merge(q3, on=["s1_id", "cand_id"], how="left"))
    del t
    R = ns9["Records"]("test")
    f = ns9["add_new"](f, R, ns9["comp_tables"]("test", R))
    v = pd.read_parquet(RD / "cand_side_v9" / "feats_test.parquet")
    assert len(v) == len(f)
    for col in V9:
        f[col] = v[col].to_numpy()
    del v
    f = f[f.p.to_numpy() >= 0.01].reset_index(drop=True)                      # learned filter = candidate set
    q2 = pd.read_parquet(RD / "reranker_qwen2" / "test_qwen.parquet")
    bge = pd.read_parquet(RD / "reranker_v3" / "test_band_scores.parquet", columns=["s1_id", "cand_id", "rr"])
    f = f.merge(q2.rename(columns={"rr2": "rr_q2"}), on=["s1_id", "cand_id"], how="left")
    f = f.merge(bge.rename(columns={"rr": "rr_bge"}), on=["s1_id", "cand_id"], how="left")
    f["d_q2"] = (sig(f.rr) - sig(f.rr_q2)).astype(np.float32)
    f["d_bge"] = (sig(f.rr) - f.rr_bge).astype(np.float32)
    models = [lgb.Booster(model_file=str(p)) for sd in SEEDS for p in sorted(OUT.glob(f"{sd}_fold*.txt"))]
    log(f"{len(models)} models, variant {ch['variant']}, tau {ch['tau']}")
    p = np.mean([m.predict(f[feats], num_iteration=m.best_iteration) for m in models], axis=0)
    sc = pd.DataFrame({"s1_id": f.s1_id, "cand_id": f.cand_id, "p": p})
    sc.to_parquet(OUT / "test_final_scores.parquet", index=False)
    pred = decide(sc, ch["tau"])
    rec = pd.read_parquet(records_path(G["cfg"], "test"), columns=["entity_id", "source", "country"])
    s1_all = rec.entity_id[rec.source == 1].tolist()
    out = Path("artifacts/submissions/v16")
    write_submission(s1_all, pred, f.groupby("s1_id").cand_id.agg(set).to_dict(), out)
    n = pd.Series({s: len(x) for s, x in pred.items()}).reindex(s1_all, fill_value=0)
    ctry = rec.set_index("entity_id").country.reindex(n.index).to_numpy()
    log({"mean": float(n.mean()), "empty": float((n == 0).mean()), **{c: float(n[ctry == c].mean()) for c in ["US", "India", "France"]}})
    vv = subprocess.run(["python3", G["cfg"]["paths"]["validator"], "--matching", str(out / "matching_results.tsv"), "--candidate",
                         str(out / "candidate_pairs.tsv"), "--test-dir", str(Path(G["cfg"]["paths"]["data_dir"]) / "test")],
                        capture_output=True, text=True)
    print(vv.stdout[-120:], f"validator exit={vv.returncode}", flush=True)


{"ablate": ablate, "final": final, "submit": submit}[ACTION]()
