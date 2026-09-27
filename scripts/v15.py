"""v15 = hidden-owner stackers (3 hidden samples, v14 / v14_s1 / v14_s2 frames) + the further-trained Qwen reranker
(reranker_qwen3) in the rr column; test probability = mean of all 3 x 5 fold models.
usage: v15.py train  -> OOF on the seed-0 test-like frame: v14 (reference) vs v15 single seed vs v15 3-seed mean
       v15.py submit -> artifacts/submissions/v15 (small candidate set, validator)"""
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
src = open("scripts/v12_combo.py").read().rsplit("\n{\"train\"", 1)[0]
G = {"__name__": "v15"}
exec(src, G)
ns9 = G["ns"]
RD, log, BASE, NEW, V9, PARAMS = G["RD"], G["log"], G["BASE"], G["NEW"], G["V9"], G["PARAMS"]
from ber.io import read_truth
from ber.matcher import decide, evaluate
from ber.normalize import records_path
from ber.predict import write_submission

FEATS = BASE + NEW + V9
QR = RD / os.environ.get("QR_SRC", "reranker_qwen3")
SEEDS = [s for s in ["v14", "v14_s1", "v14_s2"] if (RD / s / "frame_hidden.parquet").exists()]
OUT = RD / "v15"
OUT.mkdir(exist_ok=True)


def with_qwen(f: pd.DataFrame, q: pd.DataFrame) -> pd.DataFrame:
    return G["qwen_rr"](f, q)


def train() -> None:
    q = pd.read_parquet(QR / "oof_qwen.parquet").rename(columns={"rr2": "rr"})
    ref = None
    preds = {}
    for sd in SEEDS:
        f = with_qwen(pd.read_parquet(RD / sd / "frame_hidden.parquet"), q)
        if ref is None:
            ref = f[["s1_id", "cand_id", "label", "fold"]].copy()
            ref_frame = f
        assert (f.cand_id.to_numpy() == ref.cand_id.to_numpy()).all()
        for k in sorted(f.fold.unique()):
            tr, va = f.fold.to_numpy() != k, f.fold.to_numpy() == k
            m = lgb.train(PARAMS, lgb.Dataset(f.loc[tr, FEATS], f.label[tr]), 3000,
                          valid_sets=[lgb.Dataset(f.loc[va, FEATS], f.label[va])], callbacks=[lgb.early_stopping(100, verbose=False)])
            m.save_model(str(OUT / f"{sd}_fold{k}.txt"))
            p = preds.setdefault(sd, np.zeros(len(f), np.float32))
            p[va] = m.predict(ref_frame.loc[va, FEATS], num_iteration=m.best_iteration)   # all evaluated on seed-0 frame
        log(f"{sd} trained")
    old = np.zeros(len(ref), np.float32)
    of = G["qwen_rr"](pd.read_parquet(RD / "v14" / "frame_hidden.parquet"),
                      pd.read_parquet(RD / "reranker_qwen2" / "oof_qwen.parquet").rename(columns={"rr2": "rr"}))
    for k in sorted(ref.fold.unique()):
        va = ref.fold.to_numpy() == k
        m = lgb.Booster(model_file=str(RD / "v14" / f"v14_fold{k}.txt"))
        old[va] = m.predict(of.loc[va, FEATS], num_iteration=m.best_iteration)
    del of
    truth = read_truth(G["cfg"]["paths"]["data_dir"])
    sp = pd.read_parquet(RD / "splits.parquet", columns=["s1_id", "country"]).set_index("s1_id")
    s1 = sp.loc[ref.s1_id.unique()].reset_index()[["s1_id", "country"]]
    res = {}
    cands = {"v14 (old Qwen, 1 seed)": old, "v15 new Qwen, seed 0": preds[SEEDS[0]], f"v15 new Qwen, mean of {len(SEEDS)} seeds": np.mean(list(preds.values()), axis=0)}
    for name, p in cands.items():
        for tau in (0.7, 0.75, 0.8):
            e = evaluate(decide(pd.DataFrame({"s1_id": ref.s1_id, "cand_id": ref.cand_id, "p": p}), tau), truth, s1)
            res[f"{name} @ {tau}"] = e.get("macro_f05")
            print(f"{name} tau {tau}: {e.get('macro_f05'):.5f} (singletons {e.get('singletons'):.4f})", flush=True)
    (OUT / "best.json").write_text(json.dumps(res, indent=2))


def submit() -> None:
    tau = float(os.environ.get("V15_TAU", 0.75))
    t = pd.read_parquet(ns9["SCORES"]["test"])
    b = pd.read_parquet(QR / "test_qwen.parquet").rename(columns={"rr2": "rr"})
    f = ns9["stack_features"](t.merge(b, on=["s1_id", "cand_id"], how="left"))
    del t
    R = ns9["Records"]("test")
    f = ns9["add_new"](f, R, ns9["comp_tables"]("test", R))
    v = pd.read_parquet(RD / "cand_side_v9" / "feats_test.parquet")
    assert len(v) == len(f)
    for col in V9:
        f[col] = v[col].to_numpy()
    del v
    f = f[f.p.to_numpy() >= 0.01].reset_index(drop=True)                      # learned filter = candidate set
    models = [lgb.Booster(model_file=str(p)) for p in sorted(OUT.glob("v14*_fold*.txt"))]
    log(f"{len(models)} models")
    p = np.mean([m.predict(f[FEATS], num_iteration=m.best_iteration) for m in models], axis=0)
    sc = pd.DataFrame({"s1_id": f.s1_id, "cand_id": f.cand_id, "p": p})
    sc.to_parquet(OUT / "test_final_scores.parquet", index=False)
    pred = decide(sc, tau)
    rec = pd.read_parquet(records_path(G["cfg"], "test"), columns=["entity_id", "source", "country"])
    s1_all = rec.entity_id[rec.source == 1].tolist()
    out = Path("artifacts/submissions/v15")
    write_submission(s1_all, pred, f.groupby("s1_id").cand_id.agg(set).to_dict(), out)
    n = pd.Series({s: len(v) for s, v in pred.items()}).reindex(s1_all, fill_value=0)
    ctry = rec.set_index("entity_id").country.reindex(n.index).to_numpy()
    log({"tau": tau, "mean": float(n.mean()), "empty": float((n == 0).mean()), **{c: float(n[ctry == c].mean()) for c in ["US", "India", "France"]}})
    v = subprocess.run(["python3", G["cfg"]["paths"]["validator"], "--matching", str(out / "matching_results.tsv"), "--candidate",
                        str(out / "candidate_pairs.tsv"), "--test-dir", str(Path(G["cfg"]["paths"]["data_dir"]) / "test")],
                       capture_output=True, text=True)
    print(v.stdout[-120:], f"validator exit={v.returncode}", flush=True)


{"train": train, "submit": submit}[ACTION]()
