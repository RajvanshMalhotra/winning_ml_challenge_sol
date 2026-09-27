"""v12 = v9 stacker features (Model-A competition, cluster support, address competition; other session) + fine-tuned
Qwen3-Reranker-4B replacing bge in the rr column (v10). Same folds, same pairs as v8/v9/v10.
usage: v12_combo.py train | submit"""
import json
import os
import subprocess
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ACTION = sys.argv[1]
sys.argv = ["cand_side_v9.py", "none"]
ns = {"__name__": "v9"}
exec(open("scripts/cand_side_v9.py").read().rsplit("\nif sys.argv[1] ==", 1)[0], ns)
cfg, RD, log = ns["cfg"], ns["RD"], ns["log"]
BASE, NEW, V9, PARAMS = ns["BASE"], ns["NEW"], ns["V9"], ns["PARAMS"]
from ber.io import read_truth
from ber.matcher import decide, evaluate, md_table
from ber.normalize import records_path
from ber.predict import write_submission

V9D, QD = RD / "cand_side_v9", RD / "reranker_qwen2"
OUT = RD / "v12"
OUT.mkdir(exist_ok=True)
SUB = os.environ.get("SUB", "v12")
FEATS = BASE + NEW + V9


def qwen_rr(f: pd.DataFrame, q: pd.DataFrame) -> pd.DataFrame:
    f = f.drop(columns=["rr"]).merge(q, on=["s1_id", "cand_id"], how="left")
    f["has_rr"] = f.rr.notna().astype(np.int8)
    g = f.groupby("s1_id").rr
    f["rr_rank"] = g.rank(ascending=False, method="first").fillna(99).astype(np.int16)
    f["rr_gap"] = (g.transform("max") - f.rr).astype(np.float32)
    return f


def train() -> None:
    f = pd.read_parquet(RD / "cand_side" / "oof_q.parquet")
    v = pd.read_parquet(V9D / "feats_train.parquet")
    assert len(v) == len(f)
    for col in V9:
        f[col] = v[col].to_numpy()
    del v
    q9 = pd.read_parquet(V9D / "oof_q9.parquet", columns=["s1_id", "cand_id", "q9"])
    assert (q9.s1_id.to_numpy() == f.s1_id.to_numpy()).all()
    f["q9"] = q9.q9.to_numpy()
    f = qwen_rr(f, pd.read_parquet(QD / "oof_qwen.parquet").rename(columns={"rr2": "rr"}))
    q12 = np.zeros(len(f), np.float32)
    for k in sorted(f.fold.unique()):
        tr, va = f.fold.to_numpy() != k, f.fold.to_numpy() == k
        m = lgb.train(PARAMS, lgb.Dataset(f.loc[tr, FEATS], f.label[tr]), 3000,
                      valid_sets=[lgb.Dataset(f.loc[va, FEATS], f.label[va])], callbacks=[lgb.early_stopping(100, verbose=False)])
        q12[va] = m.predict(f.loc[va, FEATS], num_iteration=m.best_iteration)
        m.save_model(str(OUT / f"v12_fold{k}.txt"))
        log(f"fold {k} done ({m.best_iteration} rounds)")
    f["q12"] = q12
    f[["s1_id", "cand_id", "label", "fold", "cand_noaddr", "q", "q9", "q12"]].to_parquet(OUT / "oof_q12.parquet", index=False)
    truth = read_truth(cfg["paths"]["data_dir"])
    sp = pd.read_parquet(RD / "splits.parquet", columns=["s1_id", "country"]).set_index("s1_id")
    s1 = sp.loc[f.s1_id.unique()].reset_index()[["s1_id", "country"]]
    grid = ns["f05_grid"](f, {"v8": f.q.to_numpy(), "v9": f.q9.to_numpy(), "v12": q12}, truth, s1)
    best = {n: g.sort_values("macro_f05").iloc[-1] for n, g in grid.groupby("model")}
    (OUT / "best.json").write_text(json.dumps({n: {"tau": float(b.tau), "f05": float(b.macro_f05)} for n, b in best.items()}, indent=2))
    lab, na = f.label.to_numpy().astype(bool), f.cand_noaddr.to_numpy() == 1
    p12 = ns["mask"](f, "q12", float(best["v12"].tau))
    fmt = lambda r: (f"τ={r.tau:.2f} → **{r.macro_f05:.4f}** (singletons {r.singletons:.4f}, "
                     + ", ".join(f"{k[4:]} {v:.4f}" for k, v in r.items() if str(k).startswith("f05_")) + ")")
    md = ["# v12 = v9 features + fine-tuned Qwen3-Reranker-4B replacing bge — out-of-fold", "",
          *[f"- {n}: {fmt(best[n])}" for n in ["v8", "v9"]], f"- **v12: {fmt(best['v12'])}**", "",
          f"- v12 no-address true pairs found {(lab & na & p12).sum() / (lab & na).sum():.1%}, "
          f"precision {(lab & na & p12).sum() / max((na & p12).sum(), 1):.1%}; missed {int((lab & ~p12).sum()):,}, "
          f"wrong accepts {int((~lab & p12).sum()):,}", "", md_table(grid)]
    Path("docs/results/v12_combo.md").write_text("\n".join(md) + "\n")
    print("\n".join(md[:7]), flush=True)


def submit() -> None:
    best = json.loads((OUT / "best.json").read_text())["v12"]
    t = pd.read_parquet(ns["SCORES"]["test"])
    b = pd.read_parquet(QD / "test_qwen.parquet").rename(columns={"rr2": "rr"})
    f = ns["stack_features"](t.merge(b, on=["s1_id", "cand_id"], how="left"))
    del t
    R = ns["Records"]("test")
    f = ns["add_new"](f, R, ns["comp_tables"]("test", R))
    v = pd.read_parquet(V9D / "feats_test.parquet")
    assert len(v) == len(f)
    for col in V9:
        f[col] = v[col].to_numpy()
    del v
    models = [lgb.Booster(model_file=str(p)) for p in sorted(OUT.glob("v12_fold*.txt"))]
    p = np.mean([m.predict(f[FEATS], num_iteration=m.best_iteration) for m in models], axis=0)
    sc = pd.DataFrame({"s1_id": f.s1_id, "cand_id": f.cand_id, "p": p})
    sc.to_parquet(OUT / "test_q12.parquet", index=False)
    pred = decide(sc, best["tau"])
    rec = pd.read_parquet(records_path(cfg, "test"), columns=["entity_id", "source", "country"])
    s1_all = rec.entity_id[rec.source == 1].tolist()
    out = Path(f"artifacts/submissions/{SUB}")
    write_submission(s1_all, pred, f.groupby("s1_id").cand_id.agg(set).to_dict(), out)
    n = pd.Series({s: len(v) for s, v in pred.items()}).reindex(s1_all, fill_value=0)
    ctry = rec.set_index("entity_id").country.reindex(n.index).to_numpy()
    log({"tau": best["tau"], "mean_matches": float(n.mean()), "empty_share": float((n == 0).mean()),
         **{c: float(n[ctry == c].mean()) for c in ["US", "India", "France"]}})
    v = subprocess.run(["python3", cfg["paths"]["validator"], "--matching", str(out / "matching_results.tsv"),
                       "--candidate", str(out / "candidate_pairs.tsv"), "--test-dir",
                       str(Path(cfg["paths"]["data_dir"]) / "test")], capture_output=True, text=True)
    print(v.stdout[-800:], f"validator exit={v.returncode}", flush=True)


{"train": train, "submit": submit}[ACTION]()
