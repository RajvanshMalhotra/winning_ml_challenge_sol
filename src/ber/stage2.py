"""Model B (stage 2): re-scores every (S1, candidate) pair using Model A's probabilities of the S1's OTHER candidates
and the knowledge-graph links between candidates ("duplicate support"), plus within-S1 context.
Trained on Model A's out-of-fold probabilities with the same B-split folds (honest stacking).
CLI: python -m ber --config configs/v2.yaml stage2 {train,predict} [--a-models matcher_emb]"""
import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from ber.config import log_metrics, run_dir
from ber.io import read_truth
from ber.matcher import decide, evaluate, md_table
from ber.normalize import records_path
from ber.predict import write_submission
from ber.split import splits_path

SUP_MIN = 0.3  # neighbours count as support only if Model A gives them at least this probability
FEATURES = ["p", "logit_p", "rank_p", "p_gap", "p_max", "n_ge50", "n_ge70", "sum_p", "n_cands",
            "sup_max", "sup_mean", "sup_n70", "sup_deg", "sup_max_x_p"]


def stage2_features(scored: pd.DataFrame, edges: pd.DataFrame, code: pd.Series) -> pd.DataFrame:
    """scored: s1_id, cand_id, p. edges: record-record KG edges (a, b int codes, symmetric)."""
    f = pd.DataFrame({"s": code.reindex(scored.s1_id).to_numpy(), "c": code.reindex(scored.cand_id).to_numpy(),
                      "p": scored.p.to_numpy().astype(np.float32)})
    g = f.groupby("s").p
    f["logit_p"] = np.log(np.clip(f.p, 1e-6, 1 - 1e-6) / np.clip(1 - f.p, 1e-6, 1)).astype(np.float32)
    f["rank_p"] = g.rank(ascending=False, method="first").astype(np.int16)
    f["p_max"] = g.transform("max").astype(np.float32)
    f["p_gap"] = (f.p_max - f.p).astype(np.float32)
    f["n_ge50"] = (f.p >= 0.5).groupby(f.s).transform("sum").astype(np.int16)
    f["n_ge70"] = (f.p >= 0.7).groupby(f.s).transform("sum").astype(np.int16)
    f["sum_p"] = g.transform("sum").astype(np.float32)
    f["n_cands"] = g.transform("size").astype(np.int16)
    # duplicate support: CONFIDENT other candidates (p >= SUP_MIN) of the same S1 that are KG-neighbours of this one.
    # Walk from the confident pairs to their neighbours so the join stays small on the 190M-pair test set.
    conf = f.loc[f.p >= SUP_MIN, ["s", "c", "p"]].rename(columns={"c": "c2", "p": "p2"})
    nb = conf.merge(edges.rename(columns={"a": "c2", "b": "c"}), on="c2")
    nb["ge70"] = nb.p2 >= 0.7
    agg = nb.groupby(["s", "c"]).agg(sup_max=("p2", "max"), sup_mean=("p2", "mean"), sup_deg=("p2", "size"),
                                     sup_n70=("ge70", "sum"))
    f = f.merge(agg.reset_index(), on=["s", "c"], how="left")
    for c in ["sup_max", "sup_mean", "sup_n70", "sup_deg"]:
        f[c] = f[c].fillna(0).astype(np.float32)
    f["sup_max_x_p"] = (f.sup_max * f.p).astype(np.float32)
    return f


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("action", choices=["train", "predict"])
    parser.add_argument("--a-models", default="matcher_emb", help="Model A folder (OOF + test scores)")
    parser.add_argument("--out", default="artifacts/submissions/v3", help="submission folder for predict")


def run(cfg: dict, args: argparse.Namespace) -> None:
    t0 = time.perf_counter()
    rd = run_dir(cfg)
    out_dir = rd / "stage2"
    out_dir.mkdir(exist_ok=True)
    fam = "train" if args.action == "train" else "test"
    ids = pd.read_parquet(records_path(cfg, fam), columns=["entity_id"]).entity_id
    code = pd.Series(np.arange(len(ids)), index=ids.to_numpy())
    edges = pd.read_parquet(rd / f"kg_recrec_edges_{fam}.parquet", columns=["a", "b"])
    params = {**cfg["matcher"]["lgbm"], "num_leaves": 63, "objective": "binary", "metric": "auc", "verbose": -1,
              "num_threads": cfg["n_jobs"]}
    if args.action == "train":
        oof = pd.read_parquet(rd / args.a_models / "oof.parquet")
        f = stage2_features(oof, edges, code)
        f["label"] = oof.label.to_numpy()
        splits = pd.read_parquet(splits_path(cfg), columns=["s1_id", "country", "fold"]).set_index("s1_id")
        f["fold"] = splits.fold.reindex(oof.s1_id).to_numpy()
        print(f"stage2 train: {len(f):,} pairs ({time.perf_counter() - t0:.0f}s)", flush=True)
        p2, models = np.zeros(len(f), np.float32), []
        for k in sorted(f.fold.unique()):
            tr, va = f.fold.to_numpy() != k, f.fold.to_numpy() == k
            m = lgb.train(params, lgb.Dataset(f.loc[tr, FEATURES], f.label[tr]), 2000,
                          valid_sets=[lgb.Dataset(f.loc[va, FEATURES], f.label[va])],
                          callbacks=[lgb.early_stopping(100, verbose=False)])
            p2[va] = m.predict(f.loc[va, FEATURES], num_iteration=m.best_iteration)
            m.save_model(str(out_dir / f"lgbm_fold{k}.txt"))
            models.append(m)
            print(f"fold {k}: best_iter {m.best_iteration}, auc {m.best_score['valid_0']['auc']:.5f}", flush=True)
        truth = read_truth(cfg["paths"]["data_dir"])
        s1 = splits.loc[oof.s1_id.unique()].reset_index()[["s1_id", "country"]]
        rows = []
        for name, p in (("A", oof.p.to_numpy()), ("A+B", p2)):
            sc = pd.DataFrame({"s1_id": oof.s1_id, "cand_id": oof.cand_id, "p": p})
            for tau in np.round(np.arange(0.4, 0.91, 0.05), 2):
                rows.append({"model": name, "tau": float(tau), **evaluate(decide(sc, tau), truth, s1)})
        grid = pd.DataFrame(rows)
        best = grid[grid.model == "A+B"].sort_values("macro_f05").iloc[-1]
        base = grid[grid.model == "A"].sort_values("macro_f05").iloc[-1]
        (out_dir / "best.json").write_text(json.dumps({"tau": float(best.tau), "macro_f05": float(best.macro_f05),
                                                       "a_models": args.a_models}, indent=2))
        pd.DataFrame({"s1_id": oof.s1_id, "cand_id": oof.cand_id, "p": p2, "label": oof.label}).to_parquet(out_dir / "oof.parquet", index=False)
        imp = pd.Series(np.mean([m.feature_importance("gain") for m in models], axis=0), index=FEATURES).sort_values(ascending=False)
        md = ["# Model B (stage 2: duplicate support) — out-of-fold", "",
              f"Model A alone: best τ={base.tau:.2f} → macro F0.5 **{base.macro_f05:.4f}**", "",
              f"**A + B: best τ={best.tau:.2f} → macro F0.5 {best.macro_f05:.4f}** (singletons {best.singletons:.4f}, "
              + ", ".join(f"{k[4:]} {v:.4f}" for k, v in best.items() if str(k).startswith("f05_")) + ")", "",
              md_table(grid), "", "## Feature importance", "", md_table(imp.rename_axis("feature").reset_index(name="gain"))]
        (out_dir / "report.md").write_text("\n".join(md) + "\n")
        Path("docs/results/model_b.md").write_text("\n".join(md) + "\n")
        log_metrics(cfg, "stage2", {"A": base.to_dict(), "A+B": best.to_dict()})
        print("\n".join(md[:5]), flush=True)
    else:
        best = json.loads((out_dir / "best.json").read_text())
        scored = pd.read_parquet(rd / "test_scores.parquet")  # Model A test probabilities (from `predict`)
        f = stage2_features(scored, edges, code)
        models = [lgb.Booster(model_file=str(p)) for p in sorted(out_dir.glob("lgbm_fold*.txt"))]
        p2 = np.mean([m.predict(f[FEATURES], num_iteration=m.best_iteration) for m in models], axis=0)
        sc = pd.DataFrame({"s1_id": scored.s1_id, "cand_id": scored.cand_id, "p": p2})
        sc.to_parquet(rd / "test_scores_stage2.parquet", index=False)
        pred = decide(sc, best["tau"])
        rec = pd.read_parquet(records_path(cfg, "test"), columns=["entity_id", "source", "country"])
        s1_all = rec.entity_id[rec.source == 1].tolist()
        cands = scored.groupby("s1_id").cand_id.agg(set).to_dict()
        write_submission(s1_all, pred, cands, Path(args.out))
        n = pd.Series({s: len(v) for s, v in pred.items()}).reindex(s1_all, fill_value=0)
        print({"tau": best["tau"], "mean_matches": float(n.mean()), "empty_share": float((n == 0).mean())}, flush=True)
        import subprocess
        v = subprocess.run(["python3", cfg["paths"]["validator"], "--matching", f"{args.out}/matching_results.tsv",
                            "--candidate", f"{args.out}/candidate_pairs.tsv", "--test-dir",
                            str(Path(cfg["paths"]["data_dir"]) / "test")], capture_output=True, text=True)
        print(v.stdout[-1500:], f"validator exit={v.returncode}", flush=True)
