"""Score the TEST candidates with the Model A fold models and write the submission files.
Candidates = cand_sparse_test (+ the same extra channels as training). Probability = mean of the fold models.
Decision = cut-off tau (from the OOF report) + one-owner rule. Output: output/matching_results.tsv and
output/candidate_pairs.tsv, then the official validator.   CLI: python -m ber --config configs/v2.yaml predict"""
import argparse
import json
import subprocess
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from ber.blocking.sparse import cand_sparse_path
from ber.config import log_metrics, run_dir
from ber.features import REC_COLS, name_idf, pair_features
from ber.matcher import add_extra_channels, candidate_context, decide
from ber.normalize import records_path


def write_submission(s1_ids: list[str], pred: dict[str, set[str]], cands: dict[str, set[str]], out_dir: Path) -> None:
    """One row per test S1; comma-joined ids, empty when none; matches are a subset of candidates."""
    out_dir.mkdir(parents=True, exist_ok=True)
    m = pd.DataFrame({"source1_entity_id": s1_ids,
                      "matched_entity_ids": [",".join(sorted(pred.get(s, set()) & cands.get(s, set()))) for s in s1_ids]})
    c = pd.DataFrame({"source1_entity_id": s1_ids,
                      "candidate_entity_ids": [",".join(sorted(cands.get(s, set()))) for s in s1_ids]})
    m.to_csv(out_dir / "matching_results.tsv", sep="\t", index=False)
    c.to_csv(out_dir / "candidate_pairs.tsv", sep="\t", index=False)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--models", default="matcher", help="run_dir sub-folder holding lgbm_fold*.txt and report")
    parser.add_argument("--tau", type=float, default=None, help="cut-off; default = best tau from the OOF metrics")
    parser.add_argument("--chunk", type=int, default=200_000, help="S1s per scoring chunk")


def run(cfg: dict, args: argparse.Namespace) -> None:
    t0 = time.perf_counter()
    rd = run_dir(cfg)
    mdir = rd / args.models
    models = [lgb.Booster(model_file=str(p)) for p in sorted(mdir.glob("lgbm_fold*.txt"))]
    feat_names = models[0].feature_name()
    best = json.loads((mdir / "best.json").read_text())
    tau = args.tau if args.tau is not None else best["tau"]
    channels = best.get("extra_channels", [])
    need = {"hardname": rd / "cand_hardname_test.parquet", "graph": rd / "kg_recrec_edges_test.parquet"}
    missing = [str(need[c]) for c in channels if c in need and not need[c].exists()]
    if missing:
        raise SystemExit(f"test-side channel files missing (the models were trained with {channels}): {missing}")
    if best.get("embed_model_path"):
        raise SystemExit("models use the bi-encoder feature; test embeddings are not wired into predict yet")
    print(f"{len(models)} fold models, {len(feat_names)} features, tau={tau}, channels={channels}", flush=True)
    rec = pd.read_parquet(records_path(cfg, "test"), columns=REC_COLS).set_index("entity_id")
    idf = name_idf(rec.reset_index()) if "name_idf_jacc" in feat_names else None
    ctx = candidate_context(cand_sparse_path(cfg, "test"))
    s1_all = rec.index[rec.source == 1].tolist()
    scored = []
    for i in range(0, len(s1_all), args.chunk):
        s1 = s1_all[i:i + args.chunk]
        pairs = ds.dataset(cand_sparse_path(cfg, "test")).to_table(filter=ds.field("s1_id").isin(s1)).to_pandas()
        pairs = add_extra_channels(cfg, "test", pairs, set(s1), channels)
        pairs = pairs.join(ctx, on="cand_id")
        pairs["is_cand_best"] = (pairs.tfidf_sim >= pairs.cand_best_sim - 1e-6).astype(np.int8)
        f = pair_features(pairs, rec, workers=cfg["n_jobs"], idf=idf)
        for c in feat_names:  # a channel absent on test becomes 0 / NaN like in training fill
            if c not in f:
                f[c] = np.nan
        X = f[feat_names]
        p = np.mean([m.predict(X, num_iteration=m.best_iteration) for m in models], axis=0)
        scored.append(pd.DataFrame({"s1_id": pairs.s1_id.to_numpy(), "cand_id": pairs.cand_id.to_numpy(), "p": p}))
        print(f"  scored S1 {i + len(s1):,}/{len(s1_all):,} ({len(pairs):,} pairs, {time.perf_counter() - t0:.0f}s)", flush=True)
    scored = pd.concat(scored, ignore_index=True)
    scored.to_parquet(rd / "test_scores.parquet", index=False)
    pred = decide(scored, tau)
    cands = scored.groupby("s1_id").cand_id.agg(set).to_dict()
    out = Path("output")
    write_submission(s1_all, pred, cands, out)
    n_pred = pd.Series({s: len(v) for s, v in pred.items()}).reindex(s1_all, fill_value=0)
    stats = {"tau": tau, "test_s1": len(s1_all), "pairs_scored": len(scored), "pred_pairs": int(n_pred.sum()),
             "mean_matches_per_s1": float(n_pred.mean()), "empty_share": float((n_pred == 0).mean()),
             **{f"mean_matches_{c}": float(n_pred[rec.country.reindex(n_pred.index).to_numpy() == c].mean())
                for c in rec.country[rec.source == 1].unique()}}
    print(stats, flush=True)
    log_metrics(cfg, "predict_test", stats)
    v = subprocess.run(["python3", cfg["paths"]["validator"], "--matching", str(out / "matching_results.tsv"),
                        "--candidate", str(out / "candidate_pairs.tsv"), "--test-dir", str(Path(cfg["paths"]["data_dir"]) / "test")],
                       capture_output=True, text=True)
    print(v.stdout[-2000:], v.stderr[-2000:], f"validator exit={v.returncode}", flush=True)
