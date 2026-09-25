"""Model A: LightGBM pair matcher trained out-of-fold on a sample of B-split S1s (the bi-encoder never saw them).
Scores with the official macro F0.5 after a probability cut-off and the one-owner rule; writes a report.
CLI: python -m ber --config configs/v2.yaml matcher"""
import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds

from ber.blocking.sparse import cand_sparse_path
from ber.config import log_metrics, run_dir
from ber.evaluate import f05
from ber.features import REC_COLS, name_idf, pair_features
from ber.io import read_truth
from ber.normalize import records_path
from ber.split import splits_path
from sklearn.metrics import roc_auc_score

LABEL, NON_FEATURES = "label", {"s1_id", "cand_id", "label", "fold", "country"}


def candidate_context(cand_file: Path) -> pd.DataFrame:
    """Per candidate record, over the FULL candidate table: how many S1s list it, and its best TF-IDF sim."""
    t = ds.dataset(cand_file).to_table(columns=["cand_id", "tfidf_sim"])
    g = t.group_by("cand_id").aggregate([("tfidf_sim", "max"), ("cand_id", "count")]).to_pandas()
    return g.rename(columns={"tfidf_sim_max": "cand_best_sim", "cand_id_count": "cand_degree"}).set_index("cand_id")


def load_pairs(cfg: dict, family: str, s1_ids: list[str]) -> pd.DataFrame:
    cols = ["s1_id", "cand_id", "tfidf_sim", "tfidf_rank", "key_hits", "name_sim", "name_key"]
    d = ds.dataset(cand_sparse_path(cfg, family))
    cols = [c for c in cols if c in d.schema.names]
    return d.to_table(columns=cols, filter=pc.field("s1_id").isin(s1_ids)).to_pandas()


def embedding_cosine(pairs: pd.DataFrame, rec: pd.DataFrame, model_key: str, model_path: str, cfg: dict) -> np.ndarray:
    """Cosine of the fine-tuned bi-encoder embeddings of the two records of each pair (each record embedded once)."""
    import torch

    from ber.contrastive.encoders import ENCODERS, encode, load_encoder
    from ber.text import record_text
    spec = ENCODERS[model_key]
    model = load_encoder(spec, path=model_path, mem_fraction=cfg["gpu"]["mem_fraction"])
    s1_ids, c_ids = pd.unique(pairs.s1_id), pd.unique(pairs.cand_id)
    text = lambda ids: [record_text(n, a, c) for n, a, c in zip(rec.name_raw.reindex(ids), rec.addr_raw.reindex(ids),
                                                                  rec.country.reindex(ids))]
    bs = cfg["bakeoff"]["encode_batch_size"]
    eq = pd.Series(list(encode(model, text(s1_ids), spec.query_prefix, bs)), index=s1_ids)
    ed = pd.Series(list(encode(model, text(c_ids), spec.doc_prefix, bs)), index=c_ids)
    del model
    torch.cuda.empty_cache()
    q = np.stack(eq.reindex(pairs.s1_id).to_numpy()).astype(np.float32)
    d = np.stack(ed.reindex(pairs.cand_id).to_numpy()).astype(np.float32)
    return np.einsum("ij,ij->i", q, d).astype(np.float32)


def build_dataset(cfg: dict, truth: dict[str, set[str]], mcfg: dict) -> pd.DataFrame:
    splits = pd.read_parquet(splits_path(cfg))
    b = splits[splits.split == "B"]
    s1 = pd.concat([g.sample(min(len(g), round(mcfg["n_entities"] * len(g) / len(b))), random_state=cfg["seed"])
                    for _, g in b.groupby("country")])  # stratified by country
    pairs = load_pairs(cfg, "train", s1.s1_id.tolist())
    ctx = candidate_context(cand_sparse_path(cfg, "train"))
    pairs = pairs.join(ctx, on="cand_id")
    pairs["is_cand_best"] = (pairs.tfidf_sim >= pairs.cand_best_sim - 1e-6).astype(np.int8)
    rec = pd.read_parquet(records_path(cfg, "train"), columns=REC_COLS).set_index("entity_id")
    idf = name_idf(rec) if mcfg.get("idf_features", True) else None
    feats = pair_features(pairs, rec, workers=cfg["n_jobs"], idf=idf)
    if mcfg.get("embed_model_path"):
        feats["emb_cos"] = embedding_cosine(pairs, rec, mcfg.get("embed_model_key", "bgem3"), mcfg["embed_model_path"], cfg)
        feats["emb_cos_gap"] = (feats.groupby(pairs.s1_id.to_numpy()).emb_cos.transform("max") - feats.emb_cos).astype(np.float32)
    meta = s1.set_index("s1_id")
    feats["s1_id"], feats["cand_id"] = pairs.s1_id.to_numpy(), pairs.cand_id.to_numpy()
    feats["fold"] = meta.fold.reindex(pairs.s1_id).to_numpy()
    feats["country"] = meta.country.reindex(pairs.s1_id).to_numpy()
    feats[LABEL] = [c in truth[s] for s, c in zip(pairs.s1_id, pairs.cand_id)]
    feats.attrs["s1_sample"] = s1[["s1_id", "country", "fold"]]
    return feats


def train_oof(df: pd.DataFrame, params: dict, rounds: int) -> tuple[np.ndarray, list[lgb.Booster]]:
    X = df[[c for c in df.columns if c not in NON_FEATURES]]
    y = df[LABEL].to_numpy()
    oof, models = np.zeros(len(df), dtype=np.float32), []
    for k in sorted(df.fold.unique()):
        tr, va = df.fold.to_numpy() != k, df.fold.to_numpy() == k
        m = lgb.train(params, lgb.Dataset(X[tr], y[tr]), rounds, valid_sets=[lgb.Dataset(X[va], y[va])],
                      callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(200)])
        oof[va] = m.predict(X[va], num_iteration=m.best_iteration)
        models.append(m)
        print(f"fold {k}: best_iter {m.best_iteration}, valid auc {m.best_score['valid_0']['auc']:.5f}", flush=True)
    return oof, models


def decide(scored: pd.DataFrame, tau: float, one_owner: bool = True) -> dict[str, set[str]]:
    """scored: s1_id, cand_id, p. Keep pairs with p >= tau; each candidate goes only to its best-scoring S1."""
    s = scored[scored.p >= tau]
    if one_owner:
        s = s.loc[s.groupby("cand_id").p.idxmax()]
    return s.groupby("s1_id").cand_id.agg(set).to_dict()


def evaluate(pred: dict[str, set[str]], truth: dict[str, set[str]], s1: pd.DataFrame) -> dict:
    scores = pd.Series([f05(pred.get(x, set()), truth[x]) for x in s1.s1_id], index=s1.s1_id.to_numpy())
    single = np.array([len(truth[x]) == 0 for x in s1.s1_id])
    out = {"macro_f05": float(scores.mean()), "singletons": float(scores[single].mean()),
           "non_singletons": float(scores[~single].mean())}
    out.update({f"f05_{c}": float(scores[(s1.country == c).to_numpy()].mean()) for c in s1.country.unique()})
    return out


def md_table(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in row) + " |")
    return "\n".join(lines)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    pass


def run(cfg: dict, args: argparse.Namespace) -> None:
    t0 = time.perf_counter()
    mcfg = cfg["matcher"]
    out_dir = run_dir(cfg) / mcfg.get("out_name", "matcher")
    out_dir.mkdir(exist_ok=True)
    truth = read_truth(cfg["paths"]["data_dir"])
    df = build_dataset(cfg, truth, mcfg)
    s1 = df.attrs["s1_sample"]
    print(f"dataset: {len(df):,} pairs, {len(s1):,} S1s, positives {df[LABEL].mean():.4f} "
          f"({time.perf_counter() - t0:.0f}s)", flush=True)
    params = {**mcfg["lgbm"], "objective": "binary", "metric": "auc", "verbose": -1, "num_threads": cfg["n_jobs"]}
    oof, models = train_oof(df, params, mcfg["rounds"])
    scored = pd.DataFrame({"s1_id": df.s1_id, "cand_id": df.cand_id, "p": oof, "label": df[LABEL]})
    scored.to_parquet(out_dir / "oof.parquet", index=False)
    grid = []
    for tau in np.round(np.arange(0.05, 0.96, 0.05), 2):
        grid.append({"tau": float(tau), **evaluate(decide(scored, tau), truth, s1)})
    grid = pd.DataFrame(grid)
    best = grid.loc[grid.macro_f05.idxmax()]
    no_owner = evaluate(decide(scored, best.tau, one_owner=False), truth, s1)
    imp = pd.Series(np.mean([m.feature_importance("gain") for m in models], axis=0),
                    index=models[0].feature_name()).sort_values(ascending=False)
    for i, m in enumerate(models):
        m.save_model(str(out_dir / f"lgbm_fold{i}.txt"))
    metrics = {"n_pairs": len(df), "n_s1": len(s1), "best": best.to_dict(), "without_one_owner": no_owner,
               "oof_auc": float(roc_auc_score(df[LABEL].to_numpy(), oof))}
    log_metrics(cfg, "matcher", metrics)
    md = ["# Model A (LightGBM) — out-of-fold results on B-split sample", "",
          f"{len(s1):,} S1 businesses, {len(df):,} candidate pairs, {df[LABEL].mean():.2%} positives.", "",
          f"**Best cut-off τ = {best.tau:.2f} → macro F0.5 = {best.macro_f05:.4f}** "
          f"(singletons {best.singletons:.4f}, non-singletons {best.non_singletons:.4f}; "
          + ", ".join(f"{k[4:]} {v:.4f}" for k, v in best.items() if k.startswith("f05_")) + ")", "",
          f"Without the one-owner rule at the same τ: {no_owner['macro_f05']:.4f}", "",
          f"Out-of-fold AUC: {metrics['oof_auc']:.5f}", "",
          "## F0.5 vs cut-off", "", md_table(grid), "",
          "## Top 20 features (mean gain)", "", md_table(imp.head(20).rename_axis("feature").reset_index(name="gain"))]
    Path("docs/results").mkdir(parents=True, exist_ok=True)
    Path("docs/results").mkdir(parents=True, exist_ok=True)
    (out_dir / "report.md").write_text("\n".join(md) + "\n")
    Path(f"docs/results/{mcfg.get('report_name', 'model_a')}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md[:8]), flush=True)
    print(f"done in {time.perf_counter() - t0:.0f}s", flush=True)
