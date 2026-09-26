"""Cross-encoder reranker (BAAI/bge-reranker-v2-m3, Apache-2.0) for the uncertain band of Model A.
usage: reranker.py train | score_oof | stack | score_test | submit
  train      : fine-tune on A_train S1s' candidates (true pairs + hard look-alikes); never sees B-split entities
  score_oof  : score Model A OOF pairs with LO <= p <= HI
  stack      : OOF LightGBM on [Model A p, rerank score, context] with the B folds -> F0.5 vs Model A
  score_test : score test pairs in the band (Model A test probabilities)
  submit     : apply the stacker to test -> artifacts/submissions/v3 + official validator"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from ber.config import load_config, run_dir
from ber.io import read_truth
from ber.matcher import decide, evaluate, md_table
from ber.normalize import records_path
from ber.predict import write_submission
from ber.text import record_text

LO, HI = 0.01, 0.99
BASE = "BAAI/bge-reranker-v2-m3"
cfg = load_config("configs/v2.yaml")
RD = run_dir(cfg)
OUT = RD / os.environ.get("RR_DIR", "reranker")
OUT.mkdir(exist_ok=True)
T0 = time.perf_counter()
log = lambda *a: print(f"[{time.perf_counter() - T0:6.0f}s]", *a, flush=True)


def texts(fam: str, ids: np.ndarray) -> list[str]:
    r = pd.read_parquet(records_path(cfg, fam), columns=["entity_id", "name_raw", "addr_raw", "country"]).set_index("entity_id")
    r = r.reindex(ids)
    assert r.name_raw.notna().all(), f"{int(r.name_raw.isna().sum())} ids not found in records_{fam}"
    return [record_text(n, a, c) for n, a, c in zip(r.name_raw, r.addr_raw, r.country)]


def load_model(path: str | None = None):
    import torch
    from sentence_transformers.cross_encoder import CrossEncoder
    torch.cuda.set_per_process_memory_fraction(cfg["gpu"]["mem_fraction"])
    return CrossEncoder(path or BASE, num_labels=1, max_length=192, device="cuda",
                        model_kwargs={"dtype": torch.bfloat16})


def score_pairs(fam: str, pairs: pd.DataFrame) -> np.ndarray:
    model = load_model(str(OUT / "model"))
    a, b = texts(fam, pairs.s1_id.to_numpy()), texts(fam, pairs.cand_id.to_numpy())
    return np.asarray(model.predict(list(zip(a, b)), batch_size=256, show_progress_bar=True), dtype=np.float32)


def train() -> None:
    import torch
    from datasets import Dataset
    from sentence_transformers.cross_encoder import CrossEncoderTrainer, CrossEncoderTrainingArguments
    from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss
    truth = read_truth(cfg["paths"]["data_dir"])
    sp = pd.read_parquet(RD / "splits.parquet")
    s1 = sp[sp.split == "A_train"].sample(int(os.environ.get("RR_N", 100_000)), random_state=cfg["seed"]).s1_id.tolist()
    c = ds.dataset(RD / "cand_sparse_train.parquet").to_table(
        columns=["s1_id", "cand_id", "tfidf_rank", "key_hits"], filter=ds.field("s1_id").isin(s1)).to_pandas()
    c["label"] = [x in truth[s] for s, x in zip(c.s1_id, c.cand_id)]
    pos = c[c.label]
    hard = c[~c.label & ((c.tfidf_rank <= 5) | (c.key_hits >= 2))]
    # up to 4 hard negatives per S1 (random order + head keeps all columns; groupby.apply drops s1_id in pandas 3)
    neg = hard.assign(r=np.random.default_rng(0).random(len(hard))).sort_values("r").groupby("s1_id").head(4).drop(columns="r")
    tr = pd.concat([pos, neg]).sample(frac=1.0, random_state=0)
    assert tr.s1_id.notna().all() and tr.cand_id.notna().all(), "missing ids in reranker training pairs"
    log(f"reranker train pairs: {len(tr):,} ({tr.label.mean():.1%} positive)")
    t1, t2 = texts("train", tr.s1_id.to_numpy()), texts("train", tr.cand_id.to_numpy())
    d = Dataset.from_dict({"text1": t1, "text2": t2,
                           "label": tr.label.astype(np.float32).tolist()})
    model = load_model()
    model.model.float()  # fp32 master weights for training; bf16 autocast below
    args = CrossEncoderTrainingArguments(output_dir=str(OUT / "ckpt"), num_train_epochs=1, per_device_train_batch_size=64,
                                         learning_rate=2e-5, warmup_ratio=0.1, bf16=True, logging_steps=200,
                                         save_strategy="no", report_to="none", seed=cfg["seed"])
    CrossEncoderTrainer(model=model, args=args, train_dataset=d, loss=BinaryCrossEntropyLoss(model)).train()
    model.save(str(OUT / "model"))
    log("saved", OUT / "model", f"peak GPU {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")


def score_oof() -> None:
    o = pd.read_parquet(RD / "matcher_emb" / "oof.parquet")
    band = o[(o.p >= LO) & (o.p <= HI)].reset_index(drop=True)
    band["rr"] = score_pairs("train", band)
    band.to_parquet(OUT / "oof_band_scores.parquet", index=False)
    from sklearn.metrics import roc_auc_score
    log(f"OOF band pairs {len(band):,}: AUC Model A {roc_auc_score(band.label, band.p):.4f} | reranker {roc_auc_score(band.label, band.rr):.4f}")


FEATS = ["p", "rr", "has_rr", "rr_rank", "rr_gap", "p_rank", "p_gap", "n_cands"]


def stack_features(s: pd.DataFrame) -> pd.DataFrame:
    f = s.copy()
    f["has_rr"] = f.rr.notna().astype(np.int8)
    g = f.groupby("s1_id")
    f["rr_rank"] = g.rr.rank(ascending=False, method="first").fillna(99).astype(np.int16)
    f["rr_gap"] = (g.rr.transform("max") - f.rr).astype(np.float32)
    f["p_rank"] = g.p.rank(ascending=False, method="first").astype(np.int16)
    f["p_gap"] = (g.p.transform("max") - f.p).astype(np.float32)
    f["n_cands"] = g.p.transform("size").astype(np.int16)
    return f


def stack() -> None:
    o = pd.read_parquet(RD / "matcher_emb" / "oof.parquet")
    b = pd.read_parquet(OUT / "oof_band_scores.parquet")[["s1_id", "cand_id", "rr"]]
    f = stack_features(o.merge(b, on=["s1_id", "cand_id"], how="left"))
    sp = pd.read_parquet(RD / "splits.parquet", columns=["s1_id", "country", "fold"]).set_index("s1_id")
    f["fold"] = sp.fold.reindex(f.s1_id).to_numpy()
    params = {"objective": "binary", "metric": "auc", "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 200,
              "feature_fraction": 0.9, "verbose": -1, "num_threads": cfg["n_jobs"]}
    p2 = np.zeros(len(f), np.float32)
    for k in sorted(f.fold.unique()):
        tr, va = f.fold.to_numpy() != k, f.fold.to_numpy() == k
        m = lgb.train(params, lgb.Dataset(f.loc[tr, FEATS], f.label[tr]), 2000,
                      valid_sets=[lgb.Dataset(f.loc[va, FEATS], f.label[va])], callbacks=[lgb.early_stopping(100, verbose=False)])
        p2[va] = m.predict(f.loc[va, FEATS], num_iteration=m.best_iteration)
        m.save_model(str(OUT / f"stack_fold{k}.txt"))
    truth = read_truth(cfg["paths"]["data_dir"])
    s1 = sp.loc[o.s1_id.unique()].reset_index()[["s1_id", "country"]]
    rows = []
    for name, p in (("Model A v2", o.p.to_numpy()), ("A + reranker", p2)):
        sc = pd.DataFrame({"s1_id": o.s1_id, "cand_id": o.cand_id, "p": p})
        for tau in np.round(np.arange(0.4, 0.91, 0.05), 2):
            rows.append({"model": name, "tau": float(tau), **evaluate(decide(sc, tau), truth, s1)})
    grid = pd.DataFrame(rows)
    best = grid[grid.model == "A + reranker"].sort_values("macro_f05").iloc[-1]
    base = grid[grid.model == "Model A v2"].sort_values("macro_f05").iloc[-1]
    (OUT / "best.json").write_text(json.dumps({"tau": float(best.tau), "macro_f05": float(best.macro_f05)}, indent=2))
    md = ["# Cross-encoder reranker (bge-reranker-v2-m3) stacked on Model A v2 — out-of-fold", "",
          f"Band: Model A p in [{LO}, {HI}] ({f.has_rr.sum():,} of {len(f):,} OOF pairs re-scored).", "",
          f"Model A v2: τ={base.tau:.2f} → **{base.macro_f05:.4f}**", "",
          f"**A + reranker: τ={best.tau:.2f} → {best.macro_f05:.4f}** (singletons {best.singletons:.4f}, "
          + ", ".join(f"{k[4:]} {v:.4f}" for k, v in best.items() if str(k).startswith("f05_")) + ")", "", md_table(grid)]
    Path("docs/results/reranker.md").write_text("\n".join(md) + "\n")
    (OUT / "report.md").write_text("\n".join(md) + "\n")
    print("\n".join(md[:7]), flush=True)


def score_test() -> None:
    t = pd.read_parquet(RD / "test_scores.parquet")
    band = t[(t.p >= LO) & (t.p <= HI)].reset_index(drop=True)
    log(f"test band pairs: {len(band):,}")
    band["rr"] = score_pairs("test", band)
    band[["s1_id", "cand_id", "rr"]].to_parquet(OUT / "test_band_scores.parquet", index=False)
    log("saved test band scores")


def submit() -> None:
    t = pd.read_parquet(RD / "test_scores.parquet")
    b = pd.read_parquet(OUT / "test_band_scores.parquet")
    f = stack_features(t.merge(b, on=["s1_id", "cand_id"], how="left"))
    models = [lgb.Booster(model_file=str(p)) for p in sorted(OUT.glob("stack_fold*.txt"))]
    p2 = np.mean([m.predict(f[FEATS], num_iteration=m.best_iteration) for m in models], axis=0)
    tau = json.loads((OUT / "best.json").read_text())["tau"]
    sc = pd.DataFrame({"s1_id": f.s1_id, "cand_id": f.cand_id, "p": p2})
    pred = decide(sc, tau)
    rec = pd.read_parquet(records_path(cfg, "test"), columns=["entity_id", "source", "country"])
    s1_all = rec.entity_id[rec.source == 1].tolist()
    out = Path("artifacts/submissions/v3")
    write_submission(s1_all, pred, t.groupby("s1_id").cand_id.agg(set).to_dict(), out)
    n = pd.Series({s: len(v) for s, v in pred.items()}).reindex(s1_all, fill_value=0)
    ctry = rec.set_index("entity_id").country.reindex(n.index).to_numpy()
    log({"tau": tau, "mean_matches": float(n.mean()), "empty_share": float((n == 0).mean()),
         **{c: float(n[ctry == c].mean()) for c in ["US", "India", "France"]}})
    v = subprocess.run(["python3", cfg["paths"]["validator"], "--matching", str(out / "matching_results.tsv"),
                        "--candidate", str(out / "candidate_pairs.tsv"), "--test-dir",
                        str(Path(cfg["paths"]["data_dir"]) / "test")], capture_output=True, text=True)
    print(v.stdout[-800:], f"validator exit={v.returncode}", flush=True)


{"train": train, "score_oof": score_oof, "stack": stack, "score_test": score_test, "submit": submit}[sys.argv[1]]()
