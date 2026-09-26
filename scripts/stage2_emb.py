"""Model B v2: re-score the uncertain pairs using bge-m3 similarity between a candidate and the S1's CONFIDENT
candidates ("do the suspects agree?"). Base score = Model A + reranker stack. Honest OOF with the B folds.
usage: stage2_emb.py train | submit
  train : build stacked OOF scores, add embedding-support features, OOF LightGBM -> F0.5 vs the stack
  submit: same on test -> artifacts/submissions/v4 + official validator"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from ber.config import load_config, run_dir
from ber.io import read_truth
from ber.matcher import decide, evaluate, md_table
from ber.normalize import records_path
from ber.predict import write_submission

cfg = load_config("configs/v2.yaml")
RD = run_dir(cfg)
RR = RD / os.environ.get("RR_DIR", "reranker")
OUT = RD / os.environ.get("S2_DIR", "stage2_emb")
A_DIR = os.environ.get("A_DIR", "matcher_emb")
TEST_SCORES = os.environ.get("TEST_SCORES", "test_scores.parquet")
SUB = os.environ.get("SUB", "v4")
OUT.mkdir(exist_ok=True)
LO, HI, CONF, CHUNK = 0.01, 0.99, 0.5, 2000
T0 = time.perf_counter()
log = lambda *a: print(f"[{time.perf_counter() - T0:6.0f}s]", *a, flush=True)
STACK_FEATS = ["p", "rr", "has_rr", "rr_rank", "rr_gap", "p_rank", "p_gap", "n_cands"]
FEATS = ["q", "q_rank", "q_gap", "q_max", "n_conf", "in_band",
         "sup_cos_max", "sup_cos_mean", "sup_wcos_max", "sup_n80", "self_is_conf"]


def stacked(fam: str) -> pd.DataFrame:
    """Model A + reranker stacked probability `q` for every pair (OOF on train, fold-average on test)."""
    sys.argv = ["reranker.py", "stack"]  # reuse the reranker helpers without running an action
    ns = {}
    exec(open("scripts/reranker.py").read().split('{"train": train')[0], ns)
    if fam == "train":
        base = pd.read_parquet(RD / A_DIR / "oof.parquet")
        band = pd.read_parquet(RR / "oof_band_scores.parquet")[["s1_id", "cand_id", "rr"]]
    else:
        base = pd.read_parquet(RD / TEST_SCORES)
        band = pd.read_parquet(RR / "test_band_scores.parquet")[["s1_id", "cand_id", "rr"]]
    f = ns["stack_features"](base.merge(band, on=["s1_id", "cand_id"], how="left"))
    models = [lgb.Booster(model_file=str(p)) for p in sorted(RR.glob("stack_fold*.txt"))]
    if fam == "train":
        sp = pd.read_parquet(RD / "splits.parquet", columns=["s1_id", "fold"]).set_index("s1_id")
        f["fold"] = sp.fold.reindex(f.s1_id).to_numpy()
        q = np.zeros(len(f))
        for k, m in enumerate(models):
            va = f.fold.to_numpy() == k
            q[va] = m.predict(f.loc[va, STACK_FEATS], num_iteration=m.best_iteration)
    else:
        q = np.mean([m.predict(f[STACK_FEATS], num_iteration=m.best_iteration) for m in models], axis=0)
    f["q"] = q.astype(np.float32)
    return f


def emb_support(f: pd.DataFrame, fam: str) -> pd.DataFrame:
    """For in-band pairs: cosine between the candidate and the S1's confident candidates (q >= CONF, excluding itself)."""
    ids = pd.read_parquet(records_path(cfg, fam), columns=["entity_id"]).entity_id
    code = pd.Series(np.arange(len(ids)), index=ids.to_numpy())
    emb = np.load(RD / f"emb_{fam}_bgem3_bgem3__finetune.npy", mmap_mode="r")
    g = f.groupby("s1_id").q
    f["q_rank"] = g.rank(ascending=False, method="first").astype(np.int16)
    f["q_max"] = g.transform("max").astype(np.float32)
    f["q_gap"] = (f.q_max - f.q).astype(np.float32)
    f["n_conf"] = (f.q >= CONF).groupby(f.s1_id).transform("sum").astype(np.int16)
    f["in_band"] = ((f.q >= LO) & (f.q <= HI)).astype(np.int8)
    f["self_is_conf"] = (f.q >= CONF).astype(np.int8)
    band = f.loc[f.in_band == 1, ["s1_id", "cand_id"]]
    conf = f.loc[f.q >= CONF, ["s1_id", "cand_id", "q"]].rename(columns={"cand_id": "c2", "q": "q2"})
    j = band.merge(conf, on="s1_id")
    j = j[j.cand_id != j.c2].reset_index(drop=True)
    cos = np.empty(len(j), np.float32)
    a, b = code.reindex(j.cand_id).to_numpy(), code.reindex(j.c2).to_numpy()
    for i in range(0, len(j), 500_000):
        ea = np.asarray(emb[a[i:i + 500_000]], dtype=np.float32)
        eb = np.asarray(emb[b[i:i + 500_000]], dtype=np.float32)
        cos[i:i + 500_000] = np.einsum("ij,ij->i", ea, eb)
    j["cos"], j["wcos"] = cos, cos * j.q2.to_numpy()
    j["n80"] = j.cos >= 0.8
    agg = j.groupby(["s1_id", "cand_id"]).agg(sup_cos_max=("cos", "max"), sup_cos_mean=("cos", "mean"),
                                               sup_wcos_max=("wcos", "max"), sup_n80=("n80", "sum"))
    f = f.merge(agg.reset_index(), on=["s1_id", "cand_id"], how="left")
    log(f"{fam}: {len(band):,} in-band pairs, {len(j):,} candidate-candidate similarities")
    return f


PARAMS = {"objective": "binary", "metric": "auc", "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 200,
          "feature_fraction": 0.9, "verbose": -1, "num_threads": cfg["n_jobs"]}


def train() -> None:
    f = emb_support(stacked("train"), "train")
    f.to_parquet(OUT / "train_features.parquet", index=False)
    p3 = np.zeros(len(f), np.float32)
    for k in sorted(f.fold.unique()):
        tr, va = f.fold.to_numpy() != k, f.fold.to_numpy() == k
        m = lgb.train(PARAMS, lgb.Dataset(f.loc[tr, FEATS], f.label[tr]), 2000,
                      valid_sets=[lgb.Dataset(f.loc[va, FEATS], f.label[va])], callbacks=[lgb.early_stopping(100, verbose=False)])
        p3[va] = m.predict(f.loc[va, FEATS], num_iteration=m.best_iteration)
        m.save_model(str(OUT / f"b2_fold{k}.txt"))
    truth = read_truth(cfg["paths"]["data_dir"])
    sp = pd.read_parquet(RD / "splits.parquet", columns=["s1_id", "country"]).set_index("s1_id")
    s1 = sp.loc[f.s1_id.unique()].reset_index()[["s1_id", "country"]]
    rows = []
    for name, p in (("A + reranker", f.q.to_numpy()), ("A + reranker + B2", p3)):
        sc = pd.DataFrame({"s1_id": f.s1_id, "cand_id": f.cand_id, "p": p})
        for tau in np.round(np.arange(0.4, 0.91, 0.05), 2):
            rows.append({"model": name, "tau": float(tau), **evaluate(decide(sc, tau), truth, s1)})
    grid = pd.DataFrame(rows)
    best = grid[grid.model == "A + reranker + B2"].sort_values("macro_f05").iloc[-1]
    base = grid[grid.model == "A + reranker"].sort_values("macro_f05").iloc[-1]
    (OUT / "best.json").write_text(json.dumps({"tau": float(best.tau), "macro_f05": float(best.macro_f05),
                                               "base_tau": float(base.tau), "base_f05": float(base.macro_f05)}, indent=2))
    md = ["# Model B v2 (bge-m3 duplicate agreement) on top of A + reranker — out-of-fold", "",
          f"A + reranker: τ={base.tau:.2f} → **{base.macro_f05:.4f}**", "",
          f"**A + reranker + B2: τ={best.tau:.2f} → {best.macro_f05:.4f}** (singletons {best.singletons:.4f}, "
          + ", ".join(f"{k[4:]} {v:.4f}" for k, v in best.items() if str(k).startswith("f05_")) + ")", "", md_table(grid)]
    Path(f"docs/results/model_b2_{A_DIR}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md[:5]), flush=True)
    print(f"B2: {best.macro_f05:.4f} vs A+reranker {base.macro_f05:.4f}", flush=True)


def submit() -> None:
    best = json.loads((OUT / "best.json").read_text())
    use_b2 = best["macro_f05"] > best["base_f05"]
    f = stacked("test")
    if use_b2:
        f = emb_support(f, "test")
        models = [lgb.Booster(model_file=str(p)) for p in sorted(OUT.glob("b2_fold*.txt"))]
        p, tau, name = np.mean([m.predict(f[FEATS], num_iteration=m.best_iteration) for m in models], axis=0), best["tau"], "v4 (A+reranker+B2)"
    else:
        p, tau, name = f.q.to_numpy(), best["base_tau"], "v3 (A+reranker; B2 did not help)"
    sc = pd.DataFrame({"s1_id": f.s1_id, "cand_id": f.cand_id, "p": p})
    sc.to_parquet(OUT / "test_scores_final.parquet", index=False)
    pred = decide(sc, tau)
    rec = pd.read_parquet(records_path(cfg, "test"), columns=["entity_id", "source", "country"])
    s1_all = rec.entity_id[rec.source == 1].tolist()
    out = Path(f"artifacts/submissions/{SUB}")
    write_submission(s1_all, pred, sc.groupby("s1_id").cand_id.agg(set).to_dict(), out)
    n = pd.Series({s: len(v) for s, v in pred.items()}).reindex(s1_all, fill_value=0)
    ctry = rec.set_index("entity_id").country.reindex(n.index).to_numpy()
    log(name, {"tau": tau, "mean_matches": float(n.mean()), "empty_share": float((n == 0).mean()),
               **{c: float(n[ctry == c].mean()) for c in ["US", "India", "France"]}})
    v = subprocess.run(["python3", cfg["paths"]["validator"], "--matching", str(out / "matching_results.tsv"),
                        "--candidate", str(out / "candidate_pairs.tsv"), "--test-dir",
                        str(Path(cfg["paths"]["data_dir"]) / "test")], capture_output=True, text=True)
    print(v.stdout[-800:], f"validator exit={v.returncode}", flush=True)


{"train": train, "submit": submit}[sys.argv[1]]()
