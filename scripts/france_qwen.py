"""Route France's uncertain pairs to Qwen3-Reranker-4B (inference only, Apache-2.0); US/India keep the fine-tuned bge reranker.
Both rerankers' raw scores are mapped to "chance of a true match" (Platt scaling) so the stacker sees one comparable
`rr` column.  Qwen's calibrator is fitted on US pairs only, so the India rehearsal mimics an unseen country.
usage: france_qwen.py score_oof | rehearse | score_test | submit
  score_oof  : Qwen scores for the Model A OOF band (same pairs the bge reranker scored)
  rehearse   : calibrate both, retrain the stacker on calibrated bge, then on India compare bge vs Qwen vs blend
  score_test : Qwen scores for the France test band pairs
  submit     : stacker on test with rr = bge (US/India) or Qwen (France) -> artifacts/submissions/$SUB + validator"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from ber.config import load_config, run_dir
from ber.io import read_truth
from ber.matcher import decide, evaluate, md_table
from ber.normalize import records_path
from ber.predict import write_submission
from ber.text import record_text

LO, HI = 0.01, 0.99
QWEN = "Qwen/Qwen3-Reranker-4B"
cfg = load_config("configs/v2.yaml")
RD = run_dir(cfg)
RR = RD / os.environ.get("RR_DIR", "reranker_v3")           # bge reranker outputs (band scores) for this Model A
OUT = RD / os.environ.get("FR_DIR", "reranker_fr")
A_DIR = os.environ.get("A_DIR", "matcher_v3")
TEST_SCORES = os.environ.get("TEST_SCORES", "test_scores_matcher_v3.parquet")
SUB = os.environ.get("SUB", "v7")
ROUTE = os.environ.get("ROUTE", "France")                   # countries whose band pairs go to Qwen
OUT.mkdir(exist_ok=True)
T0 = time.perf_counter()
log = lambda *a: print(f"[{time.perf_counter() - T0:6.0f}s]", *a, flush=True)
FEATS = ["p", "rr", "has_rr", "rr_rank", "rr_gap", "p_rank", "p_gap", "n_cands"]

PREFIX = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct "
          "provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n")
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
INSTR = ("Business records come from different sources with noisy names and addresses (typos, abbreviations, legal "
         "suffixes, transliteration, reordered or missing address parts). Is the Document the same real-world "
         "business (same place) as the Query?")


def texts(fam: str, ids: np.ndarray) -> list[str]:
    r = pd.read_parquet(records_path(cfg, fam), columns=["entity_id", "name_raw", "addr_raw", "country"]).set_index("entity_id")
    r = r.reindex(ids)
    assert r.name_raw.notna().all(), f"{int(r.name_raw.isna().sum())} ids not found in records_{fam}"
    return [record_text(n, a, c) for n, a, c in zip(r.name_raw, r.addr_raw, r.country)]


def qwen_scores(fam: str, pairs: pd.DataFrame, bs: int = 128) -> np.ndarray:
    """Raw Qwen3-Reranker score = logit(yes) - logit(no) at the answer position."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.cuda.set_per_process_memory_fraction(cfg["gpu"]["mem_fraction"])
    tok = AutoTokenizer.from_pretrained(QWEN, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(QWEN, dtype=torch.bfloat16).cuda().eval()
    yes, no = tok.convert_tokens_to_ids("yes"), tok.convert_tokens_to_ids("no")
    a, b = texts(fam, pairs.s1_id.to_numpy()), texts(fam, pairs.cand_id.to_numpy())
    prompts = [f"{PREFIX}<Instruct>: {INSTR}\n<Query>: {q}\n<Document>: {d}{SUFFIX}" for q, d in zip(a, b)]
    order = np.argsort([len(p) for p in prompts])            # similar lengths per batch -> little padding
    out = np.empty(len(prompts), np.float32)
    with torch.inference_mode():
        for i in range(0, len(order), bs):
            idx = order[i:i + bs]
            enc = tok([prompts[j] for j in idx], padding=True, truncation=True, max_length=512, return_tensors="pt").to("cuda")
            lg = model(**enc, logits_to_keep=1).logits[:, -1, :]
            out[idx] = (lg[:, yes] - lg[:, no]).float().cpu().numpy()
            if (i // bs) % 200 == 0:
                log(f"qwen scored {min(i + bs, len(order)):,}/{len(order):,}")
    return out


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p.astype(np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def platt(x: np.ndarray, y: np.ndarray) -> LogisticRegression:
    return LogisticRegression(C=1e6).fit(x.reshape(-1, 1), y)


def cal(m: LogisticRegression, x: np.ndarray) -> np.ndarray:
    return m.predict_proba(x.reshape(-1, 1))[:, 1].astype(np.float32)


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


def score_oof() -> None:
    band = pd.read_parquet(RR / "oof_band_scores.parquet")
    band["qwen"] = qwen_scores("train", band)
    band.to_parquet(OUT / "oof_band_scores.parquet", index=False)
    sp = pd.read_parquet(RD / "splits.parquet", columns=["s1_id", "country"]).set_index("s1_id")
    band["country"] = sp.country.reindex(band.s1_id).to_numpy()
    for c, g in [("all", band)] + list(band.groupby("country")):
        log(f"AUC {c} ({len(g):,} pairs): Model A {roc_auc_score(g.label, g.p):.4f} | bge {roc_auc_score(g.label, g.rr):.4f}"
            f" | qwen {roc_auc_score(g.label, g.qwen):.4f}")


def calibrators(band: pd.DataFrame) -> tuple:
    us = band.country.to_numpy() == "US"
    m_bge = platt(logit(band.rr.to_numpy()), band.label.to_numpy())                 # bge: all train countries
    m_qwen = platt(band.qwen.to_numpy().astype(np.float64)[us], band.label.to_numpy()[us])  # qwen: US only
    return m_bge, m_qwen


def rehearse() -> None:
    o = pd.read_parquet(RD / A_DIR / "oof.parquet")
    band = pd.read_parquet(OUT / "oof_band_scores.parquet")
    sp = pd.read_parquet(RD / "splits.parquet", columns=["s1_id", "country", "fold"]).set_index("s1_id")
    band["country"] = sp.country.reindex(band.s1_id).to_numpy()
    m_bge, m_qwen = calibrators(band)
    band["bge_cal"] = cal(m_bge, logit(band.rr.to_numpy()))
    band["qwen_cal"] = cal(m_qwen, band.qwen.to_numpy().astype(np.float64))
    (OUT / "calibration.json").write_text(json.dumps({"bge": [float(m_bge.coef_[0, 0]), float(m_bge.intercept_[0])],
                                                     "qwen": [float(m_qwen.coef_[0, 0]), float(m_qwen.intercept_[0])]}))
    base = o.merge(band[["s1_id", "cand_id", "bge_cal", "qwen_cal"]], on=["s1_id", "cand_id"], how="left")
    base["fold"] = sp.fold.reindex(base.s1_id).to_numpy()
    base["country"] = sp.country.reindex(base.s1_id).to_numpy()
    f = stack_features(base.assign(rr=base.bge_cal))
    params = {"objective": "binary", "metric": "auc", "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 200,
              "feature_fraction": 0.9, "verbose": -1, "num_threads": cfg["n_jobs"]}
    ind = base.country.to_numpy() == "India"
    variants = {"bge (current)": base.bge_cal.to_numpy(), "qwen on India": np.where(ind, base.qwen_cal, base.bge_cal),
                "blend on India": np.where(ind, (base.bge_cal + base.qwen_cal) / 2, base.bge_cal)}
    preds = {k: np.zeros(len(f), np.float32) for k in variants}
    for k in sorted(f.fold.unique()):
        tr, va = f.fold.to_numpy() != k, f.fold.to_numpy() == k
        m = lgb.train(params, lgb.Dataset(f.loc[tr, FEATS], f.label[tr]), 2000,
                      valid_sets=[lgb.Dataset(f.loc[va, FEATS], f.label[va])], callbacks=[lgb.early_stopping(100, verbose=False)])
        m.save_model(str(OUT / f"stack_fold{k}.txt"))
        for name, rr in variants.items():   # same stacker; only the rr column of the held-out fold changes
            fv = stack_features(base.loc[va].assign(rr=rr[va]))
            preds[name][va] = m.predict(fv[FEATS], num_iteration=m.best_iteration)
    truth = read_truth(cfg["paths"]["data_dir"])
    s1 = sp.loc[o.s1_id.unique()].reset_index()[["s1_id", "country"]]
    rows = []
    for name, p in preds.items():
        sc = pd.DataFrame({"s1_id": base.s1_id, "cand_id": base.cand_id, "p": p})
        for tau in np.round(np.arange(0.5, 0.86, 0.05), 2):
            rows.append({"model": name, "tau": float(tau), **evaluate(decide(sc, tau), truth, s1)})
    grid = pd.DataFrame(rows)
    best = {n: g.sort_values("macro_f05").iloc[-1] for n, g in grid.groupby("model")}
    tau = float(best["bge (current)"].tau)
    at = grid[grid.tau == tau].set_index("model")
    (OUT / "best.json").write_text(json.dumps({"tau": tau, "india": at.f05_India.to_dict()}, indent=2))
    ib = band[band.country == "India"]
    md = ["# France routing rehearsal: Qwen3-Reranker-4B (zero-shot, calibrated on US only) on India", "",
          f"India band AUC: bge (fine-tuned, saw India) {roc_auc_score(ib.label, ib.rr):.4f} | "
          f"qwen (never saw our data) {roc_auc_score(ib.label, ib.qwen):.4f}", "",
          f"At τ={tau:.2f} (best for current bge):", "",
          "| variant | India F0.5 | overall F0.5 |", "|---|---|---|"]
    md += [f"| {n} | {r.f05_India:.4f} | {r.macro_f05:.4f} |" for n, r in at.iterrows()]
    md += ["", md_table(grid)]
    Path("docs/results/france_qwen_rehearsal.md").write_text("\n".join(md) + "\n")
    print("\n".join(md[:10 + len(at)]), flush=True)


def score_test() -> None:
    t = pd.read_parquet(RD / TEST_SCORES)
    band = t[(t.p >= LO) & (t.p <= HI)].reset_index(drop=True)
    ctry = pd.read_parquet(records_path(cfg, "test"), columns=["entity_id", "country"]).set_index("entity_id").country
    band = band[ctry.reindex(band.s1_id).isin(ROUTE.split(",")).to_numpy()].reset_index(drop=True)
    log(f"test band pairs routed to Qwen ({ROUTE}): {len(band):,}")
    band["qwen"] = qwen_scores("test", band)
    band[["s1_id", "cand_id", "qwen"]].to_parquet(OUT / "test_band_qwen.parquet", index=False)
    log("saved")


def submit() -> None:
    c = json.loads((OUT / "calibration.json").read_text())
    sig = lambda x, ab: (1 / (1 + np.exp(-(ab[0] * x + ab[1])))).astype(np.float32)
    t = pd.read_parquet(RD / TEST_SCORES)
    b = pd.read_parquet(RR / "test_band_scores.parquet")[["s1_id", "cand_id", "rr"]]
    q = pd.read_parquet(OUT / "test_band_qwen.parquet")
    t = t.merge(b, on=["s1_id", "cand_id"], how="left").merge(q, on=["s1_id", "cand_id"], how="left")
    rr = sig(logit(t.rr.to_numpy()), c["bge"])
    rr = np.where(t.rr.isna(), np.nan, rr)
    t["rr"] = np.where(t.qwen.notna(), sig(t.qwen.to_numpy(), c["qwen"]), rr)   # France: Qwen replaces bge
    log(f"band pairs: bge {int((t.qwen.isna() & t.rr.notna()).sum()):,} | qwen {int(t.qwen.notna().sum()):,}")
    f = stack_features(t)
    models = [lgb.Booster(model_file=str(p)) for p in sorted(OUT.glob("stack_fold*.txt"))]
    p2 = np.mean([m.predict(f[FEATS], num_iteration=m.best_iteration) for m in models], axis=0)
    tau = json.loads((OUT / "best.json").read_text())["tau"]
    sc = pd.DataFrame({"s1_id": f.s1_id, "cand_id": f.cand_id, "p": p2})
    pred = decide(sc, tau)
    rec = pd.read_parquet(records_path(cfg, "test"), columns=["entity_id", "source", "country"])
    s1_all = rec.entity_id[rec.source == 1].tolist()
    out = Path(f"artifacts/submissions/{SUB}")
    write_submission(s1_all, pred, t.groupby("s1_id").cand_id.agg(set).to_dict(), out)
    n = pd.Series({s: len(v) for s, v in pred.items()}).reindex(s1_all, fill_value=0)
    ctry = rec.set_index("entity_id").country.reindex(n.index).to_numpy()
    log({"tau": tau, "mean_matches": float(n.mean()), "empty_share": float((n == 0).mean()),
         **{c: float(n[ctry == c].mean()) for c in ["US", "India", "France"]},
         **{f"empty_{c}": float((n[ctry == c] == 0).mean()) for c in ["US", "India", "France"]}})
    v = subprocess.run(["python3", cfg["paths"]["validator"], "--matching", str(out / "matching_results.tsv"),
                        "--candidate", str(out / "candidate_pairs.tsv"), "--test-dir",
                        str(Path(cfg["paths"]["data_dir"]) / "test")], capture_output=True, text=True)
    print(v.stdout[-800:], f"validator exit={v.returncode}", flush=True)


{"score_oof": score_oof, "rehearse": rehearse, "score_test": score_test, "submit": submit}[sys.argv[1]]()
