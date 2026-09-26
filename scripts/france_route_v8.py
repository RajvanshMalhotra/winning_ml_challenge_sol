"""v8 with France routed to Qwen3-Reranker-4B (zero-shot): the `rr` column is bge for US/India and Qwen for France,
never both. Qwen's raw score is put on bge's scale through probability calibration (Platt on the labelled OOF band):
  P_qwen = sigmoid(a_q * qwen + b_q),  bge: P = sigmoid(a_b * logit(rr) + b_b)  ->  rr_fr = sigmoid((a_q*qwen + b_q - b_b) / a_b)
so the unchanged v8 stacker (cand_side/cs_fold*) reads a France Qwen score exactly like a bge score of equal match chance.
usage: france_route_v8.py qwen  : score France band pairs, highest Model A p first, until $DEADLINE (epoch s); checkpoints
       france_route_v8.py submit: build v8 features, wait for the Qwen file, stack, write artifacts/submissions/$SUB"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ns = {"__name__": "cs"}
exec(open("scripts/cand_side.py").read().rsplit("\n{", 1)[0], ns)
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ber.matcher import decide
from ber.normalize import records_path
from ber.predict import write_submission
from ber.text import record_text

cfg, RD, log = ns["cfg"], ns["RD"], ns["log"]
OUT = RD / "france_route"
OUT.mkdir(exist_ok=True)
SUB = os.environ.get("SUB", "v8fr")
QFILE, QDONE = OUT / "test_band_qwen.parquet", OUT / "qwen.done"
LO, HI = 0.01, 0.99
QWEN = "Qwen/Qwen3-Reranker-4B"
PREFIX = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct "
          "provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n")
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
INSTR = ("Business records come from different sources with noisy names and addresses (typos, abbreviations, legal "
         "suffixes, transliteration, reordered or missing address parts). Is the Document the same real-world "
         "business (same place) as the Query?")  # same prompt as scripts/france_qwen.py (OOF scores used for calibration)


def france_band() -> pd.DataFrame:
    t = pd.read_parquet(RD / ns["TEST_SCORES"])
    band = t[(t.p >= LO) & (t.p <= HI)]
    rec = pd.read_parquet(records_path(cfg, "test"), columns=["entity_id", "country"]).set_index("entity_id").country
    return band[(rec.reindex(band.s1_id) == "France").to_numpy()].sort_values("p", ascending=False).reset_index(drop=True)


def qwen() -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    deadline = float(os.environ["DEADLINE"])
    band = france_band()
    qout = Path(os.environ.get("QOUT", QFILE))
    if qout != QFILE and QFILE.exists():  # continuation run: skip pairs a previous run already scored
        seen = pd.read_parquet(QFILE, columns=["s1_id", "cand_id"])
        k = band.s1_id + "|" + band.cand_id
        band = band[~k.isin(set(seen.s1_id + "|" + seen.cand_id)).to_numpy()].reset_index(drop=True)
        log(f"continuing: {len(seen):,} already scored, {len(band):,} left")
    log(f"France band pairs: {len(band):,}; p>=0.5: {(band.p >= 0.5).sum():,}; p>=0.1: {(band.p >= 0.1).sum():,}")
    r = pd.read_parquet(records_path(cfg, "test"), columns=["entity_id", "name_raw", "addr_raw", "country"]).set_index("entity_id")
    txt = lambda ids: [record_text(n, a, c) for n, a, c in zip(*(r[k].reindex(ids).to_numpy() for k in ["name_raw", "addr_raw", "country"]))]
    prompts = [f"{PREFIX}<Instruct>: {INSTR}\n<Query>: {q}\n<Document>: {d}{SUFFIX}"
               for q, d in zip(txt(band.s1_id.to_numpy()), txt(band.cand_id.to_numpy()))]
    del r
    torch.cuda.set_per_process_memory_fraction(0.25)
    tok = AutoTokenizer.from_pretrained(QWEN, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(QWEN, dtype=torch.bfloat16).cuda().eval()
    yes, no = tok.convert_tokens_to_ids("yes"), tok.convert_tokens_to_ids("no")
    out = np.full(len(prompts), np.nan, np.float32)
    CH, BS = 16_384, int(os.environ.get("BS", "96"))
    log("model loaded")
    with torch.inference_mode():
        for a in range(0, len(prompts), CH):  # chunks in priority order (highest Model A p first)
            idx = np.arange(a, min(a + CH, len(prompts)))
            idx = idx[np.argsort([len(prompts[j]) for j in idx])]
            for i in range(0, len(idx), BS):
                b = idx[i:i + BS]
                enc = tok([prompts[j] for j in b], padding=True, truncation=True, max_length=384, return_tensors="pt").to("cuda")
                lg = model(**enc, logits_to_keep=1).logits[:, -1, :]
                out[b] = (lg[:, yes] - lg[:, no]).float().cpu().numpy()
            done = int(np.isfinite(out).sum())
            band.assign(qwen=out)[["s1_id", "cand_id", "p", "qwen"]].dropna().to_parquet(qout, index=False)
            log(f"qwen scored {done:,}/{len(prompts):,} (p >= {band.p.iloc[done - 1]:.3f})")
            if time.time() > deadline:
                log("deadline reached, stopping")
                break
    if qout == QFILE:
        QDONE.write_text(str(int(np.isfinite(out).sum())))


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p.astype(np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def calibration() -> dict:
    band = pd.read_parquet(RD / "reranker_fr" / "oof_band_scores.parquet")  # labelled OOF band with both scores
    y = band.label.to_numpy()
    mb = LogisticRegression(C=1e6).fit(logit(band.rr.to_numpy()).reshape(-1, 1), y)
    mq = LogisticRegression(C=1e6).fit(band.qwen.to_numpy().astype(np.float64).reshape(-1, 1), y)  # US + India
    c = {"bge": [float(mb.coef_[0, 0]), float(mb.intercept_[0])], "qwen": [float(mq.coef_[0, 0]), float(mq.intercept_[0])]}
    (OUT / "calibration.json").write_text(json.dumps(c, indent=2))
    return c


def to_bge_scale(q: np.ndarray, c: dict) -> np.ndarray:
    (ab, bb), (aq, bq) = c["bge"], c["qwen"]
    return (1 / (1 + np.exp(-((aq * q.astype(np.float64) + bq - bb) / ab)))).astype(np.float32)


def submit() -> None:
    c = calibration()
    log(f"calibration {c}")
    t = pd.read_parquet(RD / ns["TEST_SCORES"])
    b = pd.read_parquet(ns["RR"] / "test_band_scores.parquet")[["s1_id", "cand_id", "rr"]]
    t = t.merge(b, on=["s1_id", "cand_id"], how="left")
    del b
    R = ns["Records"]("test")
    t = ns["add_new"](t, R, ns["comp_tables"]("test", R))   # candidate-side features do not depend on rr
    rec = pd.read_parquet(records_path(cfg, "test"), columns=["entity_id", "source", "country"])
    ctry = rec.set_index("entity_id").country
    fr = (ctry.reindex(t.s1_id) == "France").to_numpy()
    log(f"v8 features ready; waiting for Qwen ({fr.sum():,} France pairs, band {int((fr & t.rr.notna().to_numpy()).sum()):,})")
    while not QDONE.exists():
        time.sleep(15)
    q = pd.read_parquet(QFILE)[["s1_id", "cand_id", "qwen"]]
    t = t.merge(q, on=["s1_id", "cand_id"], how="left")
    rr = t.rr.to_numpy().astype(np.float32)
    rr[fr] = np.nan                                           # France: bge weight zero
    qs = t.qwen.to_numpy()
    has_q = np.isfinite(qs)
    rr[has_q] = to_bge_scale(qs[has_q], c)                    # France: Qwen on bge's scale
    t["rr"] = rr
    t = t.drop(columns="qwen")
    log(f"rr source: bge {int((~fr & np.isfinite(rr)).sum()):,} | qwen {int(has_q.sum()):,} | France band not reached "
        f"{int((fr & (t.p.to_numpy() >= LO) & (t.p.to_numpy() <= HI) & ~has_q).sum()):,}; "
        f"France rr median {np.nanmedian(rr[has_q]) if has_q.any() else float('nan'):.3f} vs bge median {np.nanmedian(rr[~fr]):.3f}")
    f = ns["stack_features"](t)
    del t
    F = ns["BASE"] + ns["NEW"]
    models = [lgb.Booster(model_file=str(p)) for p in sorted((RD / "cand_side").glob("cs_fold*.txt"))]
    p = np.mean([m.predict(f[F], num_iteration=m.best_iteration) for m in models], axis=0)
    tau = json.loads((RD / "cand_side" / "best.json").read_text())["tau"]
    sc = pd.DataFrame({"s1_id": f.s1_id, "cand_id": f.cand_id, "p": p})
    pred = decide(sc, tau)
    s1_all = rec.entity_id[rec.source == 1].tolist()
    out = Path(f"artifacts/submissions/{SUB}")
    write_submission(s1_all, pred, f.groupby("s1_id").cand_id.agg(set).to_dict(), out)
    n = pd.Series({s: len(v) for s, v in pred.items()}).reindex(s1_all, fill_value=0)
    cc = ctry.reindex(n.index).to_numpy()
    log({"tau": tau, "mean_matches": float(n.mean()),
         **{k: round(float(n[cc == k].mean()), 4) for k in ["US", "India", "France"]},
         **{f"empty_{k}": round(float((n[cc == k] == 0).mean()), 4) for k in ["US", "India", "France"]}})
    v = subprocess.run(["python3", cfg["paths"]["validator"], "--matching", str(out / "matching_results.tsv"),
                        "--candidate", str(out / "candidate_pairs.tsv"), "--test-dir",
                        str(Path(cfg["paths"]["data_dir"]) / "test")], capture_output=True, text=True)
    print(v.stdout[-800:], f"validator exit={v.returncode}", flush=True)


{"qwen": qwen, "submit": submit}[sys.argv[1]]()
