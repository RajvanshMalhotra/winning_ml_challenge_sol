"""v10: LoRA fine-tune of Qwen3-Reranker-4B (Apache-2.0) on our pairs, used as an EXTRA reranker column next to bge.
Scores only the pairs v8 is still unsure about (bge band AND v8 stacker q in [QLO, QHI]), in OOF and test alike.
usage: qwen_reranker.py train | score_oof | stack | test_sel | score_test | submit
  train      : A_train pairs (true + hard negatives, extra no-address positives/negatives), LoRA, yes/no logit, BCE
  score_oof  : Qwen scores for the selected OOF pairs; AUC vs bge on the same pairs
  stack      : v8 stacker features + Qwen column, 5-fold OOF vs v8 on the same folds -> F0.5
  test_sel   : v8 stacker on test -> the test pairs to score (CPU)
  score_test : Qwen scores for them
  submit     : v8 features + Qwen column on test -> artifacts/submissions/$SUB + validator"""
import json
import math
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

BASE_MODEL = "Qwen/Qwen3-Reranker-4B"
cfg = load_config("configs/v2.yaml")
RD = run_dir(cfg)
OUT = RD / os.environ.get("QR_DIR", "reranker_qwen")
CS = RD / "cand_side"
SUB = os.environ.get("SUB", "v10")
QLO, QHI = 0.02, 0.98
N_S1 = int(os.environ.get("QR_N", 30_000))
BUDGET_H = float(os.environ.get("QR_HOURS", 5.5))          # hard stop for training
OUT.mkdir(exist_ok=True)
T0 = time.perf_counter()
log = lambda *a: print(f"[{time.perf_counter() - T0:6.0f}s]", *a, flush=True)

PREFIX = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct "
          "provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n")
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
INSTR = ("Records of businesses from different sources, with deliberate noise (typos, character swaps, abbreviations, "
         "legal suffixes, transliteration, reordered, truncated or missing addresses, altered names). "
         "Is the Document generated from the same real-world business as the Query?")


def texts(fam: str, ids: np.ndarray) -> list[str]:
    r = pd.read_parquet(records_path(cfg, fam), columns=["entity_id", "name_raw", "addr_raw", "country"]).set_index("entity_id")
    r = r.reindex(ids)
    assert r.name_raw.notna().all(), f"{int(r.name_raw.isna().sum())} ids not found in records_{fam}"
    return [record_text(n, a, c) for n, a, c in zip(r.name_raw, r.addr_raw, r.country)]


SHORT = os.environ.get("QR_PROMPT", "long") == "short"     # short: ~40% fewer tokens (no system/instruction text)


def prompts(fam: str, pairs: pd.DataFrame) -> list[str]:
    a, b = texts(fam, pairs.s1_id.to_numpy()), texts(fam, pairs.cand_id.to_numpy())
    if SHORT:
        return [f"<|im_start|>user\nA: {q}\nB: {d}\nSame business?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
                for q, d in zip(a, b)]
    return [f"{PREFIX}<Instruct>: {INSTR}\n<Query>: {q}\n<Document>: {d}{SUFFIX}" for q, d in zip(a, b)]


def load(adapter: bool):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get("QR_MEM", cfg["gpu"]["mem_fraction"])))
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, padding_side="left")
    tok.truncation_side = "left"            # never cut the answer prompt at the end
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16).cuda()
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(OUT / "adapter")).merge_and_unload()
    return tok, model, tok.convert_tokens_to_ids("yes"), tok.convert_tokens_to_ids("no")


def training_pairs() -> pd.DataFrame:
    """True pairs + hard negatives, with extra weight on the error types v8 still makes:
    negatives: look-alikes (sparse/dense top ranks), records that belong to ANOTHER S1 (wrong-owner), same core name
    (chain branches / namesakes), same address core (other business in the same building), no-address look-alikes;
    positives: no-address, names sharing no word (gibberish names), house-number typos -> each counted twice."""
    truth = read_truth(cfg["paths"]["data_dir"])
    owner = {c: s for s, m in truth.items() for c in m}
    sp = pd.read_parquet(RD / "splits.parquet")
    at = sp[sp.split == "A_train"]
    s1 = (at if N_S1 <= 0 or N_S1 >= len(at) else at.sample(N_S1, random_state=cfg["seed"])).s1_id.tolist()
    flt = ds.field("s1_id").isin(s1)
    c = ds.dataset(RD / "cand_sparse_train.parquet").to_table(
        columns=["s1_id", "cand_id", "tfidf_rank", "key_hits", "name_sim"], filter=flt).to_pandas()
    d = ds.dataset(RD / "cand_dense_train.parquet").to_table(columns=["s1_id", "cand_id", "dense_rank"], filter=flt).to_pandas()
    c = c.merge(d, on=["s1_id", "cand_id"], how="outer")
    rec = pd.read_parquet(records_path(cfg, "train"), columns=["entity_id", "addr_raw", "name_core", "addr_core", "house_no"]).set_index("entity_id")
    A, C = rec.reindex(c.s1_id), rec.reindex(c.cand_id)
    c["noaddr"] = C.addr_raw.fillna("").str.strip().eq("").to_numpy()
    c["label"] = [x in truth[s] for s, x in zip(c.s1_id, c.cand_id)]
    c["other_owner"] = [(not l) and owner.get(x) is not None for x, l in zip(c.cand_id, c.label)]
    c["same_name"] = (A.name_core.to_numpy() == C.name_core.to_numpy()) & (C.name_core.fillna("").to_numpy() != "")
    c["same_addr"] = (A.addr_core.to_numpy() == C.addr_core.to_numpy()) & (C.addr_core.fillna("").to_numpy() != "")
    ha, hc = A.house_no.fillna("").to_numpy(), C.house_no.fillna("").to_numpy()
    c["house_diff"] = (ha != "") & (hc != "") & (ha != hc)
    c["sim"] = -c.dense_rank.fillna(99) + c.name_sim.fillna(0)          # rough "how confusing" order
    rng = np.random.default_rng(0)
    c["r"] = rng.random(len(c))
    pos = c[c.label].copy()
    na = set(zip(pos.s1_id, pos.cand_id))
    an, cn = rec.name_core.reindex(pos.s1_id).fillna("").to_numpy(), rec.name_core.reindex(pos.cand_id).fillna("").to_numpy()
    pos["no_word"] = [not (set(x.split()) & set(y.split())) for x, y in zip(an, cn)]
    hard_pos = pos[pos.noaddr | pos.no_word | pos.house_diff]
    neg = c[~c.label]
    top = lambda x, k, by="r", asc=True: x.sort_values(by, ascending=asc).groupby("s1_id").head(k)
    parts = {
        "look-alike": top(neg[(neg.tfidf_rank <= 5) | (neg.key_hits >= 2) | (neg.dense_rank <= 5)], 4),
        "wrong-owner": top(neg[neg.other_owner], 3, "sim", False),
        "same-name": top(neg[neg.same_name], 2, "sim", False),
        "same-address": top(neg[neg.same_addr], 2, "sim", False),
        "no-address": top(neg[neg.noaddr], 2, "name_sim", False),
    }
    negs = pd.concat(parts.values()).drop_duplicates(["s1_id", "cand_id"])
    tr = pd.concat([pos, hard_pos, negs]).sample(frac=1.0, random_state=0).reset_index(drop=True)
    log(f"train pairs {len(tr):,} ({tr.label.mean():.1%} positive; no-address {tr.noaddr.mean():.1%}) from {len(s1):,} A_train S1s")
    log("  positives " + f"{len(pos):,} (+{len(hard_pos):,} hard repeated: no-address {pos.noaddr.sum():,}, "
        f"no shared word {pos.no_word.sum():,}, house typo {pos.house_diff.sum():,})")
    log("  negatives " + ", ".join(f"{k} {len(v):,}" for k, v in parts.items()) + f" -> {len(negs):,} unique")
    return tr[["s1_id", "cand_id", "label"]]


def train() -> None:
    import torch
    from peft import LoraConfig, get_peft_model
    tr = training_pairs()
    P = prompts("train", tr)
    y = torch.tensor(tr.label.to_numpy(), dtype=torch.float32)
    tok, model, yes, no = load(adapter=False)
    model.config.use_cache = False
    if os.environ.get("QR_GC", "1") == "1":
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05, task_type="CAUSAL_LM",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    model.print_trainable_parameters()
    bs = int(os.environ.get("QR_BS", 16))
    accum = max(1, 64 // bs)
    # length buckets: sort within blocks of 50 batches, then shuffle the batch order (less padding, still random)
    lens = np.array([len(p) for p in P])
    rng = np.random.default_rng(1)
    order = []
    for j in range(0, len(P), bs * 50):
        blk = np.arange(j, min(j + bs * 50, len(P)))
        blk = blk[np.argsort(lens[blk], kind="stable")]
        order += [blk[k:k + bs] for k in range(0, len(blk), bs)]
    order = [order[k] for k in rng.permutation(len(order))]
    P = [P[k] for b in order for k in b]
    y = y[torch.from_numpy(np.concatenate(order))]
    n_steps = math.ceil(len(P) / bs / accum)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=0.0)
    warm = max(1, int(0.05 * n_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, n_steps) / n_steps)))
    lossf = torch.nn.BCEWithLogitsLoss()
    model.train()
    t0, run, step = time.perf_counter(), 0.0, 0
    for i in range(0, len(P), bs):
        enc = tok(P[i:i + bs], padding=True, truncation=True, max_length=320, return_tensors="pt").to("cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = model(**enc, logits_to_keep=1).logits[:, -1, :]
            s = (lg[:, yes] - lg[:, no]).float()
        loss = lossf(s, y[i:i + bs].cuda()) / accum
        loss.backward()
        run += loss.item()
        if (i // bs + 1) % accum == 0:
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            el = time.perf_counter() - t0
            if step % 100 == 0:
                rate = (i + bs) / el
                log(f"step {step}/{n_steps} loss {run / 100:.4f} lr {sched.get_last_lr()[0]:.2e} | {rate:.1f} pairs/s | "
                    f"ETA {(len(P) - i - bs) / rate / 3600:.2f} h")
                run = 0.0
            if step % 1000 == 0:
                model.save_pretrained(str(OUT / "adapter"))
            if el > BUDGET_H * 3600:
                log(f"time budget {BUDGET_H} h reached at step {step}/{n_steps} ({i + bs:,} pairs)")
                break
    model.save_pretrained(str(OUT / "adapter"))
    log("saved adapter", OUT / "adapter", f"peak GPU {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")


def score(fam: str, pairs: pd.DataFrame, bs: int = 128) -> np.ndarray:
    if os.environ.get("QR_VLLM") == "1":
        return score_vllm(fam, pairs)
    import torch
    tok, model, yes, no = load(adapter=True)
    model.eval()
    P = prompts(fam, pairs)
    order = np.argsort([len(p) for p in P])
    out = np.empty(len(P), np.float32)
    with torch.inference_mode():
        for i in range(0, len(order), bs):
            idx = order[i:i + bs]
            enc = tok([P[j] for j in idx], padding=True, truncation=True, max_length=320, return_tensors="pt").to("cuda")
            lg = model(**enc, logits_to_keep=1).logits[:, -1, :]
            out[idx] = (lg[:, yes] - lg[:, no]).float().cpu().numpy()
            if (i // bs) % 200 == 0:
                log(f"qwen scored {min(i + bs, len(order)):,}/{len(order):,}")
    return out


def vllm_mem() -> float:
    """vLLM grabs a FRACTION OF THE WHOLE GPU at start-up and fails if that much is not free (other users' jobs).
    Ask for at most what is free now minus a 6 GB safety margin, capped at QR_VLLM_MEM; a 4B model needs ~20 GB."""
    import torch
    free, total = torch.cuda.mem_get_info()
    frac = min(float(os.environ.get("QR_VLLM_MEM", 0.6)), (free - 6e9) / total)
    assert frac * total > 20e9, f"only {free / 1e9:.0f} GB free on the GPU; vLLM needs ~20 GB"
    log(f"vLLM memory fraction {frac:.2f} ({frac * total / 1e9:.0f} GB; {free / 1e9:.0f} GB free)")
    return round(frac, 3)


def score_vllm(fam: str, pairs: pd.DataFrame) -> np.ndarray:
    """Merge the LoRA adapter once, then score with vLLM in its own venv (~/vllm-sol); ~5x faster than HF."""
    merged = OUT / "merged"
    if not (merged / "config.json").exists():
        tok, model, *_ = load(adapter=True)
        model.save_pretrained(str(merged), safe_serialization=True)
        tok.save_pretrained(str(merged))
        del model
        import torch
        torch.cuda.empty_cache()
        log("merged adapter ->", merged)
    tag = f"{fam}_{len(pairs)}"
    src, dst = OUT / f"prompts_{tag}.parquet", OUT / f"vllm_{tag}.parquet"
    pp = pairs[["s1_id", "cand_id"]].copy()
    pp["prompt"] = prompts(fam, pp)
    pp.to_parquet(src, index=False)
    if dst.exists() and len(pd.read_parquet(dst, columns=["s1_id"])) == len(pp):
        log(f"reusing {dst}")
        got = pd.read_parquet(dst)
        assert (got.s1_id.to_numpy() == pp.s1_id.to_numpy()).all()
        return got.score.to_numpy().astype(np.float32)
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONNOUSERSITE"] = "1"
    r = subprocess.run([os.path.expanduser("~/vllm-sol/bin/python"), "-u", "scripts/vllm_score.py", str(src), str(merged),
                        str(dst), str(vllm_mem())], env=env)
    assert r.returncode == 0, "vLLM scoring failed"
    got = pd.read_parquet(dst)
    assert (got.s1_id.to_numpy() == pp.s1_id.to_numpy()).all()
    return got.score.to_numpy().astype(np.float32)


def oof_frame() -> pd.DataFrame:
    return pd.read_parquet(CS / "oof_q.parquet")                # v8 features + OOF q (scripts/v8_errors.py)


def selected(f: pd.DataFrame) -> pd.Series:
    return (f.has_rr == 1) & (f.q >= QLO) & (f.q <= QHI)


def score_oof() -> None:
    """Scores ALL bge-band OOF pairs, so both uses can be judged: extra column (v8-unsure pairs only) and replacing bge."""
    from sklearn.metrics import roc_auc_score
    f = oof_frame()
    sel = f[f.has_rr == 1].reset_index(drop=True)
    log(f"OOF band pairs to score: {len(sel):,} (v8-unsure subset {selected(sel).sum():,})")
    sel["rr2"] = score("train", sel)
    sel[["s1_id", "cand_id", "rr2"]].to_parquet(OUT / "oof_qwen.parquet", index=False)
    for name, m in (("all band", np.ones(len(sel), bool)), ("v8-unsure", selected(sel).to_numpy()),
                    ("no-address band", (sel.cand_noaddr == 1).to_numpy())):
        x = sel[m]
        log(f"AUC {name} ({len(x):,}): Model A {roc_auc_score(x.label, x.p):.4f} | bge {roc_auc_score(x.label, x.rr):.4f} | "
            f"qwen fine-tuned {roc_auc_score(x.label, x.rr2):.4f} | v8 q {roc_auc_score(x.label, x.q):.4f}")


FEATS2 = ["rr2", "has_rr2", "rr2_rank", "rr2_gap"]


def add_rr2(f: pd.DataFrame, q: pd.DataFrame) -> pd.DataFrame:
    f = f.drop(columns=[c for c in FEATS2 if c in f]).merge(q, on=["s1_id", "cand_id"], how="left")
    f["has_rr2"] = f.rr2.notna().astype(np.int8)
    g = f.groupby("s1_id").rr2
    f["rr2_rank"] = g.rank(ascending=False, method="first").fillna(99).astype(np.int16)
    f["rr2_gap"] = (g.transform("max") - f.rr2).astype(np.float32)
    return f


def stack() -> None:
    ns = {}
    exec(open("scripts/cand_side.py").read().rsplit("\n{", 1)[0], ns)
    BASE, NEW, PARAMS = ns["BASE"], ns["NEW"], ns["PARAMS"]
    base = oof_frame()
    qall = pd.read_parquet(OUT / "oof_qwen.parquet")
    keep = base.loc[selected(base), ["s1_id", "cand_id"]]
    f = add_rr2(base, qall.merge(keep, on=["s1_id", "cand_id"]))      # extra column only where test will have it
    # replacement: Qwen's score takes bge's place in the rr column for every band pair
    r = base.drop(columns=["rr"]).merge(qall.rename(columns={"rr2": "rr"}), on=["s1_id", "cand_id"], how="left")
    g = r.groupby("s1_id").rr
    r["rr_rank"] = g.rank(ascending=False, method="first").fillna(99).astype(np.int16)
    r["rr_gap"] = (g.transform("max") - r.rr).astype(np.float32)
    assert (r.s1_id.to_numpy() == f.s1_id.to_numpy()).all()
    runs = (("v8", f, BASE + NEW, None), ("v8 + qwen", f, BASE + NEW + FEATS2, "stack_fold"),
            ("qwen replaces bge", r, BASE + NEW, "repl_fold"))
    preds = {n: np.zeros(len(f), np.float32) for n, *_ in runs}
    for k in sorted(f.fold.unique()):
        tr, va = f.fold.to_numpy() != k, f.fold.to_numpy() == k
        for name, d, feats, save in runs:
            m = lgb.train(PARAMS, lgb.Dataset(d.loc[tr, feats], d.label[tr]), 3000,
                          valid_sets=[lgb.Dataset(d.loc[va, feats], d.label[va])], callbacks=[lgb.early_stopping(100, verbose=False)])
            preds[name][va] = m.predict(d.loc[va, feats], num_iteration=m.best_iteration)
            if save:
                m.save_model(str(OUT / f"{save}{k}.txt"))
        log(f"fold {k} done")
    truth = read_truth(cfg["paths"]["data_dir"])
    sp = pd.read_parquet(RD / "splits.parquet", columns=["s1_id", "country"]).set_index("s1_id")
    s1 = sp.loc[f.s1_id.unique()].reset_index()[["s1_id", "country"]]
    rows = []
    for name, p in preds.items():
        sc = pd.DataFrame({"s1_id": f.s1_id, "cand_id": f.cand_id, "p": p})
        for tau in np.round(np.arange(0.5, 0.91, 0.05), 2):
            rows.append({"model": name, "tau": float(tau), **evaluate(decide(sc, tau), truth, s1)})
    grid = pd.DataFrame(rows)
    best = {n: g.sort_values("macro_f05").iloc[-1] for n, g in grid.groupby("model")}
    b0, b1, b2 = best["v8"], best["v8 + qwen"], best["qwen replaces bge"]
    (OUT / "best.json").write_text(json.dumps({"tau": float(b1.tau), "macro_f05": float(b1.macro_f05),
                                               "base_tau": float(b0.tau), "base_f05": float(b0.macro_f05),
                                               "repl_tau": float(b2.tau), "repl_f05": float(b2.macro_f05)}, indent=2))
    fmt = lambda r: (f"τ={r.tau:.2f} → **{r.macro_f05:.4f}** (singletons {r.singletons:.4f}, "
                     + ", ".join(f"{k[4:]} {v:.4f}" for k, v in r.items() if str(k).startswith("f05_")) + ")")
    md = ["# v10: fine-tuned Qwen3-Reranker-4B (LoRA) as an extra stacker column — out-of-fold", "",
          f"- v8: {fmt(b0)}", f"- **v8 + qwen (extra column on v8-unsure pairs): {fmt(b1)}**",
          f"- **qwen replaces bge (all band pairs): {fmt(b2)}**", "", md_table(grid)]
    Path("docs/results/qwen_reranker.md").write_text("\n".join(md) + "\n")
    print("\n".join(md[:5]), flush=True)


def v8_test() -> pd.DataFrame:
    """v8 features + v8 stacker probability q for every test pair."""
    ns = {}
    exec(open("scripts/cand_side.py").read().rsplit("\n{", 1)[0], ns)
    t = pd.read_parquet(RD / "test_scores_matcher_v3.parquet")
    b = pd.read_parquet(RD / "reranker_v3" / "test_band_scores.parquet")[["s1_id", "cand_id", "rr"]]
    f = ns["stack_features"](t.merge(b, on=["s1_id", "cand_id"], how="left"))
    del t
    R = ns["Records"]("test")
    f = ns["add_new"](f, R, ns["comp_tables"]("test", R))
    feats = ns["BASE"] + ns["NEW"]
    models = [lgb.Booster(model_file=str(p)) for p in sorted(CS.glob("cs_fold*.txt"))]
    f["q"] = np.mean([m.predict(f[feats], num_iteration=m.best_iteration) for m in models], axis=0).astype(np.float32)
    return f


def test_sel() -> None:
    f = v8_test()
    sel = f.loc[selected(f), ["s1_id", "cand_id"]]
    sel.to_parquet(OUT / "test_sel.parquet", index=False)
    log(f"test pairs selected for Qwen: {len(sel):,} of {len(f):,}")


def score_test() -> None:
    """Qwen scores EVERY bge-band test pair (same pairs bge read), so Qwen can replace bge."""
    band = pd.read_parquet(RD / "reranker_v3" / "test_band_scores.parquet", columns=["s1_id", "cand_id"])
    log(f"test band pairs to score: {len(band):,}")
    band["rr2"] = score("test", band)
    band.to_parquet(OUT / "test_qwen.parquet", index=False)
    log("saved")


def submit() -> None:
    """Uses whichever Qwen variant won out-of-fold: 'qwen replaces bge' (the plan) or 'v8 + qwen' extra column."""
    ns = {}
    exec(open("scripts/cand_side.py").read().rsplit("\n{", 1)[0], ns)
    best = json.loads((OUT / "best.json").read_text())
    repl = best["repl_f05"] >= best["macro_f05"]
    f = v8_test()
    q = pd.read_parquet(OUT / "test_qwen.parquet")
    if repl:
        f = f.drop(columns=["rr"]).merge(q.rename(columns={"rr2": "rr"}), on=["s1_id", "cand_id"], how="left")
        g = f.groupby("s1_id").rr
        f["rr_rank"] = g.rank(ascending=False, method="first").fillna(99).astype(np.int16)
        f["rr_gap"] = (g.transform("max") - f.rr).astype(np.float32)
        feats, pat, tau, name = ns["BASE"] + ns["NEW"], "repl_fold*.txt", best["repl_tau"], "qwen replaces bge"
    else:
        keep = pd.read_parquet(OUT / "test_sel.parquet")
        f = add_rr2(f, q.merge(keep, on=["s1_id", "cand_id"]))
        feats, pat, tau, name = ns["BASE"] + ns["NEW"] + FEATS2, "stack_fold*.txt", best["tau"], "v8 + qwen"
    log(f"variant: {name} (OOF repl {best['repl_f05']:.4f} | extra {best['macro_f05']:.4f} | v8 {best['base_f05']:.4f})")
    models = [lgb.Booster(model_file=str(p)) for p in sorted(OUT.glob(pat))]
    p = np.mean([m.predict(f[feats], num_iteration=m.best_iteration) for m in models], axis=0)
    pred = decide(pd.DataFrame({"s1_id": f.s1_id, "cand_id": f.cand_id, "p": p}), tau)
    rec = pd.read_parquet(records_path(cfg, "test"), columns=["entity_id", "source", "country"])
    s1_all = rec.entity_id[rec.source == 1].tolist()
    out = Path(f"artifacts/submissions/{SUB}")
    write_submission(s1_all, pred, f.groupby("s1_id").cand_id.agg(set).to_dict(), out)
    n = pd.Series({s: len(v) for s, v in pred.items()}).reindex(s1_all, fill_value=0)
    ctry = rec.set_index("entity_id").country.reindex(n.index).to_numpy()
    log({"variant": name, "tau": tau, "mean_matches": float(n.mean()), "empty_share": float((n == 0).mean()),
         **{c: float(n[ctry == c].mean()) for c in ["US", "India", "France"]}})
    v = subprocess.run(["python3", cfg["paths"]["validator"], "--matching", str(out / "matching_results.tsv"),
                        "--candidate", str(out / "candidate_pairs.tsv"), "--test-dir",
                        str(Path(cfg["paths"]["data_dir"]) / "test")], capture_output=True, text=True)
    print(v.stdout[-800:], f"validator exit={v.returncode}", flush=True)


{"train": train, "score_oof": score_oof, "stack": stack, "test_sel": test_sel, "score_test": score_test,
 "submit": submit}[sys.argv[1]]()
