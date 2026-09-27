"""Short French fine-tune of the reranker (continue from reranker_qwen3) on French practice pairs mixed with US/India
A_train pairs, then an honest check and France-only test scoring.
usage: qwen_fr.py train | check | score_test
  train      : LoRA continue-training, time budget QF_MIN minutes -> reranker_qwen_fr/adapter
  check      : AUC old (qwen3) vs new on held-out French pairs AND on a US/India validation sample (vLLM)
  score_test : new model on France's uncertain test pairs -> reranker_qwen_fr/test_qwen_fr.parquet"""
import math
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

os.environ.setdefault("QR_PROMPT", "short")
os.environ["QR_DIR"] = "reranker_qwen_fr"
ACTION = sys.argv[1]
sys.argv = ["qwen_reranker.py", "none"]
ns = {"__name__": "qf"}
exec(open("scripts/qwen_reranker.py").read().rsplit("\n{", 1)[0], ns)
OUT, RD, cfg, log = ns["OUT"], ns["RD"], ns["cfg"], ns["log"]
wrap = lambda q, d: f"<|im_start|>user\nA: {q}\nB: {d}\nSame business?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
OLD = RD / "reranker_qwen3"


def train() -> None:
    import torch
    from peft import PeftModel
    fr = pd.read_parquet(OUT / "fr_train.parquet")
    P = [wrap(a, b) for a, b in zip(fr.s1_text, fr.cand_text)]
    y = fr.label.tolist()
    ns["N_S1"] = 3000                                                      # keep US/India in the mix (scale anchor)
    ui = ns["training_pairs"]()
    P += ns["prompts"]("train", ui)
    y += ui.label.astype(int).tolist()
    order = np.random.default_rng(0).permutation(len(P))
    P, y = [P[i] for i in order], torch.tensor([y[i] for i in order], dtype=torch.float32)
    log(f"train prompts {len(P):,} (French {len(fr):,}, US/India {len(ui):,})")
    tok, model, yes, no = ns["load"](adapter=False)
    model.config.use_cache = False
    model = PeftModel.from_pretrained(model, str(OLD / "adapter"), is_trainable=True)
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    bs = int(os.environ.get("QF_BS", 16)); accum = max(1, 64 // bs); budget = float(os.environ.get("QF_MIN", 45)) * 60
    n_steps = math.ceil(len(P) / bs / accum)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=float(os.environ.get("QF_LR", 3e-5)))
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
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % 100 == 0:
                el = time.perf_counter() - t0
                log(f"step {step}/{n_steps} loss {run / 100:.4f} | {(i + bs) / el:.1f} pairs/s | ETA {(len(P) - i) / ((i + bs) / el) / 60:.0f} min")
                run = 0.0
            if time.perf_counter() - t0 > budget:
                log(f"time budget reached at step {step}/{n_steps}")
                break
    model.save_pretrained(str(OUT / "adapter"))
    log("saved adapter")


def vllm(model_dir, prompts: list, tag: str) -> np.ndarray:
    src, dst = OUT / f"prompts_{tag}.parquet", OUT / f"vllm_{tag}.parquet"
    pd.DataFrame({"s1_id": [str(i) for i in range(len(prompts))], "cand_id": [str(i) for i in range(len(prompts))],
                  "prompt": prompts}).to_parquet(src, index=False)
    for f in OUT.glob(f"vllm_{tag}.parquet*"):
        f.unlink()
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONNOUSERSITE"] = "1"
    r = subprocess.run([os.environ.get("VLLM_PY", os.path.expanduser("~/vllm-sol/bin/python")), "-u", "scripts/vllm_score.py",
                        str(src), str(model_dir), str(dst), str(ns["vllm_mem"]())], env=env)
    assert r.returncode == 0
    return pd.read_parquet(dst).score.to_numpy()


def merged_new():
    m = OUT / "merged"
    if not (m / "config.json").exists():
        from peft import PeftModel
        import torch
        tok, model, *_ = ns["load"](adapter=False)
        model = PeftModel.from_pretrained(model, str(OUT / "adapter")).merge_and_unload()
        model.save_pretrained(str(m), safe_serialization=True)
        tok.save_pretrained(str(m))
        del model
        torch.cuda.empty_cache()
    return m


def check() -> None:
    from sklearn.metrics import roc_auc_score
    fr = pd.read_parquet(OUT / "fr_check.parquet")
    pf = [wrap(a, b) for a, b in zip(fr.s1_text, fr.cand_text)]
    o = pd.read_parquet(RD / "cand_side" / "oof_q.parquet", columns=["s1_id", "cand_id", "label", "has_rr"])
    ui = o[o.has_rr == 1].sample(20_000, random_state=0).reset_index(drop=True)
    pu = ns["prompts"]("train", ui)
    new = merged_new()
    for name, prompts, lab in (("French held-out", pf, fr.label.to_numpy()), ("US/India validation", pu, ui.label.to_numpy())):
        a_old = roc_auc_score(lab, vllm(OLD / "merged", prompts, "chk_old"))
        a_new = roc_auc_score(lab, vllm(new, prompts, "chk_new"))
        print(f"{name}: AUC old {a_old:.4f} -> new {a_new:.4f} ({len(lab):,} pairs)", flush=True)


def score_test() -> None:
    t = pd.read_parquet(OLD / "test_qwen.parquet", columns=["s1_id", "cand_id"])
    rec = pd.read_parquet(ns["records_path"](cfg, "test"), columns=["entity_id", "country"]).set_index("entity_id").country
    t = t[rec.reindex(t.s1_id).to_numpy() == "France"].reset_index(drop=True)
    log(f"France uncertain test pairs: {len(t):,}")
    t["rr_fr"] = vllm(merged_new(), ns["prompts"]("test", t), "test_fr")
    t.to_parquet(OUT / "test_qwen_fr.parquet", index=False)
    log("saved")


{"train": train, "check": check, "score_test": score_test, "merge": lambda: print("merged", merged_new(), flush=True)}[ACTION]()
