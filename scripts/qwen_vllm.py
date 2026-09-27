"""Helpers (run in ber-sol) for fast Qwen3-Reranker scoring with vLLM, which lives in its own venv (~/vllm-sol).
usage: qwen_vllm.py merge                              -> LoRA adapter merged into $QR_DIR/merged (for vLLM)
       qwen_vllm.py prompts <fam> <pairs.parquet> <out.parquet>   -> s1_id, cand_id, prompt (same prompt as training)
       qwen_vllm.py hf <prompts.parquet> <model|base> <out.parquet> -> HF reference scores (for the agreement check)"""
import sys

import numpy as np
import pandas as pd

ACTION = sys.argv[1:]
sys.argv = [sys.argv[0], "score_oof"]
ns = {"__name__": "qr"}
exec(open("scripts/qwen_reranker.py").read().rsplit("\n{", 1)[0], ns)
OUT, log = ns["OUT"], ns["log"]

if ACTION[0] == "merge":
    from transformers import AutoTokenizer
    tok, model, *_ = ns["load"](adapter=True)
    model.save_pretrained(str(OUT / "merged"), safe_serialization=True)
    AutoTokenizer.from_pretrained(ns["BASE_MODEL"]).save_pretrained(str(OUT / "merged"))
    log("saved merged model", OUT / "merged")
elif ACTION[0] == "prompts":
    fam, src, dst = ACTION[1:4]
    p = pd.read_parquet(src, columns=["s1_id", "cand_id"])
    p["prompt"] = ns["prompts"](fam, p)
    p.to_parquet(dst, index=False)
    log(f"{len(p):,} prompts -> {dst}")
elif ACTION[0] == "hf":
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    src, model_path, dst = ACTION[1:4]
    model_path = ns["BASE_MODEL"] if model_path == "base" else model_path
    tok = AutoTokenizer.from_pretrained(model_path, padding_side="left"); tok.truncation_side = "left"
    m = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16).cuda().eval()
    yes, no = tok.convert_tokens_to_ids("yes"), tok.convert_tokens_to_ids("no")
    P = pd.read_parquet(src)
    out = np.empty(len(P), np.float32)
    t0 = ns["time"].perf_counter()
    with torch.inference_mode():
        for i in range(0, len(P), 128):
            enc = tok(P.prompt.iloc[i:i + 128].tolist(), padding=True, truncation=True, max_length=320, return_tensors="pt").to("cuda")
            lg = m(**enc, logits_to_keep=1).logits[:, -1, :]
            out[i:i + 128] = (lg[:, yes] - lg[:, no]).float().cpu().numpy()
    log(f"HF {len(P) / (ns['time'].perf_counter() - t0):.1f} pairs/s")
    P[["s1_id", "cand_id"]].assign(score=out).to_parquet(dst, index=False)
