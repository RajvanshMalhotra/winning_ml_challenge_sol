"""Run in the vLLM venv (~/vllm-sol): score prompts with a Qwen3-Reranker model. score = logprob(yes) - logprob(no)
logit(yes) - logit(no) of the first answer token (processed logits, only yes/no allowed). Writes partial results every chunk and resumes from them.
usage: vllm_score.py <prompts.parquet> <model_path> <out.parquet> [mem_fraction]"""
import os
import sys
import time

import numpy as np
import pandas as pd

src, model, dst = sys.argv[1:4]
mem = float(sys.argv[4]) if len(sys.argv) > 4 else 0.3
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

tok = AutoTokenizer.from_pretrained(model)
tok.truncation_side = "left"
yes, no = tok.convert_tokens_to_ids("yes"), tok.convert_tokens_to_ids("no")
P = pd.read_parquet(src)
llm = LLM(model=model, dtype="bfloat16", gpu_memory_utilization=mem, max_model_len=512, enable_prefix_caching=True,
          max_num_seqs=512, max_num_batched_tokens=32768,
          logprobs_mode="processed_logits")
sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=2, allowed_token_ids=[yes, no])  # returns the 2 logits
CH = 100_000
parts, t0 = [], time.perf_counter()
for i in range(0, len(P), CH):
    part = f"{dst}.part{i // CH:04d}.npy"
    if os.path.exists(part):
        parts.append(np.load(part)); continue
    ids = tok(P.prompt.iloc[i:i + CH].tolist(), truncation=True, max_length=320)["input_ids"]   # batched (fast), same truncation as HF
    outs = llm.generate([{"prompt_token_ids": x} for x in ids], sp, use_tqdm=False)
    s = np.empty(len(outs), np.float32)
    for j, o in enumerate(outs):
        lp = o.outputs[0].logprobs[0]
        s[j] = lp[yes].logprob - lp[no].logprob
    np.save(part, s); parts.append(s)
    done = min(i + CH, len(P))
    print(f"[{time.perf_counter() - t0:6.0f}s] scored {done:,}/{len(P):,} | {done / (time.perf_counter() - t0):.1f} pairs/s", flush=True)
P[["s1_id", "cand_id"]].assign(score=np.concatenate(parts)).to_parquet(dst, index=False)
print("saved", dst, flush=True)
os._exit(0)   # vLLM can hang at interpreter shutdown; results are already on disk
