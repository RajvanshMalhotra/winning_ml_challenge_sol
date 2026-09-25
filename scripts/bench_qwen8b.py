"""Measure Qwen3-Embedding-8B: exact parameter count and records/s on real name+address texts (H100, shared)."""
import time

import pandas as pd
import torch

from ber.contrastive.encoders import ENCODERS, encode, load_encoder
from ber.text import record_text

spec = ENCODERS["qwen8b"]
model = load_encoder(spec, mem_fraction=0.4)
n_params = sum(p.numel() for p in model.parameters())
print(f"parameters: {n_params:,} ({n_params / 1e9:.3f} B)", flush=True)
rec = pd.read_parquet("artifacts/v2/records_train.parquet", columns=["name_raw", "addr_raw", "country"]).sample(20_000, random_state=0)
import sys
NAME_ONLY = "--names" in sys.argv
texts = rec.name_raw.tolist() if NAME_ONLY else [record_text(n, a, c) for n, a, c in zip(rec.name_raw, rec.addr_raw, rec.country)]
print("name-only" if NAME_ONLY else "name+address", flush=True)
encode(model, texts[:512], spec.doc_prefix, 256)  # warm-up
for bs in ((512, 1024) if NAME_ONLY else (128, 256, 512)):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    encode(model, texts[:8000], spec.doc_prefix, bs)
    torch.cuda.synchronize(); dt = time.perf_counter() - t0
    print(f"batch {bs}: {8000 / dt:,.0f} records/s  -> 12.5M train records ~ {12.5e6 / (8000 / dt) / 3600:.1f} h, "
          f"24.2M train+test ~ {24.2e6 / (8000 / dt) / 3600:.1f} h | peak mem {torch.cuda.max_memory_allocated() / 1e9:.1f} GB", flush=True)
