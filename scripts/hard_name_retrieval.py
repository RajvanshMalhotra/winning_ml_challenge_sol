"""Embedding retrieval for HARD S2/S3 records (native-script names or empty address): embed names with
Qwen3-Embedding (default 0.6B), find each hard record's nearest S1 names within its country, save the new pairs,
and report block recall before/after on train.   usage: hard_name_retrieval.py <family> [model_id] [top_k]"""
import sys
import time

import numpy as np
import pandas as pd
import torch

from ber.blocking.sparse import cand_sparse_path
from ber.config import load_config, run_dir
from ber.contrastive.encoders import EncoderSpec, encode, load_encoder
from ber.io import read_truth
from ber.normalize import records_path

FAMILY = sys.argv[1] if len(sys.argv) > 1 else "train"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "Qwen/Qwen3-Embedding-0.6B"
TOP_K = int(sys.argv[3]) if len(sys.argv) > 3 else 10
INSTRUCT = ("Instruct: Given a business name, retrieve the same business written differently "
            "(other script, abbreviation, typos)\nQuery: ")
T0 = time.perf_counter()
log = lambda *a: print(f"[{time.perf_counter() - T0:6.0f}s]", *a, flush=True)
cfg = load_config("configs/v2.yaml")
RD = run_dir(cfg)

rec = pd.read_parquet(records_path(cfg, FAMILY), columns=["entity_id", "source", "country", "name_raw", "addr_raw"])
native = rec.name_raw.str.contains(r"[ऀ-෿]", regex=True)
hard = rec[(rec.source != 1) & (native | (rec.addr_raw == ""))]
s1 = rec[rec.source == 1]
log(f"{FAMILY}: {len(s1):,} S1 names + {len(hard):,} hard records to embed")

spec = EncoderSpec("qwen-name", MODEL, padding_side="left", max_seq_length=48)
model = load_encoder(spec, mem_fraction=0.4)
e_s1 = encode(model, s1.name_raw.tolist(), "", 512)
log("S1 names embedded")
e_hd = encode(model, hard.name_raw.tolist(), INSTRUCT, 512)
log("hard records embedded")
del model
torch.cuda.empty_cache()

parts = []
for country in hard.country.unique():
    qi = np.flatnonzero(hard.country.to_numpy() == country)
    di = np.flatnonzero(s1.country.to_numpy() == country)
    if len(qi) == 0 or len(di) == 0:
        continue
    docs = torch.from_numpy(e_s1[di]).cuda()  # float16, L2-normalized
    sims, idx = [], []
    for a in range(0, len(qi), 8192):
        q = torch.from_numpy(e_hd[qi[a:a + 8192]]).cuda()
        v, j = torch.topk(q @ docs.T, k=min(TOP_K, len(di)), dim=1)
        sims.append(v.float().cpu().numpy()); idx.append(j.cpu().numpy())
    sims, idx = np.concatenate(sims), np.concatenate(idx)
    parts.append(pd.DataFrame({"s1_id": s1.entity_id.to_numpy()[di][idx.ravel()],
                               "cand_id": np.repeat(hard.entity_id.to_numpy()[qi], idx.shape[1]),
                               "emb_sim": sims.ravel().astype(np.float32),
                               "emb_rank": np.tile(np.arange(1, idx.shape[1] + 1, dtype=np.int16), len(qi))}))
    del docs
    log(f"[{country}] {len(qi):,} hard records x {len(di):,} S1 names searched")
new = pd.concat(parts, ignore_index=True)
out = RD / f"cand_hardname_{FAMILY}.parquet"
new.to_parquet(out, index=False)
log(f"saved {len(new):,} pairs -> {out}")

if FAMILY == "train":
    base = pd.read_parquet(cand_sparse_path(cfg, "train"), columns=["s1_id", "cand_id"])
    extra = [pd.read_parquet(p, columns=["s1_id", "cand_id"]) for p in sorted((RD / "block_cache_train").glob("nameonly_*.parquet"))]
    base = pd.concat([base] + extra, ignore_index=True).drop_duplicates()
    truth = read_truth(cfg["paths"]["data_dir"])
    tp = pd.DataFrame([(s, c) for s, m in truth.items() for c in m], columns=["s1_id", "cand_id"])
    tp["country"] = s1.set_index("entity_id").country.reindex(tp.s1_id).to_numpy()
    hit_b = tp.merge(base.assign(b=1), how="left").b.notna().to_numpy()
    hit_n = tp.merge(new[["s1_id", "cand_id"]].assign(n=1), how="left").n.notna().to_numpy()
    tp["before"], tp["after"] = hit_b, hit_b | hit_n
    print(f"name-only caches included in 'before': {len(extra)}")
    for c, d in [("ALL", tp)] + list(tp.groupby("country")):
        print(f"  {c:<6} recall before {d.before.mean():.4f} -> after {d.after.mean():.4f} "
              f"(+{d.after.mean() - d.before.mean():.4f}); missed pairs {int((~d.before).sum()):,} -> {int((~d.after).sum()):,}")
    for k in (1, 3, 5, 10):
        kk = new[new.emb_rank <= k]
        h = tp.merge(kk[["s1_id", "cand_id"]].assign(n=1), how="left").n.notna()
        print(f"  top-{k}: {len(kk):,} new pairs, recovers {int((h & ~tp.before).sum()):,} missed true pairs")
