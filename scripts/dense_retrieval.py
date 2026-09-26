"""Dense candidate channel: fine-tuned bge-m3 embeddings (cached .npy), top-K S2/S3 per S1 within its state group
(+ unknown-state records; S1s with unknown state search the whole country). Writes cand_dense_<family>.parquet and,
on train, the recall it adds over the existing channels.   usage: dense_retrieval.py <family> [K]"""
import json
import sys
import time

import numpy as np
import pandas as pd
import torch

from ber.config import load_config, run_dir
from ber.normalize import records_path

FAM = sys.argv[1]
K = int(sys.argv[2]) if len(sys.argv) > 2 else 30
cfg = load_config("configs/v2.yaml")
RD = run_dir(cfg)
T0 = time.perf_counter()
log = lambda *a: print(f"[{time.perf_counter() - T0:6.0f}s]", *a, flush=True)

rec = pd.read_parquet(records_path(cfg, FAM), columns=["entity_id", "source", "country", "state"])
emb = np.load(RD / f"emb_{FAM}_bgem3_bgem3__finetune.npy", mmap_mode="r")
assert emb.shape[0] == len(rec), "embedding rows must follow records order"
groups = json.loads((RD / "state_groups.json").read_text())
rec["grp"] = [groups.get(c, {}).get(s, s) for c, s in zip(rec.country, rec.state)]
ids = rec.entity_id.to_numpy()
torch.cuda.set_per_process_memory_fraction(cfg["gpu"]["mem_fraction"])
parts = []
for country, cr in rec.groupby("country"):
    s1 = cr[cr.source == 1]
    oth = cr[cr.source != 1]
    unk = oth.index[oth.grp.to_numpy() == ""]
    for g, q in s1.groupby("grp"):
        d_idx = oth.index.to_numpy() if g == "" else np.concatenate([oth.index[oth.grp.to_numpy() == g], unk])
        if len(d_idx) == 0:
            continue
        D = torch.from_numpy(np.asarray(emb[d_idx], dtype=np.float16)).cuda()
        k = min(K, len(d_idx))
        q_idx = q.index.to_numpy()
        for a in range(0, len(q_idx), 512):
            Q = torch.from_numpy(np.asarray(emb[q_idx[a:a + 512]], dtype=np.float16)).cuda()
            v, j = torch.topk(Q @ D.T, k=k, dim=1)
            parts.append(pd.DataFrame({"s1_id": np.repeat(ids[q_idx[a:a + 512]], k),
                                       "cand_id": ids[d_idx[j.cpu().numpy().ravel()]],
                                       "dense_sim": v.float().cpu().numpy().ravel(),
                                       "dense_rank": np.tile(np.arange(1, k + 1, dtype=np.int16), len(q_idx[a:a + 512]))}))
        del D
        torch.cuda.empty_cache()
    log(f"[{country}] {len(s1):,} S1 searched")
dense = pd.concat(parts, ignore_index=True)
dense.to_parquet(RD / f"cand_dense_{FAM}.parquet", index=False)
log(f"saved {len(dense):,} dense pairs")

if FAM == "train":
    from ber.io import read_truth
    truth = read_truth(cfg["paths"]["data_dir"])
    tp = pd.DataFrame([(s, c) for s, m in truth.items() for c in m], columns=["s1_id", "cand_id"])
    tp["country"] = rec.set_index("entity_id").country.reindex(tp.s1_id).to_numpy()
    base = pd.concat([pd.read_parquet(RD / "cand_sparse_train.parquet", columns=["s1_id", "cand_id"]),
                      pd.read_parquet(RD / "cand_hardname_train.parquet", columns=["s1_id", "cand_id"])]).drop_duplicates()
    hb = tp.merge(base.assign(b=1), how="left").b.notna().to_numpy()
    for kk in (10, 20, 30):
        hd = tp.merge(dense[dense.dense_rank <= kk][["s1_id", "cand_id"]].assign(d=1), how="left").d.notna().to_numpy()
        for c in ["India", "US"]:
            m = tp.country.to_numpy() == c
            print(f"  top-{kk} {c}: dense alone {hd[m].mean():.4f} | sparse+hardname {hb[m].mean():.4f} -> "
                  f"with dense {(hb | hd)[m].mean():.4f}", flush=True)
