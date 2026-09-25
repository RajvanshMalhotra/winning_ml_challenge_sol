"""Per-blocking-method candidate counts and recall on train (plus name-only caches and hard-name pairs if present)."""
from pathlib import Path

import numpy as np
import pandas as pd

from ber.blocking.sparse import cand_sparse_path
from ber.config import load_config, run_dir
from ber.io import read_truth
from ber.split import splits_path

cfg = load_config("configs/v2.yaml")
RD = run_dir(cfg)
c = pd.read_parquet(cand_sparse_path(cfg, "train"), columns=["s1_id", "cand_id", "tfidf_rank", "key_hits"])
no = [pd.read_parquet(p) for p in sorted((RD / "block_cache_train").glob("nameonly_*.parquet"))]
hn = RD / "cand_hardname_train.parquet"
sp = pd.read_parquet(splits_path(cfg), columns=["s1_id", "country"])
country = sp.set_index("s1_id").country
truth = read_truth(cfg["paths"]["data_dir"])
tp = pd.DataFrame([(s, x) for s, m in truth.items() for x in m], columns=["s1_id", "cand_id"])
tp["country"] = country.reindex(tp.s1_id).to_numpy()
n_s1 = sp.country.value_counts()

methods = {f"TF-IDF top-{k}": c[c.tfidf_rank <= k] for k in (10, 20, 50)}
methods["Key blocks"] = c[c.key_hits > 0]
methods["TF-IDF top-50 + keys (current)"] = c
if no:
    methods["Name-only (empty-address)"] = pd.concat(no)
if hn.exists():
    h = pd.read_parquet(hn)
    methods["Qwen-0.6B hard names (top-10)"] = h
union = [c[["s1_id", "cand_id"]]] + [m[["s1_id", "cand_id"]] for k, m in methods.items() if k.startswith(("Name-only", "Qwen"))]
if len(union) > 1:
    methods["UNION of all"] = pd.concat(union).drop_duplicates()

rows = []
for name, m in methods.items():
    m = m[["s1_id", "cand_id"]].drop_duplicates()
    hit = tp.merge(m, on=["s1_id", "cand_id"])
    mc = country.reindex(m.s1_id).to_numpy()
    for ct in ["ALL"] + sorted(n_s1.index):
        sel_t = tp if ct == "ALL" else tp[tp.country == ct]
        sel_h = hit if ct == "ALL" else hit[hit.country == ct]
        n_pairs = len(m) if ct == "ALL" else int((mc == ct).sum())
        ents = n_s1.sum() if ct == "ALL" else n_s1[ct]
        rows.append((name, ct, n_pairs, n_pairs / ents, len(sel_h) / len(sel_t), len(sel_h) / max(n_pairs, 1)))
df = pd.DataFrame(rows, columns=["method", "country", "pairs", "cands_per_S1", "recall", "pair_quality"])
pd.set_option("display.width", 200)
print(f"name-only caches: {len(no)} | hard-name file: {hn.exists()}")
print(df.to_string(index=False, formatters={"pairs": "{:,}".format, "cands_per_S1": "{:.1f}".format,
                                            "recall": "{:.4f}".format, "pair_quality": "{:.4f}".format}))
