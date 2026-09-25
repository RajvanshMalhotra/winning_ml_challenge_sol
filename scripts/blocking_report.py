"""Blocking quality on train: recall ceiling, pair quality, candidates per entity, and oracle macro F0.5
(the score a perfect matcher would get on this candidate set). usage: blocking_report.py <config>"""
import sys

import numpy as np
import pandas as pd

from ber.blocking.sparse import cand_sparse_path
from ber.config import load_config
from ber.io import read_truth
from ber.split import splits_path

cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "configs/v2.yaml")
import pyarrow.parquet as pq
_cols = pq.read_schema(cand_sparse_path(cfg, "train")).names
cands = pd.read_parquet(cand_sparse_path(cfg, "train"),
                        columns=[c for c in ["s1_id", "cand_id", "tfidf_rank", "key_hits", "name_sim", "name_key"] if c in _cols])
truth = read_truth(cfg["paths"]["data_dir"])
splits = pd.read_parquet(splits_path(cfg), columns=["s1_id", "country", "n_matches"])

# vectorized: per S1 -> number of candidates, number of true matches found among them
tp = pd.DataFrame([(s, c) for s, m in truth.items() for c in m], columns=["s1_id", "cand_id"])
hit = tp.merge(cands, on=["s1_id", "cand_id"], how="inner")
r = splits.set_index("s1_id")
r["n_true"] = r.n_matches
r["n_found"] = hit.groupby("s1_id").size().reindex(r.index, fill_value=0)
r["n_cands"] = cands.groupby("s1_id").size().reindex(r.index, fill_value=0)
# oracle: a perfect matcher predicts exactly the found true pairs -> precision 1, recall = found/true
rec = np.where(r.n_true > 0, r.n_found / r.n_true.clip(lower=1), 1.0)
r["oracle_f05"] = np.where(r.n_true == 0, 1.0, np.where(r.n_found == 0, 0.0, 1.25 * rec / (0.25 + rec)))
r = r.reset_index()

def summary(g: pd.DataFrame) -> dict:
    return {"entities": len(g),
            "block_recall": g.n_found.sum() / max(g.n_true.sum(), 1),
            "pair_quality": g.n_found.sum() / max(g.n_cands.sum(), 1),
            "cands_mean": g.n_cands.mean(), "cands_p95": g.n_cands.quantile(0.95),
            "entities_all_found": (g.n_found == g.n_true).mean(),
            "oracle_macro_f05": g.oracle_f05.mean()}

out = pd.DataFrame({"ALL": summary(r), **{c: summary(g) for c, g in r.groupby("country")}}).T
extra = {"recall_tfidf_only": (hit.tfidf_rank < 999).sum() / r.n_true.sum(),
         "recall_keys_only": (hit.key_hits > 0).sum() / r.n_true.sum()}
if "name_sim" in hit:
    extra["recall_name_only"] = ((hit.name_sim > 0) | (hit.name_key > 0)).sum() / r.n_true.sum()
for k in (10, 20, 30, 50):
    extra[f"recall_tfidf_top{k}"] = (hit.tfidf_rank <= k).sum() / r.n_true.sum()
pd.set_option("display.width", 200)
print(out.round(4).to_string())
print({k: round(v, 4) for k, v in extra.items()})
print("oracle F0.5 by match-count bucket:",
      r.groupby(pd.cut(r.n_matches, [-1, 0, 1, 3, 5, 99], labels=["0", "1", "2-3", "4-5", "6+"]),
                observed=True).oracle_f05.mean().round(4).to_dict())
