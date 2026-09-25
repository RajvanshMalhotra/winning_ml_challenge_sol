"""Blocking quality on train: recall ceiling, pair quality, candidates per entity, and oracle macro F0.5
(the score a perfect matcher would get on this candidate set). usage: blocking_report.py <config>"""
import sys

import numpy as np
import pandas as pd

from ber.blocking.sparse import cand_sparse_path
from ber.config import load_config
from ber.evaluate import f05
from ber.io import read_truth
from ber.split import splits_path

cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "configs/v2.yaml")
cands = pd.read_parquet(cand_sparse_path(cfg, "train"), columns=["s1_id", "cand_id", "tfidf_rank", "key_hits"])
truth = read_truth(cfg["paths"]["data_dir"])
splits = pd.read_parquet(splits_path(cfg), columns=["s1_id", "country", "n_matches"])

cand_sets = cands.groupby("s1_id").cand_id.agg(set)
rows = []
for s1, country, n in zip(splits.s1_id, splits.country, splits.n_matches):
    t = truth[s1]
    c = cand_sets.get(s1, set())
    found = t & c
    rows.append((s1, country, n, len(t), len(found), len(c), f05(found, t)))  # oracle: predict exactly the found true pairs
r = pd.DataFrame(rows, columns=["s1_id", "country", "n_matches", "n_true", "n_found", "n_cands", "oracle_f05"])

def summary(g: pd.DataFrame) -> dict:
    return {"entities": len(g),
            "block_recall": g.n_found.sum() / max(g.n_true.sum(), 1),
            "pair_quality": g.n_found.sum() / max(g.n_cands.sum(), 1),
            "cands_mean": g.n_cands.mean(), "cands_p95": g.n_cands.quantile(0.95),
            "entities_all_found": (g.n_found == g.n_true).mean(),
            "oracle_macro_f05": g.oracle_f05.mean()}

out = pd.DataFrame({"ALL": summary(r), **{c: summary(g) for c, g in r.groupby("country")}}).T
tp = cands.merge(pd.DataFrame([(s, c) for s, m in truth.items() for c in m], columns=["s1_id", "cand_id"]))
extra = {"recall_tfidf_only": (tp.tfidf_rank < 999).sum() / r.n_true.sum(),
         "recall_keys_only": (tp.key_hits > 0).sum() / r.n_true.sum()}
for k in (10, 20, 30, 50):
    extra[f"recall_tfidf_top{k}"] = (tp.tfidf_rank <= k).sum() / r.n_true.sum()
pd.set_option("display.width", 200)
print(out.round(4).to_string())
print({k: round(v, 4) for k, v in extra.items()})
print("oracle F0.5 by match-count bucket:",
      r.groupby(pd.cut(r.n_matches, [-1, 0, 1, 3, 5, 99], labels=["0", "1", "2-3", "4-5", "6+"]),
                observed=True).oracle_f05.mean().round(4).to_dict())
