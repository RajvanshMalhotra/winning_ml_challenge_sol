import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

from ber.config import log_metrics, run_dir
from ber.evaluate import candidates_per_entity, pair_completeness
from ber.io import read_truth, truth_pairs
from ber.normalize import records_path
from ber.split import splits_path

EMPTY = pd.DataFrame({"s1_id": pd.Series(dtype=str), "cand_id": pd.Series(dtype=str)})


def tfidf_topk(s1: pd.DataFrame, others: pd.DataFrame, cfg_tfidf: dict) -> pd.DataFrame:
    if len(s1) == 0 or len(others) == 0:
        return EMPTY.assign(tfidf_sim=pd.Series(dtype="float32"))
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=tuple(cfg_tfidf["ngram_range"]),
                          min_df=cfg_tfidf["min_df"], max_df=cfg_tfidf.get("max_df", 1.0),
                          sublinear_tf=True, dtype=np.float32)
    vec.fit(pd.concat([s1.block_text, others.block_text]))
    a, b = vec.transform(s1.block_text), vec.transform(others.block_text)
    c = sp_matmul_topn(a, b.T.tocsr(), top_n=cfg_tfidf["top_k"], threshold=cfg_tfidf["min_sim"],
                       sort=True, n_threads=cfg_tfidf["n_threads"]).tocoo()
    return pd.DataFrame({"s1_id": s1.entity_id.values[c.row], "cand_id": others.entity_id.values[c.col],
                         "tfidf_sim": c.data.astype(np.float32)})


def _rarest(tokens: pd.Series, n: int, min_len: int) -> pd.DataFrame:
    """The n lowest-document-frequency tokens per row. `tokens` holds token lists on a RangeIndex."""
    ex = tokens.explode().dropna().rename("tok").rename_axis("row").reset_index()
    ex = ex[(ex.tok.str.len() >= min_len) & ~ex.tok.str.isdigit()].drop_duplicates()
    ex["df"] = ex.groupby("tok").tok.transform("size")
    return ex.sort_values(["row", "df", "tok"]).groupby("row").head(n)[["row", "tok"]]


def key_blocks(recs: pd.DataFrame, cfg_keys: dict) -> pd.DataFrame:
    recs = recs.reset_index(drop=True)
    name = recs.name_domain.where(recs.name_domain != "", recs.name_core)
    name_t = _rarest(name.str.split(), cfg_keys["n_name_tokens"], cfg_keys["min_token_len"])
    addr_t = _rarest(recs.addr_norm.str.split(), cfg_keys["n_addr_tokens"], cfg_keys["min_token_len"])
    hn = recs.house_no.rename_axis("row").reset_index()
    hn = hn[hn.house_no != ""]
    k1 = addr_t.merge(hn, on="row")
    k1["key"] = "h|" + k1.house_no + "|" + k1.tok
    k2 = name_t.merge(addr_t, on="row", suffixes=("_n", "_a"))
    k2["key"] = "n|" + k2.tok_n + "|" + k2.tok_a
    keys = pd.concat([k1[["row", "key"]], k2[["row", "key"]]]).drop_duplicates()
    keys = keys[keys.groupby("key").row.transform("size") <= cfg_keys["max_block_size"]]
    is_s1 = recs.source.values[keys.row.values] == 1
    pairs = keys[is_s1].merge(keys[~is_s1], on="key", suffixes=("_s1", "_c"))
    pairs = pairs.groupby(["row_s1", "row_c"]).size().rename("key_hits").reset_index()
    return pd.DataFrame({"s1_id": recs.entity_id.values[pairs.row_s1.values],
                         "cand_id": recs.entity_id.values[pairs.row_c.values],
                         "key_hits": pairs.key_hits.values.astype(np.int16)})


def block_family(recs: pd.DataFrame, cfg_blocking: dict) -> pd.DataFrame:
    parts = []
    for country in recs.loc[recs.source == 1, "country"].unique():  # open set: whatever labels appear
        sub = recs[recs.country == country]
        s1, others = sub[sub.source == 1], sub[sub.source != 1]
        t = tfidf_topk(s1, others, cfg_blocking["tfidf"])
        t = t.sort_values(["s1_id", "tfidf_sim"], ascending=[True, False])
        t["tfidf_rank"] = t.groupby("s1_id").cumcount() + 1
        k = key_blocks(sub, cfg_blocking["keys"])
        parts.append(t.merge(k, on=["s1_id", "cand_id"], how="outer"))
    out = pd.concat(parts, ignore_index=True)
    out["tfidf_sim"] = out.tfidf_sim.fillna(0).astype(np.float32)
    out["tfidf_rank"] = out.tfidf_rank.fillna(999).astype(np.int16)
    out["key_hits"] = out.key_hits.fillna(0).astype(np.int16)
    return out[["s1_id", "cand_id", "tfidf_sim", "tfidf_rank", "key_hits"]]


def cand_sparse_path(cfg: dict, family: str) -> Path:
    return run_dir(cfg) / f"cand_sparse_{family}.parquet"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--family", nargs="+", default=["train", "test"], choices=["train", "test"])


def run(cfg: dict, args: argparse.Namespace) -> None:
    for family in args.family:
        recs = pd.read_parquet(records_path(cfg, family))
        cands = block_family(recs, cfg["blocking"])
        cands.to_parquet(cand_sparse_path(cfg, family), index=False)
        s1_ids = recs.entity_id[recs.source == 1].tolist()
        stats = {"n_pairs": len(cands), "per_entity": candidates_per_entity(cands, s1_ids)}
        if family == "train":
            truth, splits = read_truth(cfg["paths"]["data_dir"]), pd.read_parquet(splits_path(cfg))
            tp = truth_pairs(truth)
            stats["pc_all"] = pair_completeness(cands, tp)
            stats["pc_tfidf_only"] = pair_completeness(cands[cands.tfidf_rank < 999], tp)
            stats["pc_keys_only"] = pair_completeness(cands[cands.key_hits > 0], tp)
            for c, grp in splits.groupby("country"):
                stats[f"pc_{c}"] = pair_completeness(cands, truth_pairs(truth, grp.s1_id.tolist()))
        print(family, stats, flush=True)
        log_metrics(cfg, f"block_sparse_{family}", stats)
