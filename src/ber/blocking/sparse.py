import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize
from sparse_dot_topn import sp_matmul_topn

from ber.config import log_metrics, run_dir
from ber.evaluate import candidates_per_entity, pair_completeness
from ber.io import read_truth, truth_pairs
from ber.normalize import records_path
from ber.split import splits_path

EMPTY = pd.DataFrame({"s1_id": pd.Series(dtype=str), "cand_id": pd.Series(dtype=str)})


N_HASH_FEATURES = 2 ** 23
_HASHER: HashingVectorizer | None = None


def _hash_counts(texts: list[str]) -> sp.csr_matrix:
    return _HASHER.transform(texts)


def _tfidf_matrices(s1_text: pd.Series, other_text: pd.Series, cfg_tfidf: dict) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """Char n-gram TF-IDF (sublinear tf, smooth idf, min_df/max_df, L2) built from hashed counts in parallel.
    Equivalent to TfidfVectorizer up to hash collisions, but uses every core instead of one."""
    global _HASHER
    _HASHER = HashingVectorizer(analyzer="char_wb", ngram_range=tuple(cfg_tfidf["ngram_range"]),
                                n_features=N_HASH_FEATURES, alternate_sign=False, norm=None, dtype=np.float32)
    texts = pd.concat([s1_text, other_text]).tolist()
    n_jobs = cfg_tfidf["n_threads"]
    if n_jobs > 1 and len(texts) > 10_000:
        bounds = np.linspace(0, len(texts), n_jobs * 4 + 1, dtype=int)
        with ProcessPoolExecutor(n_jobs, mp_context=get_context("fork")) as ex:
            x = sp.vstack(list(ex.map(_hash_counts, [texts[a:b] for a, b in zip(bounds[:-1], bounds[1:])])), format="csr")
    else:
        x = _hash_counts(texts)
    n_docs = x.shape[0]
    df = np.bincount(x.indices, minlength=x.shape[1])
    keep = (df >= cfg_tfidf["min_df"]) & (df <= cfg_tfidf.get("max_df", 1.0) * n_docs)
    idf = (np.log((1 + n_docs) / (1 + df)) + 1).astype(np.float32) * keep
    x.data = 1 + np.log(x.data)  # sublinear tf, as in TfidfVectorizer(sublinear_tf=True)
    x = normalize(x @ sp.diags(idf), norm="l2", copy=False).tocsr()
    x.eliminate_zeros()
    return x[: len(s1_text)], x[len(s1_text):]


def tfidf_topk(s1: pd.DataFrame, others: pd.DataFrame, cfg_tfidf: dict,
               s1_group: np.ndarray | None = None, oth_group: np.ndarray | None = None,
               log: str = "") -> pd.DataFrame:
    """Top-k char TF-IDF neighbours of each S1 among `others`. With groups, an S1 in group g only searches
    others in g plus others whose group is unknown (''); an S1 with unknown group searches everything.
    IDF is always fit on the whole (country) input, so scores are comparable across groups."""
    if len(s1) == 0 or len(others) == 0:
        return EMPTY.assign(tfidf_sim=pd.Series(dtype="float32"))
    a, b = _tfidf_matrices(s1.block_text, others.block_text, cfg_tfidf)
    s1_group = np.full(len(s1), "", dtype=object) if s1_group is None else np.asarray(s1_group, dtype=object)
    oth_group = np.full(len(others), "", dtype=object) if oth_group is None else np.asarray(oth_group, dtype=object)
    unknown = oth_group == ""
    groups = pd.Series(s1_group).value_counts().index.tolist()  # biggest first
    parts, t0 = [], time.perf_counter()
    for i, g in enumerate(groups, 1):
        qi = np.flatnonzero(s1_group == g)
        pi = np.arange(len(others)) if g == "" else np.flatnonzero((oth_group == g) | unknown)
        c = sp_matmul_topn(a[qi], b[pi].T.tocsr(), top_n=cfg_tfidf["top_k"], threshold=cfg_tfidf["min_sim"],
                           sort=True, n_threads=cfg_tfidf["n_threads"]).tocoo()
        parts.append(pd.DataFrame({"s1_id": s1.entity_id.values[qi[c.row]], "cand_id": others.entity_id.values[pi[c.col]],
                                   "tfidf_sim": c.data.astype(np.float32)}))
        if log:
            print(f"{log} group {i}/{len(groups)} '{g or 'unknown'}': {len(qi):,} x {len(pi):,} "
                  f"({time.perf_counter() - t0:.0f}s elapsed)", flush=True)
    return pd.concat(parts, ignore_index=True)


def learn_state_groups(recs: pd.DataFrame, tp: pd.DataFrame, min_share: float) -> dict[str, dict[str, str]]:
    """Per country, merge states whose cross-state true pairs are >= min_share of one state's pairs (union-find).
    Returns {country: {state: group_label}}; states never confused keep their own name as the label."""
    st = recs.set_index("entity_id")
    pairs = tp.assign(country=st.country.reindex(tp.s1_id).values,
                      a=st.state.reindex(tp.s1_id).values, b=st.state.reindex(tp.cand_id).values)
    out: dict[str, dict[str, str]] = {}
    for country, g in pairs.groupby("country"):
        g = g[(g.a != "") & (g.b != "")]
        parent = {x: x for x in set(g.a) | set(g.b)}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        per_state = g.groupby("a").size()
        cross = g[g.a != g.b].groupby(["a", "b"]).size()
        for (x, y), n in cross.items():
            if n >= min_share * min(per_state.get(x, n), per_state.get(y, n)):
                parent[find(x)] = find(y)
        members: dict[str, list[str]] = {}
        for x in parent:
            members.setdefault(find(x), []).append(x)
        out[country] = {x: "+".join(sorted(members[find(x)])) for x in parent}
    return out


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


def block_family(recs: pd.DataFrame, cfg_blocking: dict, state_groups: dict[str, dict[str, str]] | None = None
                 ) -> pd.DataFrame:
    parts = []
    for country in recs.loc[recs.source == 1, "country"].unique():  # open set: whatever labels appear
        sub = recs[recs.country == country]
        s1, others = sub[sub.source == 1], sub[sub.source != 1]
        grp = None
        if state_groups is not None and "state" in sub:
            m = state_groups.get(country, {})
            grp = (s1.state.map(lambda x: m.get(x, x)).to_numpy(), others.state.map(lambda x: m.get(x, x)).to_numpy())
        t = tfidf_topk(s1, others, cfg_blocking["tfidf"], *(grp or (None, None)), log=f"[{country}]")
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
    sg_cfg = cfg["blocking"].get("state_groups", {})
    sg_path = run_dir(cfg) / "state_groups.json"
    for family in args.family:
        recs = pd.read_parquet(records_path(cfg, family))
        groups = None
        if sg_cfg.get("enabled") and "state" in recs:
            if family == "train":
                groups = learn_state_groups(recs, truth_pairs(read_truth(cfg["paths"]["data_dir"])), sg_cfg["min_share"])
                sg_path.write_text(json.dumps(groups, indent=2, sort_keys=True))
                print("state groups:", {c: sorted({v for v in m.values() if "+" in v}) for c, m in groups.items()}, flush=True)
            else:
                groups = json.loads(sg_path.read_text())  # learned on train; unseen countries get no grouping
        cands = block_family(recs, cfg["blocking"], groups)
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
