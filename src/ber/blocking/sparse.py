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
    x.data *= idf[x.indices]  # column scaling in O(nnz); `x @ sp.diags(idf)` is a slow single-core sparse matmul
    x = normalize(x, norm="l2", copy=False).tocsr()
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


def learn_state_groups(recs: pd.DataFrame, tp: pd.DataFrame, min_share: float, min_pairs: int = 100
                       ) -> dict[str, dict[str, str]]:
    """Per country, merge states whose cross-state true pairs are >= min_share of the larger state's pairs (union-find).
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
            # relative to the LARGER state: stray pairs from tiny states must not chain everything together
            if n >= max(min_pairs, min_share * max(per_state.get(x, n), per_state.get(y, n))):
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


def _name_text(df: pd.DataFrame) -> pd.Series:
    return df.name_domain.where(df.name_domain != "", df.name_core)


def name_only_candidates(s1: pd.DataFrame, others: pd.DataFrame, cfg_name: dict, n_threads: int) -> pd.DataFrame:
    """Candidates for S2/S3 records with an EMPTY address: their only evidence is the name, so compare names alone
    (the S1's long address would otherwise dilute the TF-IDF similarity) and add a two-rare-name-word key block."""
    pool = others[others.addr_norm == ""]
    if len(s1) == 0 or len(pool) == 0:
        return pd.DataFrame({"s1_id": pd.Series(dtype=str), "cand_id": pd.Series(dtype=str),
                             "name_sim": pd.Series(dtype="float32"), "name_key": pd.Series(dtype="int8")})
    q = s1.assign(block_text=_name_text(s1))
    d = pool.assign(block_text=_name_text(pool))
    tf = {**cfg_name, "ngram_range": cfg_name.get("ngram_range", [3, 5]), "n_threads": n_threads}
    t = tfidf_topk(q, d, tf).rename(columns={"tfidf_sim": "name_sim"})
    # key: the two rarest name tokens (document frequency counted over S1 names + the empty-address pool)
    both = pd.concat([q[["entity_id", "source", "block_text"]], d[["entity_id", "source", "block_text"]]], ignore_index=True)
    toks = _rarest(both.block_text.str.split(), 2, cfg_name.get("min_token_len", 3))
    keyed = toks.groupby("row").tok.agg(lambda x: "|".join(sorted(x))).rename("key").reset_index()
    keyed = keyed[keyed.key.str.contains("|", regex=False)]
    keyed = keyed[keyed.groupby("key").row.transform("size") <= cfg_name.get("max_block_size", 100)]
    is_s1 = both.source.values[keyed.row.values] == 1
    kp = keyed[is_s1].merge(keyed[~is_s1], on="key", suffixes=("_s1", "_c"))
    k = pd.DataFrame({"s1_id": both.entity_id.values[kp.row_s1.values], "cand_id": both.entity_id.values[kp.row_c.values],
                      "name_key": np.int8(1)}).drop_duplicates(["s1_id", "cand_id"])
    out = t.merge(k, on=["s1_id", "cand_id"], how="outer")
    out["name_sim"] = out.name_sim.fillna(0).astype(np.float32)
    out["name_key"] = out.name_key.fillna(0).astype(np.int8)
    return out


_KB_SUB: pd.DataFrame | None = None  # set before forking so workers read it without pickling
_KB_CFG: dict | None = None


def _key_blocks_group(idx: np.ndarray) -> pd.DataFrame:
    return key_blocks(_KB_SUB.iloc[idx], _KB_CFG)


def key_blocks_grouped(sub: pd.DataFrame, cfg_keys: dict, s1_group: np.ndarray, oth_group: np.ndarray,
                       n_jobs: int, log: str = "") -> pd.DataFrame:
    """key_blocks run separately per state group (S1s of the group + others of the group or of unknown state),
    in parallel processes. An S1 with unknown state is blocked against the whole country."""
    global _KB_SUB, _KB_CFG
    sub = sub.reset_index(drop=True)
    is_s1 = (sub.source == 1).to_numpy()
    s1_pos, oth_pos = np.flatnonzero(is_s1), np.flatnonzero(~is_s1)
    unknown = oth_group == ""
    tasks = []
    for g in pd.Series(s1_group).value_counts().index:  # biggest first
        q = s1_pos[s1_group == g]
        o = oth_pos if g == "" else oth_pos[(oth_group == g) | unknown]
        tasks.append((g, np.concatenate([q, o])))
    _KB_SUB, _KB_CFG = sub, cfg_keys
    t0, parts = time.perf_counter(), []
    with ProcessPoolExecutor(max(1, min(n_jobs, len(tasks))), mp_context=get_context("fork")) as ex:
        for i, ((g, _), df) in enumerate(zip(tasks, ex.map(_key_blocks_group, [t[1] for t in tasks])), 1):
            parts.append(df)
            if log:
                print(f"{log} key blocks {i}/{len(tasks)} '{g or 'unknown'}' done ({time.perf_counter() - t0:.0f}s)", flush=True)
    return pd.concat(parts, ignore_index=True).drop_duplicates(["s1_id", "cand_id"])


def block_family(recs: pd.DataFrame, cfg_blocking: dict, state_groups: dict[str, dict[str, str]] | None = None,
                 cache_dir: Path | None = None) -> pd.DataFrame:
    parts = []
    for country in recs.loc[recs.source == 1, "country"].unique():  # open set: whatever labels appear
        cache = cache_dir / f"blocks_{country}.parquet" if cache_dir else None
        name_cache = cache_dir / f"nameonly_{country}.parquet" if cache_dir else None
        sub = recs[recs.country == country]
        s1, others = sub[sub.source == 1], sub[sub.source != 1]
        name_part = None
        if cfg_blocking.get("name_only", {}).get("enabled"):
            if name_cache is not None and name_cache.exists():
                name_part = pd.read_parquet(name_cache)
            else:
                name_part = name_only_candidates(s1, others, cfg_blocking["name_only"], cfg_blocking["tfidf"]["n_threads"])
                if name_cache is not None:
                    name_part.to_parquet(name_cache, index=False)
            print(f"[{country}] name-only candidates: {len(name_part):,}", flush=True)
        if cache is not None and cache.exists():  # resume: this country already finished in an earlier run
            print(f"[{country}] loaded from cache", flush=True)
            part = pd.read_parquet(cache)
            parts.append(part if name_part is None else part.merge(name_part, on=["s1_id", "cand_id"], how="outer"))
            continue
        grp = None
        if state_groups is not None and "state" in sub:
            m = state_groups.get(country, {})
            grp = (s1.state.map(lambda x: m.get(x, x)).to_numpy(), others.state.map(lambda x: m.get(x, x)).to_numpy())
        t = tfidf_topk(s1, others, cfg_blocking["tfidf"], *(grp or (None, None)), log=f"[{country}]")
        t = t.sort_values(["s1_id", "tfidf_sim"], ascending=[True, False])
        t["tfidf_rank"] = t.groupby("s1_id").cumcount() + 1
        if grp is not None:
            # key_blocks_grouped needs group labels aligned with sub's order (S1 rows and other rows)
            k = key_blocks_grouped(pd.concat([s1, others]), cfg_blocking["keys"], grp[0], grp[1],
                                   cfg_blocking["keys"].get("n_jobs", 16), log=f"[{country}]")
        else:
            k = key_blocks(sub, cfg_blocking["keys"])
        part = t.merge(k, on=["s1_id", "cand_id"], how="outer")
        if cache is not None:
            part.to_parquet(cache, index=False)
        parts.append(part if name_part is None else part.merge(name_part, on=["s1_id", "cand_id"], how="outer"))
    out = pd.concat(parts, ignore_index=True)
    out["tfidf_sim"] = out.tfidf_sim.fillna(0).astype(np.float32)
    out["tfidf_rank"] = out.tfidf_rank.fillna(999).astype(np.int16)
    out["key_hits"] = out.key_hits.fillna(0).astype(np.int16)
    cols = ["s1_id", "cand_id", "tfidf_sim", "tfidf_rank", "key_hits"]
    if "name_sim" in out:
        out["name_sim"] = out.name_sim.fillna(0).astype(np.float32)
        out["name_key"] = out.name_key.fillna(0).astype(np.int8)
        cols += ["name_sim", "name_key"]
    return out[cols]


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
                groups = learn_state_groups(recs, truth_pairs(read_truth(cfg["paths"]["data_dir"])), sg_cfg["min_share"],
                                            sg_cfg.get("min_pairs", 100))
                sg_path.write_text(json.dumps(groups, indent=2, sort_keys=True))
                print("state groups:", {c: sorted({v for v in m.values() if "+" in v}) for c, m in groups.items()}, flush=True)
            else:
                groups = json.loads(sg_path.read_text())  # learned on train; unseen countries get no grouping
        cache_dir = run_dir(cfg) / f"block_cache_{family}"
        cache_dir.mkdir(exist_ok=True)
        cands = block_family(recs, cfg["blocking"], groups, cache_dir)
        cands.to_parquet(cand_sparse_path(cfg, family), index=False)
        s1_ids = recs.entity_id[recs.source == 1].tolist()
        stats = {"n_pairs": len(cands), "per_entity": candidates_per_entity(cands, s1_ids)}
        if family == "train":
            truth, splits = read_truth(cfg["paths"]["data_dir"]), pd.read_parquet(splits_path(cfg))
            tp = truth_pairs(truth)
            stats["pc_all"] = pair_completeness(cands, tp)
            stats["pc_tfidf_only"] = pair_completeness(cands[cands.tfidf_rank < 999], tp)
            stats["pc_keys_only"] = pair_completeness(cands[cands.key_hits > 0], tp)
            if "name_sim" in cands:
                stats["pc_name_only"] = pair_completeness(cands[(cands.name_sim > 0) | (cands.name_key > 0)], tp)
            for c, grp in splits.groupby("country"):
                stats[f"pc_{c}"] = pair_completeness(cands, truth_pairs(truth, grp.s1_id.tolist()))
        print(family, stats, flush=True)
        log_metrics(cfg, f"block_sparse_{family}", stats)
