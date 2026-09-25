"""Pair features for the matcher (Model A). Everything is vectorised: RapidFuzz `cpdist` scores row-aligned pairs
of strings on all cores; categorical comparisons are pandas ops. One row per (s1_id, cand_id)."""
import re

import numpy as np
import pandas as pd
from rapidfuzz import distance, fuzz
from rapidfuzz.process import cpdist

NATIVE_RE = "[ऀ-෿]"  # Indic scripts, as real characters (pyarrow regex has no \\u escapes)
REC_COLS = ["entity_id", "source", "country", "state", "name_raw", "addr_raw", "name_core", "name_domain",
            "name_skeleton", "name_phonetic", "addr_norm", "addr_core", "house_no", "num_tokens", "legal_suffix", "unit"]
BLOCK_COLS = ["tfidf_sim", "tfidf_rank", "key_hits", "name_sim", "name_key"]


def _tri(a: pd.Series, b: pd.Series) -> np.ndarray:
    """0 = missing on either side, 1 = equal, 2 = different."""
    a, b = a.to_numpy(), b.to_numpy()
    return np.where((a == "") | (b == ""), 0, np.where(a == b, 1, 2)).astype(np.int8)


def _sim(a: pd.Series, b: pd.Series, scorer, workers: int) -> np.ndarray:
    return cpdist(a.tolist(), b.tolist(), scorer=scorer, workers=workers, dtype=np.float32).astype(np.float32)


def name_idf(rec: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Per country: token -> ln(N / df) over that country's names (name_domain if set, else name_core)."""
    name = rec.name_domain.where(rec.name_domain != "", rec.name_core)
    out = {}
    for c, s in name.groupby(rec.country):
        df = s.str.split().apply(lambda t: list(set(t))).explode().dropna().value_counts()  # document frequency
        out[c] = (np.log(len(s) / df)).astype(np.float32).to_dict()
    return out


def _lead_int(s: str) -> int:
    m = re.match(r"\d+", s)
    return int(m.group()) if m else -1


def _idf_feats(na: pd.Series, nb: pd.Series, country: np.ndarray, idf: dict[str, dict[str, float]]):
    jac, rare = np.zeros(len(na), np.float32), np.zeros(len(na), np.float32)
    for i, (x, y, c) in enumerate(zip(na.to_numpy(), nb.to_numpy(), country)):
        w = idf.get(c, {})
        a, b = set(x.split()), set(y.split())
        u = a | b
        if not u:
            continue
        wu = sum(w.get(t, 12.0) for t in u)
        jac[i] = sum(w.get(t, 12.0) for t in a & b) / wu if wu else 0.0
        diff = u - (a & b)
        rare[i] = max((w.get(t, 12.0) for t in diff), default=0.0)
    return jac, rare


def pair_features(pairs: pd.DataFrame, rec: pd.DataFrame, workers: int = -1,
                  idf: dict[str, dict[str, float]] | None = None) -> pd.DataFrame:
    """pairs: s1_id, cand_id (+ blocking scores, + optional context columns). rec: records indexed by entity_id.
    idf (optional, from name_idf): adds IDF-weighted name overlap and the rarest non-shared name word."""
    a = rec.reindex(pairs.s1_id.to_numpy())
    b = rec.reindex(pairs.cand_id.to_numpy())
    na = a.name_domain.where(a.name_domain != "", a.name_core)
    nb = b.name_domain.where(b.name_domain != "", b.name_core)
    f = pd.DataFrame(index=pairs.index)
    for c in BLOCK_COLS:
        f[c] = pairs[c].to_numpy() if c in pairs else 0
    f["name_ratio"] = _sim(na, nb, fuzz.ratio, workers)
    f["name_token_set"] = _sim(na, nb, fuzz.token_set_ratio, workers)
    f["name_token_sort"] = _sim(na, nb, fuzz.token_sort_ratio, workers)
    f["name_partial"] = _sim(na, nb, fuzz.partial_ratio, workers)
    f["name_jw"] = _sim(na, nb, distance.JaroWinkler.normalized_similarity, workers)
    f["name_skeleton_ts"] = _sim(a.name_skeleton, b.name_skeleton, fuzz.token_set_ratio, workers)
    f["name_phonetic_ts"] = _sim(a.name_phonetic, b.name_phonetic, fuzz.token_set_ratio, workers)
    f["addr_token_set"] = _sim(a.addr_norm, b.addr_norm, fuzz.token_set_ratio, workers)
    f["addr_ratio"] = _sim(a.addr_norm, b.addr_norm, fuzz.ratio, workers)
    f["addr_core_ts"] = _sim(a.addr_core, b.addr_core, fuzz.token_set_ratio, workers)
    f["num_token_set"] = _sim(a.num_tokens, b.num_tokens, fuzz.token_set_ratio, workers)
    f["house_cmp"] = _tri(a.house_no, b.house_no)
    ha = np.array([_lead_int(x) for x in a.house_no.to_numpy()]); hb = np.array([_lead_int(x) for x in b.house_no.to_numpy()])
    f["house_num_diff"] = np.where((ha < 0) | (hb < 0), -1, np.minimum(np.abs(ha - hb), 1000)).astype(np.int16)
    if idf is not None:
        f["name_idf_jacc"], f["name_rare_mismatch"] = _idf_feats(na, nb, a.country.to_numpy(), idf)
    f["state_cmp"] = _tri(a.state, b.state)
    f["legal_cmp"] = _tri(a.legal_suffix, b.legal_suffix)
    f["unit_cmp"] = _tri(a.unit, b.unit)
    f["cand_source"] = b.source.to_numpy().astype(np.int8)
    f["cand_addr_empty"] = (b.addr_raw.to_numpy() == "").astype(np.int8)
    f["cand_native"] = b.name_raw.str.contains(NATIVE_RE, regex=True).to_numpy().astype(np.int8)
    f["s1_native"] = a.name_raw.str.contains(NATIVE_RE, regex=True).to_numpy().astype(np.int8)
    la, lb = na.str.len().to_numpy(), nb.str.len().to_numpy()
    f["name_len_ratio"] = (np.minimum(la, lb) / np.maximum(np.maximum(la, lb), 1)).astype(np.float32)
    f["cand_name_len"] = lb.astype(np.int16)
    f["cand_addr_len"] = b.addr_norm.str.len().to_numpy().astype(np.int16)
    # per-S1 context: how this candidate compares with the S1's other candidates
    s1 = pairs.s1_id.to_numpy()
    for c in ["tfidf_sim", "name_token_set", "addr_token_set"]:
        best = f.groupby(s1)[c].transform("max")
        f[f"{c}_gap"] = (best - f[c]).astype(np.float32)
    f["s1_n_cands"] = f.groupby(s1)["tfidf_sim"].transform("size").astype(np.int16)
    # per-candidate context (computed on the FULL candidate table upstream, passed through if present)
    for c in ["cand_degree", "cand_best_sim", "is_cand_best"]:
        if c in pairs:
            f[c] = pairs[c].to_numpy()
    if "cand_best_sim" in f:
        f["mutual_best"] = ((f.tfidf_rank == 1) & (f.is_cand_best == 1)).astype(np.int8)
    return f
