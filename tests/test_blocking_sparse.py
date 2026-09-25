import pandas as pd

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from ber.blocking.sparse import _tfidf_matrices, block_family, key_blocks, tfidf_topk
from ber.normalize import build_records

CFG = {
    "tfidf": {"ngram_range": [3, 5], "min_df": 1, "top_k": 3, "min_sim": 0.1, "n_threads": 1},
    "keys": {"n_name_tokens": 2, "n_addr_tokens": 3, "min_token_len": 3, "max_block_size": 100},
}


def _recs():
    rows = [
        ("S1-1", 1, "Galaxy Solutions Pvt Ltd", "47/1 Airport Road, Kolhapur, MH", "India"),
        ("S2-1", 2, "Galaxy Solutiocns Pvt Ltd", "4-7/1 AIRPORT ROAD, KOLHAPUR", "India"),
        ("S3-1", 3, "Pvt Ltd Galaxy", "Kolhapur, 47/1 Airport Rd", "India"),
        ("S2-2", 2, "Sunrise Traders", "12 MG Road, Pune", "India"),
        ("S2-9", 2, "Galaxy Solutions Pvt Ltd", "47/1 Airport Road, Kolhapur", "France"),  # other country
        ("S1-2", 1, "Morgan Dental PC", "412 Madrona Avenue, Salem, OR", "US"),
        ("S3-2", 3, "Morgan Dental (PC)", "412 Madrona Ave, Salem, Oregon", "US"),
    ]
    raw = pd.DataFrame(rows, columns=["entity_id", "source", "business_name", "business_address", "country"])
    raw["source"] = raw.source.astype("int8")
    return build_records(raw[["entity_id", "business_name", "business_address", "country", "source"]], n_jobs=1)


def test_tfidf_finds_noisy_match_within_country():
    rec = _recs()
    india = rec[rec.country == "India"]
    out = tfidf_topk(india[india.source == 1], india[india.source != 1], CFG["tfidf"])
    got = set(out.cand_id[out.s1_id == "S1-1"])
    assert {"S2-1", "S3-1"} <= got and "S2-9" not in got


def test_key_blocks_house_number_and_name():
    rec = _recs()
    out = key_blocks(rec[rec.country == "India"], CFG["keys"])
    assert {"S2-1", "S3-1"} <= set(out.cand_id[out.s1_id == "S1-1"])
    assert (out.key_hits >= 1).all()


def test_block_family_respects_country_and_fills_defaults():
    out = block_family(_recs(), CFG)
    assert list(out.columns) == ["s1_id", "cand_id", "tfidf_sim", "tfidf_rank", "key_hits"]
    assert not ((out.s1_id == "S1-1") & (out.cand_id == "S2-9")).any()
    assert not ((out.s1_id == "S1-2") & out.cand_id.isin(["S2-1", "S3-1"])).any()
    assert ((out.s1_id == "S1-2") & (out.cand_id == "S3-2")).any()
    assert out.tfidf_rank.min() >= 1


def test_parallel_tfidf_matches_sklearn_single_and_multiprocess():
    rng = np.random.default_rng(0)
    words = ["galaxy", "solutions", "airport", "road", "kolhapur", "morgan", "dental", "salem", "rue", "lille"]
    texts = pd.Series([" ".join(rng.choice(words, size=4)) + f" {i % 97}" for i in range(12_000)])  # >10k -> process pool
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_df=0.5, sublinear_tf=True)
    x = vec.fit_transform(texts.tolist())
    ref = (x[:50] @ x[50:].T).toarray()
    for n_threads in (1, 4):
        a, b = _tfidf_matrices(texts[:50], texts[50:], {"ngram_range": [3, 5], "min_df": 2, "max_df": 0.5, "n_threads": n_threads})
        np.testing.assert_allclose((a @ b.T).toarray(), ref, atol=1e-4)


def test_learn_state_groups_merges_confused_states():
    from ber.blocking.sparse import learn_state_groups
    recs = pd.DataFrame({"entity_id": ["S1-1", "S1-2", "S1-3", "S2-1", "S2-2", "S2-3"],
                         "country": ["India"] * 6,
                         "state": ["delhi", "delhi", "kerala", "uttar pradesh", "delhi", "kerala"]})
    tp = pd.DataFrame({"s1_id": ["S1-1", "S1-2", "S1-3"], "cand_id": ["S2-1", "S2-2", "S2-3"]})
    g = learn_state_groups(recs, tp, min_share=0.1, min_pairs=1)["India"]
    assert g["delhi"] == g["uttar pradesh"] == "delhi+uttar pradesh" and g["kerala"] == "kerala"


def test_grouped_tfidf_restricts_pool_but_keeps_unknown_state():
    s1 = pd.DataFrame({"entity_id": ["S1-a"], "block_text": ["galaxy solutions airport road"]})
    oth = pd.DataFrame({"entity_id": ["S2-same", "S2-other", "S2-unk"],
                        "block_text": ["galaxy solutions airport road"] * 3})
    cfg = CFG["tfidf"]
    got = set(tfidf_topk(s1, oth, cfg, np.array(["texas"]), np.array(["texas", "ohio", ""])).cand_id)
    assert got == {"S2-same", "S2-unk"}
    assert set(tfidf_topk(s1, oth, cfg).cand_id) == {"S2-same", "S2-other", "S2-unk"}  # no groups = whole pool


def test_learn_state_groups_does_not_chain_through_tiny_states():
    from ber.blocking.sparse import learn_state_groups
    # 100 delhi pairs, 100 kerala pairs, and a tiny state with 1 pair to each: must not merge delhi with kerala
    ids = [(f"S1-d{i}", "delhi", f"S2-d{i}", "delhi") for i in range(100)] + \
          [(f"S1-k{i}", "kerala", f"S2-k{i}", "kerala") for i in range(100)] + \
          [("S1-t1", "sikkim", "S2-t1", "delhi"), ("S1-t2", "sikkim", "S2-t2", "kerala")]
    recs = pd.DataFrame([(a, "India", sa) for a, sa, _, _ in ids] + [(b, "India", sb) for _, _, b, sb in ids],
                        columns=["entity_id", "country", "state"])
    tp = pd.DataFrame([(a, b) for a, _, b, _ in ids], columns=["s1_id", "cand_id"])
    g = learn_state_groups(recs, tp, min_share=0.002)["India"]
    assert g["delhi"] != g["kerala"]


def test_key_blocks_grouped_matches_ungrouped_when_one_group():
    from ber.blocking.sparse import key_blocks_grouped
    rec = _recs()
    india = rec[rec.country == "India"]
    s1, oth = india[india.source == 1], india[india.source != 1]
    sub = pd.concat([s1, oth])
    got = key_blocks_grouped(sub, CFG["keys"], np.array(["mh"] * len(s1)), np.array(["mh"] * len(oth)), n_jobs=2)
    ref = key_blocks(sub, CFG["keys"])
    key = lambda d: set(zip(d.s1_id, d.cand_id))
    assert key(got) == key(ref)


def test_key_blocks_grouped_keeps_unknown_state_records():
    from ber.blocking.sparse import key_blocks_grouped
    rec = _recs()
    india = rec[rec.country == "India"]
    s1, oth = india[india.source == 1], india[india.source != 1]
    groups = np.array(["karnataka" if e == "S2-1" else "" if e == "S3-1" else "kerala" for e in oth.entity_id])
    got = key_blocks_grouped(pd.concat([s1, oth]), CFG["keys"], np.array(["kerala"] * len(s1)), groups, n_jobs=2)
    cands = set(got.cand_id[got.s1_id == "S1-1"])
    assert "S3-1" in cands and "S2-1" not in cands  # unknown-state record kept, other-state record excluded


def test_block_family_cache_resumes(tmp_path):
    out1 = block_family(_recs(), CFG, None, tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["blocks_India.parquet", "blocks_US.parquet"]
    out2 = block_family(_recs(), CFG, None, tmp_path)
    pd.testing.assert_frame_equal(out1.sort_values(["s1_id", "cand_id"]).reset_index(drop=True),
                                  out2.sort_values(["s1_id", "cand_id"]).reset_index(drop=True))
