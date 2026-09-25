import pandas as pd

from ber.blocking.sparse import block_family, key_blocks, tfidf_topk
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
