import numpy as np
import pandas as pd

from ber.features import pair_features
from ber.matcher import decide, evaluate
from ber.normalize import build_records


def _recs():
    rows = [("S1-1", 1, "Galaxy Solutions Pvt Ltd", "47/1 Airport Road, Kolhapur, MH", "India"),
            ("S2-1", 2, "Galaxy Solutiocns Pvt Ltd", "4-7/1 AIRPORT ROAD, KOLHAPUR, Maharashtra", "India"),
            ("S3-1", 3, "Sunrise Traders", "12 MG Road, Pune, MH", "India"),
            ("S3-2", 3, "गैलेक्सी सॉल्यूशंस", "", "India")]
    raw = pd.DataFrame(rows, columns=["entity_id", "source", "business_name", "business_address", "country"])
    raw["source"] = raw.source.astype("int8")
    return build_records(raw[["entity_id", "business_name", "business_address", "country", "source"]], n_jobs=1)


def test_pair_features_values():
    rec = _recs().set_index("entity_id")
    pairs = pd.DataFrame({"s1_id": ["S1-1"] * 3, "cand_id": ["S2-1", "S3-1", "S3-2"],
                          "tfidf_sim": [0.9, 0.2, 0.0], "tfidf_rank": [1, 2, 999], "key_hits": [2, 0, 0],
                          "name_sim": [0.0, 0.0, 0.0], "name_key": [0, 0, 0]})
    f = pair_features(pairs, rec, workers=1)
    assert f.loc[0, "house_cmp"] == 1 and f.loc[1, "house_cmp"] == 2   # 47/1 equal; 12 different
    assert f.loc[0, "state_cmp"] == 1                                    # maharashtra both
    assert f.loc[0, "name_token_set"] > f.loc[1, "name_token_set"]
    assert f.loc[2, "cand_native"] == 1 and f.loc[2, "cand_addr_empty"] == 1 and f.loc[0, "cand_native"] == 0
    assert f.loc[0, "tfidf_sim_gap"] == 0 and np.isclose(f.loc[1, "tfidf_sim_gap"], 0.7)
    assert (f.s1_n_cands == 3).all()


def test_decide_one_owner_and_evaluate():
    scored = pd.DataFrame({"s1_id": ["S1-a", "S1-b", "S1-b", "S1-c"], "cand_id": ["x", "x", "y", "z"],
                           "p": [0.9, 0.8, 0.95, 0.3]})
    pred = decide(scored, tau=0.5)
    assert pred == {"S1-a": {"x"}, "S1-b": {"y"}}          # x goes only to its best S1; z below cut-off
    truth = {"S1-a": {"x"}, "S1-b": {"y", "w"}, "S1-c": set()}
    s1 = pd.DataFrame({"s1_id": ["S1-a", "S1-b", "S1-c"], "country": ["US", "US", "India"]})
    m = evaluate(pred, truth, s1)
    assert m["singletons"] == 1.0 and np.isclose(m["f05_US"], (1.0 + 1.25 * 0.5 / (0.25 + 0.5)) / 2)
