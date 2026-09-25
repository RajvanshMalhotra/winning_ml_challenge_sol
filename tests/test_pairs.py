import numpy as np
import pandas as pd

from ber.contrastive.encoders import EncoderSpec
from ber.contrastive.pairs import hard_negatives, render_triplets, sample_triplets

TRUTH = {"S1-a": {"S2-a1", "S2-a2", "S3-a1"}, "S1-b": {"S3-b1"}, "S1-z": set()}
IDC = {"S1-a": "US", "S2-a1": "US", "S2-a2": "US", "S3-a1": "US", "S1-b": "US", "S3-b1": "US",
       "S2-x": "US", "S3-y": "US", "S1-z": "US"}
POOL = {"US": np.array(["S2-a1", "S2-a2", "S3-a1", "S3-b1", "S2-x", "S3-y"])}


def test_hard_negatives_exclude_true_matches():
    cands = pd.DataFrame({"s1_id": ["S1-a", "S1-a", "S1-b"], "cand_id": ["S2-a1", "S2-x", "S3-y"]})
    assert hard_negatives(cands, TRUTH, ["S1-a", "S1-b"]) == {"S1-a": ["S2-x"], "S1-b": ["S3-y"]}


def test_sample_triplets_valid_and_deterministic():
    hn = {"S1-a": ["S2-x"], "S1-b": []}
    t = sample_triplets(TRUTH, ["S1-a", "S1-b", "S1-z"], hn, POOL, IDC, n_rounds=50, p_intra=0.5, seed=3)
    t2 = sample_triplets(TRUTH, ["S1-a", "S1-b", "S1-z"], hn, POOL, IDC, n_rounds=50, p_intra=0.5, seed=3)
    pd.testing.assert_frame_equal(t, t2)
    assert len(t) == 100  # S1-z (singleton) contributes nothing
    groups = {"S1-a": {"S1-a"} | TRUTH["S1-a"], "S1-b": {"S1-b"} | TRUTH["S1-b"]}
    for r in t.itertuples():
        g = next(v for v in groups.values() if r.anchor_id in v)
        assert r.positive_id in g and r.positive_id != r.anchor_id and r.negative_id not in g
    assert ((t.anchor_id != "S1-a") & (t.anchor_id != "S1-b")).any()  # intra S2/S3 pairs occur
    assert (t[t.anchor_id == "S1-a"].negative_id == "S2-x").all()


def test_render_prefixes():
    rec = pd.DataFrame({"entity_id": ["S1-a", "S2-a1", "S2-x"], "name_raw": ["A Co", "A CO", "B Inc"],
                        "addr_raw": ["1 Main St, X", "1 MAIN ST", "2 Elm"], "country": ["US"] * 3}).set_index("entity_id")
    trip = pd.DataFrame({"anchor_id": ["S1-a"], "positive_id": ["S2-a1"], "negative_id": ["S2-x"]})
    spec = EncoderSpec("t", "t", query_prefix="q: ", doc_prefix="d: ")
    out = render_triplets(trip, rec, spec, aug_prob=0.0, seed=0)
    assert out.iloc[0].tolist() == ["q: name: A Co | address: 1 Main St, X | country: US",
                                    "d: name: A CO | address: 1 MAIN ST | country: US",
                                    "d: name: B Inc | address: 2 Elm | country: US"]
