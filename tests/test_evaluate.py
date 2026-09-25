import pandas as pd
import pytest

from ber.evaluate import candidates_per_entity, f05, f05_breakdown, macro_f05, pair_completeness


def test_readme_example():
    pred = {"S2-00047", "S2-00193", "S3-00812"}
    truth = {"S2-00047", "S3-00812"}
    assert f05(pred, truth) == pytest.approx(0.7142857, abs=1e-6)


def test_singleton_rules():
    assert f05(set(), set()) == 1.0
    assert f05({"S2-1"}, set()) == 0.0
    assert f05(set(), {"S2-1"}) == 0.0
    assert f05({"S2-9"}, {"S2-1"}) == 0.0


def test_macro_averages_over_truth_keys_and_missing_pred_is_empty():
    truth = {"S1-a": {"S2-1"}, "S1-b": set(), "S1-c": {"S3-1", "S3-2"}}
    pred = {"S1-a": {"S2-1"}, "S1-c": {"S3-1"}}  # S1-b missing -> empty -> 1.0
    expected = (1.0 + 1.0 + f05({"S3-1"}, {"S3-1", "S3-2"})) / 3
    assert macro_f05(pred, truth) == pytest.approx(expected)


def test_breakdown():
    truth = {"a": {"x"}, "b": set()}
    pred = {"a": set(), "b": set()}
    assert f05_breakdown(pred, truth, {"a": "US", "b": "India"}) == {"US": 0.0, "India": 1.0}


def test_pair_completeness_and_counts():
    truth = pd.DataFrame({"s1_id": ["a", "a", "b"], "cand_id": ["x", "y", "z"]})
    cands = pd.DataFrame({"s1_id": ["a", "a", "b"], "cand_id": ["x", "q", "z"]})
    assert pair_completeness(cands, truth) == pytest.approx(2 / 3)
    stats = candidates_per_entity(cands, ["a", "b", "c"])
    assert stats["mean"] == pytest.approx(1.0) and stats["max"] == 2
