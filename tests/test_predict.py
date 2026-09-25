import pandas as pd

from ber.predict import write_submission


def test_write_submission_format(tmp_path):
    pred = {"S1-1": {"S2-9", "S3-1"}, "S1-3": {"S3-7"}}
    cands = {"S1-1": {"S3-1", "S2-9", "S2-4"}, "S1-2": {"S2-5"}}   # S1-3's prediction is not a candidate -> dropped
    write_submission(["S1-1", "S1-2", "S1-3"], pred, cands, tmp_path)
    m = pd.read_csv(tmp_path / "matching_results.tsv", sep="\t", dtype=str, keep_default_na=False)
    c = pd.read_csv(tmp_path / "candidate_pairs.tsv", sep="\t", dtype=str, keep_default_na=False)
    assert list(m.columns) == ["source1_entity_id", "matched_entity_ids"]
    assert list(c.columns) == ["source1_entity_id", "candidate_entity_ids"]
    assert m.matched_entity_ids.tolist() == ["S2-9,S3-1", "", ""]
    assert c.candidate_entity_ids.tolist() == ["S2-4,S2-9,S3-1", "S2-5", ""]
