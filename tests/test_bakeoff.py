import pandas as pd

from ber.contrastive.bakeoff import build_eval_set, select_winner, write_report


def test_eval_set_contains_all_matches_and_sampled_distractors():
    rec = pd.DataFrame({
        "entity_id": ["S1-a", "S1-b", "S2-a", "S3-a", "S2-b", "S2-x1", "S2-x2", "S2-x3", "S2-f"],
        "source": [1, 1, 2, 3, 2, 2, 2, 2, 2],
        "country": ["US", "US", "US", "US", "US", "US", "US", "US", "France"],
    })
    splits = pd.DataFrame({"s1_id": ["S1-a", "S1-b"], "split": ["A_val", "B"], "country": ["US", "US"]})
    truth = {"S1-a": {"S2-a", "S3-a"}, "S1-b": {"S2-b"}}
    ev = build_eval_set(rec, splits, truth, n_distractors=2, seed=0)
    q = ev[ev.role == "query"]
    d = ev[ev.role == "doc"]
    assert q.entity_id.tolist() == ["S1-a"]
    assert {"S2-a", "S3-a"} <= set(d.entity_id)
    assert len(d) == 4 and "S2-f" not in set(d.entity_id)  # 2 matches + 2 same-country distractors


def _res(model, tag, us, india, rps=100.0):
    return {"model": model, "tag": tag, "dim": 768, "throughput_rps": rps, "test_vectors_gb": 1.0,
            "peak_train_mem_gb": 1.0, "train_seconds": 1.0,
            "recall": {"US": {"10": us, "50": us, "100": us}, "India": {"10": india, "50": india, "100": india}}}


def test_select_winner_rules():
    # clear winner on mean fine-tuned recall@50
    assert select_winner([_res("a", "finetune", 0.90, 0.90), _res("b", "finetune", 0.95, 0.95)]) == "b"
    # within 0.005 -> leave-one-country-out India recall decides
    res = [_res("a", "finetune", 0.950, 0.950), _res("b", "finetune", 0.952, 0.950),
           _res("a", "finetune_US", 0.9, 0.93), _res("b", "finetune_US", 0.9, 0.90)]
    assert select_winner(res) == "a"


def test_write_report(tmp_path):
    out = tmp_path / "r.md"
    write_report([_res("a", "zeroshot", 0.5, 0.4), _res("a", "finetune", 0.9, 0.8)], out)
    text = out.read_text()
    assert "| a | finetune |" in text and "Winner: **a**" in text
