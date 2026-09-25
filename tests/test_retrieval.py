import numpy as np
import pytest

from ber.contrastive.retrieval import recall_at_k


def test_recall_at_k_micro():
    rng = np.random.default_rng(0)
    d = rng.normal(size=(50, 8)).astype(np.float32)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    d_ids = [f"S2-{i}" for i in range(50)]
    # q0 equals doc 0 exactly; its truth = {doc0, doc1}. q1 equals doc 5; truth = {doc5}.
    q = d[[0, 5]].copy()
    truth = {"S1-0": {"S2-0", "S2-1"}, "S1-1": {"S2-5"}}
    r = recall_at_k(["S1-0", "S1-1"], q, d_ids, d, truth, ks=[1, 50])
    assert r[1] == pytest.approx(2 / 3)
    assert r[50] == pytest.approx(1.0)


def test_found_rank():
    from ber.contrastive.retrieval import found_rank
    rng = np.random.default_rng(0)
    d = rng.normal(size=(20, 8)).astype(np.float32)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    ids = [f"S2-{i}" for i in range(20)]
    f = found_rank(["S1-0"], d[[3]], ids, d, {"S1-0": {"S2-3", "S2-99"}}, k=5)
    assert f == {("S1-0", "S2-3"): 1}
