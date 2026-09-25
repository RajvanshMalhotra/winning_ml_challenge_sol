import faiss
import numpy as np


def recall_at_k(q_ids: list[str], q_emb: np.ndarray, d_ids: list[str], d_emb: np.ndarray,
                truth: dict[str, set[str]], ks: list[int], n_threads: int = 16) -> dict[int, float]:
    """Micro recall: found true pairs / all true pairs, over the top-k inner-product neighbours."""
    faiss.omp_set_num_threads(n_threads)
    index = faiss.IndexFlatIP(d_emb.shape[1])
    index.add(np.ascontiguousarray(d_emb, dtype=np.float32))
    _, nn = index.search(np.ascontiguousarray(q_emb, dtype=np.float32), max(ks))
    d_ids = np.asarray(d_ids)
    hits, total = {k: 0 for k in ks}, 0
    for qid, row in zip(q_ids, nn):
        t = truth[qid]
        total += len(t)
        found = np.cumsum([j >= 0 and d_ids[j] in t for j in row])
        for k in ks:
            hits[k] += int(found[k - 1])
    return {k: hits[k] / total for k in ks} if total else {k: 1.0 for k in ks}
