from collections import defaultdict

import numpy as np
import pandas as pd


def f05(pred: set[str], truth: set[str]) -> float:
    """Per-entity F0.5 exactly as the challenge defines it (singletons included)."""
    if not truth:
        return 1.0 if not pred else 0.0
    tp = len(pred & truth)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(truth)
    return 1.25 * p * r / (0.25 * p + r)


def macro_f05(pred: dict[str, set[str]], truth: dict[str, set[str]]) -> float:
    return float(np.mean([f05(pred.get(k, set()), t) for k, t in truth.items()]))


def f05_breakdown(pred: dict[str, set[str]], truth: dict[str, set[str]], groups: dict[str, str]) -> dict[str, float]:
    scores: dict[str, list[float]] = defaultdict(list)
    for k, t in truth.items():
        scores[groups[k]].append(f05(pred.get(k, set()), t))
    return {g: float(np.mean(v)) for g, v in scores.items()}


def pair_completeness(cands: pd.DataFrame, truth_pairs: pd.DataFrame) -> float:
    if len(truth_pairs) == 0:
        return 1.0
    hit = truth_pairs[["s1_id", "cand_id"]].merge(
        cands[["s1_id", "cand_id"]].drop_duplicates(), how="left", indicator=True
    )
    return float((hit["_merge"] == "both").mean())


def candidates_per_entity(cands: pd.DataFrame, s1_ids: list[str]) -> dict:
    counts = cands.groupby("s1_id").size().reindex(s1_ids, fill_value=0)
    return {"mean": float(counts.mean()), "p95": float(counts.quantile(0.95)), "max": int(counts.max())}
