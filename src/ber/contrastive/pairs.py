import numpy as np
import pandas as pd

from ber.contrastive.encoders import EncoderSpec
from ber.noise import augment
from ber.text import record_text


def hard_negatives(cands: pd.DataFrame, truth: dict[str, set[str]], s1_ids: list[str]) -> dict[str, list[str]]:
    sub = cands[cands.s1_id.isin(set(s1_ids))]
    out: dict[str, list[str]] = {}
    for s1, c in zip(sub.s1_id, sub.cand_id):
        if c not in truth.get(s1, ()):
            out.setdefault(s1, []).append(c)
    return out


def sample_triplets(truth: dict[str, set[str]], s1_ids: list[str], hard_negs: dict[str, list[str]],
                    country_pool: dict[str, np.ndarray], id_country: dict[str, str],
                    n_rounds: int, p_intra: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_rounds):
        for s1 in s1_ids:
            matches = sorted(truth.get(s1, ()))
            if not matches:
                continue
            group = set(matches) | {s1}
            if len(matches) >= 2 and rng.random() < p_intra:
                i, j = rng.choice(len(matches), size=2, replace=False)
                anchor, pos = matches[i], matches[j]
            else:
                anchor, pos = s1, matches[int(rng.integers(len(matches)))]
            negs = hard_negs.get(s1) or []
            if negs:
                neg = negs[int(rng.integers(len(negs)))]
            else:
                pool = country_pool[id_country[s1]]
                neg = pool[int(rng.integers(len(pool)))]
                while neg in group:
                    neg = pool[int(rng.integers(len(pool)))]
            rows.append((anchor, pos, neg))
    return pd.DataFrame(rows, columns=["anchor_id", "positive_id", "negative_id"])


def render_triplets(trip: pd.DataFrame, records: pd.DataFrame, spec: EncoderSpec, aug_prob: float,
                    seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    name, addr, country = records["name_raw"], records["addr_raw"], records["country"]

    def text(eid: str, prefix: str, noisy: bool) -> str:
        n, a, c = name[eid], addr[eid], country[eid]
        if noisy and rng.random() < aug_prob:
            n, a = augment(n, a, c, rng)
        return prefix + record_text(n, a, c)

    return pd.DataFrame({
        "anchor": [text(e, spec.query_prefix, False) for e in trip.anchor_id],
        "positive": [text(e, spec.doc_prefix, True) for e in trip.positive_id],
        "negative": [text(e, spec.doc_prefix, True) for e in trip.negative_id],
    })
