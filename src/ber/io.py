import csv
from pathlib import Path

import pandas as pd


def _read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)


def read_family(data_dir: str | Path, family: str) -> pd.DataFrame:
    parts = []
    for s in (1, 2, 3):
        df = _read_tsv(Path(data_dir) / family / f"{family}_source{s}.tsv")
        df["source"] = s
        parts.append(df)
    out = pd.concat(parts, ignore_index=True)
    out["source"] = out["source"].astype("int8")
    return out


def read_truth(data_dir: str | Path) -> dict[str, set[str]]:
    gt = _read_tsv(Path(data_dir) / "train" / "train_ground_truth.tsv")
    return {
        s1: {x for x in ids.split(",") if x}
        for s1, ids in zip(gt["source1_entity_id"], gt["matched_entity_ids"])
    }


def truth_pairs(truth: dict[str, set[str]], s1_ids: list[str] | None = None) -> pd.DataFrame:
    keys = truth.keys() if s1_ids is None else s1_ids
    rows = [(s1, c) for s1 in keys for c in truth.get(s1, ())]
    return pd.DataFrame(rows, columns=["s1_id", "cand_id"])
