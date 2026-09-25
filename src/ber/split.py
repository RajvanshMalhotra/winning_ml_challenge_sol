import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ber.config import log_metrics, run_dir
from ber.io import read_truth
from ber.normalize import records_path


def _bucket(n: int) -> str:
    return "0" if n == 0 else "1" if n == 1 else "2-3" if n <= 3 else "4-5" if n <= 5 else "6+"


def assign_splits(truth: dict[str, set[str]], s1_country: pd.Series, cfg_split: dict, seed: int) -> pd.DataFrame:
    df = pd.DataFrame({"s1_id": sorted(truth)})
    df["country"] = s1_country.reindex(df.s1_id).values
    df["n_matches"] = [len(truth[s]) for s in df.s1_id]
    df["split"], df["fold"] = "B", -1
    rng = np.random.default_rng(seed)
    for _, idx in df.groupby([df.country, df.n_matches.map(_bucket)]).groups.items():
        idx = rng.permutation(np.asarray(idx))
        n_tr, n_val = round(len(idx) * cfg_split["a_train"]), round(len(idx) * cfg_split["a_val"])
        df.loc[idx[:n_tr], "split"] = "A_train"
        df.loc[idx[n_tr:n_tr + n_val], "split"] = "A_val"
        b = idx[n_tr + n_val:]
        df.loc[b, "fold"] = (np.arange(len(b)) + int(rng.integers(cfg_split["n_folds"]))) % cfg_split["n_folds"]
    df["fold"] = df["fold"].astype("int8")
    return df


def splits_path(cfg: dict) -> Path:
    return run_dir(cfg) / "splits.parquet"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    pass


def run(cfg: dict, args: argparse.Namespace) -> None:
    rec = pd.read_parquet(records_path(cfg, "train"), columns=["entity_id", "source", "country"])
    s1 = rec[rec.source == 1].set_index("entity_id").country
    df = assign_splits(read_truth(cfg["paths"]["data_dir"]), s1, cfg["split"], cfg["seed"])
    df.to_parquet(splits_path(cfg), index=False)
    summary = df.groupby(["split", "country"]).size().unstack(fill_value=0)
    print(summary)
    log_metrics(cfg, "split", {f"{s}|{c}": int(v) for (s, c), v in summary.stack().items()})
