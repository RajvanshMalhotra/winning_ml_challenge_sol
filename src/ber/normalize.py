import argparse
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

from ber.config import run_dir
from ber.io import read_family
from ber.lexicons import load_lexicon
from ber.text import domain_body, is_domain_name, normalize_address, normalize_name, segment

COLUMNS = ["entity_id", "source", "country", "name_raw", "addr_raw", "name_norm", "name_core",
           "legal_suffix", "name_domain", "addr_norm", "house_no", "num_tokens", "block_text"]


def _normalize_chunk(raw: pd.DataFrame) -> pd.DataFrame:
    lex = load_lexicon()
    names = [normalize_name(n, lex) for n in raw["business_name"]]
    addrs = [normalize_address(a, c, lex) for a, c in zip(raw["business_address"], raw["country"])]
    out = pd.DataFrame({
        "entity_id": raw["entity_id"].values,
        "source": raw["source"].values,
        "country": raw["country"].values,
        "name_raw": raw["business_name"].values,
        "addr_raw": raw["business_address"].values,
    })
    out["name_norm"], out["name_core"], out["legal_suffix"] = map(list, zip(*names)) if names else ([], [], [])
    out["addr_norm"], out["house_no"], out["num_tokens"] = map(list, zip(*addrs)) if addrs else ([], [], [])
    out["is_domain"] = [is_domain_name(n) for n in raw["business_name"]]
    return out


def _segment_domains(rec: pd.DataFrame) -> pd.Series:
    """Split domain-style names using a per-country vocabulary of ordinary name tokens."""
    result = pd.Series("", index=rec.index)
    for country, grp in rec[rec["is_domain"]].groupby("country"):
        plain = rec.loc[(rec["country"] == country) & ~rec["is_domain"], "name_core"]
        vocab = {t for t in plain.str.split().explode().dropna() if len(t) >= 2}
        result.loc[grp.index] = [" ".join(segment(domain_body(n), vocab)) for n in grp["name_raw"]]
    return result


def build_records(raw: pd.DataFrame, n_jobs: int) -> pd.DataFrame:
    if n_jobs <= 1:
        rec = _normalize_chunk(raw)
    else:
        bounds = np.linspace(0, len(raw), n_jobs * 4 + 1, dtype=int)
        chunks = [raw.iloc[a:b] for a, b in zip(bounds[:-1], bounds[1:]) if b > a]
        with ProcessPoolExecutor(n_jobs, mp_context=get_context("fork")) as ex:
            rec = pd.concat(list(ex.map(_normalize_chunk, chunks)), ignore_index=True)
    rec["name_domain"] = _segment_domains(rec)
    name_for_block = rec["name_domain"].where(rec["name_domain"] != "", rec["name_core"])
    rec["block_text"] = name_for_block + " " + rec["addr_norm"]
    return rec[COLUMNS].reset_index(drop=True)


def records_path(cfg: dict, family: str) -> Path:
    return run_dir(cfg) / f"records_{family}.parquet"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--family", nargs="+", default=["train", "test"], choices=["train", "test"])


def run(cfg: dict, args: argparse.Namespace) -> None:
    for family in args.family:
        rec = build_records(read_family(cfg["paths"]["data_dir"], family), cfg["n_jobs"])
        rec.to_parquet(records_path(cfg, family), index=False)
        print(family, len(rec), "records; domain names:", int((rec.name_domain != "").sum()),
              "; house_no present:", round(float((rec.house_no != "").mean()), 3))
