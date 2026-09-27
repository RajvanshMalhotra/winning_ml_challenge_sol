"""Stage-2 learned filter: the final matcher only runs on pairs with Model A p >= CAND_MIN; candidate_pairs.tsv is
that filtered set. Rewrites a submission from saved final scores (same s1/cand row order as the Model A scores).
usage: refilter.py <final_scores.parquet> <tau> <out_dir> [cand_min=0.01]"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ber.config import load_config, run_dir
from ber.matcher import decide
from ber.normalize import records_path
from ber.predict import write_submission

src, tau, out = sys.argv[1], float(sys.argv[2]), Path(sys.argv[3])
cmin = float(sys.argv[4]) if len(sys.argv) > 4 else 0.01
cfg = load_config("configs/v2.yaml")
RD = run_dir(cfg)
sc = pd.read_parquet(src)
pa = pq.read_table(RD / "test_scores_matcher_v3.parquet", columns=["s1_id", "cand_id", "p"]).to_pandas()
assert len(pa) == len(sc) and (pa.cand_id.to_numpy() == sc.cand_id.to_numpy()).all(), "row order differs"
keep = pa.p.to_numpy() >= cmin
sc = sc[keep]
pred = decide(sc, tau)
rec = pd.read_parquet(records_path(cfg, "test"), columns=["entity_id", "source", "country"])
s1_all = rec.entity_id[rec.source == 1].tolist()
write_submission(s1_all, pred, sc.groupby("s1_id").cand_id.agg(set).to_dict(), out)
print(f"cand_min {cmin}: {keep.sum():,} candidate pairs ({keep.sum() / len(s1_all):.2f}/S1), "
      f"{sum(len(v) for v in pred.values()):,} matches", flush=True)
v = subprocess.run(["python3", cfg["paths"]["validator"], "--matching", str(out / "matching_results.tsv"), "--candidate",
                    str(out / "candidate_pairs.tsv"), "--test-dir", str(Path(cfg["paths"]["data_dir"]) / "test")],
                   capture_output=True, text=True)
print(v.stdout[-300:], f"validator exit={v.returncode}", flush=True)
