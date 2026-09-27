"""Rewrite a submission from saved final test scores at a different cut-off (prior-shift probe).
usage: retau.py <scores.parquet> <tau> <out_dir>"""
import subprocess
import sys
from pathlib import Path

import pandas as pd

from ber.config import load_config
from ber.matcher import decide
from ber.normalize import records_path
from ber.predict import write_submission

src, tau, out = sys.argv[1], float(sys.argv[2]), Path(sys.argv[3])
cfg = load_config("configs/v2.yaml")
sc = pd.read_parquet(src)
pred = decide(sc, tau)
rec = pd.read_parquet(records_path(cfg, "test"), columns=["entity_id", "source", "country"])
s1_all = rec.entity_id[rec.source == 1].tolist()
write_submission(s1_all, pred, sc.groupby("s1_id").cand_id.agg(set).to_dict(), out)
n = pd.Series({s: len(v) for s, v in pred.items()}).reindex(s1_all, fill_value=0)
ctry = rec.set_index("entity_id").country.reindex(n.index).to_numpy()
print({"tau": tau, "mean_matches": round(float(n.mean()), 4), "empty": round(float((n == 0).mean()), 4),
       **{c: round(float(n[ctry == c].mean()), 4) for c in ["US", "India", "France"]}}, flush=True)
v = subprocess.run(["python3", cfg["paths"]["validator"], "--matching", str(out / "matching_results.tsv"), "--candidate",
                    str(out / "candidate_pairs.tsv"), "--test-dir", str(Path(cfg["paths"]["data_dir"]) / "test")],
                   capture_output=True, text=True)
print(v.stdout[-300:], f"validator exit={v.returncode}", flush=True)
