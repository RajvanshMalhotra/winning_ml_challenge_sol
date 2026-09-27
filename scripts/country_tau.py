"""Per-country cut-off from saved final scores (same filter/candidates). usage: country_tau.py <scores> <out_dir> <tau> <country=tau,...>
e.g. country_tau.py artifacts/v2/v12/test_q12.parquet artifacts/submissions/v12_fr90 0.75 France=0.90"""
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

src, out, tau = sys.argv[1], Path(sys.argv[2]), float(sys.argv[3])
over = dict((k, float(v)) for k, v in (x.split("=") for x in sys.argv[4:]))
cfg = load_config("configs/v2.yaml"); RD = run_dir(cfg)
sc = pd.read_parquet(src)
pa = pq.read_table(RD / "test_scores_matcher_v3.parquet", columns=["cand_id", "p"]).to_pandas()
assert len(pa) == len(sc) and (pa.cand_id.to_numpy() == sc.cand_id.to_numpy()).all()
sc = sc[pa.p.to_numpy() >= 0.01].reset_index(drop=True)          # learned filter (candidate set)
rec = pd.read_parquet(records_path(cfg, "test"), columns=["entity_id", "source", "country"])
ctry = rec.set_index("entity_id").country.reindex(sc.s1_id).to_numpy()
t = np.full(len(sc), tau)
for k, v in over.items():
    t[ctry == k] = v
p = np.clip(sc.p.to_numpy() * (tau / t), 0, 1)                    # per-row cut-off == global tau after rescaling
pred = decide(sc.assign(p=p), tau)
s1_all = rec.entity_id[rec.source == 1].tolist()
write_submission(s1_all, pred, sc.groupby("s1_id").cand_id.agg(set).to_dict(), out)
n = pd.Series({s: len(v) for s, v in pred.items()}).reindex(s1_all, fill_value=0)
c1 = rec.set_index("entity_id").country.reindex(n.index).to_numpy()
print({"tau": tau, **over, **{c: round(float(n[c1 == c].mean()), 4) for c in ["US", "India", "France"]},
       "pairs": int(n.sum())}, flush=True)
v = subprocess.run(["python3", cfg["paths"]["validator"], "--matching", str(out / "matching_results.tsv"), "--candidate",
                    str(out / "candidate_pairs.tsv"), "--test-dir", str(Path(cfg["paths"]["data_dir"]) / "test")],
                   capture_output=True, text=True)
print(v.stdout[-120:], f"validator exit={v.returncode}", flush=True)
