"""Decision rule = per-S1 expected-F0.5 maximisation instead of a fixed cut-off.
1) one-owner: each record is kept only for the S1 with its highest probability;
2) per S1, candidates sorted by probability; for k = 0..K pick the top-k maximising E[F0.5] under independent
   Bernoulli(p_i) truth (Monte Carlo), where F0.5 = 1 for an empty prediction when there is no true match.
usage: expected_f.py oof            -> macro F0.5 on OOF vs the fixed cut-off
       expected_f.py test <scores.parquet> <out_dir>"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ber.config import load_config, run_dir
from ber.io import read_truth
from ber.matcher import decide, evaluate

cfg = load_config("configs/v2.yaml")
RD = run_dir(cfg)
LOW, K, NS, B2 = 0.02, 12, 400, 0.25
SHRINK = float(__import__("os").environ.get("EF_SHRINK", 1.0))     # optional calibration shrink of p


def choose(sc: pd.DataFrame) -> dict:
    sc = sc[sc.p >= LOW]
    sc = sc.sort_values("p", ascending=False).drop_duplicates("cand_id")          # one-owner first
    sc = sc.sort_values(["s1_id", "p"], ascending=[True, False])
    rng = np.random.default_rng(0)
    pred = {}
    for s, g in sc.groupby("s1_id", sort=False):
        p = np.clip(g.p.to_numpy()[:K] * SHRINK, 0, 1)
        ids = g.cand_id.to_numpy()[:K]
        y = rng.random((NS, len(p))) < p                       # sampled truths (independent)
        T = y.sum(1)
        best_k, best_v = 0, (T == 0).mean()                   # empty prediction: F=1 iff no true match
        tp = np.cumsum(y, axis=1)
        for k in range(1, len(p) + 1):
            f = (1 + B2) * tp[:, k - 1] / (B2 * T + k)          # F0.5 = (1+b2) TP / (b2 * |truth| + |pred|)
            v = f.mean()
            if v > best_v:
                best_k, best_v = k, v
        pred[s] = set(ids[:best_k])
    return pred


if sys.argv[1] == "oof":
    f = pd.read_parquet(RD / "v12" / "oof_q12.parquet", columns=["s1_id", "cand_id", "q12"]).rename(columns={"q12": "p"})
    truth = read_truth(cfg["paths"]["data_dir"])
    sp = pd.read_parquet(RD / "splits.parquet", columns=["s1_id", "country"]).set_index("s1_id")
    s1 = sp.loc[f.s1_id.unique()].reset_index()[["s1_id", "country"]]
    base = evaluate(decide(f, 0.75), truth, s1)
    print(f"fixed cut-off 0.75: {base.get('macro_f05'):.5f} (singletons {base.get('singletons'):.4f})", flush=True)
    for sh in [1.0, 0.97, 0.94]:
        SHRINK = sh
        e = evaluate(choose(f), truth, s1)
        print(f"expected-F rule (p x {sh}): {e.get('macro_f05'):.5f} (singletons {e.get('singletons'):.4f}, "
              f"India {e.get('f05_India'):.4f}, US {e.get('f05_US'):.4f})", flush=True)
else:
    from ber.normalize import records_path
    from ber.predict import write_submission
    sc = pd.read_parquet(sys.argv[2])
    pred = choose(sc)
    rec = pd.read_parquet(records_path(cfg, "test"), columns=["entity_id", "source"])
    s1_all = rec.entity_id[rec.source == 1].tolist()
    write_submission(s1_all, pred, sc.groupby("s1_id").cand_id.agg(set).to_dict(), Path(sys.argv[3]))
    print("written", sys.argv[3], sum(len(v) for v in pred.values()), "matches", flush=True)
