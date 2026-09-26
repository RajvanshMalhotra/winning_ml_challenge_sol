"""Test has more orphan S2/S3 records (owned by no S1) than train, so every pair's prior odds drop. Two label-free ways to
pick the test cut-off for the v11 ensemble: (a) Bayes label-shift: odds x (test owned-odds / train owned-odds);
(b) match test's matches-per-S1 to OOF's at τ=0.75 (equal recall). Saves test q; writes submissions at chosen cut-offs.
usage: CUDA_VISIBLE_DEVICES= python -u scripts/prior_shift.py FAM"""
import json, os, subprocess, sys
import numpy as np, pandas as pd
from pathlib import Path
G = {"__name__": "x"}
exec(open(os.path.expanduser("~/claude_v11/scripts/stacker_v11_submit.py")).read().rsplit('\nif __name__ == "__main__":', 1)[0], G)
fam = sys.argv[1] if len(sys.argv) > 1 else "testfr"
OUT, ALL, log, cfg = G["OUT"], G["ALL"], G["log"], G["cfg"]
qpath = OUT / f"q_ens_{fam}.parquet"
if not qpath.exists():
    X = pd.read_parquet(OUT / f"X_{fam}.parquet", columns=["s1_id", "cand_id", "country"] + ALL, filters=[("p", ">=", 0.001)])
    q = G["predict_models"](X, ["full", "ens0", "ens1", "ens2"], ALL)
    X[["s1_id", "cand_id", "country"]].assign(q=q).to_parquet(qpath, index=False)
t = pd.read_parquet(qpath)
# (a) Bayes label shift
from ber.io import read_truth
from ber.normalize import records_path
truth = read_truth(cfg["paths"]["data_dir"])
owned = {c for m in truth.values() for c in m}
rtr = pd.read_parquet(records_path(cfg, "train"), columns=["entity_id", "source", "country"])
o = rtr[rtr.source != 1]
p_tr = float(o.entity_id.isin(owned).mean())
rte = pd.read_parquet(records_path(cfg, "test"), columns=["entity_id", "source", "country"])
n_s1 = rte[rte.source == 1].groupby("country").size(); n_oth = rte[rte.source != 1].groupby("country").size()
mean_true = 3.444                                    # generator's matches per S1 (train truth, identical for US and India)
p_te = (n_s1 * mean_true / n_oth).clip(upper=1)      # share of S2/S3 records owned by some S1 on test
ratio = (p_te / (1 - p_te)) / (p_tr / (1 - p_tr))
tau_bayes = {c: float(0.75 / (0.75 + 0.25 * ratio[c])) for c in ratio.index}  # q' = r q / (r q + 1 - q) >= .75  <=>  q >= .75 / (.75 + .25 r)
log(f"owned share train {p_tr:.3f}; test {p_te.round(3).to_dict()}; odds ratio {ratio.round(3).to_dict()}")
log(f"Bayes cut-off (q' >= 0.75): {({k: round(v, 3) for k, v in tau_bayes.items()})}")
# (b) matches-per-S1 matching against OOF
f = pd.read_parquet(OUT / "X_train.parquet", columns=["s1_id", "cand_id", "country", "label"])
qo = np.load(OUT / "oof_q.npy")[3]
def mean_matches(s1, cand, q, country, tau):
    d = pd.DataFrame({"s": s1, "c": cand, "q": q, "k": country}); d = d[d.q >= tau]; a = d.loc[d.groupby("c").q.idxmax()]
    return a.groupby("k").size()
oof_s1 = f.groupby("s1_id").country.first().value_counts()
oof_mean = mean_matches(f.s1_id, f.cand_id, qo, f.country, 0.75) / oof_s1
rows = []
for tau in [0.75, 0.78, 0.8, 0.82, 0.84, 0.86, 0.88, 0.9]:
    m = mean_matches(t.s1_id, t.cand_id, t.q.to_numpy(), t.country, tau) / n_s1
    om = mean_matches(f.s1_id, f.cand_id, qo, f.country, tau) / oof_s1
    rows.append({"tau": tau, **{f"test {c}": m.get(c, np.nan) for c in ["US", "India", "France"]}, **{f"OOF {c}": om.get(c, np.nan) for c in ["US", "India"]}})
tab = pd.DataFrame(rows)
print(tab.round(4).to_string(index=False), flush=True)
print("OOF matches per S1 at τ=0.75:", oof_mean.round(4).to_dict(), flush=True)
match = {}
for c in ["US", "India"]:
    x = tab[["tau", f"test {c}"]].to_numpy()
    match[c] = float(np.interp(oof_mean[c], x[::-1, 1], x[::-1, 0]))
log(f"cut-off where test matches/S1 = OOF matches/S1 at .75: {({k: round(v, 3) for k, v in match.items()})}")
json.dump({"tau_bayes": tau_bayes, "tau_match": match, "table": rows}, open(OUT / f"prior_shift_{fam}.json", "w"), indent=2, default=float)
