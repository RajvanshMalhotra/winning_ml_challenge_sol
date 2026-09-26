"""True pairs still MISSED by submission v3 (Model A v2 + reranker stack), for one country, on the 50k B-split OOF sample.
Writes artifacts/v2/eda/v3_missed_<country>.csv and prints a summary.  usage: v3_misses.py [country] [tau]"""
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

COUNTRY = sys.argv[1] if len(sys.argv) > 1 else "India"
TAU = float(sys.argv[2]) if len(sys.argv) > 2 else 0.70
sys.argv = ["reranker.py", "stack"]
ns = {}
exec(open("scripts/reranker.py").read().split('{"train": train')[0], ns)
RD, RR, cfg = ns["RD"], ns["OUT"], ns["cfg"]
from ber.io import read_truth
from ber.matcher import decide

o = pd.read_parquet(RD / "matcher_emb" / "oof.parquet")
b = pd.read_parquet(RR / "oof_band_scores.parquet")[["s1_id", "cand_id", "rr"]]
f = ns["stack_features"](o.merge(b, on=["s1_id", "cand_id"], how="left"))
sp = pd.read_parquet(RD / "splits.parquet", columns=["s1_id", "country", "fold"]).set_index("s1_id")
f["fold"] = sp.fold.reindex(f.s1_id).to_numpy()
q = np.zeros(len(f))
for k in range(5):
    m = lgb.Booster(model_file=str(RR / f"stack_fold{k}.txt"))
    va = f.fold.to_numpy() == k
    q[va] = m.predict(f.loc[va, ns["FEATS"]], num_iteration=m.best_iteration)
f["q"] = q
pred = decide(f[["s1_id", "cand_id"]].assign(p=f.q), TAU)
truth = read_truth(cfg["paths"]["data_dir"])

rec = pd.read_parquet(RD / "records_train.parquet", columns=["entity_id", "source", "country", "state", "name_raw", "addr_raw",
                                                               "name_core", "addr_norm", "house_no"]).set_index("entity_id")
s1s = [s for s in f.s1_id.unique() if sp.country.get(s) == COUNTRY]
fi = f.set_index(["s1_id", "cand_id"])
owner = f[f.q >= TAU].sort_values("q").groupby("cand_id").s1_id.last()  # who won each candidate (one-owner)
rows = []
for s in s1s:
    for c in truth[s]:
        if c in pred.get(s, set()):
            continue
        if (s, c) not in fi.index:
            status, pa, rr, qq = "never a candidate", np.nan, np.nan, np.nan
        else:
            r = fi.loc[(s, c)]
            pa, rr, qq = r.p, r.rr, r.q
            status = "given to another business" if (qq >= TAU and owner.get(c) != s) else "rejected by model"
        rows.append((s, c, status, pa, rr, qq))
d = pd.DataFrame(rows, columns=["s1_id", "cand_id", "status", "p_modelA", "p_reranker", "p_final"])
a, c = rec.reindex(d.s1_id), rec.reindex(d.cand_id)
jac = lambda x, y: len(set(x.split()) & set(y.split())) / max(len(set(x.split()) | set(y.split())), 1)
d["s1_name"], d["cand_name"], d["s1_addr"], d["cand_addr"] = a.name_raw.values, c.name_raw.values, a.addr_raw.values, c.addr_raw.values
d["cand_source"] = d.cand_id.str[:2]
d["cand_native_script"] = c.name_raw.str.contains("[ऀ-෿]", regex=True).values
d["cand_addr_empty"] = (c.addr_raw == "").values
d["name_overlap"] = [jac(x, y) for x, y in zip(a.name_core, c.name_core)]
d["addr_overlap"] = [jac(x, y) for x, y in zip(a.addr_norm, c.addr_norm)]
d["house"] = np.where((a.house_no.values == "") | (c.house_no.values == ""), "missing",
                      np.where(a.house_no.values == c.house_no.values, "equal", "different"))
def why(r):
    if r.cand_native_script: return "Indian-script name"
    if r.cand_addr_empty: return "empty address"
    if r.name_overlap == 0: return "name shares no words (alias/website/truncated)"
    if r.addr_overlap < 0.2: return "address very different"
    return "similar text, still below cut-off"
d["likely_cause"] = d.apply(why, axis=1)
n_true = sum(len(truth[s]) for s in s1s)
print(f"{COUNTRY}: {len(s1s):,} businesses, {n_true:,} true pairs, missed {len(d):,} ({len(d)/n_true:.2%}) at tau={TAU}")
print(pd.crosstab(d.likely_cause, d.status, margins=True).to_string())
print("\nrejected-by-model score quantiles (final p):", d[d.status == "rejected by model"].p_final.quantile([.25, .5, .75, .9]).round(3).to_dict())
out = RD / "eda" / f"v3_missed_{COUNTRY}.csv"
d.sort_values(["status", "likely_cause", "p_final"], ascending=[True, True, False]).to_csv(out, index=False)
print("saved", out)
for st in ["rejected by model", "never a candidate", "given to another business"]:
    x = d[d.status == st]
    print(f"\n== {st} ({len(x):,}) — examples")
    for r in x.sample(min(6, len(x)), random_state=1).itertuples():
        print(f"  [{r.likely_cause}] p={r.p_final:.2f}  S1: {r.s1_name} | {r.s1_addr[:60]}")
        print(f"      {r.cand_source}: {r.cand_name} | {r.cand_addr[:60]}")
