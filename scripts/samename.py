"""Missed no-address matches: can the true S1 be told apart from other S1s sharing the same core name?"""
import numpy as np, pandas as pd
from ber.config import load_config, run_dir
from ber.io import read_truth
from ber.matcher import decide
cfg = load_config("configs/v2.yaml"); RD = run_dir(cfg)
truth = read_truth(cfg["paths"]["data_dir"])
f = pd.read_parquet(RD / "cand_side" / "oof_q.parquet", columns=["s1_id", "cand_id", "q", "label", "cand_noaddr", "s1_name_cnt"])
pred = decide(f[["s1_id", "cand_id"]].assign(p=f.q), 0.75)
m = f[(f.cand_noaddr == 1) & f.label.astype(bool)].copy()
m["found"] = [c in pred.get(s, set()) for s, c in zip(m.s1_id, m.cand_id)]
rec = pd.read_parquet(RD / "records_train.parquet", columns=["entity_id", "source", "country", "name_raw", "name_norm", "name_core", "legal_suffix"]).set_index("entity_id")
s1 = rec[rec.source == 1]
by_core = s1.reset_index().groupby(["country", "name_core"]).entity_id.agg(list)
rows = []
for r in m[~m.found].itertuples():
    c = rec.loc[r.cand_id]; t = rec.loc[r.s1_id]
    comp = [x for x in by_core.get((t.country, c.name_core), []) if x != r.s1_id]
    if not comp:
        rows.append(("no other S1 with this name", 0)); continue
    others = s1.loc[comp]
    same_norm_true = t.name_norm == c.name_norm
    same_norm_other = (others.name_norm == c.name_norm).sum()
    same_raw_all = (others.name_raw == t.name_raw).all()
    if same_raw_all: k = "other S1s have IDENTICAL raw name (unresolvable by name)"
    elif same_norm_true and same_norm_other == 0: k = "full name (incl. suffix) matches ONLY the true S1"
    elif same_norm_true: k = "full name matches true S1 AND others"
    else: k = "full name matches neither exactly"
    rows.append((k, len(comp)))
d = pd.DataFrame(rows, columns=["case", "n_other"])
print(f"missed no-address true pairs: {len(d):,}")
print(d.case.value_counts().to_string())
print("other same-core-name S1s: median", d.n_other.median(), "| quantiles", d.n_other.quantile([.25, .75, .9]).to_dict())
# do other same-name S1s already have their own copy? (are they 'full')
ex = m[~m.found].sample(5, random_state=1)
for r in ex.itertuples():
    c = rec.loc[r.cand_id]; t = rec.loc[r.s1_id]
    comp = [x for x in by_core.get((t.country, c.name_core), []) if x != r.s1_id][:3]
    print(f"\ncand {r.cand_id}: '{c.name_raw}' (no address) -> true S1 '{t.name_raw}'")
    for x in comp: print(f"   other S1 '{s1.loc[x].name_raw}' (true matches: {len(truth[x])})")
