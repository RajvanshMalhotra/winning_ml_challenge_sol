"""France: v12-REJECTED candidate pairs with the same core name + same house number + same région and a fuzzy-similar
address (street typo). Counts by address similarity, plus the same statistic on US/India train OOF (true share)."""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

RD = "artifacts/v2"


def pattern(pairs, r):
    a, b = r.reindex(pairs.s), r.reindex(pairs.c)
    m = (a.name_core.to_numpy() == b.name_core.to_numpy()) & (b.addr_norm.fillna("").to_numpy() != "") \
        & (a.house_no.to_numpy() == b.house_no.to_numpy()) & (a.house_no.fillna("").to_numpy() != "") \
        & (a.state.to_numpy() == b.state.to_numpy()) & (a.addr_norm.to_numpy() != b.addr_norm.to_numpy())
    x = pairs[m].copy()
    x["sim"] = cpdist(a.addr_norm.to_numpy()[m].tolist(), b.addr_norm.to_numpy()[m].tolist(), scorer=fuzz.token_set_ratio, workers=-1)
    x["bin"] = pd.cut(x.sim, [0, 50, 80, 90, 101], labels=["<50", "50-80", "80-90", "90+"], right=False)
    return x


cols = ["entity_id", "source", "country", "addr_norm", "name_core", "house_no", "state"]
# train OOF (US/India): true share by bin
tr = pd.read_parquet(f"{RD}/records_train.parquet", columns=cols).set_index("entity_id")
o = pd.read_parquet(f"{RD}/v12/oof_q12.parquet", columns=["s1_id", "cand_id", "label", "q12"]).rename(columns={"s1_id": "s", "cand_id": "c"})
x = pattern(o, tr)
x["country"] = tr.country.reindex(x.s).to_numpy()
print("TRAIN OOF (same name+house+state, different address):")
print(x.groupby(["country", "bin"], observed=True).agg(pairs=("label", "size"), true=("label", "mean"), accepted=("q12", lambda q: (q >= 0.75).mean())).round(3).to_string())
# test France: all candidate pairs (filtered pool), v12 decisions
te = pd.read_parquet(f"{RD}/records_testfr.parquet", columns=cols).set_index("entity_id")
t = pq.read_table(f"{RD}/v12/test_q12.parquet").to_pandas().rename(columns={"s1_id": "s", "cand_id": "c", "p": "q"})
fr = set(te.index[(te.source == 1) & (te.country == "France")])
t = t[t.s.isin(fr)]
m = pd.read_csv("artifacts/submissions/v12/matching_results.tsv", sep="\t", dtype=str, keep_default_na=False)
acc = {(s, c) for s, v in zip(m.source1_entity_id, m.matched_entity_ids) if v for c in v.split(",")}
taken = {c for _, c in acc}
y = pattern(t, te)
y["accepted"] = [(s, c) in acc for s, c in zip(y.s, y.c)]
y["owned_elsewhere"] = [(c in taken) and ((s, c) not in acc) for s, c in zip(y.s, y.c)]
print("\nTEST France (same name+house+région, different address):")
print(y.groupby("bin", observed=True).agg(pairs=("q", "size"), accepted=("accepted", "mean"), owned_elsewhere=("owned_elsewhere", "mean"),
                                          median_q=("q", "median")).round(3).to_string())
rej = y[~y.accepted & ~y.owned_elsewhere & (y.sim >= 80)]
print(f"\nrejected & free, address sim >= 80: {len(rej):,} pairs on {rej.s.nunique():,} France S1s ({rej.c.nunique():,} records)")
for z in rej.sample(min(10, len(rej)), random_state=0).itertuples():
    print(f"  q={z.q:.2f} {te.name_core[z.s]!r} | S1: {te.addr_norm[z.s]!r}  <->  {te.addr_norm[z.c]!r}")
rej.to_parquet(f"{RD}/v12/france_typo_candidates.parquet", index=False)
