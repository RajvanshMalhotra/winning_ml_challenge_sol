"""Shared addresses: how many S1s sit at the same normalized address (s1_addr_cnt) and how many S1s share the candidate's
address (cand_addr_s1cnt), OOF (US/India, with truth) vs test by country (v14 decisions)."""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ber.config import load_config, run_dir
from ber.matcher import decide

cfg = load_config("configs/v2.yaml")
RD = run_dir(cfg)
bins, labels = [0, 1, 3, 10, 10 ** 9], ["1 (alone)", "2-3", "4-10", "11+"]

f = pd.read_parquet(RD / "v12" / "oof_q12.parquet", columns=["s1_id", "cand_id", "label", "q12"])
v = pd.read_parquet(RD / "cand_side_v9" / "feats_train.parquet", columns=["s1_addr_cnt", "cand_addr_s1cnt"])
f["s1_addr_cnt"], f["cand_addr_s1cnt"] = v.s1_addr_cnt.to_numpy(), v.cand_addr_s1cnt.to_numpy()
del v
pred = decide(f[["s1_id", "cand_id"]].assign(p=f.q12), 0.75)
f["acc"] = [c in pred.get(s, set()) for s, c in zip(f.s1_id, f.cand_id)]
f["b"] = pd.cut(f.s1_addr_cnt, bins, labels=labels)
s1b = f.drop_duplicates("s1_id").b.value_counts(normalize=True).reindex(labels)
a = f[f.acc]
mid = a[a.q12 < 0.9]
print("OOF US/India, by #S1s at the S1's address:")
print(pd.DataFrame({"S1 share": s1b.round(4), "accepted": a.groupby("b", observed=False).size(),
                    "precision": a.groupby("b", observed=False).label.mean().round(4),
                    "accepted 0.75-0.9": mid.groupby("b", observed=False).size(),
                    "precision 0.75-0.9": mid.groupby("b", observed=False).label.mean().round(3)}).to_string())
t = f[f.label.astype(bool)]
print("recall by bucket:", t.groupby("b", observed=False).acc.mean().round(4).to_dict())

pm = pq.read_table(RD / "test_scores_matcher_v3.parquet", columns=["p"]).column("p").to_numpy() >= 0.01
v = pq.read_table(RD / "cand_side_v9" / "feats_test.parquet", columns=["s1_addr_cnt", "cand_addr_s1cnt"]).to_pandas()[pm].reset_index(drop=True)
sc = pd.read_parquet(RD / "v14" / "test_final_scores.parquet")
assert len(sc) == len(v)
sc["s1_addr_cnt"] = v.s1_addr_cnt.to_numpy()
del v
rec = pd.read_parquet(RD / "records_test.parquet", columns=["entity_id", "source", "country"])
ctry = rec.set_index("entity_id").country
sc["country"] = ctry.reindex(sc.s1_id).to_numpy()
pred = decide(sc[["s1_id", "cand_id", "p"]], 0.75)
sc["acc"] = [c in pred.get(s, set()) for s, c in zip(sc.s1_id, sc.cand_id)]
sc["b"] = pd.cut(sc.s1_addr_cnt, bins, labels=labels)
print("\nTEST (v14), by country and #S1s at the S1's address:")
for c, g in sc.groupby("country"):
    s1b = g.drop_duplicates("s1_id").b.value_counts(normalize=True).reindex(labels)
    a = g[g.acc]
    print(f"{c}: S1 share " + ", ".join(f"{k} {v:.2%}" for k, v in s1b.items()) +
          " | accepted pairs " + ", ".join(f"{k} {v:,}" for k, v in a.b.value_counts().reindex(labels).items()) +
          " | uncertain-accepted (0.75-0.9) share " + ", ".join(f"{k} {v:.2%}" for k, v in
                                                            a.groupby("b", observed=False).p.apply(lambda x: (x < 0.9).mean()).items()), flush=True)
