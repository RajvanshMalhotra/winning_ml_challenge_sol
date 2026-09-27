"""Copy-group (record<->record) signal for orphan decoys. For each OOF pair (S1 s, record c) use c's S2/S3 neighbours
(kg_recrec_edges: TF-IDF top-10 + name keys) and Model A p over ALL S1s (OOF p for sample S1s) to compute:
  nb_n        #neighbours of c
  nb_s        #neighbours that s also wants (p(s, nb) >= 0.5)
  nb_orphan   #neighbours no S1 wants (max p < 0.1)
  nb_other    #neighbours whose best S1 is a different S1 with p >= 0.5
Compare true vs wrong ACCEPTED pairs of v12 (and missed vs correctly rejected)."""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ber.config import load_config, run_dir
from ber.matcher import decide

cfg = load_config("configs/v2.yaml")
RD = run_dir(cfg)
rec = pd.read_parquet(RD / "records_train.parquet", columns=["entity_id"])
code = pd.Series(np.arange(len(rec), dtype=np.int64), index=rec.entity_id)
t = pq.read_table(RD / "train_scores_matcher_v3.parquet", columns=["s1_id", "cand_id", "p"]).to_pandas()
t["s"], t["c"] = code.reindex(t.s1_id).to_numpy(), code.reindex(t.cand_id).to_numpy()
t = t[["s", "c", "p"]]
o = pd.read_parquet(RD / "matcher_v3" / "oof.parquet", columns=["s1_id", "cand_id", "p"])
o["s"], o["c"] = code.reindex(o.s1_id).to_numpy(), code.reindex(o.cand_id).to_numpy()
t = pd.concat([t.merge(o[["s", "c"]], on=["s", "c"], how="left", indicator=True).query("_merge == 'left_only'").drop(columns="_merge"),
               o[["s", "c", "p"]]], ignore_index=True)
print(f"all train pairs with p: {len(t):,}", flush=True)
best = t.sort_values("p").groupby("c").agg(bp=("p", "last"), bs=("s", "last"))
E = pd.read_parquet(RD / "kg_recrec_edges_train.parquet", columns=["a", "b"]).astype(np.int64)
f = pd.read_parquet(RD / "v12" / "oof_q12.parquet", columns=["s1_id", "cand_id", "label", "q12", "cand_noaddr"])
pred = decide(f[["s1_id", "cand_id"]].assign(p=f.q12), 0.75)
f["acc"] = [c in pred.get(s, set()) for s, c in zip(f.s1_id, f.cand_id)]
f["s"], f["c"] = code.reindex(f.s1_id).to_numpy(), code.reindex(f.cand_id).to_numpy()
g = f[f.q12 >= 0.3][["s", "c"]].drop_duplicates()
nb = g.merge(E, left_on="c", right_on="a")[["s", "c", "b"]]
nb = nb.merge(best, left_on="b", right_index=True, how="left")
nb = nb.merge(t.rename(columns={"c": "b", "p": "ps"}), on=["s", "b"], how="left")
nb["ps"] = nb.ps.fillna(0)
agg = nb.assign(n=1, same=(nb.ps >= 0.5), orphan=(nb.bp.fillna(0) < 0.1), other=(nb.bs != nb.s) & (nb.bp >= 0.5)) \
        .groupby(["s", "c"])[["n", "same", "orphan", "other"]].sum().reset_index()
f = f.merge(agg, on=["s", "c"], how="left").fillna({"n": 0, "same": 0, "orphan": 0, "other": 0})
y = f.label.astype(bool)
for name, m in [("TRUE accepted", y & f.acc), ("WRONG accepted", ~y & f.acc), ("TRUE missed (q>=0.3)", y & ~f.acc & (f.q12 >= 0.3)),
                ("wrong rejected (q>=0.3)", ~y & ~f.acc & (f.q12 >= 0.3))]:
    d = f[m]
    print(f"{name:26s} n={len(d):7,} | neighbours median {d.n.median():.0f} | >=1 neighbour s wants {(d.same > 0).mean():.1%} "
          f"| >=1 unwanted neighbour {(d.orphan > 0).mean():.1%} | all neighbours unwanted {((d.n > 0) & (d.orphan == d.n)).mean():.1%} "
          f"| >=1 neighbour owned by another S1 {(d.other > 0).mean():.1%} | no neighbours {(d.n == 0).mean():.1%}", flush=True)
