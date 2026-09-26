"""Export v9's still-missed true pairs (OOF) with raw text and the S1 that Model A preferred -> eda_v9/v9_still_missed.parquet"""
import numpy as np, pandas as pd, pyarrow.dataset as ds
from ber.config import load_config, run_dir
cfg = load_config("configs/v2.yaml"); RD = run_dir(cfg)
d = pd.read_parquet(RD / "eda_v9" / "v9_changes.parquet")
d = d[d.grp == "still missed"].copy()
t = ds.dataset(RD / "train_scores_matcher_v3.parquet").to_table(filter=ds.field("cand_id").isin(d.cand_id.unique().tolist())).to_pandas()
t = t.sort_values("p", ascending=False)
best = {}
for c, g in t.groupby("cand_id", sort=False):
    best[c] = list(zip(g.s1_id.to_numpy()[:3], g.p.to_numpy()[:3]))
def other(r):
    for s, p in best.get(r.cand_id, []):
        if s != r.s1_id: return s, p
    return None, np.nan
o = [other(r) for r in d.itertuples()]
d["top_other_s1"] = [x[0] for x in o]; d["top_other_p"] = [x[1] for x in o]
rec = pd.read_parquet(RD / "records_train.parquet", columns=["entity_id", "name_raw", "addr_raw"]).set_index("entity_id")
for pre, col in [("s1", "s1_id"), ("cand", "cand_id"), ("other", "top_other_s1")]:
    r = rec.reindex(d[col].to_numpy())
    d[f"{pre}_name"], d[f"{pre}_addr"] = r.name_raw.to_numpy(), r.addr_raw.to_numpy()
d.to_parquet(RD / "eda_v9" / "v9_still_missed.parquet", index=False)
print(len(d), d.cause.value_counts().to_dict())
