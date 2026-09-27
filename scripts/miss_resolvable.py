"""Are v12's missed matches resolvable? For each missed (and, as control, each found) true pair, gather ALL S1s that
have the same record as a candidate (sparse + dense, whole train) and check whether the TRUE S1 ranks first by finer
raw-name comparisons than the ones the stacker already has."""
import numpy as np, pandas as pd, pyarrow.dataset as ds
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist
from ber.config import load_config, run_dir
from ber.matcher import decide
cfg = load_config("configs/v2.yaml"); RD = run_dir(cfg)
f = pd.read_parquet(RD / "v12" / "oof_q12.parquet")
pred = decide(f[["s1_id", "cand_id"]].assign(p=f.q12), 0.75)
f["pred"] = [c in pred.get(s, set()) for s, c in zip(f.s1_id, f.cand_id)]
y = f.label.astype(bool)
miss = f[y & ~f.pred].sample(4000, random_state=0)
found = f[y & f.pred & (f.cand_noaddr == 1)].sample(4000, random_state=0)
rec = pd.read_parquet(RD / "records_train.parquet", columns=["entity_id", "name_raw", "name_norm", "name_core", "legal_suffix"]).set_index("entity_id")
cands = pd.concat([miss.cand_id, found.cand_id]).unique().tolist()
flt = ds.field("cand_id").isin(cands)
comp = pd.concat([ds.dataset(RD / "cand_sparse_train.parquet").to_table(columns=["s1_id", "cand_id"], filter=flt).to_pandas(),
                  ds.dataset(RD / "cand_dense_train.parquet").to_table(columns=["s1_id", "cand_id"], filter=flt).to_pandas()]).drop_duplicates()
a, b = rec.reindex(comp.s1_id), rec.reindex(comp.cand_id)
low = lambda s: s.fillna("").str.lower().tolist()
comp["raw_ratio"] = cpdist(low(a.name_raw), low(b.name_raw), scorer=fuzz.ratio, workers=-1)
comp["norm_ratio"] = cpdist(a.name_norm.fillna("").tolist(), b.name_norm.fillna("").tolist(), scorer=fuzz.ratio, workers=-1)
comp["core_ratio"] = cpdist(a.name_core.fillna("").tolist(), b.name_core.fillna("").tolist(), scorer=fuzz.ratio, workers=-1)
comp["suffix_eq"] = (a.legal_suffix.fillna("").to_numpy() == b.legal_suffix.fillna("").to_numpy()).astype(float)
comp["combo"] = comp.raw_ratio + comp.norm_ratio + comp.core_ratio + 10 * comp.suffix_eq
for name, d in (("MISSED (all kinds)", miss), ("missed no-address", miss[miss.cand_noaddr == 1]), ("FOUND no-address (control)", found)):
    t = d[["s1_id", "cand_id"]].assign(true=1)
    m = comp.merge(t, on=["s1_id", "cand_id"], how="left")
    m["true"] = m["true"].fillna(0)
    m = m[m.cand_id.isin(set(d.cand_id))]
    out = {}
    for col in ["raw_ratio", "norm_ratio", "core_ratio", "combo"]:
        r = m.groupby("cand_id")[col].rank(ascending=False, method="min")
        top = m.assign(r=r)[m["true"] == 1]
        # unique top: the true S1 is strictly first
        tie = m.assign(r=r).groupby("cand_id").r.apply(lambda x: (x == 1).sum())
        uniq = top.r.eq(1) & top.cand_id.map(tie).eq(1)
        out[col] = f"top {top.r.eq(1).mean():.0%} (unique {uniq.mean():.0%})"
    n = m.groupby("cand_id").size()
    print(f"{name}: n={len(d):,}, competing S1s per record median {n.median():.0f} | " + " | ".join(f"{k}: {v}" for k, v in out.items()), flush=True)
