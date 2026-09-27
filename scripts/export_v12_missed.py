"""Every true pair v12 misses on OOF (150k B-split S1): shortlist misses (with each stage's score) + blocking misses.
-> artifacts/v2/v12_missed/v12_missed.parquet   usage: CUDA_VISIBLE_DEVICES= python -u scripts/export_v12_missed.py"""
import json
import re

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from rapidfuzz.distance import Levenshtein

from ber.config import load_config, run_dir
from ber.io import read_truth

cfg = load_config("configs/v2.yaml"); RD = run_dir(cfg); OUT = RD / "v12_missed"; OUT.mkdir(exist_ok=True)
tau = json.loads((RD / "v12" / "best.json").read_text())["v12"]["tau"]
f = pd.read_parquet(RD / "v12" / "oof_q12.parquet")
own = f[f.q12 >= tau]; own = own.loc[own.groupby("cand_id").q12.idxmax()]
acc = set(zip(own.s1_id, own.cand_id)); winner = dict(zip(own.cand_id, own.s1_id))
lab = f.label.to_numpy().astype(bool)
miss = f[lab & ~np.array([(s, c) in acc for s, c in zip(f.s1_id, f.cand_id)])].copy()
miss["stage"] = "rejected by v12"
# scores from each stage
a = pd.read_parquet(RD / "matcher_v3" / "oof.parquet", columns=["s1_id", "cand_id", "p"])
qw = pd.read_parquet(RD / "reranker_qwen2" / "oof_qwen.parquet").rename(columns={"rr2": "qwen_rr"})
cs = pd.read_parquet(RD / "cand_side" / "oof_q.parquet", columns=["s1_id", "cand_id", "rr", "s1_name_cnt"])
v9 = pd.read_parquet(RD / "cand_side_v9" / "feats_train.parquet", columns=["a_crank", "a_share", "a_ncomp"])
v9[["s1_id", "cand_id"]] = f[["s1_id", "cand_id"]].to_numpy()
s2 = pd.read_parquet(RD / "stage2_check" / "pool.parquet", columns=["s1_id", "cand_id", "s2"])
for t in [a, qw[["s1_id", "cand_id", "qwen_rr"]], cs, v9, s2]:
    miss = miss.merge(t, on=["s1_id", "cand_id"], how="left")
# blocking misses
truth = read_truth(cfg["paths"]["data_dir"])
short = set(zip(f.s1_id, f.cand_id))
blk = pd.DataFrame([(s, c) for s in f.s1_id.unique() for c in truth[s] if (s, c) not in short], columns=["s1_id", "cand_id"])
blk["stage"] = "never a candidate (blocking)"
miss = pd.concat([miss, blk], ignore_index=True)
rec = pd.read_parquet(RD / "records_train.parquet", columns=["entity_id", "source", "country", "name_raw", "addr_raw", "name_core",
                                                              "addr_norm", "house_no"]).set_index("entity_id")
A, B = rec.reindex(miss.s1_id), rec.reindex(miss.cand_id)
miss["country"] = A.country.to_numpy()
noaddr = B.addr_raw.fillna("").str.strip().eq("").to_numpy()
native = B.name_raw.fillna("").str.contains("[ऀ-෿]", regex=True).to_numpy()
noword = np.array([not (set(x.split()) & set(y.split())) for x, y in zip(A.name_core.fillna(""), B.name_core.fillna(""))])
ha, hb = A.house_no.fillna("").to_numpy(), B.house_no.fillna("").to_numpy()
typo = np.array([x != "" and y != "" and x != y and Levenshtein.distance(x, y) == 1 for x, y in zip(ha, hb)])
same = miss.s1_name_cnt.fillna(0).to_numpy() >= 2
miss["cause"] = np.select([noaddr & same, noaddr, native, noword, typo],
                          ["no address, 2+ S1s share the name", "no address, name unique", "Indian script name", "names share no word",
                           "house no. 1-digit typo"], "other / similar text")
# the S1 that competes for the record: v12's winner if any, else the best other S1 by Model A (all train S1s)
t = ds.dataset(RD / "train_scores_matcher_v3.parquet").to_table(filter=ds.field("cand_id").isin(miss.cand_id.unique().tolist())).to_pandas()
t = t.sort_values("p", ascending=False)
top = {c: list(zip(g.s1_id.to_numpy()[:3], g.p.to_numpy()[:3])) for c, g in t.groupby("cand_id", sort=False)}
oth, othp, why = [], [], []
for s, c in zip(miss.s1_id, miss.cand_id):
    if c in winner and winner[c] != s:
        oth.append(winner[c]); othp.append(np.nan); why.append("v12 gave the record to this S1")
        continue
    x = next(((o, p) for o, p in top.get(c, []) if o != s), (None, np.nan))
    oth.append(x[0]); othp.append(x[1]); why.append("best other S1 by Model A" if x[0] else "")
miss["other_s1"], miss["other_p"], miss["other_kind"] = oth, othp, why
O = rec.reindex(miss.other_s1)
miss["s1_name"], miss["s1_addr"] = A.name_raw.to_numpy(), A.addr_raw.to_numpy()
miss["cand_name"], miss["cand_addr"] = B.name_raw.to_numpy(), B.addr_raw.to_numpy()
miss["other_name"], miss["other_addr"] = O.name_raw.to_numpy(), O.addr_raw.to_numpy()
miss.to_parquet(OUT / "v12_missed.parquet", index=False)
print(len(miss), miss.stage.value_counts().to_dict())
print(pd.crosstab(miss.cause, miss.country, margins=True).to_string())
