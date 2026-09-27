"""Same-address, different-name records next to a lone S1: renamed COPY of that S1 (true) or a DIFFERENT business in the
same building (false)? Structural, label-free signal: a different business usually has copies in both S2 and S3, so its
(cleaned) name appears on >= 2 records at that address; a renamed copy's new name appears once.
Train: true share by sibling count (labels). Test: counts per country and v12 acceptance by sibling count.
usage: CUDA_VISIBLE_DEVICES= python -u scripts/sibling_check.py"""
import re

import numpy as np
import pandas as pd
from anyascii import anyascii

from ber.config import load_config, run_dir
from ber.io import read_truth

cfg = load_config("configs/v2.yaml"); RD = run_dir(cfg)
DROP = {"pvt", "private", "ltd", "limited", "llc", "inc", "incorporated", "corp", "corporation", "co", "company", "pc", "llp", "plc",
        "lp", "sarl", "sas", "sasu", "eurl", "sci", "sa", "snc", "selarl", "ei", "cie", "the", "and", "of", "de", "des", "du", "la", "le",
        "les", "et", "d", "l", "shri", "smt", "sri", "m", "s", "dr", "france", "india"}


def key(n):
    toks = [t for t in re.findall(r"[a-z0-9]+", anyascii(n or "").lower()) if t not in DROP]
    return " ".join(sorted(toks))


def pairs(fam):
    r = pd.read_parquet(RD / f"records_{fam}.parquet", columns=["entity_id", "source", "country", "name_raw", "addr_norm"])
    r = r[r.addr_norm.fillna("") != ""].copy()
    r["k"] = [key(n) for n in r.name_raw]
    s1 = r[r.source == 1]
    lone = s1.groupby(["country", "addr_norm"]).entity_id.agg(["first", "size"])
    lone = lone[lone["size"] == 1]["first"].rename("s1_id").reset_index()
    oth = r[r.source != 1]
    sib = oth.groupby(["country", "addr_norm", "k"]).size().rename("n_same_name").reset_index()
    x = oth.merge(lone, on=["country", "addr_norm"]).merge(sib, on=["country", "addr_norm", "k"])
    x = x.merge(s1[["entity_id", "k"]].rename(columns={"entity_id": "s1_id", "k": "s1k"}), on="s1_id")
    x = x[x.k != x.s1k]  # different name from the S1
    x["sibling"] = np.where(x.n_same_name >= 2, "name on >=2 records here", "name on 1 record")
    return x.rename(columns={"entity_id": "cand_id"}), s1.groupby("country").size()


x, n1 = pairs("train")
truth = read_truth(cfg["paths"]["data_dir"])
x["label"] = [c in truth.get(s, ()) for s, c in zip(x.s1_id, x.cand_id)]
g = x.groupby(["country", "sibling"])
print("TRAIN (labels): lone S1, same-address record with a different name")
print(pd.DataFrame({"pairs": g.size(), "per 1k S1": 1000 * g.size() / n1.reindex(g.size().index.get_level_values(0)).to_numpy(),
                    "true share": g.label.mean()}).round(3).to_string(), flush=True)
t, n1t = pairs("testfr")
sub = pd.read_csv("artifacts/submissions/v12/matching_results.tsv", sep="\t", dtype=str, keep_default_na=False)
acc = {(s, c) for s, v in zip(sub.source1_entity_id, sub.matched_entity_ids) for c in v.split(",") if c}
t["v12_accepted"] = [(s, c) in acc for s, c in zip(t.s1_id, t.cand_id)]
g = t.groupby(["country", "sibling"])
print("\nTEST: same pattern, v12 acceptance")
print(pd.DataFrame({"pairs": g.size(), "per 1k S1": 1000 * g.size() / n1t.reindex(g.size().index.get_level_values(0)).to_numpy(),
                    "v12 accepted": g.v12_accepted.mean()}).round(3).to_string(), flush=True)
fr = t[(t.country == "France")]
rn = pd.read_parquet(RD / "records_testfr.parquet", columns=["entity_id", "name_raw", "addr_raw"]).set_index("entity_id")
for sname in ["name on 1 record", "name on >=2 records here"]:
    print(f"\nFrance examples, {sname}, rejected by v12:")
    for r in fr[(fr.sibling == sname) & ~fr.v12_accepted].sample(5, random_state=3).itertuples():
        print(f"   S1 {rn.name_raw[r.s1_id]!r} | {rn.addr_raw[r.s1_id]!r}  <-  {rn.name_raw[r.cand_id]!r} | {rn.addr_raw[r.cand_id]!r}")
t.to_parquet(RD / "v11" / "sibling_test.parquet", index=False)
