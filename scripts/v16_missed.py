"""What does the final model (v16 stacker; France = v16frmixacr) still miss?
Part 1 (labels, US/India OOF): rebuild v16's OOF scores exactly as v16.py final() does (3 hidden-owner frames, each
scored by its own fold models, averaged), list every missed true pair (+ blocking misses) with its cause, compare with v12.
Part 2 (no labels, France test): what France still leaves unassigned after the French Qwen blend + acronym routing.
usage (repo root on the HPC): CUDA_VISIBLE_DEVICES= python -u scripts/v16_missed.py"""
import json
import re
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein

sys.argv = ["v16.py", "none"]
H = {"__name__": "v16m"}
exec(open("scripts/v16.py").read().split("\ndef final()")[0], H)
RD, G, log = H["RD"], H["G"], H["log"]
from ber.io import read_truth

OUT = RD / "v16_missed"
OUT.mkdir(exist_ok=True)
ch = json.loads((RD / "v16" / "choice.json").read_text())
feats = H["BASE"] + H["NEW"] + H["V9"] + (H["EXTRA"] if ch["extra"] else [])
tau = ch["tau"]
inp = H["oof_inputs"]()
preds, ref = [], None
for sd in H["SEEDS"]:
    f = H["add_rr"](pd.read_parquet(RD / sd / "frame_hidden.parquet"), *inp)
    if ref is None:
        ref = f[["s1_id", "cand_id", "label", "fold", "p", "rr", "cand_noaddr", "s1_name_cnt", "a_crank", "a_share", "a_ncomp"]].copy()
    assert (f.cand_id.to_numpy() == ref.cand_id.to_numpy()).all()
    p = np.zeros(len(f), np.float32)
    for k in range(5):
        m = lgb.Booster(model_file=str(RD / "v16" / f"{sd}_fold{k}.txt"))
        va = f.fold.to_numpy() == k
        p[va] = m.predict(f.loc[va, feats], num_iteration=m.best_iteration)
    preds.append(p)
    log(f"{sd} scored")
f = ref
f["q16"] = np.mean(preds, axis=0)
truth = read_truth(G["cfg"]["paths"]["data_dir"])
s1 = pd.Index(f.s1_id.unique()); nt = np.array([len(truth[s]) for s in s1]); code = s1.get_indexer(f.s1_id)
lab = f.label.to_numpy().astype(bool)
own = f[f.q16 >= tau]; own = own.loc[own.groupby("cand_id").q16.idxmax()]
acc = np.zeros(len(f), bool); acc[own.index.to_numpy()] = True
tp = np.bincount(code, weights=acc & lab, minlength=len(s1)); npred = np.bincount(code, weights=acc, minlength=len(s1))
sc = np.where(nt == 0, (npred == 0) * 1.0, 1.25 * tp / np.maximum(0.25 * nt + npred, 1e-9))
log(f"v16 OOF (test-like frames, tau {tau}): {sc.mean():.5f} (choice.json: {ch['oof'][str(tau)]:.5f}); "
    f"missed {int((lab & ~acc).sum()):,}, wrong accepts {int((~lab & acc).sum()):,}")
q12 = pd.read_parquet(RD / "v12" / "oof_q12.parquet", columns=["s1_id", "cand_id", "q12"])
f = f.merge(q12, on=["s1_id", "cand_id"], how="left")
o12 = f[f.q12 >= 0.75]; o12 = o12.loc[o12.groupby("cand_id").q12.idxmax()]
acc12 = np.zeros(len(f), bool); acc12[o12.index.to_numpy()] = True
print(f"vs v12: misses fixed {int((lab & ~acc12 & acc).sum()):,}, newly missed {int((lab & acc12 & ~acc).sum()):,}; "
      f"wrong accepts removed {int((~lab & acc12 & ~acc).sum()):,}, new {int((~lab & ~acc12 & acc).sum()):,}", flush=True)
# ---- causes of the remaining misses
rec = pd.read_parquet(RD / "records_train.parquet", columns=["entity_id", "country", "name_raw", "addr_raw", "name_core", "house_no"]).set_index("entity_id")
m = f[lab & ~acc].copy(); m["stage"] = "rejected"
short = set(zip(f.s1_id, f.cand_id))
blk = pd.DataFrame([(s, c) for s in s1 for c in truth[s] if (s, c) not in short], columns=["s1_id", "cand_id"]); blk["stage"] = "blocking"
m = pd.concat([m, blk], ignore_index=True)
A, B = rec.reindex(m.s1_id), rec.reindex(m.cand_id)
m["country"] = A.country.to_numpy()
noaddr = B.addr_raw.fillna("").str.strip().eq("").to_numpy()
native = B.name_raw.fillna("").str.contains("[ऀ-෿]", regex=True).to_numpy()
noword = np.array([not (set(x.split()) & set(y.split())) for x, y in zip(A.name_core.fillna(""), B.name_core.fillna(""))])
ha, hb = A.house_no.fillna("").to_numpy(), B.house_no.fillna("").to_numpy()
typo = np.array([x != "" and y != "" and x != y and Levenshtein.distance(x, y) == 1 for x, y in zip(ha, hb)])
acr = B.name_raw.fillna("").str.replace(".", "", regex=False).str.strip().str.match(r"^[A-Z]{2,6}$").to_numpy()
same = m.s1_name_cnt.fillna(0).to_numpy() >= 2
m["cause"] = np.select([m.stage.to_numpy() == "blocking", noaddr & same, noaddr, acr, native, noword, typo],
                       ["never a candidate (blocking)", "no address, 2+ S1s share the name", "no address, name unique", "acronym record",
                        "Indian script name", "names share no word", "house no. 1-digit typo"], "other / similar text")
m["zone"] = pd.cut(m.p, [-1, 0.01, 0.99, 2], labels=["Model A < 0.01 (no reranker)", "reranked band", "Model A > 0.99"]).astype(str)
m.loc[m.stage == "blocking", "zone"] = "never a candidate"
m["q16_bin"] = pd.cut(m.q16, [-0.01, 0.05, 0.2, 0.5, 0.75], labels=["< 0.05", "0.05-0.2", "0.2-0.5", "0.5-0.75 (near miss)"]).astype(str)
m.loc[m.stage == "blocking", "q16_bin"] = "n/a"
print("\n## v16 misses by cause and country (US/India OOF, test-like)")
print(pd.crosstab(m.cause, m.country, margins=True).to_string())
print("\n## by cause x final score")
print(pd.crosstab(m.cause, m.q16_bin, margins=True).to_string())
print("\n## by cause x Model A zone")
print(pd.crosstab(m.cause, m.zone, margins=True).to_string(), flush=True)
m.to_parquet(OUT / "v16_missed_oof.parquet", index=False)

# ---- Part 2: France after v16frmixacr (no labels)
rt = pd.read_parquet(RD / "records_test.parquet", columns=["entity_id", "source", "country", "name_raw", "addr_raw"]).set_index("entity_id")
sub = pd.read_csv("submissions/v16frmixacr/matching_results.tsv", sep="\t", dtype=str, keep_default_na=False)
got = {c for v in sub.matched_entity_ids for c in v.split(",") if c}
oth = rt[rt.source != 1].copy()
oth["assigned"] = oth.index.isin(got)
oth["noaddr"] = oth.addr_raw.fillna("").str.strip().eq("")
oth["acronym"] = oth.name_raw.fillna("").str.replace(".", "", regex=False).str.strip().str.match(r"^[A-Z]{2,6}$")
n1 = rt[rt.source == 1].groupby("country").size()
g = oth.groupby("country")
print("\n## Test S2/S3 records after the final submission (no labels)")
print(pd.DataFrame({"records per S1": g.size() / n1, "assigned share": g.assigned.mean(),
                    "assigned | has address": g.apply(lambda d: d.assigned[~d.noaddr].mean()),
                    "assigned | no address": g.apply(lambda d: d.assigned[d.noaddr].mean()),
                    "acronym records per 1k S1": 1000 * g.acronym.sum() / n1,
                    "acronym assigned": g.apply(lambda d: d.assigned[d.acronym].mean()),
                    "matches per S1": g.assigned.sum() / n1}).round(3).to_string(), flush=True)
