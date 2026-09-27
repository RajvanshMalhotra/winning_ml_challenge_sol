"""Remove France accepts of the pattern: same core name + same house number + same région, but DISJOINT distinctive street
words (street/city stop-words and common city/région words dropped). In US/India train this pattern is rare and
68-96% true; in France test it is 5-10x more frequent per S1 -> mostly decoys. US/India rows untouched.
usage: france_p2_remove.py <base_sub_dir> <out_dir>"""
import subprocess
import sys
from pathlib import Path

import pandas as pd

RD = "artifacts/v2"
STOP = set("""rue avenue boulevard allee place chemin route impasse cours quai square promenade porte residence faubourg lotissement cite
de du des la le les l d et saint sainte bis ter street road drive lane court circle avenue way place parkway highway boulevard trail terrace
unit suite apartment floor n no nagar marg road cross main layout colony sector block phase near opp behind""".split())
src, out = Path(sys.argv[1]), Path(sys.argv[2])
r = pd.read_parquet(f"{RD}/records_testfr.parquet", columns=["entity_id", "source", "country", "addr_norm", "name_core", "house_no", "state"]).set_index("entity_id")
s1 = r[(r.source == 1) & (r.country == "France")]
w = s1.addr_norm.str.split().explode().value_counts()
drop = set(w.index[w > 0.002 * len(s1)])
distinct = lambda a: frozenset(t for t in a.split() if not t[0].isdigit() and t not in STOP and t not in drop and len(t) > 1)
m = pd.read_csv(src / "matching_results.tsv", sep="\t", dtype=str, keep_default_na=False)
pairs = pd.DataFrame([(s, c) for s, v in zip(m.source1_entity_id, m.matched_entity_ids) if v for c in v.split(",")], columns=["s", "c"])
a, b = r.reindex(pairs.s), r.reindex(pairs.c)
sel = (a.country.to_numpy() == "France") & (a.name_core.to_numpy() == b.name_core.to_numpy()) & (b.addr_norm.fillna("").to_numpy() != "") \
      & (a.house_no.to_numpy() == b.house_no.to_numpy()) & (a.house_no.fillna("").to_numpy() != "") & (a.state.to_numpy() == b.state.to_numpy()) \
      & (a.addr_norm.to_numpy() != b.addr_norm.to_numpy())
cand = pairs[sel].copy()
da = [distinct(x) for x in a.addr_norm.to_numpy()[sel]]
db = [distinct(x) for x in b.addr_norm.to_numpy()[sel]]
cand["disjoint"] = [bool(x) and bool(y) and not (x & y) for x, y in zip(da, db)]
rm = cand[cand.disjoint]
print(f"France accepted pairs with same name+house+région, different address: {len(cand):,}; disjoint street words: {len(rm):,} "
      f"(on {rm.s.nunique():,} France S1s)", flush=True)
drop_set = set(zip(rm.s, rm.c))
pred = {}
for s, v in zip(m.source1_entity_id, m.matched_entity_ids):
    pred[s] = [c for c in (v.split(",") if v else []) if (s, c) not in drop_set]
out.mkdir(parents=True, exist_ok=True)
pd.DataFrame({"source1_entity_id": m.source1_entity_id, "matched_entity_ids": [",".join(pred[s]) for s in m.source1_entity_id]}) \
  .to_csv(out / "matching_results.tsv", sep="\t", index=False)
subprocess.run(["cp", str(src / "candidate_pairs.tsv"), str(out / "candidate_pairs.tsv")], check=True)
for x in rm.sample(min(8, len(rm)), random_state=0).itertuples():
    print(f"  {r.name_core[x.s]!r} | S1: {r.addr_norm[x.s]!r}  <->  cand: {r.addr_norm[x.c]!r}", flush=True)
v = subprocess.run(["python3", "data/student_resource/utils/validate_submission.py", "--matching", str(out / "matching_results.tsv"),
                    "--candidate", str(out / "candidate_pairs.tsv"), "--test-dir", "data/student_resource/dataset/test"],
                   capture_output=True, text=True)
print(v.stdout[-120:], f"validator exit={v.returncode}", flush=True)
