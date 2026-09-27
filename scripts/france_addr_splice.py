"""v12 + French address rule (other session's france_addr_pairs, ~/claude_v11): add France pairs with the same name,
house number and région whose street is written differently, when the record is not already given to another S1.
US/India rows are untouched. Candidates = v12's filtered set + the rule's pairs (a French-address blocking key).
usage: france_addr_splice.py <base_sub_dir> <out_dir>"""
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

src, out = Path(sys.argv[1]), Path(sys.argv[2])
sys.argv = ["x", "noop"]
G = {"__name__": "x"}
exec(open(os.path.expanduser("~/claude_v11/scripts/stacker_v11_submit.py")).read().rsplit('\nif __name__ == "__main__":', 1)[0], G)
fa = G["france_addr_pairs"]("testfr")
out.mkdir(parents=True, exist_ok=True)
m = pd.read_csv(src / "matching_results.tsv", sep="\t", dtype=str, keep_default_na=False)
c = pd.read_csv(src / "candidate_pairs.tsv", sep="\t", dtype=str, keep_default_na=False)
pred = {s: set(v.split(",")) - {""} for s, v in zip(m.source1_entity_id, m.matched_entity_ids)}
cand = {s: set(v.split(",")) - {""} for s, v in zip(c.source1_entity_id, c.candidate_entity_ids)}
taken = {x for v in pred.values() for x in v}
add = fa[~fa.cand_id.isin(taken)].drop_duplicates("cand_id")
newc = 0
for s, x in zip(add.s1_id, add.cand_id):
    pred[s].add(x)
    if x not in cand[s]:
        cand[s].add(x); newc += 1
print(f"rule pairs {len(fa):,}; already accepted {int(fa.cand_id.isin(taken).sum()):,}; added {len(add):,} to "
      f"{add.s1_id.nunique():,} France S1s; {newc:,} of them new candidates", flush=True)
pd.DataFrame({"source1_entity_id": m.source1_entity_id,
              "matched_entity_ids": [",".join(sorted(pred[s])) for s in m.source1_entity_id]}).to_csv(out / "matching_results.tsv", sep="\t", index=False)
pd.DataFrame({"source1_entity_id": c.source1_entity_id,
              "candidate_entity_ids": [",".join(sorted(cand[s])) for s in c.source1_entity_id]}).to_csv(out / "candidate_pairs.tsv", sep="\t", index=False)
n = sum(len(v) for v in cand.values())
print(f"candidates {n:,} ({n / len(cand):.2f}/S1)", flush=True)
v = subprocess.run(["python3", "data/student_resource/utils/validate_submission.py", "--matching", str(out / "matching_results.tsv"),
                    "--candidate", str(out / "candidate_pairs.tsv"), "--test-dir", "data/student_resource/dataset/test"],
                   capture_output=True, text=True)
print(v.stdout[-200:], f"validator exit={v.returncode}", flush=True)
