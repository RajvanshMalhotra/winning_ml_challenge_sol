"""Why were true pairs missed by blocking? Classifies missed pairs per country and prints examples."""
import json
import sys

import numpy as np
import pandas as pd

from ber.blocking.sparse import cand_sparse_path
from ber.config import load_config, run_dir
from ber.io import read_truth
from ber.normalize import records_path

cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "configs/v2.yaml")
cands = pd.read_parquet(cand_sparse_path(cfg, "train"), columns=["s1_id", "cand_id"])
truth = read_truth(cfg["paths"]["data_dir"])
rec = pd.read_parquet(records_path(cfg, "train"),
                      columns=["entity_id", "source", "country", "state", "name_raw", "addr_raw", "name_core", "addr_norm", "house_no"]).set_index("entity_id")
groups = json.loads((run_dir(cfg) / "state_groups.json").read_text())

tp = pd.DataFrame([(s, c) for s, m in truth.items() for c in m], columns=["s1_id", "cand_id"])
tp = tp.merge(cands.assign(hit=True), on=["s1_id", "cand_id"], how="left")
tp["hit"] = tp.hit.fillna(False).astype(bool)
tp["country"] = rec.country.reindex(tp.s1_id).to_numpy()
g = lambda st, c: groups.get(c, {}).get(st, st)
s1_state, c_state = rec.state.reindex(tp.s1_id).to_numpy(), rec.state.reindex(tp.cand_id).to_numpy()
tp["state_blocked"] = [a != "" and b != "" and g(a, c) != g(b, c) for a, b, c in zip(s1_state, c_state, tp.country)]
tp["cand_addr_empty"] = rec.addr_raw.reindex(tp.cand_id).to_numpy() == ""
sn, cn = rec.name_core.reindex(tp.s1_id).to_numpy(), rec.name_core.reindex(tp.cand_id).to_numpy()
tp["name_overlap"] = [len(set(a.split()) & set(b.split())) / max(len(set(a.split()) | set(b.split())), 1) for a, b in zip(sn, cn)]
sa, ca = rec.addr_norm.reindex(tp.s1_id).to_numpy(), rec.addr_norm.reindex(tp.cand_id).to_numpy()
tp["addr_overlap"] = [len(set(a.split()) & set(b.split())) / max(len(set(a.split()) | set(b.split())), 1) for a, b in zip(sa, ca)]
tp["cand_source"] = tp.cand_id.str[:2]

for c, d in tp.groupby("country"):
    m = d[~d.hit]
    print(f"\n==== {c}: {len(m):,} missed of {len(d):,} true pairs ({len(m)/len(d):.2%})")
    print(" missed because state group differs :", f"{m.state_blocked.mean():.1%}")
    print(" candidate address empty             :", f"{m.cand_addr_empty.mean():.1%}")
    print(" name token overlap = 0              :", f"{(m.name_overlap == 0).mean():.1%}", "| found pairs:", f"{(d[d.hit].name_overlap == 0).mean():.1%}")
    print(" address token overlap < 0.2         :", f"{(m.addr_overlap < 0.2).mean():.1%}", "| found pairs:", f"{(d[d.hit].addr_overlap < 0.2).mean():.1%}")
    print(" missed by source                    :", m.cand_source.value_counts(normalize=True).round(3).to_dict())
    print(" miss rate by source                 :", (~d.hit).groupby(d.cand_source).mean().round(4).to_dict())
    ex = m.sample(min(10, len(m)), random_state=0)
    for s, k in zip(ex.s1_id, ex.cand_id):
        print("  S1 :", rec.name_raw[s], "|", rec.addr_raw[s][:110], "| state:", rec.state[s] or "-")
        print("  -> :", rec.name_raw[k], "|", rec.addr_raw[k][:110], "| state:", rec.state[k] or "-")
