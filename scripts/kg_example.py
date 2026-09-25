"""Pick one India S1 whose missed true match is reachable in the knowledge graph through one of its strong
candidates; dump its neighbourhood as JSON for docs/results/kg_example.html."""
import json

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from ber.io import read_truth

RD = "artifacts/v2"
rec = pd.read_parquet(f"{RD}/records_train.parquet", columns=["entity_id", "source", "country", "name_raw", "addr_raw"])
ids = rec.entity_id.to_numpy()
rec = rec.set_index("entity_id")
truth = read_truth("data/student_resource/dataset")
miss = pd.read_parquet(f"{RD}/eda/missed_India.parquet", columns=["s1_id", "cand_id", "reason"])
E = pd.read_parquet(f"{RD}/kg_recrec_edges_train.parquet")  # int codes a,b,kind (symmetric)
code = pd.Series(np.arange(len(ids)), index=ids)

pick = None
for s1 in miss.s1_id.drop_duplicates().sample(3000, random_state=7):
    cands = ds.dataset(f"{RD}/cand_sparse_train.parquet").to_table(filter=ds.field("s1_id") == s1).to_pandas()
    strong = cands[(cands.tfidf_rank <= 3) | (cands.key_hits >= 2)]
    missed = set(miss.cand_id[miss.s1_id == s1])
    sc = code.reindex(strong.cand_id).to_numpy()
    nb = E[E.a.isin(sc)]
    reach = nb[nb.b.isin(code.reindex(list(missed)).to_numpy())]
    if len(reach) and len(truth[s1]) >= 3:
        pick = (s1, cands, strong, reach)
        break
s1, cands, strong, reach = pick
true = truth[s1]
top = cands.sort_values("tfidf_rank").head(8)
nodes, edges = {}, []
def node(eid, role):
    r = rec.loc[eid]
    nodes[eid] = {"id": eid, "role": role, "true": eid in true, "name": r.name_raw, "addr": r.addr_raw[:80]}
node(s1, "query")
for r in top.itertuples():
    node(r.cand_id, "candidate")
    edges.append({"a": s1, "b": r.cand_id, "kind": "candidate", "w": round(float(r.tfidf_sim), 3), "rank": int(r.tfidf_rank)})
for r in reach.itertuples():
    a, b = ids[r.a], ids[r.b]
    node(a, "candidate") if a not in nodes else None
    node(b, "reached via graph")
    edges.append({"a": a, "b": b, "kind": "record-record", "w": None, "rank": None})
for e in true:  # show every true match, found or not
    if e not in nodes:
        node(e, "missed")
out = {"s1": s1, "n_true": len(true), "n_cands": len(cands), "nodes": list(nodes.values()), "edges": edges}
json.dump(out, open(f"{RD}/kg_example.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1)[:4000])
