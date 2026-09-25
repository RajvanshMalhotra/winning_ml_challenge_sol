"""Knowledge-graph candidate expansion (train): add record<->record edges among S2/S3 and expand each S1's candidates
with the graph neighbours of its strongest candidates. Reports block recall / candidates / oracle F0.5 before vs after,
and saves the graph for later use (Model B).  usage: kg_expand.py [config]"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ber.blocking.sparse import _rarest, cand_sparse_path, tfidf_topk
from ber.config import load_config, run_dir
from ber.io import read_truth
from ber.normalize import records_path

T0 = time.perf_counter()
log = lambda *a: print(f"[{time.perf_counter() - T0:6.0f}s]", *a, flush=True)
cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "configs/v2.yaml")
FAMILY = sys.argv[2] if len(sys.argv) > 2 else "train"
RD = run_dir(cfg)
CC = {"ngram_range": [3, 5], "min_df": 2, "max_df": 0.05, "top_k": 10, "min_sim": 0.5,
      "n_threads": cfg["blocking"]["tfidf"]["n_threads"]}
SEED_RANK, SEED_KEYS = 3, 2

rec = pd.read_parquet(records_path(cfg, FAMILY),
                      columns=["entity_id", "source", "country", "state", "name_core", "name_domain", "addr_norm", "block_text"])
ids = rec.entity_id.to_numpy()
code = pd.Series(np.arange(len(rec), dtype=np.int32), index=rec.entity_id)
groups = json.loads((RD / "state_groups.json").read_text())
rec["grp"] = [groups.get(c, {}).get(s, s) for c, s in zip(rec.country, rec.state)]

# ---- existing S1 -> candidate edges (plus name-only caches if the re-run has produced them) ----
cands = pd.read_parquet(cand_sparse_path(cfg, FAMILY), columns=["s1_id", "cand_id", "tfidf_rank", "key_hits"])
extra = [pd.read_parquet(p, columns=["s1_id", "cand_id"]) for p in sorted((RD / f"block_cache_{FAMILY}").glob("nameonly_*.parquet"))]
base = pd.concat([cands[["s1_id", "cand_id"]]] + extra, ignore_index=True)
base = pd.DataFrame({"s": code.reindex(base.s1_id).to_numpy(), "c": code.reindex(base.cand_id).to_numpy()}).drop_duplicates()
seeds = cands[(cands.tfidf_rank <= SEED_RANK) | (cands.key_hits >= SEED_KEYS)]
seeds = pd.DataFrame({"s": code.reindex(seeds.s1_id).to_numpy(), "c": code.reindex(seeds.cand_id).to_numpy()})
log(f"base pairs {len(base):,} (name-only files: {len(extra)}), seed pairs {len(seeds):,}")

# ---- record <-> record edges among S2/S3 ----
edges = []
for country in rec.country.unique():
    pool = rec[(rec.country == country) & (rec.source != 1)]
    known = pool[pool.grp != ""]
    g = known.grp.to_numpy()
    t = tfidf_topk(known, known, CC, g, g, log=f"[{country} rec-rec]")
    t = t[t.s1_id != t.cand_id]
    edges.append(pd.DataFrame({"a": code.reindex(t.s1_id).to_numpy(), "b": code.reindex(t.cand_id).to_numpy(), "kind": np.int8(1)}))
    # records with unknown state (mostly empty address): two-rarest-name-word key against every pool record
    name = pool.name_domain.where(pool.name_domain != "", pool.name_core).reset_index(drop=True)
    toks = _rarest(name.str.split(), 2, 3)
    keyed = toks.groupby("row").tok.agg(lambda x: "|".join(sorted(x))).rename("key").reset_index()
    keyed = keyed[keyed.key.str.contains("|", regex=False)]
    keyed = keyed[keyed.groupby("key").row.transform("size") <= 50]
    unk = (pool.grp.to_numpy() == "")
    pairs = keyed[unk[keyed.row.to_numpy()]].merge(keyed, on="key", suffixes=("_u", "_o"))
    pairs = pairs[pairs.row_u != pairs.row_o]
    pc = code.reindex(pool.entity_id).to_numpy()
    edges.append(pd.DataFrame({"a": pc[pairs.row_u.to_numpy()], "b": pc[pairs.row_o.to_numpy()], "kind": np.int8(2)}))
    log(f"[{country}] rec-rec tfidf edges {len(t):,}, unknown-state name-key edges {len(pairs):,}")
E = pd.concat(edges, ignore_index=True)
E = pd.concat([E, E.rename(columns={"a": "b", "b": "a"})], ignore_index=True).drop_duplicates(["a", "b"])
E.to_parquet(RD / f"kg_recrec_edges_{FAMILY}.parquet", index=False)
log(f"graph: {len(rec):,} nodes, {len(base):,} S1-candidate edges, {len(E) // 2:,} record-record edges (saved)")

if FAMILY != "train":  # no labels: the graph edges are all we need (the matcher expands per S1)
    sys.exit(0)
# ---- expansion: neighbours of each S1's strongest candidates ----
exp = seeds.merge(E[["a", "b"]], left_on="c", right_on="a")[["s", "b"]].rename(columns={"b": "c"}).drop_duplicates()
new = exp.merge(base, how="left", indicator=True)
new = new[new._merge == "left_only"][["s", "c"]]
full = pd.concat([base, new], ignore_index=True)
log(f"expansion added {len(new):,} pairs")

# ---- recall / candidates / oracle before vs after ----
truth = read_truth(cfg["paths"]["data_dir"])
tp = pd.DataFrame([(s, c) for s, m in truth.items() for c in m], columns=["s1_id", "cand_id"])
tp = pd.DataFrame({"s": code.reindex(tp.s1_id).to_numpy(), "c": code.reindex(tp.cand_id).to_numpy()})
s1 = rec[rec.source == 1]
s1c = code.reindex(s1.entity_id).to_numpy()
n_true = pd.Series([len(truth[x]) for x in s1.entity_id], index=s1c)


def report(pairs: pd.DataFrame, label: str) -> None:
    hit = tp.merge(pairs, on=["s", "c"])
    found = hit.groupby("s").size().reindex(s1c, fill_value=0)
    ncand = pairs.groupby("s").size().reindex(s1c, fill_value=0)
    rec_e = np.where(n_true > 0, found / n_true.clip(lower=1), 1.0)
    oracle = np.where(n_true == 0, 1.0, np.where(found == 0, 0.0, 1.25 * rec_e / (0.25 + rec_e)))
    df = pd.DataFrame({"country": s1.country.to_numpy(), "t": n_true.to_numpy(), "f": found.to_numpy(),
                       "n": ncand.to_numpy(), "o": oracle})
    for c, d in [("ALL", df)] + list(df.groupby("country")):
        print(f"  {label:<8} {c:<6} recall {d.f.sum() / d.t.sum():.4f} | all-found {(d.f == d.t).mean():.4f} | "
              f"cands mean {d.n.mean():6.1f} p95 {d.n.quantile(.95):5.0f} | pair quality {d.f.sum() / max(d.n.sum(), 1):.4f} | "
              f"oracle F0.5 {d.o.mean():.4f}", flush=True)


report(base, "before")
report(full, "after")
new_hit = tp.merge(new, on=["s", "c"])
print(f"  expansion: {len(new):,} new pairs, {len(new_hit):,} of them true ({len(new_hit) / max(len(new), 1):.2%})")
