"""Recall of every candidate channel alone, all channels combined, and each channel's MARGINAL gain
(recall lost if that channel is removed), per country, on train. Writes docs/results/blocking_channels.md.
Channels: tfidf (top-50), keys, name-only, qwen hard-names (top-10), graph expansion (neighbours of strong candidates)."""
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ber.io import read_truth

T0 = time.perf_counter()
log = lambda *a: print(f"[{time.perf_counter() - T0:6.0f}s]", *a, flush=True)
RD = Path("artifacts/v2")
rec = pd.read_parquet(RD / "records_train.parquet", columns=["entity_id", "source", "country"])
code = pd.Series(np.arange(len(rec), dtype=np.int64), index=rec.entity_id)
country = rec.country.to_numpy()
M = np.int64(len(rec))
key = lambda s, c: code.reindex(s).to_numpy() * M + code.reindex(c).to_numpy()  # one int64 per (S1, candidate)

c = pd.read_parquet(RD / "cand_sparse_train.parquet", columns=["s1_id", "cand_id", "tfidf_rank", "key_hits", "name_sim", "name_key"])
k_all = key(c.s1_id, c.cand_id)
ch = {"tfidf top-50": k_all[(c.tfidf_rank < 999).to_numpy()],
      "key blocks": k_all[(c.key_hits > 0).to_numpy()],
      "name-only": k_all[((c.name_sim > 0) | (c.name_key > 0)).to_numpy()]}
seeds = pd.DataFrame({"s": code.reindex(c.s1_id).to_numpy(), "c": code.reindex(c.cand_id).to_numpy()})[
    ((c.tfidf_rank <= 3) | (c.key_hits >= 2)).to_numpy()]
del c
h = pd.read_parquet(RD / "cand_hardname_train.parquet", columns=["s1_id", "cand_id"])
ch["qwen hard-names"] = key(h.s1_id, h.cand_id)
del h
E = pd.read_parquet(RD / "kg_recrec_edges_train.parquet", columns=["a", "b"])
exp = seeds.merge(E, left_on="c", right_on="a")
ch["graph expansion"] = exp.s.to_numpy() * M + exp.b.to_numpy()
del E, exp, seeds
ch = {k: np.unique(v) for k, v in ch.items()}
log("channels loaded:", {k: f"{len(v):,}" for k, v in ch.items()})

truth = read_truth("data/student_resource/dataset")
tp = pd.DataFrame([(s, x) for s, m in truth.items() for x in m], columns=["s1_id", "cand_id"])
tk = key(tp.s1_id, tp.cand_id)
t_country = country[code.reindex(tp.s1_id).to_numpy()]
n_s1 = pd.Series(country[rec.source.to_numpy() == 1]).value_counts()
hit = {k: np.isin(tk, v) for k, v in ch.items()}
allhit = np.logical_or.reduce(list(hit.values()))
union = np.unique(np.concatenate(list(ch.values())))
u_country = country[(union // M).astype(np.int64)]
log("union pairs:", f"{len(union):,}")

rows = []
for ct in ["ALL", "India", "US"]:
    sel = np.ones_like(allhit) if ct == "ALL" else (t_country == ct)
    ents = n_s1.sum() if ct == "ALL" else n_s1[ct]
    for name, v in ch.items():
        vc = country[(v // M).astype(np.int64)]
        n = len(v) if ct == "ALL" else int((vc == ct).sum())
        others = np.logical_or.reduce([hit[o] for o in ch if o != name])
        rows.append((ct, name, n / ents, hit[name][sel].mean(), (allhit & ~others)[sel].mean()))
    n = len(union) if ct == "ALL" else int((u_country == ct).sum())
    rows.append((ct, "ALL CHANNELS", n / ents, allhit[sel].mean(), np.nan))
df = pd.DataFrame(rows, columns=["country", "channel", "cands_per_S1", "recall_alone", "marginal_gain"])
pd.set_option("display.width", 200)
print(df.to_string(index=False, formatters={"cands_per_S1": "{:.1f}".format, "recall_alone": "{:.4f}".format,
                                            "marginal_gain": lambda x: "" if pd.isna(x) else f"+{x:.4f}"}))
md = ["# Blocking channels: recall alone, combined, and marginal gain (train)", "",
      "Marginal gain = recall lost if that one channel is removed while all others stay.", "",
      "| country | channel | candidates per S1 | recall alone | marginal gain |", "|---|---|---|---|---|"]
for r in df.itertuples():
    md.append(f"| {r.country} | {r.channel} | {r.cands_per_S1:.1f} | {r.recall_alone:.2%} | "
              f"{'' if pd.isna(r.marginal_gain) else f'+{r.marginal_gain:.2%}'} |")
Path("docs/results").mkdir(parents=True, exist_ok=True)
Path("docs/results/blocking_channels.md").write_text("\n".join(md) + "\n")
log("wrote docs/results/blocking_channels.md")
