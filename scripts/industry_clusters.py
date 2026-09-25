"""Industry clustering of business names with an LLM embedding model (Qwen3-Embedding-8B, Apache-2.0).
Tests the hypothesis "records of the same business fall in the same industry cluster" and exports
data for docs/results/industry_clusters.html."""
import json
import sys
from collections import Counter

import faiss
import numpy as np
import pandas as pd

from ber.contrastive.encoders import ENCODERS, EncoderSpec, encode, load_encoder
from ber.io import read_truth

# usage: industry_clusters.py [qwen8b|qwen06b] [n_s1] [n_fr]   (qwen06b runs fine on CPU)
MODEL = sys.argv[1] if len(sys.argv) > 1 else "qwen8b"
K, N_S1, N_FR, N_MAP, SEED = 30, int(sys.argv[2]) if len(sys.argv) > 2 else 50_000, \
    int(sys.argv[3]) if len(sys.argv) > 3 else 25_000, 20_000, 0
INSTRUCT = "Instruct: Identify the industry or line of business of this company from its name\nQuery: "
rng = np.random.default_rng(SEED)
cols = ["entity_id", "source", "country", "name_raw", "name_core"]
tr = pd.read_parquet("artifacts/v2/records_train.parquet", columns=cols).set_index("entity_id")
te = pd.read_parquet("artifacts/v2/records_test.parquet", columns=cols)
truth = read_truth("data/student_resource/dataset")

s1_all = [s for s, m in truth.items() if m]
s1 = list(rng.choice(s1_all, size=N_S1, replace=False))
cands = sorted({c for s in s1 for c in truth[s]})
fr = te[(te.source == 1) & (te.country == "France")].sample(N_FR, random_state=SEED).set_index("entity_id")

names = pd.concat([tr.loc[s1 + cands], fr])
names["role"] = ["s1"] * len(s1) + ["cand"] * len(cands) + ["fr"] * len(fr)
print("records to embed:", len(names), flush=True)

spec = ENCODERS["qwen8b"] if MODEL == "qwen8b" else EncoderSpec(
    "qwen06b", "Qwen/Qwen3-Embedding-0.6B", padding_side="left", max_seq_length=48)
model = load_encoder(spec, mem_fraction=0.4)
emb = encode(model, names.name_raw.tolist(), INSTRUCT, 128).astype(np.float32)

# k-means on S1 + France names (the "reference" businesses); candidates are only assigned
ref = np.flatnonzero(names.role.isin(["s1", "fr"]).to_numpy())
km = faiss.Kmeans(emb.shape[1], K, niter=25, spherical=True, seed=SEED, verbose=False)
km.train(emb[ref])
_, lab = km.index.search(emb, 1)
names["cluster"] = lab[:, 0]

# hypothesis test: do true pairs share a cluster?
cl = names.cluster
pairs = pd.DataFrame([(s, c) for s in s1 for c in truth[s]], columns=["s1", "cand"])
pairs["same"] = cl.loc[pairs.s1].to_numpy() == cl.loc[pairs.cand].to_numpy()
pairs["country"] = names.country.loc[pairs.s1].to_numpy()
share = names[names.role == "s1"].cluster.value_counts(normalize=True)
chance = float((share ** 2).sum())

# cluster labels: most distinctive core-name tokens (lift vs. all reference names)
tok = names.iloc[ref].assign(t=names.iloc[ref].name_core.str.split()).explode("t")
tok = tok[tok.t.str.len() >= 3]
glob = tok.t.value_counts()
labels, clusters = {}, []
for k in range(K):
    ck = tok[tok.cluster == k].t.value_counts()
    ck = ck[ck >= 15]
    lift = (ck / ck.sum()) / (glob.reindex(ck.index) / glob.sum())
    labels[k] = " · ".join(lift.sort_values(ascending=False).head(4).index)
    members = names[(names.cluster == k) & names.role.isin(["s1", "fr"])]
    kp = pairs[cl.loc[pairs.s1].to_numpy() == k]
    clusters.append({
        "id": k, "label": labels[k],
        "n": {c: int(v) for c, v in members.country.value_counts().items()},
        "same_rate": float(kp.same.mean()) if len(kp) else None, "n_pairs": int(len(kp)),
        "examples": members.name_raw.sample(min(6, len(members)), random_state=SEED).tolist(),
    })

# 2-D map of a subsample (UMAP, cosine)
import umap
sub = rng.choice(ref, size=min(N_MAP, len(ref)), replace=False)
xy = umap.UMAP(n_neighbors=30, min_dist=0.1, metric="cosine", random_state=SEED).fit_transform(emb[sub])
points = [{"x": round(float(a), 3), "y": round(float(b), 3), "c": int(names.cluster.iloc[i]),
           "country": names.country.iloc[i], "name": names.name_raw.iloc[i]} for (a, b), i in zip(xy, sub)]

# where do mismatched pairs go? examples for inspection
miss = pairs[~pairs.same].sample(min(12, int((~pairs.same).sum())), random_state=SEED)
miss_ex = [{"s1": names.name_raw.loc[a], "s1_cluster": labels[int(cl.loc[a])],
            "cand": names.name_raw.loc[b], "cand_cluster": labels[int(cl.loc[b])]} for a, b in zip(miss.s1, miss.cand)]

out = {"k": K, "model": spec.hf_id, "n_pairs": int(len(pairs)),
       "same_rate": float(pairs.same.mean()), "chance": chance,
       "same_rate_by_country": pairs.groupby("country").same.mean().round(4).to_dict(),
       "clusters": clusters, "points": points, "mismatch_examples": miss_ex}
json.dump(out, open(f"artifacts/v2/industry_clusters_{MODEL}.json", "w"))
print(json.dumps({k: out[k] for k in ["n_pairs", "same_rate", "chance", "same_rate_by_country"]}), flush=True)
