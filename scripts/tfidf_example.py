"""Worked TF-IDF example for ONE query: pieces, tf, idf, weights, cosine with candidates, where true matches rank."""
import numpy as np
import pandas as pd

from ber.blocking.sparse import _tfidf_matrices
from ber.io import read_truth

rec = pd.read_parquet("artifacts/v2/records_train.parquet", columns=["entity_id", "source", "state", "name_raw", "addr_raw", "block_text"])
kl = rec[rec.state == "kerala"].reset_index(drop=True)
truth = read_truth("data/student_resource/dataset")
s1 = kl[kl.source == 1]
pick = next(e for e in s1.entity_id.sample(2000, random_state=3) if len(truth[e]) >= 3)
qi = kl.index[kl.entity_id == pick][0]
others = kl[kl.source != 1]
cfg = {"ngram_range": [3, 5], "min_df": 2, "max_df": 0.05, "n_threads": 32}
A, B = _tfidf_matrices(kl.loc[[qi], "block_text"], others.block_text, cfg)

# exact per-piece numbers via TfidfVectorizer (same maths as the pipeline; verified equal in tests)
from sklearn.feature_extraction.text import CountVectorizer
docs = pd.concat([kl.loc[[qi], "block_text"], others.block_text]).tolist()
N = len(docs)
cv = CountVectorizer(analyzer="char_wb", ngram_range=(3, 5))
X = cv.fit_transform(docs)
vocab = cv.get_feature_names_out()
df_all = np.bincount(X.indices, minlength=X.shape[1])
q = X[0].tocoo()
q_text = kl.at[qi, "block_text"]
print(f"QUERY  {pick}: {kl.at[qi, 'name_raw']} | {kl.at[qi, 'addr_raw']}")
print(f"cleaned text: '{q_text}'   ({q.nnz} distinct letter pieces, N = {N:,} Kerala records)\n")
rows = []
for j, cnt in zip(q.col, q.data):
    dfg = int(df_all[j]); idf = np.log((1 + N) / (1 + dfg)) + 1
    keep = 2 <= dfg <= 0.05 * N
    rows.append((repr(vocab[j]), int(cnt), 1 + np.log(cnt), dfg, idf, (1 + np.log(cnt)) * idf if keep else 0.0,
                 "kept" if keep else ("too common (>5%)" if dfg > 0.05 * N else "too rare (<2)")))
t = pd.DataFrame(rows, columns=["piece", "count", "tf=1+ln(count)", "df", "idf=ln((1+N)/(1+df))+1", "weight", "status"])
t["normalized"] = t.weight / np.sqrt((t.weight ** 2).sum())
pd.set_option("display.width", 200)
print("STEP 1-3: pieces -> tf x idf -> L2-normalise (10 highest-weight pieces, then 5 dropped ones)")
print(t.sort_values("normalized", ascending=False).head(10).round(3).to_string(index=False))
print(t[t.status != "kept"].sort_values("df", ascending=False).head(5).round(3).to_string(index=False))
print(f"kept {int((t.status == 'kept').sum())} of {len(t)} pieces\n")

sims = np.asarray((A @ B.T).todense()).ravel()
order = np.argsort(-sims)
true_ids = truth[pick]
print("STEP 4: cosine(query, record) = sum of products of shared piece weights -> top 8 of", f"{len(others):,}")
for r, j in enumerate(order[:8], 1):
    e = others.iloc[j]
    print(f"  #{r}  sim {sims[j]:.3f}  {'TRUE MATCH' if e.entity_id in true_ids else '          '}  {e.name_raw} | {e.addr_raw[:70]}")
print("\nwhere the TRUE matches rank:")
rank = {others.entity_id.iloc[j]: r for r, j in enumerate(order, 1)}
for e in true_ids:
    rr = others[others.entity_id == e]
    if len(rr):
        print(f"  rank {rank[e]:>6}  sim {sims[rr.index[0] - others.index[0]] if False else sims[others.index.get_loc(rr.index[0])]:.3f}  "
              f"{'IN top-50' if rank[e] <= 50 else 'MISSED by top-50'}  {rr.name_raw.iloc[0]} | {rr.addr_raw.iloc[0][:70]}")
    else:
        print(f"  (true match {e} is outside Kerala)")
j0 = order[0]
bq, bd = A.toarray().ravel(), B[j0].toarray().ravel()
shared = np.flatnonzero((bq > 0) & (bd > 0))
contrib = pd.Series(bq[shared] * bd[shared]).sort_values(ascending=False)
print(f"\nSTEP 4 detail for #1: {len(shared)} shared pieces, sum of q_g*d_g = {contrib.sum():.3f}; top 5 contributions:",
      contrib.head(5).round(4).tolist())
