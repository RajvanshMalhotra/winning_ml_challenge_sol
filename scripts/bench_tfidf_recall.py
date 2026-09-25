"""Pair completeness of char TF-IDF top-k for max_df variants: 5k US S1 queries vs their true matches + 1M distractors."""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

from ber.io import read_truth

rec = pd.read_parquet("artifacts/v1/records_train.parquet", columns=["entity_id", "source", "country", "block_text"])
truth = read_truth("data/student_resource/dataset")
us = rec[rec.country == "US"]
s1 = us[(us.source == 1) & us.entity_id.map(lambda s: len(truth.get(s, ())) > 0)]
q = s1.sample(5000, random_state=0)
matches = set().union(*(truth[s] for s in q.entity_id))
oth = us[us.source != 1]
docs = pd.concat([oth[oth.entity_id.isin(matches)], oth[~oth.entity_id.isin(matches)].sample(1_000_000, random_state=0)])
fit_text = pd.concat([us.block_text.sample(2_000_000, random_state=1)])
for max_df in (1.0, 0.05, 0.01):
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_df=max_df, sublinear_tf=True, dtype=np.float32).fit(fit_text)
    c = sp_matmul_topn(vec.transform(q.block_text), vec.transform(docs.block_text).T.tocsr(), top_n=50, threshold=0.1, sort=True, n_threads=32).tocoo()
    got = set(zip(q.entity_id.values[c.row], docs.entity_id.values[c.col]))
    tp = [(s, m) for s in q.entity_id for m in truth[s]]
    print(f"max_df={max_df}: pair completeness@50 = {np.mean([p in got for p in tp]):.4f} ({len(tp)} true pairs)", flush=True)
