"""Benchmark char TF-IDF top-k on a slice of US S1 against a slice of US S2/S3, with and without max_df."""
import sys
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

rec = pd.read_parquet("artifacts/v1/records_train.parquet", columns=["entity_id", "source", "country", "block_text"])
us = rec[rec.country == "US"]
s1, oth = us[us.source == 1], us[us.source != 1]
rng = np.random.default_rng(0)
q = s1.sample(5000, random_state=0)
docs = oth.sample(1_000_000, random_state=0)  # 1M of 6.2M candidate docs
n_threads = int(sys.argv[1]) if len(sys.argv) > 1 else 16
for max_df in (1.0, 0.05, 0.01):
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_df=max_df, sublinear_tf=True, dtype=np.float32)
    t0 = time.perf_counter(); vec.fit(pd.concat([q.block_text, docs.block_text])); fit = time.perf_counter() - t0
    a, b = vec.transform(q.block_text), vec.transform(docs.block_text)
    t0 = time.perf_counter()
    c = sp_matmul_topn(a, b.T.tocsr(), top_n=50, threshold=0.1, sort=True, n_threads=n_threads)
    mm = time.perf_counter() - t0
    print(f"max_df={max_df}: vocab={len(vec.vocabulary_)} nnz/row={a.nnz/a.shape[0]:.0f} fit={fit:.0f}s "
          f"matmul 5k x 1M = {mm:.1f}s -> full US est {mm * (len(s1)/5000) * (len(oth)/1e6) / 3600:.1f} h @ {n_threads} thr", flush=True)
