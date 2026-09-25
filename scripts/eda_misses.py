"""EDA of true pairs still missed by blocking (default: India, train). Writes every missed pair with diagnostics to
artifacts/v2/eda/missed_<country>.parquet (+ a CSV sample) and prints summary tables.
usage: eda_misses.py [country] [sample_csv_rows]"""
import json
import re
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from ber.blocking.sparse import cand_sparse_path
from ber.config import load_config, run_dir
from ber.io import read_truth
from ber.normalize import records_path

COUNTRY = sys.argv[1] if len(sys.argv) > 1 else "India"
N_CSV = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
cfg = load_config("configs/v2.yaml")
RD = run_dir(cfg)
out_dir = RD / "eda"
out_dir.mkdir(exist_ok=True)

cols = ["entity_id", "source", "country", "state", "name_raw", "addr_raw", "name_core", "name_domain", "addr_norm",
        "house_no", "legal_suffix", "name_parts", "landmark", "unit", "block_text"]
rec = pd.read_parquet(records_path(cfg, "train"), columns=cols)
rec = rec[rec.country == COUNTRY].set_index("entity_id")
cands = pd.read_parquet(cand_sparse_path(cfg, "train"), columns=["s1_id", "cand_id", "tfidf_sim", "tfidf_rank"])
cands = cands[cands.s1_id.isin(rec.index)]
groups = json.loads((RD / "state_groups.json").read_text()).get(COUNTRY, {})
truth = read_truth(cfg["paths"]["data_dir"])

tp = pd.DataFrame([(s, c) for s, m in truth.items() if s in rec.index for c in m], columns=["s1_id", "cand_id"])
tp = tp.merge(cands[["s1_id", "cand_id"]].assign(hit=True), how="left")
miss = tp[tp.hit.isna()][["s1_id", "cand_id"]].reset_index(drop=True)
print(f"{COUNTRY}: {len(tp):,} true pairs, {len(miss):,} missed ({len(miss) / len(tp):.2%}) by current blocking")

# the S1's candidate list: how many, best score, cut-off (score of its last / 50th TF-IDF candidate)
tf = cands[cands.tfidf_rank < 999].groupby("s1_id").tfidf_sim
s1_stats = pd.DataFrame({"s1_n_cands": cands.groupby("s1_id").size(), "s1_best_sim": tf.max(), "s1_cutoff_sim": tf.min()})
miss = miss.join(s1_stats, on="s1_id")

a, b = rec.reindex(miss.s1_id), rec.reindex(miss.cand_id)
for side, d in (("s1", a), ("c", b)):
    for c in ["name_raw", "addr_raw", "state", "house_no", "legal_suffix", "name_core", "addr_norm", "name_parts", "landmark", "unit"]:
        miss[f"{side}_{c}"] = d[c].to_numpy()
miss["c_source"] = b.source.map({2: "S2", 3: "S3"}).to_numpy()
miss["s1_n_true"] = miss.s1_id.map(lambda s: len(truth[s]))

# pair similarity with the SAME kind of char TF-IDF as blocking (fit on this country's text)
vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_df=0.05, sublinear_tf=True, dtype=np.float32)
vec.fit(rec.block_text.sample(min(len(rec), 2_000_000), random_state=0))
va, vb = vec.transform(a.block_text.to_numpy()), vec.transform(b.block_text.to_numpy())
miss["pair_tfidf_sim"] = np.asarray(va.multiply(vb).sum(axis=1)).ravel()

def jacc(x, y):
    x, y = set(x.split()), set(y.split())
    return len(x & y) / len(x | y) if x | y else 0.0

def grams(s, n=3):
    s = f" {s} "
    return {s[i:i + n] for i in range(len(s) - n + 1)} if len(s) >= n else set()

miss["name_word_jacc"] = [jacc(x, y) for x, y in zip(miss.s1_name_core, miss.c_name_core)]
miss["name_3gram_jacc"] = [len(grams(x) & grams(y)) / max(len(grams(x) | grams(y)), 1) for x, y in zip(miss.s1_name_core, miss.c_name_core)]
miss["addr_word_jacc"] = [jacc(x, y) for x, y in zip(miss.s1_addr_norm, miss.c_addr_norm)]
miss["c_native_script"] = miss.c_name_raw.str.contains(r"[ऀ-෿]", regex=True)
miss["c_addr_native"] = miss.c_addr_raw.str.contains(r"[ऀ-෿]", regex=True)
miss["c_addr_empty"] = miss.c_addr_raw == ""
miss["c_domain_style"] = miss.c_name_raw.str.contains(r"\.(com|in|net|org)\b|^#", regex=True)
g = lambda s: groups.get(s, s)
miss["state_group_diff"] = [x != "" and y != "" and g(x) != g(y) for x, y in zip(miss.s1_state, miss.c_state)]
miss["house_equal"] = np.where((miss.s1_house_no != "") & (miss.c_house_no != ""),
                               np.where(miss.s1_house_no == miss.c_house_no, "equal", "different"), "missing")
miss["c_truncated"] = [0 < len(set(y.split())) < len(set(x.split())) and set(y.split()) <= set(x.split())
                       for x, y in zip(miss.s1_name_core, miss.c_name_core)]
miss["below_cutoff"] = miss.pair_tfidf_sim < miss.s1_cutoff_sim.fillna(0)

def reason(r):
    if r.c_native_script:
        return "native-script name"
    if r.state_group_diff:
        return "different state"
    if r.c_addr_empty:
        return "empty address"
    if r.c_domain_style:
        return "website-style name"
    if r.c_truncated:
        return "truncated name"
    if r.name_word_jacc == 0 and r.name_3gram_jacc < 0.2:
        return "name very different"
    if r.addr_word_jacc < 0.2:
        return "address very different"
    return "similar but ranked below top-50"
miss["reason"] = miss.apply(reason, axis=1)

miss.to_parquet(out_dir / f"missed_{COUNTRY}.parquet", index=False)
show = ["reason", "c_source", "s1_name_raw", "c_name_raw", "s1_addr_raw", "c_addr_raw", "s1_state", "c_state",
        "s1_house_no", "c_house_no", "pair_tfidf_sim", "s1_cutoff_sim", "s1_best_sim", "s1_n_cands", "s1_n_true",
        "name_word_jacc", "name_3gram_jacc", "addr_word_jacc", "house_equal", "c_native_script", "c_addr_native",
        "c_addr_empty", "c_domain_style", "c_truncated", "state_group_diff", "s1_id", "cand_id"]
miss.sample(min(N_CSV, len(miss)), random_state=0)[show].to_csv(out_dir / f"missed_{COUNTRY}_sample.csv", index=False)

pd.set_option("display.width", 220, "display.max_colwidth", 60)
print("\n== reason (first matching rule)"); print(miss.reason.value_counts().to_frame("pairs").assign(share=lambda d: (d.pairs / len(miss)).round(3)).to_string())
print("\n== flags (not exclusive)")
for f in ["c_native_script", "c_addr_native", "c_addr_empty", "c_domain_style", "c_truncated", "state_group_diff", "below_cutoff"]:
    print(f"  {f:<18} {miss[f].mean():.1%}")
print("  house number:", miss.house_equal.value_counts(normalize=True).round(3).to_dict())
print("  candidate source:", miss.c_source.value_counts(normalize=True).round(3).to_dict())
print("\n== candidate value: pair TF-IDF similarity vs the S1's cut-off (50th candidate) and best candidate")
print(miss[["pair_tfidf_sim", "s1_cutoff_sim", "s1_best_sim", "s1_n_cands", "name_word_jacc", "name_3gram_jacc", "addr_word_jacc"]]
      .describe(percentiles=[.1, .25, .5, .75, .9]).round(3).to_string())
print("  gap to cut-off (cutoff - pair sim) quantiles:",
      (miss.s1_cutoff_sim - miss.pair_tfidf_sim).quantile([.1, .25, .5, .75, .9]).round(3).to_dict())
print("\n== by S1 state (top 12, missed pairs and miss rate)")
tp_state = rec.state.reindex(tp.s1_id).to_numpy()
rate = pd.Series(tp.hit.isna().to_numpy()).groupby(tp_state).mean()
by = miss.s1_state.value_counts().head(12).to_frame("missed").assign(miss_rate=lambda d: rate.reindex(d.index).round(3))
print(by.to_string())
print("\n== by number of true matches of the S1:", miss.groupby(pd.cut(miss.s1_n_true, [0, 1, 3, 5, 99])).size().to_dict())
print("\n== examples per reason")
for rsn, d in miss.groupby("reason"):
    print(f"-- {rsn}")
    for r in d.sample(min(3, len(d)), random_state=1).itertuples():
        print(f"   S1 : {r.s1_name_raw} | {r.s1_addr_raw[:90]}")
        print(f"   {r.c_source} : {r.c_name_raw} | {r.c_addr_raw[:90]}   (sim {r.pair_tfidf_sim:.2f} vs cut-off {r.s1_cutoff_sim:.2f})")
print(f"\nsaved: {out_dir}/missed_{COUNTRY}.parquet ({len(miss):,} rows) and missed_{COUNTRY}_sample.csv")
