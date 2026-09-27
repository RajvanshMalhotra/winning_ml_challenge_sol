"""Label-free check: are the neural similarities (bge-m3 dense / emb_cos, hard-name) and Model A's p lower for France on
OBVIOUS pairs (identical name core + identical address) than for US/India? Model A features for 20k test S1 per country.
usage: CUDA_VISIBLE_DEVICES= python -u scripts/diag_neural.py [family]  -> artifacts/v2/v11/diag_neural_<family>.parquet"""
import json, sys
import lightgbm as lgb, numpy as np, pandas as pd, pyarrow.dataset as ds
from ber.blocking.sparse import cand_sparse_path
from ber.config import load_config, run_dir
from ber.features import REC_COLS, name_idf, pair_features
from ber.matcher import add_extra_channels, candidate_context
from ber.predict import record_embeddings
fam = sys.argv[1] if len(sys.argv) > 1 else "testfr"
cfg = load_config("configs/v2.yaml"); RD = run_dir(cfg); MD = RD / "matcher_v3"
models = [lgb.Booster(model_file=str(MD / f"lgbm_fold{k}.txt")) for k in range(5)]
fn = models[0].feature_name(); best = json.loads((MD / "best.json").read_text())
emb, code = record_embeddings(cfg, fam, best.get("embed_model_key", "bgem3"), best["embed_model_path"])
rec = pd.read_parquet(RD / f"records_{fam}.parquet", columns=REC_COLS).set_index("entity_id")
idf = name_idf(rec.reset_index()); ctx = candidate_context(cand_sparse_path(cfg, fam))
s1 = rec[rec.source == 1]
pick = pd.concat([g.sample(20_000, random_state=0) for _, g in s1.groupby("country")]).index.tolist()
pairs = ds.dataset(cand_sparse_path(cfg, fam)).to_table(filter=ds.field("s1_id").isin(pick)).to_pandas()
pairs = add_extra_channels(cfg, fam, pairs, set(pick), best.get("extra_channels", []))
pairs = pairs.join(ctx, on="cand_id"); pairs["is_cand_best"] = (pairs.tfidf_sim >= pairs.cand_best_sim - 1e-6).astype(np.int8)
f = pair_features(pairs, rec, workers=cfg["n_jobs"], idf=idf)
a, b = code.reindex(pairs.s1_id).to_numpy(), code.reindex(pairs.cand_id).to_numpy()
f["emb_cos"] = np.einsum("ij,ij->i", np.asarray(emb[a], dtype=np.float32), np.asarray(emb[b], dtype=np.float32))
f["emb_cos_gap"] = (f.groupby(pairs.s1_id.to_numpy()).emb_cos.transform("max") - f.emb_cos).astype(np.float32)
for c in fn:
    if c not in f: f[c] = np.nan
f["p"] = np.mean([m.predict(f[fn], num_iteration=m.best_iteration) for m in models], axis=0)
f["s1_id"], f["cand_id"] = pairs.s1_id.to_numpy(), pairs.cand_id.to_numpy()
f["country"] = rec.country.reindex(f.s1_id).to_numpy()
sa, sb = rec.reindex(f.s1_id), rec.reindex(f.cand_id)
f["same_name"] = (sa.name_core.to_numpy() == sb.name_core.to_numpy()); f["same_addr"] = (sa.addr_norm.to_numpy() == sb.addr_norm.to_numpy()) & (sb.addr_norm.to_numpy() != "")
f.to_parquet(RD / "v11" / f"diag_neural_{fam}.parquet", index=False)
cols = ["p", "dense_sim", "dense_rank", "emb_cos", "hard_emb_sim", "addr_core_ts", "name_token_set", "tfidf_sim", "s1_n_cands"]
for nm, m in [("obvious: same name core + same address", f.same_name & f.same_addr), ("same address, different name", ~f.same_name & f.same_addr),
              ("same name, no/other address", f.same_name & ~f.same_addr), ("all pairs", np.ones(len(f), bool))]:
    print(f"\n## {nm}"); g = f[m].groupby("country")
    print(pd.concat([g.size().rename("pairs"), g[cols].median()], axis=1).round(3).to_string())
print("\nshare of pairs with 0.01<=p<=0.99:", f.groupby("country").p.apply(lambda x: float(((x >= .01) & (x <= .99)).mean())).round(4).to_dict())
