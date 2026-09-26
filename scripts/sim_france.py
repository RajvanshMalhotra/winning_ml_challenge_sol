"""How much do France's data gaps cost? Re-score the OOF pairs (US/India, labelled) with Model A after damaging the
features the way France test data is damaged, and compare F0.5 (Model A alone and the v5 stacker on top).
  base   : unchanged (must reproduce matcher_v3/oof.parquet)
  nostate: state unknown on every record (France: région never extracted)
  nohouse: house number lost on 4.9% of S2/S3 records (France: 'N°8' -> 'ndeg8')
  both   : nostate + nohouse
usage: CUDA_VISIBLE_DEVICES= python -u scripts/sim_france.py  -> artifacts/v2/sim_france/report.md"""
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from ber.blocking.sparse import cand_sparse_path
from ber.config import load_config, run_dir
from ber.features import REC_COLS, name_idf, pair_features
from ber.io import read_truth
from ber.matcher import add_extra_channels, candidate_context, decide, evaluate
from ber.predict import record_embeddings

cfg = load_config("configs/v2.yaml")
RD = run_dir(cfg)
OUT = RD / "sim_france"
OUT.mkdir(exist_ok=True)
T0 = time.perf_counter()
log = lambda *a: print(f"[{time.perf_counter() - T0:6.0f}s]", *a, flush=True)
MD = RD / "matcher_v3"
models = [lgb.Booster(model_file=str(MD / f"lgbm_fold{k}.txt")) for k in range(5)]
feat_names = models[0].feature_name()
best = json.loads((MD / "best.json").read_text())
channels = best.get("extra_channels", [])
emb, emb_code = record_embeddings(cfg, "train", best.get("embed_model_key", "bgem3"), best["embed_model_path"])
rec = pd.read_parquet(RD / "records_train.parquet", columns=REC_COLS).set_index("entity_id")
idf = name_idf(rec.reset_index())
ctx = candidate_context(cand_sparse_path(cfg, "train"))
oof = pd.read_parquet(MD / "oof.parquet")
sp = pd.read_parquet(RD / "splits.parquet", columns=["s1_id", "country", "fold"]).set_index("s1_id")
s1_all = oof.s1_id.unique().tolist()
rng = np.random.default_rng(0)
lost_house = set(rec.index[(rec.source.to_numpy() != 1) & (rng.random(len(rec)) < 0.049)])
log(f"{len(s1_all):,} OOF S1, {len(oof):,} OOF pairs, {len(feat_names)} features; house lost on {len(lost_house):,} S2/S3")
VAR = ["base", "nostate", "nohouse", "both"]
res = {v: [] for v in VAR}
for i in range(0, len(s1_all), 30_000):
    s1 = s1_all[i:i + 30_000]
    pairs = ds.dataset(cand_sparse_path(cfg, "train")).to_table(filter=ds.field("s1_id").isin(s1)).to_pandas()
    pairs = add_extra_channels(cfg, "train", pairs, set(s1), channels)
    pairs = pairs.join(ctx, on="cand_id")
    pairs["is_cand_best"] = (pairs.tfidf_sim >= pairs.cand_best_sim - 1e-6).astype(np.int8)
    f = pair_features(pairs, rec, workers=cfg["n_jobs"], idf=idf)
    a, b = emb_code.reindex(pairs.s1_id).to_numpy(), emb_code.reindex(pairs.cand_id).to_numpy()
    cos = np.empty(len(pairs), np.float32)
    for j in range(0, len(pairs), 1_000_000):
        cos[j:j + 1_000_000] = np.einsum("ij,ij->i", np.asarray(emb[a[j:j + 1_000_000]], dtype=np.float32),
                                         np.asarray(emb[b[j:j + 1_000_000]], dtype=np.float32))
    f["emb_cos"] = cos
    f["emb_cos_gap"] = (f.groupby(pairs.s1_id.to_numpy()).emb_cos.transform("max") - f.emb_cos).astype(np.float32)
    for c in feat_names:
        if c not in f:
            f[c] = np.nan
    fold = sp.fold.reindex(pairs.s1_id).to_numpy()
    hit = pairs.cand_id.isin(lost_house).to_numpy()
    for v in VAR:
        X = f[feat_names].copy()
        if v in ("nostate", "both"):
            X["state_cmp"] = 0
        if v in ("nohouse", "both"):
            X.loc[hit, "house_cmp"] = 0
            X.loc[hit, "house_num_diff"] = -1
        p = np.zeros(len(X), np.float32)
        for k in range(5):
            m = fold == k
            if m.any():
                p[m] = models[k].predict(X[m], num_iteration=models[k].best_iteration)
        res[v].append(pd.DataFrame({"s1_id": pairs.s1_id.to_numpy(), "cand_id": pairs.cand_id.to_numpy(), "p": p}))
    log(f"S1 {i + len(s1):,}/{len(s1_all):,} ({len(pairs):,} pairs)")
truth = read_truth(cfg["paths"]["data_dir"])
s1meta = sp.loc[s1_all].reset_index()[["s1_id", "country"]]
band = pd.read_parquet(RD / "reranker_v3" / "oof_band_scores.parquet")[["s1_id", "cand_id", "rr"]]
FEATS = ["p", "rr", "has_rr", "rr_rank", "rr_gap", "p_rank", "p_gap", "n_cands"]
v5m = [lgb.Booster(model_file=str(RD / "reranker_v3" / f"stack_fold{k}.txt")) for k in range(5)]
rows = []
for v in VAR:
    sc = pd.concat(res[v], ignore_index=True)
    sc.to_parquet(OUT / f"p_{v}.parquet", index=False)
    if v == "base":
        chk = sc.merge(oof[["s1_id", "cand_id", "p"]], on=["s1_id", "cand_id"], suffixes=("", "_oof"))
        log(f"base reproduces OOF p: max |diff| {np.abs(chk.p - chk.p_oof).max():.2e} over {len(chk):,} pairs")
    r = {"variant": v, "Model A F0.5 (τ .70)": evaluate(decide(sc, 0.70), truth, s1meta)["macro_f05"]}
    # v5 stacker: rr only exists for the ORIGINAL band pairs; a damaged p can move pairs out of / into the band
    s = sc.merge(band, on=["s1_id", "cand_id"], how="left")
    s.loc[(s.p < 0.01) | (s.p > 0.99), "rr"] = np.nan
    s["has_rr"] = s.rr.notna().astype(np.int8)
    g = s.groupby("s1_id")
    s["rr_rank"] = g.rr.rank(ascending=False, method="first").fillna(99).astype(np.int16)
    s["rr_gap"] = (g.rr.transform("max") - s.rr).astype(np.float32)
    s["p_rank"] = g.p.rank(ascending=False, method="first").astype(np.int16)
    s["p_gap"] = (g.p.transform("max") - s.p).astype(np.float32)
    s["n_cands"] = g.p.transform("size").astype(np.int16)
    fold = sp.fold.reindex(s.s1_id).to_numpy()
    q = np.zeros(len(s), np.float32)
    for k in range(5):
        m = fold == k
        q[m] = v5m[k].predict(s.loc[m, FEATS], num_iteration=v5m[k].best_iteration)
    e = evaluate(decide(s[["s1_id", "cand_id"]].assign(p=q), 0.65), truth, s1meta)
    r.update({"v5 F0.5 (τ .65)": e["macro_f05"], "v5 US": e["f05_US"], "v5 India": e["f05_India"],
              "band pairs": int(((sc.p >= 0.01) & (sc.p <= 0.99)).sum())})
    rows.append(r)
    log(r)
t = pd.DataFrame(rows)
for c in ["Model A F0.5 (τ .70)", "v5 F0.5 (τ .65)"]:
    t[f"Δ {c.split()[0]}"] = t[c] - t[c].iloc[0]
md = ["# France-gap simulation on labelled US/India OOF pairs", "",
      "| " + " | ".join(t.columns) + " |", "|" + "---|" * len(t.columns)]
md += ["| " + " | ".join(f"{x:.4f}" if isinstance(x, float) else str(x) for x in r) + " |" for r in t.itertuples(index=False)]
(OUT / "report.md").write_text("\n".join(md) + "\n")
print("\n".join(md), flush=True)
