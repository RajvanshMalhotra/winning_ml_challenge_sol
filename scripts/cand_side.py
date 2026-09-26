"""Candidate-side features for the stacker: "among all S1s that want this S2/S3 record, is this S1 the clear best?"
Motivation: S2/S3 records with an empty address belong to some S1 97.7% of the time, yet v5 finds only 44% of them.
Competition is measured over ALL S1s (dense top-30 and sparse channels exist for every S1 in train and test), never
over the OOF sample, so train and test features mean the same thing.
usage: cand_side.py train | submit
  train : OOF LightGBM stacker (v5 features + candidate-side) vs the v5 stacker on the same folds -> F0.5, no-address recall
  submit: apply to test -> artifacts/submissions/$SUB + validator"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from ber.config import load_config, run_dir
from ber.io import read_truth
from ber.matcher import decide, evaluate, md_table
from ber.normalize import records_path
from ber.predict import write_submission

cfg = load_config("configs/v2.yaml")
RD = run_dir(cfg)
RR = RD / os.environ.get("RR_DIR", "reranker_v3")
A_DIR = os.environ.get("A_DIR", "matcher_v3")
TEST_SCORES = os.environ.get("TEST_SCORES", "test_scores_matcher_v3.parquet")
OUT = RD / os.environ.get("CS_DIR", "cand_side")
SUB = os.environ.get("SUB", "v8")
OUT.mkdir(exist_ok=True)
T0 = time.perf_counter()
log = lambda *a: print(f"[{time.perf_counter() - T0:6.0f}s]", *a, flush=True)
BASE = ["p", "rr", "has_rr", "rr_rank", "rr_gap", "p_rank", "p_gap", "n_cands"]
NEW = ["cand_noaddr", "name_exact", "s1_name_cnt", "cand_name_cnt",
       "d_crank", "d_cmargin", "d_ncomp", "t_crank", "t_cmargin", "t_ncomp", "n_crank", "n_cmargin", "n_ncomp"]
PARAMS = {"objective": "binary", "metric": "auc", "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 200,
          "feature_fraction": 0.9, "verbose": -1, "num_threads": cfg["n_jobs"]}


def stack_features(s: pd.DataFrame) -> pd.DataFrame:
    f = s.copy()
    f["has_rr"] = f.rr.notna().astype(np.int8)
    g = f.groupby("s1_id")
    f["rr_rank"] = g.rr.rank(ascending=False, method="first").fillna(99).astype(np.int16)
    f["rr_gap"] = (g.rr.transform("max") - f.rr).astype(np.float32)
    f["p_rank"] = g.p.rank(ascending=False, method="first").astype(np.int16)
    f["p_gap"] = (g.p.transform("max") - f.p).astype(np.float32)
    f["n_cands"] = g.p.transform("size").astype(np.int16)
    return f


class Records:
    """Integer codes for entity ids plus per-record name/address facts."""

    def __init__(self, fam: str):
        rec = pd.read_parquet(records_path(cfg, fam), columns=["entity_id", "source", "country", "name_core", "addr_raw"])
        assert len(rec) < 1 << 24
        self.ids = pa.array(rec.entity_id.to_numpy(), type=pa.large_string())
        self.noaddr = rec.addr_raw.fillna("").str.strip().eq("").to_numpy()
        self.nk = pd.factorize(rec.country.astype(str) + "\x00" + rec.name_core.fillna("").astype(str))[0]
        s1 = rec.source.to_numpy() == 1
        self.s1cnt = np.bincount(self.nk[s1], minlength=self.nk.max() + 1).astype(np.int32)
        self.ocnt = np.bincount(self.nk[~s1], minlength=self.nk.max() + 1).astype(np.int32)

    def codes(self, col) -> np.ndarray:
        x = pc.index_in(pc.cast(col, pa.large_string()), value_set=self.ids)
        assert x.null_count == 0, f"{x.null_count} ids not in records"
        return np.asarray(x.to_numpy()).astype(np.int64)


def fnum(col) -> np.ndarray:
    return col.to_pandas().to_numpy(dtype=np.float32, na_value=np.nan)


def comp_table(s: np.ndarray, c: np.ndarray, sim: np.ndarray) -> tuple:
    """Per (S1, cand): rank of this S1 among all S1s that have cand (by sim), margin over the best OTHER S1, #S1s."""
    m = ~np.isnan(sim)
    s, c, sim = s[m], c[m], sim[m]
    o = np.lexsort((-sim, c))
    s, c, sim = s[o], c[o], sim[o]
    start = np.r_[0, np.flatnonzero(np.diff(c)) + 1]
    cnt = np.diff(np.r_[start, len(c)])
    gid = np.repeat(np.arange(len(start)), cnt)
    rank = np.arange(len(c)) - start[gid] + 1
    best = sim[start][gid]
    second = np.where(cnt > 1, sim[np.minimum(start + 1, len(c) - 1)], np.nan)[gid]
    margin = sim - np.where(rank == 1, second, best)
    key = (s << 24) | c
    ko = np.argsort(key, kind="stable")
    return key[ko], rank[ko].astype(np.int16), margin[ko].astype(np.float32), cnt[gid][ko].astype(np.int16)


def lookup(tab: tuple, qkey: np.ndarray) -> tuple:
    ks, rank, margin, n = tab
    i = np.minimum(np.searchsorted(ks, qkey), len(ks) - 1)
    hit = ks[i] == qkey
    return (np.where(hit, rank[i], 99).astype(np.int16), np.where(hit, margin[i], np.nan).astype(np.float32),
            np.where(hit, n[i], 0).astype(np.int16))


def comp_tables(fam: str, R: Records) -> dict:
    tabs = {}
    t = pq.read_table(RD / f"cand_dense_{fam}.parquet", columns=["s1_id", "cand_id", "dense_sim"])
    tabs["d"] = comp_table(R.codes(t["s1_id"]), R.codes(t["cand_id"]), fnum(t["dense_sim"]))
    log(f"{fam}: dense competition table {len(tabs['d'][0]):,} pairs")
    t = pq.read_table(RD / f"cand_sparse_{fam}.parquet", columns=["s1_id", "cand_id", "tfidf_sim", "name_sim"])
    s, c = R.codes(t["s1_id"]), R.codes(t["cand_id"])
    tabs["t"] = comp_table(s, c, fnum(t["tfidf_sim"]))
    tabs["n"] = comp_table(s, c, fnum(t["name_sim"]))
    log(f"{fam}: sparse competition tables {len(tabs['t'][0]):,} / name {len(tabs['n'][0]):,} pairs")
    return tabs


def add_new(f: pd.DataFrame, R: Records, tabs: dict) -> pd.DataFrame:
    s = R.codes(pa.array(f.s1_id.to_numpy(), type=pa.large_string()))
    c = R.codes(pa.array(f.cand_id.to_numpy(), type=pa.large_string()))
    f["cand_noaddr"] = R.noaddr[c].astype(np.int8)
    f["name_exact"] = (R.nk[s] == R.nk[c]).astype(np.int8)
    f["s1_name_cnt"] = R.s1cnt[R.nk[c]]
    f["cand_name_cnt"] = R.ocnt[R.nk[c]]
    key = (s << 24) | c
    for p, tab in tabs.items():
        f[f"{p}_crank"], f[f"{p}_cmargin"], f[f"{p}_ncomp"] = lookup(tab, key)
    return f


def train() -> None:
    o = pd.read_parquet(RD / A_DIR / "oof.parquet")
    b = pd.read_parquet(RR / "oof_band_scores.parquet")[["s1_id", "cand_id", "rr"]]
    f = stack_features(o.merge(b, on=["s1_id", "cand_id"], how="left"))
    R = Records("train")
    f = add_new(f, R, comp_tables("train", R))
    sp = pd.read_parquet(RD / "splits.parquet", columns=["s1_id", "country", "fold"]).set_index("s1_id")
    f["fold"] = sp.fold.reindex(f.s1_id).to_numpy()
    log(f"OOF pairs {len(f):,}; no-address candidates {f.cand_noaddr.mean():.2%}; "
        f"dense competition found for {(f.d_ncomp > 0).mean():.1%}")
    preds = {"v5 stacker": np.zeros(len(f), np.float32), "v5 + candidate-side": np.zeros(len(f), np.float32)}
    imp = []
    for k in sorted(f.fold.unique()):
        tr, va = f.fold.to_numpy() != k, f.fold.to_numpy() == k
        for name, feats in (("v5 stacker", BASE), ("v5 + candidate-side", BASE + NEW)):
            m = lgb.train(PARAMS, lgb.Dataset(f.loc[tr, feats], f.label[tr]), 3000,
                          valid_sets=[lgb.Dataset(f.loc[va, feats], f.label[va])], callbacks=[lgb.early_stopping(100, verbose=False)])
            preds[name][va] = m.predict(f.loc[va, feats], num_iteration=m.best_iteration)
            if name != "v5 stacker":
                m.save_model(str(OUT / f"cs_fold{k}.txt"))
                imp.append(pd.Series(m.feature_importance("gain"), index=feats))
        log(f"fold {k} done")
    truth = read_truth(cfg["paths"]["data_dir"])
    s1 = sp.loc[o.s1_id.unique()].reset_index()[["s1_id", "country"]]
    rows, na = [], {}
    for name, p in preds.items():
        sc = pd.DataFrame({"s1_id": f.s1_id, "cand_id": f.cand_id, "p": p})
        for tau in np.round(np.arange(0.4, 0.91, 0.05), 2):
            rows.append({"model": name, "tau": float(tau), **evaluate(decide(sc, tau), truth, s1)})
    grid = pd.DataFrame(rows)
    best = {n: g.sort_values("macro_f05").iloc[-1] for n, g in grid.groupby("model")}
    noaddr_ids = set(f.cand_id[f.cand_noaddr == 1])
    for name, p in preds.items():   # recall / precision on no-address candidates at each model's best tau
        pred = decide(pd.DataFrame({"s1_id": f.s1_id, "cand_id": f.cand_id, "p": p}), best[name].tau)
        tp_ = [(c in pred.get(s, set())) for s in s1.s1_id for c in truth[s] if c in noaddr_ids]
        pk = [(c in truth[s]) for s in s1.s1_id for c in pred.get(s, set()) if c in noaddr_ids]
        na[name] = (float(np.mean(tp_)), float(np.mean(pk)))
    b0, b1 = best["v5 stacker"], best["v5 + candidate-side"]
    (OUT / "best.json").write_text(json.dumps({"tau": float(b1.tau), "macro_f05": float(b1.macro_f05),
                                               "base_tau": float(b0.tau), "base_f05": float(b0.macro_f05)}, indent=2))
    gi = pd.concat(imp, axis=1).mean(axis=1)
    gi = (gi / gi.sum() * 100).sort_values(ascending=False).round(1)
    fmt = lambda r: (f"τ={r.tau:.2f} → **{r.macro_f05:.4f}** (singletons {r.singletons:.4f}, "
                     + ", ".join(f"{k[4:]} {v:.4f}" for k, v in r.items() if str(k).startswith("f05_")) + ")")
    md = ["# Candidate-side features in the stacker — out-of-fold (same folds, same pairs)", "",
          f"- v5 stacker: {fmt(b0)} | no-address recall {na['v5 stacker'][0]:.1%}, precision {na['v5 stacker'][1]:.1%}",
          f"- **v5 + candidate-side: {fmt(b1)}** | no-address recall {na['v5 + candidate-side'][0]:.1%}, "
          f"precision {na['v5 + candidate-side'][1]:.1%}", "",
          "Feature importance (gain %): " + ", ".join(f"{k} {v}" for k, v in gi.items()), "", md_table(grid)]
    Path("docs/results/cand_side.md").write_text("\n".join(md) + "\n")
    print("\n".join(md[:6]), flush=True)


def submit() -> None:
    best = json.loads((OUT / "best.json").read_text())
    t = pd.read_parquet(RD / TEST_SCORES)
    b = pd.read_parquet(RR / "test_band_scores.parquet")[["s1_id", "cand_id", "rr"]]
    f = stack_features(t.merge(b, on=["s1_id", "cand_id"], how="left"))
    del t
    R = Records("test")
    f = add_new(f, R, comp_tables("test", R))
    models = [lgb.Booster(model_file=str(p)) for p in sorted(OUT.glob("cs_fold*.txt"))]
    p = np.mean([m.predict(f[BASE + NEW], num_iteration=m.best_iteration) for m in models], axis=0)
    sc = pd.DataFrame({"s1_id": f.s1_id, "cand_id": f.cand_id, "p": p})
    pred = decide(sc, best["tau"])
    rec = pd.read_parquet(records_path(cfg, "test"), columns=["entity_id", "source", "country"])
    s1_all = rec.entity_id[rec.source == 1].tolist()
    out = Path(f"artifacts/submissions/{SUB}")
    write_submission(s1_all, pred, f.groupby("s1_id").cand_id.agg(set).to_dict(), out)
    n = pd.Series({s: len(v) for s, v in pred.items()}).reindex(s1_all, fill_value=0)
    ctry = rec.set_index("entity_id").country.reindex(n.index).to_numpy()
    noaddr = set(f.cand_id[f.cand_noaddr == 1])
    n_na = sum(c in noaddr for v in pred.values() for c in v)
    log({"tau": best["tau"], "mean_matches": float(n.mean()), "empty_share": float((n == 0).mean()),
         "pred_noaddr_pairs": n_na, **{c: float(n[ctry == c].mean()) for c in ["US", "India", "France"]}})
    v = subprocess.run(["python3", cfg["paths"]["validator"], "--matching", str(out / "matching_results.tsv"),
                        "--candidate", str(out / "candidate_pairs.tsv"), "--test-dir",
                        str(Path(cfg["paths"]["data_dir"]) / "test")], capture_output=True, text=True)
    print(v.stdout[-800:], f"validator exit={v.returncode}", flush=True)


{"train": train, "submit": submit}[sys.argv[1]]()
