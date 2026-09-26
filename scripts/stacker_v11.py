"""v11 stacker: v10a features (+ acronym feature) on OOF; France-robust feature set; seed ensemble; calibrated
expected-F0.5 decisions per S1. Everything is scored on the same OOF folds as v8/v9/v10a (nested where tuned).
usage (cwd = repo, CPU only):
  stacker_v11.py scores            : testfr_scores = test Model A scores with France rows from the fixed-text re-score
  stacker_v11.py feats FAM         : full stacker feature table for FAM in {train, test, testfr} -> v11/X_FAM.parquet
  stacker_v11.py shift             : France vs US/India distribution shift of every feature on testfr (label-free)
  stacker_v11.py train             : OOF models + decision rules -> v11/report.md, v11/best.json
  stacker_v11.py submit FAM NAME   : write artifacts/submissions/NAME from the chosen variant (+ validator)"""
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

G = {"__name__": "v10a"}
exec(open(os.path.expanduser("~/claude_v11/scripts/clean_name_v10a.py")).read().rsplit('\nif __name__ == "__main__":', 1)[0], G)
import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

cfg, RD = G["cfg"], G["RD"]
N9, N8 = G["ns"], G["ns"]["ns"]                    # v9 namespace, v8 (cand_side) namespace
BASE, NEW, V9, CN = G["BASE"], G["NEW"], G["V9"], G["CN"]
ACR = ["acr_cand", "acr_match", "acr_part"]
OUT = RD / "v11"
OUT.mkdir(exist_ok=True)
T0 = time.perf_counter()
log = lambda *a: print(f"[{time.perf_counter() - T0:6.0f}s]", *a, flush=True)
SCORES = {"train": RD / "train_scores_matcher_v3.parquet", "test": RD / "test_scores_matcher_v3.parquet",
          "testfr": RD / "testfr_scores_matcher_v3.parquet"}
N9["SCORES"].update(SCORES)
_all_pairs_v9 = N9["all_pairs"]


def all_pairs(fam, R):  # test-like families: every pair is a focus pair (v9 treats only "test" that way)
    if fam == "train":
        return _all_pairs_v9(fam, R)
    t = pq.read_table(SCORES[fam], columns=["s1_id", "cand_id", "p"])
    return R.codes(t["s1_id"]), R.codes(t["cand_id"]), np.array(N8["fnum"](t["p"]), dtype=np.float32), np.arange(t.num_rows)


N9["all_pairs"] = G["all_pairs"] = all_pairs


def _clean_chunk(xs):  # defined in this (__main__) module so multiprocessing can pickle it
    return [G["clean_name"](x) for x in xs]


def _clean_all(names):
    from multiprocessing import Pool
    with Pool(48) as pool:
        out = pool.map(_clean_chunk, np.array_split(names, 192))
    return np.array([y for part in out for y in part], dtype=object)


G["clean_all"] = _clean_all
STOP = {"de", "des", "du", "la", "le", "les", "et", "d", "l", "of", "the", "and", "&", "sarl", "sas", "sasu", "eurl", "sci",
        "sa", "snc", "selarl", "ei", "cie", "inc", "llc", "ltd", "pvt", "private", "limited", "corp", "co", "pc", "llp", "lp"}
_ACR = re.compile(r"^[A-Z]{2,6}$")
_W = re.compile(r"[A-Za-zÀ-ÿ]+")


def initials(name: str) -> str:
    return "".join(w[0] for w in _W.findall(G["anyascii"](name or "")) if w.lower() not in STOP).upper()


def acr_features(s1_ids: np.ndarray, cand_ids: np.ndarray, fam: str) -> dict:
    rec = pd.read_parquet(G["records_path"](cfg, fam), columns=["entity_id", "name_raw"]).set_index("entity_id").name_raw
    raw = rec.fillna("").str.replace(".", "", regex=False).str.strip()
    is_acr = raw.str.match(_ACR)
    acr_ids = set(rec.index[is_acr.to_numpy()])
    m = pd.Series(cand_ids).isin(acr_ids).to_numpy()
    out = {k: np.zeros(len(cand_ids), np.int8) for k in ACR}
    out["acr_cand"][m] = 1
    if m.any():
        ini = np.array([initials(x) for x in rec.reindex(s1_ids[m]).to_numpy()])
        cand = raw.reindex(cand_ids[m]).to_numpy()
        out["acr_match"][m] = (ini == cand).astype(np.int8)
        out["acr_part"][m] = np.array([bool(c) and (c in i or i in c) for i, c in zip(ini, cand)]).astype(np.int8)
    log(f"{fam}: acronym candidates {int(m.sum()):,} pairs, initials match {int(out['acr_match'].sum()):,}")
    return out


def scores() -> None:
    t = pd.read_parquet(SCORES["test"])
    fr = pd.read_parquet(RD / "testfr_scores_matcher_v3_France.parquet")
    cty = pd.read_parquet(G["records_path"](cfg, "test"), columns=["entity_id", "country"]).set_index("entity_id").country
    isfr = (cty.reindex(t.s1_id) == "France").to_numpy()
    m = t[isfr].merge(fr, on=["s1_id", "cand_id"], how="left", suffixes=("_old", ""))
    assert len(m) == int(isfr.sum()) and m.p.notna().all(), "France re-score must cover the same pairs"
    old, new = m.p_old.to_numpy(), m.p.to_numpy()
    band = lambda x: (x >= 0.01) & (x <= 0.99)
    log(f"France pairs {len(m):,}: band {int(band(old).sum()):,} -> {int(band(new).sum()):,}; p>=0.7 {int((old >= .7).sum()):,} -> "
        f"{int((new >= .7).sum()):,}; mean |dp| {np.abs(new - old).mean():.4f}")
    t = t.copy()
    t.loc[isfr, "p"] = pd.Series(new, index=t.index[isfr]).to_numpy()  # same row order as the merge (left join keeps order)
    t.to_parquet(SCORES["testfr"], index=False)
    log("wrote testfr_scores_matcher_v3.parquet")


def feats(fam: str) -> None:
    if fam == "train":
        f = pd.read_parquet(RD / "cand_side" / "oof_q.parquet")
        for path, cols in [(RD / "cand_side_v9" / "feats_train.parquet", V9), (RD / "clean_v10a" / "feats_train.parquet", CN)]:
            v = pd.read_parquet(path, columns=cols)
            for k in cols:
                f[k] = v[k].to_numpy()
            del v
    else:
        t = pd.read_parquet(SCORES[fam])
        b = pd.read_parquet(N8["RR"] / "test_band_scores.parquet")[["s1_id", "cand_id", "rr"]]
        t = t.merge(b, on=["s1_id", "cand_id"], how="left")
        t.loc[(t.p < 0.01) | (t.p > 0.99), "rr"] = np.nan  # rr only where the pair is in Model A's band (as in training)
        del b
        f = N8["stack_features"](t)
        del t
        R = N8["Records"](fam)
        f = N8["add_new"](f, R, N8["comp_tables"](fam, R))
        log(f"{fam}: v8 features done")
        v9p, cnp = RD / "cand_side_v9" / f"feats_{fam}.parquet", RD / "clean_v10a" / f"feats_{fam}.parquet"
        if not v9p.exists():
            N9["feats"](fam)
        if not cnp.exists():
            G["feats"](fam)
        for path, cols in [(v9p, V9), (cnp, CN)]:
            v = pd.read_parquet(path, columns=cols)
            assert len(v) == len(f)
            for k in cols:
                f[k] = v[k].to_numpy()
            del v
    a = acr_features(f.s1_id.to_numpy(), f.cand_id.to_numpy(), "train" if fam == "train" else fam)
    for k in ACR:
        f[k] = a[k]
    cty = pd.read_parquet(G["records_path"](cfg, fam), columns=["entity_id", "country"]).set_index("entity_id").country
    f["country"] = cty.reindex(f.s1_id).to_numpy()
    f.to_parquet(OUT / f"X_{fam}.parquet", index=False)
    log(f"wrote X_{fam}.parquet {f.shape}")


ALL = BASE + NEW + V9 + CN + ACR


def shift() -> None:
    cols = ALL + ["country"]
    f = pd.read_parquet(OUT / "X_testfr.parquet", columns=cols, filters=[("p", ">=", 0.01)])
    fr, ot = f[f.country == "France"], f[f.country != "France"]
    rows = []
    for c in ALL:
        a, b = ot[c].to_numpy(np.float64), fr[c].to_numpy(np.float64)
        qs = np.unique(np.nanquantile(a, np.linspace(0, 1, 11)))
        if len(qs) < 3:
            qs = np.unique(np.r_[np.nanmin(np.r_[a, b]) - 1, np.nanquantile(a, [.5]), np.nanmax(np.r_[a, b]) + 1])
        edges = np.r_[-np.inf, qs[1:-1], np.inf]
        pa = np.histogram(a[~np.isnan(a)], edges)[0] / max((~np.isnan(a)).sum(), 1) + 1e-4
        pb = np.histogram(b[~np.isnan(b)], edges)[0] / max((~np.isnan(b)).sum(), 1) + 1e-4
        na_a, na_b = np.isnan(a).mean() + 1e-4, np.isnan(b).mean() + 1e-4
        psi = float(np.sum((pb - pa) * np.log(pb / pa)) + (na_b - na_a) * np.log(na_b / na_a))
        rows.append({"feature": c, "PSI France vs US/India": psi, "median US/India": np.nanmedian(a), "median France": np.nanmedian(b),
                     "missing US/India": float(np.isnan(a).mean()), "missing France": float(np.isnan(b).mean())})
    t = pd.DataFrame(rows).sort_values("PSI France vs US/India", ascending=False)
    t.to_csv(OUT / "shift.csv", index=False)
    print(t.round(4).to_string(index=False), flush=True)


PARAMS = dict(G["PARAMS"])


def lgb_oof(f: pd.DataFrame, feats: list, params: dict, tag: str) -> np.ndarray:
    q = np.zeros(len(f), np.float32)
    for k in range(5):
        tr, va = f.fold.to_numpy() != k, f.fold.to_numpy() == k
        m = lgb.train(params, lgb.Dataset(f.loc[tr, feats], f.label[tr]), 4000,
                      valid_sets=[lgb.Dataset(f.loc[va, feats], f.label[va])], callbacks=[lgb.early_stopping(100, verbose=False)])
        q[va] = m.predict(f.loc[va, feats], num_iteration=m.best_iteration)
        m.save_model(str(OUT / f"{tag}_fold{k}.txt"))
    log(f"{tag}: trained")
    return q


def owner_mask(cand: np.ndarray, q: np.ndarray) -> np.ndarray:
    """each candidate kept only for its best-scoring S1"""
    d = pd.DataFrame({"c": cand, "q": q})
    m = np.zeros(len(q), bool)
    m[d.groupby("c").q.idxmax().to_numpy()] = True
    return m


def decide_tau(code, cand, q, tau):
    return owner_mask(cand, q) & (q >= tau)


def decide_ef(code, cand, q, n_s1, lam=1.0, floor=0.0):
    """Per S1: among its owned candidates sorted by q, pick the top-k maximising expected F0.5
    E[F](k) ~ 1.25 * sum_topk q / (0.25 * lam * sum_all q + k); k = 0 scores prod(1 - q)."""
    own = owner_mask(cand, q) & (q >= floor)
    idx = np.flatnonzero(own)
    d = pd.DataFrame({"s": code[idx], "q": q[idx], "i": idx}).sort_values(["s", "q"], ascending=[True, False])
    s, qq = d.s.to_numpy(), d.q.to_numpy()
    start = np.r_[0, np.flatnonzero(np.diff(s)) + 1]
    grp = np.repeat(np.arange(len(start)), np.diff(np.r_[start, len(s)]))
    k = np.arange(len(s)) - start[grp] + 1
    cs = np.cumsum(qq)
    cs_g = cs - np.r_[0, cs][start][grp]
    tot = np.bincount(code, weights=q, minlength=n_s1)[s]          # all candidates of the S1 (expected #true)
    ef = 1.25 * cs_g / (0.25 * lam * tot + k)
    lp = np.log1p(-np.clip(q, 0, 1 - 1e-7))
    ef0 = np.exp(np.bincount(code, weights=lp, minlength=n_s1))[s]  # P(no true match) among all candidates
    best = pd.Series(ef).groupby(grp).cummax().to_numpy()
    gmax = pd.Series(ef).groupby(grp).transform("max").to_numpy()
    kbest = pd.Series(np.where(ef >= gmax - 1e-12, k, 10 ** 6)).groupby(grp).transform("min").to_numpy()
    pick = (k <= kbest) & (gmax > ef0)
    m = np.zeros(len(q), bool)
    m[d.i.to_numpy()[pick]] = True
    return m


def per_s1(code, lab, pred, nt):
    tp = np.bincount(code, weights=pred & lab, minlength=len(nt))
    npred = np.bincount(code, weights=pred, minlength=len(nt))
    return np.where(nt == 0, (npred == 0) * 1.0, 1.25 * tp / np.maximum(0.25 * nt + npred, 1e-9))


def isotonic_oof(q, lab, fold):
    from sklearn.isotonic import IsotonicRegression
    out = np.zeros(len(q), np.float32)
    for k in range(5):
        tr, va = fold != k, fold == k
        sub = np.flatnonzero(tr)[::7]
        ir = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(q[sub], lab[sub])
        out[va] = ir.predict(q[va])
    return out


def train() -> None:
    f = pd.read_parquet(OUT / "X_train.parquet")
    q9 = pd.read_parquet(RD / "cand_side_v9" / "oof_q9.parquet", columns=["q9"]).q9.to_numpy()
    q10 = pd.read_parquet(RD / "clean_v10a" / "oof_q10.parquet", columns=["q10"]).q10.to_numpy()
    truth = G["read_truth"](cfg["paths"]["data_dir"])
    s1 = pd.Index(f.s1_id.unique())
    nt = np.array([len(truth[s]) for s in s1])
    code = s1.get_indexer(f.s1_id)
    fold_s1 = f.groupby("s1_id").fold.first().reindex(s1).to_numpy()
    ctry_s1 = f.groupby("s1_id").country.first().reindex(s1).to_numpy()
    lab = f.label.to_numpy().astype(bool)
    cand = f.cand_id.to_numpy()
    fold = f.fold.to_numpy()
    shift_t = pd.read_csv(OUT / "shift.csv") if (OUT / "shift.csv").exists() else None
    shifted = [] if shift_t is None else shift_t.loc[shift_t["PSI France vs US/India"] > 0.25, "feature"].tolist()
    robust = [c for c in ALL if c not in shifted]
    log(f"features: all {len(ALL)}, France-shifted (PSI > 0.25) {shifted}")
    Q = {"v9": q9, "v10a": q10}
    Q["v11 (v10a + acronym)"] = lgb_oof(f, ALL, PARAMS, "full")
    if shifted:
        Q["v11 robust (no shifted feats)"] = lgb_oof(f, robust, PARAMS, "robust")
    ens = [Q["v11 (v10a + acronym)"]]
    for i, (nl, ff, mcs) in enumerate([(63, 0.8, 100), (15, 0.9, 400), (31, 0.7, 200)]):
        p = dict(PARAMS, num_leaves=nl, feature_fraction=ff, min_child_samples=mcs, bagging_fraction=0.8, bagging_freq=1, seed=11 + i)
        ens.append(lgb_oof(f, ALL, p, f"ens{i}"))
    Q["v11 ensemble (4 LightGBMs)"] = np.mean(ens, axis=0).astype(np.float32)
    np.save(OUT / "oof_q.npy", np.vstack([Q[k] for k in Q]))
    acc = decide_tau(code, cand, Q["v11 ensemble (4 LightGBMs)"], 0.5)
    log(f"accepted OOF pairs (τ .5) with Model A p < 0.001: {int((acc & (f.p.to_numpy() < 0.001)).sum())} of {int(acc.sum()):,}")
    rows, sc_keep = [], {}
    for name, q in Q.items():
        best = None
        for tau in np.round(np.arange(0.5, 0.91, 0.05), 2):
            sc = per_s1(code, lab, decide_tau(code, cand, q, tau), nt)
            if best is None or sc.mean() > best[1].mean():
                best = (tau, sc)
        rows.append({"model": name, "decision": f"global τ={best[0]}", "F0.5": best[1].mean(),
                     "US": best[1][ctry_s1 == "US"].mean(), "India": best[1][ctry_s1 == "India"].mean(),
                     "singletons": best[1][nt == 0].mean()})
        sc_keep[(name, "tau")] = best
        log(rows[-1])
    # expected-F0.5 decisions on the ensemble, calibrated with isotonic (fit on other folds); lam/floor nested-CV
    qe = Q["v11 ensemble (4 LightGBMs)"]
    qc = isotonic_oof(qe, lab, fold)
    grid = {}
    for lam in [0.8, 1.0, 1.2]:
        for floor in [0.2, 0.35, 0.5]:
            grid[(lam, floor)] = per_s1(code, lab, decide_ef(code, cand, qc, len(s1), lam, floor), nt)
    out, picks = np.zeros(len(s1)), []
    for k in range(5):
        tr, te = fold_s1 != k, fold_s1 == k
        b = max(grid, key=lambda g: grid[g][tr].mean())
        picks.append(b)
        out[te] = grid[b][te]
    rows.append({"model": "v11 ensemble", "decision": f"expected-F0.5 per S1 (calibrated; nested CV, picks {picks})", "F0.5": out.mean(),
                 "US": out[ctry_s1 == "US"].mean(), "India": out[ctry_s1 == "India"].mean(), "singletons": out[nt == 0].mean()})
    log(rows[-1])
    # test-like orphan density: test has ~1.9x more records owned by no S1 per S1 than train, so every wrong accept
    # of an orphan record is counted twice here (a second, similar orphan would be accepted the same way)
    owned = {c for m in truth.values() for c in m}
    orphan = ~pd.Series(cand).isin(owned).to_numpy()
    orows = []
    for name in ["v9", "v11 ensemble (4 LightGBMs)"]:
        for tau in np.round(np.arange(0.5, 0.96, 0.05), 2):
            pr = decide_tau(code, cand, Q[name], tau)
            sc1 = per_s1(code, lab, pr, nt)
            extra = np.bincount(code, weights=pr & orphan, minlength=len(nt))
            tp = np.bincount(code, weights=pr & lab, minlength=len(nt))
            npred = np.bincount(code, weights=pr, minlength=len(nt)) + extra
            sc2 = np.where(nt == 0, (npred == 0) * 1.0, 1.25 * tp / np.maximum(0.25 * nt + npred, 1e-9))
            orows.append({"model": name, "tau": float(tau), "F0.5 (train density)": sc1.mean(), "F0.5 (test-like orphans x2)": sc2.mean(),
                          "orphan wrong accepts": int((pr & orphan).sum()), "other wrong accepts": int((pr & ~lab & ~orphan).sum())})
    ot = pd.DataFrame(orows)
    ot.to_csv(OUT / "orphan_density.csv", index=False)
    print(ot.round(5).to_string(index=False), flush=True)
    t = pd.DataFrame(rows)
    base = sc_keep[("v9", "tau")][1]
    rng = np.random.default_rng(0)
    cis = []
    for r in rows:
        sc = sc_keep.get((r["model"], "tau"), (None, out))[1] if "expected" not in r["decision"] else out
        d = sc - base
        bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(300)]
        cis.append(f"{d.mean():+.5f} [{np.percentile(bs, 2.5):+.5f}, {np.percentile(bs, 97.5):+.5f}]")
    t["Δ vs v9 (95% CI)"] = cis
    best_row = t.sort_values("F0.5").iloc[-1]
    j = {"best_model": best_row.model, "best_decision": best_row.decision, "best_f05": float(best_row["F0.5"]),
         "shifted": shifted, "robust": robust, "ef_pick": list(max(grid, key=lambda g: grid[g].mean())),
         "taus": {k[0]: float(v[0]) for k, v in sc_keep.items()}}
    (OUT / "best.json").write_text(json.dumps(j, indent=2, default=str))
    md = ["# v11 stacker: OOF comparison (150k B-split S1, same folds)", "", G["md_table"](t), "",
          f"France-shifted features dropped in the robust model: {shifted}", "", "## Cut-off under test-like orphan density", "", G["md_table"](ot)]
    (OUT / "report.md").write_text("\n".join(md) + "\n")
    print("\n".join(md), flush=True)


def robust() -> None:
    f = pd.read_parquet(OUT / "X_train.parquet")
    sh = pd.read_csv(OUT / "shift.csv")
    shifted = sh.loc[sh["PSI France vs US/India"] > 0.25, "feature"].tolist()
    feats = [c for c in ALL if c not in shifted]
    q = lgb_oof(f, feats, PARAMS, "robust")
    truth = G["read_truth"](cfg["paths"]["data_dir"])
    s1 = pd.Index(f.s1_id.unique()); nt = np.array([len(truth[s]) for s in s1]); code = s1.get_indexer(f.s1_id)
    lab = f.label.to_numpy().astype(bool); cand = f.cand_id.to_numpy()
    res = {float(t): per_s1(code, lab, decide_tau(code, cand, q, t), nt).mean() for t in np.round(np.arange(0.5, 0.91, 0.05), 2)}
    tau = max(res, key=res.get)
    j = json.loads((OUT / "best.json").read_text())
    j.update({"shifted": shifted, "robust": feats, "tau_robust": tau, "f05_robust": res[tau]})
    (OUT / "best.json").write_text(json.dumps(j, indent=2, default=str))
    line = f"robust stacker (dropped {shifted}): τ={tau} → OOF F0.5 {res[tau]:.5f}"
    with open(OUT / "report.md", "a") as fh:
        fh.write("\n" + line + "\n")
    log(line)


def france_addr_pairs(fam: str) -> pd.DataFrame:
    """France S1s that are the only S1 at their exact (normalized) address, paired with every S2/S3 record at that same
    address. In train (US/India) such pairs are 96.4% true; in France the models reject 38% of them (renamed copies
    with realistic French names), vs 3.9% in the US (the true orphan rate)."""
    r = pd.read_parquet(G["records_path"](cfg, fam), columns=["entity_id", "source", "country", "addr_norm"])
    r = r[(r.country == "France") & (r.addr_norm != "")]
    s1 = r[r.source == 1]
    lone = s1.groupby("addr_norm").entity_id.agg(["first", "size"])
    lone = lone[lone["size"] == 1]["first"]
    oth = r[r.source != 1]
    pr = oth.merge(lone.rename("s1_id"), left_on="addr_norm", right_index=True)[["s1_id", "entity_id"]]
    return pr.rename(columns={"entity_id": "cand_id"})


def predict_models(X: pd.DataFrame, tags: list, feats: list) -> np.ndarray:
    ps = []
    for tag in tags:
        ms = [lgb.Booster(model_file=str(OUT / f"{tag}_fold{k}.txt")) for k in range(5)]
        ps.append(np.mean([m.predict(X[feats], num_iteration=m.best_iteration) for m in ms], axis=0))
    return np.mean(ps, axis=0).astype(np.float32)


def submit(fam: str, name: str, variant: str) -> None:
    """variant: full | ensemble | ensemble_ef | ensemble_frrobust (France S1s scored by the robust model)"""
    j = json.loads((OUT / "best.json").read_text())
    X = pd.read_parquet(OUT / f"X_{fam}.parquet")
    keep = X.p.to_numpy() >= 0.001                 # below this Model A score no OOF pair is ever accepted
    Xk = X[keep].reset_index(drop=True)
    tags = ["full"] if variant == "full" else ["full", "ens0", "ens1", "ens2"]
    q = predict_models(Xk, tags, ALL)
    if variant == "ensemble_frrobust":
        fr = (Xk.country == "France").to_numpy()
        q[fr] = predict_models(Xk[fr], ["robust"], j["robust"])
        tau_fr = j["tau_robust"]
    log(f"{name}: scored {len(Xk):,} of {len(X):,} pairs")
    code = pd.Index(Xk.s1_id.unique()).get_indexer(Xk.s1_id)
    if variant == "ensemble_ef":
        from sklearn.isotonic import IsotonicRegression
        f = pd.read_parquet(OUT / "X_train.parquet", columns=["label"])
        qo = np.load(OUT / "oof_q.npy")[-1]
        sub = np.arange(0, len(qo), 7)
        qc = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(qo[sub], f.label.to_numpy()[sub]).predict(q)
        lam, floor = j["ef_pick"]
        pred = decide_ef(code, Xk.cand_id.to_numpy(), qc.astype(np.float32), code.max() + 1, lam, floor)
    else:
        tau = j["taus"]["v11 ensemble (4 LightGBMs)" if variant != "full" else "v11 (v10a + acronym)"]
        tau = float(os.environ.get("TAU", tau))
        tau_row = np.where((Xk.country == "France").to_numpy(), tau_fr, tau) if variant == "ensemble_frrobust" else tau
        pred = owner_mask(Xk.cand_id.to_numpy(), q) & (q >= tau_row)
        log(f"cut-off {tau}" + (f", France {tau_fr}" if variant == "ensemble_frrobust" else ""))
    pr = Xk[pred]
    pred_d = pr.groupby("s1_id").cand_id.agg(set).to_dict()
    cands = X.groupby("s1_id").cand_id.agg(set).to_dict()
    if os.environ.get("FR_ADDR") == "1":
        fa = france_addr_pairs(fam)
        taken = set(pr.cand_id)
        add = fa[~fa.cand_id.isin(taken)]
        n_new_cand = 0
        for s_, c_ in zip(add.s1_id, add.cand_id):
            pred_d.setdefault(s_, set()).add(c_)
            if c_ not in cands.setdefault(s_, set()):
                cands[s_].add(c_)
                n_new_cand += 1
        log(f"France same-address rule: {len(fa):,} pairs, already accepted {int(fa.cand_id.isin(taken).sum()):,}, "
            f"added {len(add):,} matches ({n_new_cand:,} were outside the candidate list) for {add.s1_id.nunique():,} S1s")
    rec = pd.read_parquet(G["records_path"](cfg, "test"), columns=["entity_id", "source", "country"])
    s1_all = rec.entity_id[rec.source == 1].tolist()
    out = Path(f"artifacts/submissions/{name}")
    G["write_submission"](s1_all, pred_d, cands, out)
    n = pd.Series({s: len(v) for s, v in pred_d.items()}).reindex(s1_all, fill_value=0)
    cc = rec.set_index("entity_id").country.reindex(n.index).to_numpy()
    log({"name": name, "variant": variant, "mean_matches": round(float(n.mean()), 4),
         **{k: round(float(n[cc == k].mean()), 4) for k in ["US", "India", "France"]},
         **{f"empty_{k}": round(float((n[cc == k] == 0).mean()), 4) for k in ["US", "India", "France"]}})
    v = subprocess.run(["python3", cfg["paths"]["validator"], "--matching", str(out / "matching_results.tsv"),
                        "--candidate", str(out / "candidate_pairs.tsv"), "--test-dir",
                        str(Path(cfg["paths"]["data_dir"]) / "test")], capture_output=True, text=True)
    print(v.stdout[-400:], f"validator exit={v.returncode}", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "scores":
        scores()
    elif cmd == "feats":
        feats(sys.argv[2])
    elif cmd == "shift":
        shift()
    elif cmd == "train":
        train()
    elif cmd == "robust":
        robust()
    elif cmd == "submit":
        submit(sys.argv[2], sys.argv[3], sys.argv[4])
