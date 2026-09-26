"""v9 stacker: v8 (v5 stack + candidate-side) plus features that look at ALL S1s through Model A's score p.
Motivation (docs/results/eda_leftover.md): 75% of v8's missed pairs are no-address S2/S3 records wanted by ~33 S1s;
the dense/TF-IDF competition features only see raw similarity, not Model A's full judgement.
New features, computed over every S1 of the family (train or test) so they mean the same in both:
  a_crank, a_cmargin, a_ncomp : rank / margin / #S1s of this S1 among all S1s having the candidate, by Model A p
  a_nhi, a_share              : #OTHER S1s with p >= 0.5 for the candidate; p / sum of p over all S1s for it
  s1_nconf                    : #other records this S1 already matches confidently (p >= 0.9)
  sib_name_max, sib_crank, sib_cmargin : best name similarity of the candidate to this S1's confident matches
                                (cluster support), and how this S1 ranks on it among the competing S1s
  addr_sim, addr_crank, addr_cmargin   : address similarity and its competition (for gibberish names)
  s1_addr_cnt, cand_addr_s1cnt         : #S1s sharing this S1's exact address / the candidate's exact address
Train uses Model A's OOF p for the 150k sample S1s and the fold-model mean for all other train S1s (never trained on);
test uses the fold-model mean everywhere.
usage: cand_side_v9.py feats train|test | train | submit"""
import json
import os
import subprocess
import sys
from pathlib import Path

ns = {"__name__": "cs"}
exec(open("scripts/cand_side.py").read().rsplit("\n{", 1)[0], ns)
import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

from ber.io import read_truth
from ber.matcher import decide, evaluate, md_table
from ber.normalize import records_path
from ber.predict import write_submission

cfg, RD, log = ns["cfg"], ns["RD"], ns["log"]
Records, comp_table, lookup, fnum, stack_features, add_new, comp_tables = (
    ns[k] for k in ["Records", "comp_table", "lookup", "fnum", "stack_features", "add_new", "comp_tables"])
BASE, NEW, PARAMS = ns["BASE"], ns["NEW"], ns["PARAMS"]
OUT = RD / os.environ.get("CS9_DIR", "cand_side_v9")
SUB = os.environ.get("SUB", "v9")
OUT.mkdir(exist_ok=True)
V9 = ["a_crank", "a_cmargin", "a_ncomp", "a_nhi", "a_share", "s1_nconf", "sib_name_max", "sib_crank", "sib_cmargin",
      "addr_sim", "addr_crank", "addr_cmargin", "s1_addr_cnt", "cand_addr_s1cnt"]
SCORES = {"train": RD / "train_scores_matcher_v3.parquet", "test": RD / ns["TEST_SCORES"]}
LO, CONF = 0.01, 0.9


def all_pairs(fam: str, R) -> tuple:
    """Codes + Model A p for every candidate pair of the family; on train, OOF p replaces it for the sample S1s.
    Returns (s, c, p, focus): focus = row index of the pairs the stacker works on, in its row order."""
    t = pq.read_table(SCORES[fam], columns=["s1_id", "cand_id", "p"])
    s, c, p = R.codes(t["s1_id"]), R.codes(t["cand_id"]), np.array(fnum(t["p"]), dtype=np.float32)
    del t
    log(f"{fam}: {len(s):,} scored pairs")
    if fam == "test":
        return s, c, p, np.arange(len(s))
    o = pq.read_table(RD / ns["A_DIR"] / "oof.parquet", columns=["s1_id", "cand_id", "p"])
    ok = (R.codes(o["s1_id"]) << 24) | R.codes(o["cand_id"])
    op = fnum(o["p"])
    key = (s << 24) | c
    order = np.argsort(key, kind="stable")
    i = np.minimum(np.searchsorted(key, ok, sorter=order), len(key) - 1)
    hit = key[order[i]] == ok
    log(f"OOF pairs found among all train pairs: {hit.mean():.4%} ({(~hit).sum():,} missing)")
    if not hit.all():  # a handful of OOF pairs not re-scored: append them so every OOF row has features
        extra = np.flatnonzero(~hit)
        s = np.r_[s, ok[extra] >> 24]
        c = np.r_[c, ok[extra] & ((1 << 24) - 1)]
        p = np.r_[p, op[extra]]
        focus = np.empty(len(ok), np.int64)
        focus[hit] = order[i[hit]]
        focus[extra] = len(key) + np.arange(len(extra))
    else:
        focus = order[i]
    d = np.abs(p[focus] - op)
    log(f"OOF p vs fold-mean p on sample pairs: mean |diff| {d.mean():.4f}")
    p[focus] = op
    return s, c, p, focus


def v9_features(fam: str) -> pd.DataFrame:
    R = Records(fam)
    rec = pd.read_parquet(records_path(cfg, fam), columns=["entity_id", "source", "country", "name_norm", "addr_norm"])
    name = rec.name_norm.fillna("").to_numpy(object)
    addr = rec.addr_norm.fillna("").to_numpy(object)
    akey = pd.factorize(rec.country.astype(str) + "\x00" + rec.addr_norm.fillna("").astype(str))[0]
    s1rec = rec.source.to_numpy() == 1
    acnt = np.bincount(akey[s1rec & (addr != "")], minlength=akey.max() + 1).astype(np.int32)
    del rec
    s, c, p, focus = all_pairs(fam, R)
    n = len(s)
    out = {}
    # 1. competition by Model A p
    tab = comp_table(s, c, p)
    key = (s << 24) | c
    out["a_crank"], out["a_cmargin"], out["a_ncomp"] = lookup(tab, key[focus])
    del tab
    nrec = len(R.noaddr)
    hi = np.bincount(c[p >= 0.5], minlength=nrec)
    psum = np.bincount(c, weights=p, minlength=nrec)
    out["a_nhi"] = (hi[c[focus]] - (p[focus] >= 0.5)).astype(np.int16)
    out["a_share"] = (p[focus] / np.maximum(psum[c[focus]], 1e-6)).astype(np.float32)
    log("A-competition done")
    # 2. cluster support: similarity of the candidate's name to this S1's other confident matches
    conf = p >= CONF
    nconf = np.bincount(s[conf], minlength=nrec)
    out["s1_nconf"] = (nconf[s[focus]] - conf[focus]).astype(np.int16)
    m = np.flatnonzero(p >= LO)
    sib = pd.DataFrame({"i": m, "s": s[m], "c": c[m]}).merge(pd.DataFrame({"s": s[conf], "c2": c[conf]}), on="s")
    sib = sib[sib.c.to_numpy() != sib.c2.to_numpy()]
    log(f"cluster support: {len(m):,} pairs with p >= {LO}, {len(sib):,} sibling comparisons")
    sim = cpdist(name[sib.c.to_numpy()], name[sib.c2.to_numpy()], scorer=fuzz.token_set_ratio, workers=-1)
    sib_max = np.full(n, np.nan, np.float32)
    g = pd.Series(sim, index=sib.i.to_numpy()).groupby(level=0).max()
    sib_max[g.index.to_numpy()] = g.to_numpy()
    del sib, sim, g
    out["sib_name_max"] = sib_max[focus]
    tab = comp_table(s[m], c[m], sib_max[m])
    out["sib_crank"], out["sib_cmargin"], _ = lookup(tab, key[focus])
    del tab
    log("cluster support done")
    # 3. address similarity and its competition (pairs with p >= LO only)
    asim = np.full(n, np.nan, np.float32)
    asim[m] = cpdist(addr[s[m]], addr[c[m]], scorer=fuzz.token_set_ratio, workers=-1)
    asim[m[addr[c[m]] == ""]] = np.nan
    out["addr_sim"] = asim[focus]
    tab = comp_table(s[m], c[m], asim[m])
    out["addr_crank"], out["addr_cmargin"], _ = lookup(tab, key[focus])
    del tab
    out["s1_addr_cnt"] = acnt[akey[s[focus]]]
    cf = c[focus]
    out["cand_addr_s1cnt"] = np.where(addr[cf] == "", 0, acnt[akey[cf]]).astype(np.int32)
    log("address features done")
    f = pd.DataFrame(out)[V9]
    if fam == "train":
        f.insert(0, "cand_code", c[focus])
        f.insert(0, "s1_code", s[focus])
    return f


def feats(fam: str) -> None:
    f = v9_features(fam)
    f.to_parquet(OUT / f"feats_{fam}.parquet", index=False)
    log(f"wrote feats_{fam}.parquet {f.shape}")
    print(f[V9].describe().T.round(3).to_string(), flush=True)


def f05_grid(f: pd.DataFrame, preds: dict, truth: dict, s1: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, p in preds.items():
        sc = pd.DataFrame({"s1_id": f.s1_id, "cand_id": f.cand_id, "p": p})
        for tau in np.round(np.arange(0.5, 0.91, 0.05), 2):
            rows.append({"model": name, "tau": float(tau), **evaluate(decide(sc, tau), truth, s1)})
    return pd.DataFrame(rows)


def train() -> None:
    f = pd.read_parquet(RD / "cand_side" / "oof_q.parquet")  # v8 features + fold + v8 OOF score q, row order = matcher oof
    v = pd.read_parquet(OUT / "feats_train.parquet")
    R = Records("train")
    assert len(v) == len(f)
    assert (v.s1_code.to_numpy() == R.codes(pa_str(f.s1_id))).all() and (v.cand_code.to_numpy() == R.codes(pa_str(f.cand_id))).all()
    for col in V9:
        f[col] = v[col].to_numpy()
    del v
    feats9 = BASE + NEW + V9
    q9 = np.zeros(len(f), np.float32)
    imp = []
    for k in sorted(f.fold.unique()):
        tr, va = f.fold.to_numpy() != k, f.fold.to_numpy() == k
        m = lgb.train(PARAMS, lgb.Dataset(f.loc[tr, feats9], f.label[tr]), 3000,
                      valid_sets=[lgb.Dataset(f.loc[va, feats9], f.label[va])], callbacks=[lgb.early_stopping(100, verbose=False)])
        q9[va] = m.predict(f.loc[va, feats9], num_iteration=m.best_iteration)
        m.save_model(str(OUT / f"cs9_fold{k}.txt"))
        imp.append(pd.Series(m.feature_importance("gain"), index=feats9))
        log(f"fold {k} done ({m.best_iteration} rounds)")
    f["q9"] = q9
    f[["s1_id", "cand_id", "label", "fold", "q", "q9"]].to_parquet(OUT / "oof_q9.parquet", index=False)
    truth = read_truth(cfg["paths"]["data_dir"])
    sp = pd.read_parquet(RD / "splits.parquet", columns=["s1_id", "country"]).set_index("s1_id")
    s1 = sp.loc[f.s1_id.unique()].reset_index()[["s1_id", "country"]]
    grid = f05_grid(f, {"v8": f.q.to_numpy(), "v9": q9}, truth, s1)
    best = {n: g.sort_values("macro_f05").iloc[-1] for n, g in grid.groupby("model")}
    b8, b9 = best["v8"], best["v9"]
    (OUT / "best.json").write_text(json.dumps({"tau": float(b9.tau), "macro_f05": float(b9.macro_f05),
                                               "v8_tau": float(b8.tau), "v8_f05": float(b8.macro_f05)}, indent=2))
    # what happened to v8's leftovers
    lab = f.label.to_numpy().astype(bool)
    p8 = mask(f, "q", float(b8.tau))
    p9 = mask(f, "q9", float(b9.tau))
    na = f.cand_noaddr.to_numpy() == 1
    lines = [f"- v8 missed {int((lab & ~p8).sum()):,} → v9 missed {int((lab & ~p9).sum()):,} "
             f"(fixed {int((lab & ~p8 & p9).sum()):,}, newly lost {int((lab & p8 & ~p9).sum()):,})",
             f"- wrong accepts: v8 {int((~lab & p8).sum()):,} → v9 {int((~lab & p9).sum()):,}",
             f"- no-address true pairs found: v8 {(lab & na & p8).sum() / (lab & na).sum():.1%} → v9 {(lab & na & p9).sum() / (lab & na).sum():.1%}; "
             f"no-address precision v8 {(lab & na & p8).sum() / max((na & p8).sum(), 1):.1%} → v9 {(lab & na & p9).sum() / max((na & p9).sum(), 1):.1%}"]
    gi = pd.concat(imp, axis=1).mean(axis=1)
    gi = (gi / gi.sum() * 100).sort_values(ascending=False).round(1)
    fmt = lambda r: (f"τ={r.tau:.2f} → **{r.macro_f05:.4f}** (singletons {r.singletons:.4f}, "
                     + ", ".join(f"{k[4:]} {v:.4f}" for k, v in r.items() if str(k).startswith("f05_")) + ")")
    md = ["# v9 stacker: + Model-A competition, cluster support, address competition — out-of-fold (same folds, same pairs)", "",
          f"- v8: {fmt(b8)}", f"- **v9: {fmt(b9)}**", "", *lines, "",
          "Feature importance (gain %): " + ", ".join(f"{k} {v}" for k, v in gi.items()), "", md_table(grid)]
    Path("docs/results").mkdir(parents=True, exist_ok=True)
    (OUT / "report.md").write_text("\n".join(md) + "\n")
    print("\n".join(md[:9]) + "\n\n" + md[9], flush=True)


def pa_str(x: pd.Series):
    import pyarrow as pa
    return pa.array(x.to_numpy(), type=pa.large_string())


def mask(f: pd.DataFrame, col: str, tau: float) -> np.ndarray:
    s = f.loc[f[col] >= tau, ["cand_id", col]]
    out = np.zeros(len(f), bool)
    out[s.groupby("cand_id")[col].idxmax().to_numpy()] = True
    return out


def submit() -> None:
    best = json.loads((OUT / "best.json").read_text())
    t = pd.read_parquet(SCORES["test"])
    b = pd.read_parquet(ns["RR"] / "test_band_scores.parquet")[["s1_id", "cand_id", "rr"]]
    f = stack_features(t.merge(b, on=["s1_id", "cand_id"], how="left"))
    del t
    R = Records("test")
    f = add_new(f, R, comp_tables("test", R))
    v = pd.read_parquet(OUT / "feats_test.parquet")
    assert len(v) == len(f)
    for col in V9:
        f[col] = v[col].to_numpy()
    del v
    models = [lgb.Booster(model_file=str(p)) for p in sorted(OUT.glob("cs9_fold*.txt"))]
    p = np.mean([m.predict(f[BASE + NEW + V9], num_iteration=m.best_iteration) for m in models], axis=0)
    sc = pd.DataFrame({"s1_id": f.s1_id, "cand_id": f.cand_id, "p": p})
    sc.to_parquet(OUT / "test_q9.parquet", index=False)
    pred = decide(sc, best["tau"])
    rec = pd.read_parquet(records_path(cfg, "test"), columns=["entity_id", "source", "country"])
    s1_all = rec.entity_id[rec.source == 1].tolist()
    out = Path(f"artifacts/submissions/{SUB}")
    write_submission(s1_all, pred, f.groupby("s1_id").cand_id.agg(set).to_dict(), out)
    n = pd.Series({s: len(v) for s, v in pred.items()}).reindex(s1_all, fill_value=0)
    ctry = rec.set_index("entity_id").country.reindex(n.index).to_numpy()
    log({"tau": best["tau"], "mean_matches": float(n.mean()), "empty_share": float((n == 0).mean()),
         **{c: float(n[ctry == c].mean()) for c in ["US", "India", "France"]}})
    v = subprocess.run(["python3", cfg["paths"]["validator"], "--matching", str(out / "matching_results.tsv"),
                        "--candidate", str(out / "candidate_pairs.tsv"), "--test-dir",
                        str(Path(cfg["paths"]["data_dir"]) / "test")], capture_output=True, text=True)
    print(v.stdout[-800:], f"validator exit={v.returncode}", flush=True)


if sys.argv[1] == "feats":
    feats(sys.argv[2])
else:
    {"train": train, "submit": submit}[sys.argv[1]]()
