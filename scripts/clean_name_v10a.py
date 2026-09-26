"""v10a: noise-cleaned names for the pairs the v9 stacker rejects.
Cleaning (same on the S1 query side and the S2/S3 record side): accents, `(ID: 123)` / `#123` tags, web domains,
brackets, look-alike digits inside words (8uilding -> building), honorifics (Shri, Smt, M/s ...), legal suffixes and
generic filler words (Pvt, Ltd, LLC, Co, Services, Group ...).
Features for every pair with Model A p >= 0.01, over ALL S1s of the family (train or test):
  cn_exact, cn_ratio, cn_tset, cn_jw, cn_sim (mean of the three), cn_crank / cn_cmargin (competition by cn_sim among all
  S1s having the record), cn_s1cnt (#S1s in the country whose cleaned name equals the record's cleaned name)
Two ways to use them, both scored on the same OOF folds as v9:
  rule   : among v9-REJECTED pairs (all of them, true and false) accept when v9 score >= t0, this S1 is Model A's #1 for
           the record, cn_sim >= T1 and cn_cmargin >= T2. t0/T1/T2 chosen on 4 folds, scored on the 5th.
  stacker: v9 features + cn_* in the LightGBM stacker (5-fold OOF, same folds).
usage: clean_name_v10a.py feats train|test | eval | submit"""
import json
import os
import re
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

ns = {"__name__": "cs9"}
_V9 = "scripts/cand_side_v9.py" if os.path.exists("scripts/cand_side_v9.py") else os.path.expanduser("~/claude_v9/cand_side_v9.py")
exec(open(_V9).read().rsplit("\nif sys.argv[1] ==", 1)[0], ns)
import lightgbm as lgb
import numpy as np
import pandas as pd
from anyascii import anyascii
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.process import cpdist

from ber.io import read_truth
from ber.matcher import decide, evaluate, md_table
from ber.normalize import records_path
from ber.predict import write_submission

cfg, RD, log = ns["cfg"], ns["RD"], ns["log"]
Records, comp_table, lookup, all_pairs, mask = ns["Records"], ns["comp_table"], ns["lookup"], ns["all_pairs"], ns["mask"]
BASE, NEW, V9, PARAMS = ns["BASE"], ns["NEW"], ns["V9"], ns["PARAMS"]
OUT = RD / "clean_v10a"
OUT.mkdir(exist_ok=True)
SUB = os.environ.get("SUB", "v10a")
CN = ["cn_exact", "cn_ratio", "cn_tset", "cn_jw", "cn_sim", "cn_crank", "cn_cmargin", "cn_s1cnt"]
LO = 0.01

LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b"})
HON = {"shri", "shree", "sri", "smt", "ms", "mr", "mrs", "dr", "the", "m"}   # "m/s" tokenises to "m", "s"
DROP = {"pvt", "private", "ltd", "limited", "llc", "inc", "incorporated", "corp", "corporation", "co", "company", "pc",
        "llp", "plc", "lp", "group", "services", "service", "enterprises", "enterprise", "and", "of", "the"}
_ID = re.compile(r"\(?\bid\s*[:#.]?\s*\d+\)?|#\s*\d+")
_DOM = re.compile(r"\b(?:www\.)?([a-z0-9-]+)\.(?:com|in|net|org|co|biz|info|us)\b")
_TOK = re.compile(r"[a-z0-9]+")


def clean_name(x) -> str:
    x = anyascii(x or "").lower()
    x = _DOM.sub(r" \1 ", _ID.sub(" ", x))
    toks = [t.translate(LEET) if (any(ch.isalpha() for ch in t) and any(ch.isdigit() for ch in t)) else t
            for t in _TOK.findall(x.replace("m/s", " "))]
    while toks and toks[0] in HON:
        toks.pop(0)
    core = [t for t in toks if t not in DROP]
    return " ".join(core or toks)


def _clean_chunk(xs):
    return [clean_name(x) for x in xs]


def clean_all(names: np.ndarray) -> np.ndarray:
    parts = np.array_split(names, 192)
    with Pool(48) as pool:
        out = pool.map(_clean_chunk, parts)
    return np.array([y for p in out for y in p], dtype=object)


def feats(fam: str) -> None:
    R = Records(fam)
    rec = pd.read_parquet(records_path(cfg, fam), columns=["source", "country", "name_raw"])
    cn = clean_all(rec.name_raw.to_numpy(object))
    log(f"{fam}: cleaned {len(cn):,} names")
    key = pd.factorize(rec.country.astype(str).to_numpy() + "\x00" + cn)[0]
    s1rec = rec.source.to_numpy() == 1
    kcnt = np.bincount(key[s1rec], minlength=key.max() + 1).astype(np.int32)
    del rec
    s, c, p, focus = all_pairs(fam, R)
    n, m = len(s), np.flatnonzero(p >= LO)
    a, b = cn[s[m]], cn[c[m]]
    cols = {"cn_exact": (a == b).astype(np.float32),
            "cn_ratio": cpdist(a, b, scorer=fuzz.ratio, workers=-1).astype(np.float32),
            "cn_tset": cpdist(a, b, scorer=fuzz.token_set_ratio, workers=-1).astype(np.float32),
            "cn_jw": (100 * cpdist(a, b, scorer=JaroWinkler.normalized_similarity, workers=-1)).astype(np.float32)}
    cols["cn_sim"] = (cols["cn_ratio"] + cols["cn_tset"] + cols["cn_jw"]) / 3
    log(f"{fam}: similarities for {len(m):,} pairs")
    full = {}
    for k, v in cols.items():
        x = np.full(n, np.nan, np.float32)
        x[m] = v
        full[k] = x[focus]
    tab = comp_table(s[m], c[m], cols["cn_sim"])
    full["cn_crank"], full["cn_cmargin"], _ = lookup(tab, ((s << 24) | c)[focus])
    full["cn_s1cnt"] = kcnt[key[c[focus]]]
    f = pd.DataFrame(full)[CN]
    f.to_parquet(OUT / f"feats_{fam}.parquet", index=False)
    log(f"wrote feats_{fam}.parquet {f.shape}")
    print(f.describe().T.round(3).to_string(), flush=True)


def load_oof() -> pd.DataFrame:
    f = pd.read_parquet(RD / "cand_side" / "oof_q.parquet")
    f["q9"] = pd.read_parquet(RD / "cand_side_v9" / "oof_q9.parquet", columns=["q9"]).q9.to_numpy()
    v = pd.read_parquet(RD / "cand_side_v9" / "feats_train.parquet", columns=V9)
    for k in V9:
        f[k] = v[k].to_numpy()
    v = pd.read_parquet(OUT / "feats_train.parquet")
    for k in CN:
        f[k] = v[k].to_numpy()
    return f


def eval_() -> None:
    f = load_oof()
    tau = json.loads((RD / "cand_side_v9" / "best.json").read_text())["tau"]
    truth = read_truth(cfg["paths"]["data_dir"])
    ntrue = pd.Series({s: len(truth[s]) for s in f.s1_id.unique()})
    fold_of = f.groupby("s1_id").fold.first().reindex(ntrue.index).to_numpy()
    lab = f.label.to_numpy().astype(bool)

    def f05_s1(score):
        sel = score >= tau
        s = pd.DataFrame({"cand_id": f.cand_id.to_numpy()[sel], "sc": score[sel]}, index=np.flatnonzero(sel))
        pred = np.zeros(len(f), bool)
        pred[s.groupby("cand_id").sc.idxmax().to_numpy()] = True
        t = pd.DataFrame({"s1_id": f.s1_id, "tp": pred & lab, "np": pred}).groupby("s1_id")[["tp", "np"]].sum().reindex(ntrue.index, fill_value=0)
        nt = ntrue.to_numpy()
        return np.where(nt == 0, (t.np.to_numpy() == 0).astype(float), 1.25 * t.tp.to_numpy() / np.maximum(0.25 * nt + t.np.to_numpy(), 1e-9)), pred

    q9 = f.q9.to_numpy()
    base, pred9 = f05_s1(q9)
    rej = q9 < tau
    lines = [f"v9 OOF F0.5 {base.mean():.5f} (τ={tau}); rejected pairs: {int(rej.sum()):,} "
             f"(true {int((rej & lab).sum()):,}, false {int((rej & ~lab).sum()):,})"]
    # --- how well does the cleaned name separate true from false among ALL rejected pairs?
    r = f[rej & np.isfinite(f.cn_sim.to_numpy())]
    lines.append(f"rejected pairs with p >= {LO} (have cleaned-name features): {len(r):,}, true share {r.label.mean():.3f}")
    for cond, name in [(r.cn_exact == 1, "cleaned names identical"), ((r.cn_exact == 1) & (r.cn_s1cnt <= 1), "identical AND no other S1 has that cleaned name"),
                       ((r.cn_crank == 1) & (r.cn_cmargin >= 10), "closest cleaned name by >= 10 points"),
                       ((r.cn_sim >= 90) & (r.a_crank == 1), "cn_sim >= 90 and Model A's #1")]:
        lines.append(f"  {name}: {int(cond.sum()):,} pairs, true share {r.label[cond].mean():.3f}")
    # --- rule: nested CV over (t0, T1, T2); every rejected pair is eligible
    a1 = f.a_crank.to_numpy() == 1
    sim = np.nan_to_num(f.cn_sim.to_numpy(), nan=-1)
    mg = np.nan_to_num(f.cn_cmargin.to_numpy(), nan=100.0)   # no competitor -> unlimited margin
    res = {("none",): base}
    for t0 in [0.1, 0.2, 0.3, 0.4, 0.5]:
        for t1 in [85, 90, 95, 100]:
            for t2 in [0, 5, 10, 20]:
                rule = rej & (q9 >= t0) & a1 & (sim >= t1) & (mg >= t2)
                res[(t0, t1, t2)] = f05_s1(np.where(rule, np.maximum(q9, tau), q9))[0]
    log(f"{len(res)} rule variants scored")
    best_all = sorted(res, key=lambda k: -res[k].mean())[:5]
    out, picks = np.zeros(len(ntrue)), []
    for k in range(5):
        tr, te = fold_of != k, fold_of == k
        b = max(res, key=lambda kk: res[kk][tr].mean())
        picks.append(b)
        out[te] = res[b][te]
    lines.append("rule, best on all data: " + ", ".join(f"{k} {res[k].mean():.5f}" for k in best_all))
    lines.append(f"**rule, nested CV: {out.mean():.5f} (gain {out.mean() - base.mean():+.5f}); picks {picks}**")
    # --- stacker with the cleaned-name features
    feats10 = BASE + NEW + V9 + CN
    q10 = np.zeros(len(f), np.float32)
    imp = []
    for k in range(5):
        tr, va = f.fold.to_numpy() != k, f.fold.to_numpy() == k
        mdl = lgb.train(PARAMS, lgb.Dataset(f.loc[tr, feats10], f.label[tr]), 3000,
                        valid_sets=[lgb.Dataset(f.loc[va, feats10], f.label[va])], callbacks=[lgb.early_stopping(100, verbose=False)])
        q10[va] = mdl.predict(f.loc[va, feats10], num_iteration=mdl.best_iteration)
        mdl.save_model(str(OUT / f"cs10_fold{k}.txt"))
        imp.append(pd.Series(mdl.feature_importance("gain"), index=feats10))
        log(f"stacker fold {k} done")
    sp = pd.read_parquet(RD / "splits.parquet", columns=["s1_id", "country"]).set_index("s1_id")
    s1 = sp.loc[f.s1_id.unique()].reset_index()[["s1_id", "country"]]
    rows = []
    for name, p in {"v9": q9, "v10a stacker": q10}.items():
        sc = pd.DataFrame({"s1_id": f.s1_id, "cand_id": f.cand_id, "p": p})
        for t in np.round(np.arange(0.5, 0.91, 0.05), 2):
            rows.append({"model": name, "tau": float(t), **evaluate(decide(sc, t), truth, s1)})
    grid = pd.DataFrame(rows)
    best = {n: g.sort_values("macro_f05").iloc[-1] for n, g in grid.groupby("model")}
    b10 = best["v10a stacker"]
    m10 = mask(f.assign(q10=q10), "q10", float(b10.tau))
    m9 = pred9
    lines.append(f"**stacker v10a: τ={b10.tau:.2f} → {b10.macro_f05:.5f} (v9 {best['v9'].macro_f05:.5f}); "
                 f"misses {int((lab & ~m9).sum()):,} → {int((lab & ~m10).sum()):,}, wrong accepts {int((~lab & m9).sum()):,} → {int((~lab & m10).sum()):,}**")
    gi = pd.concat(imp, axis=1).mean(axis=1)
    gi = (gi / gi.sum() * 100).sort_values(ascending=False).round(1)
    lines.append("stacker gain %: " + ", ".join(f"{k} {v}" for k, v in gi.head(12).items()))
    choice = {"rule": {"gain": float(out.mean() - base.mean()), "params": list(max(res, key=lambda kk: res[kk].mean()))},
              "stacker": {"gain": float(b10.macro_f05 - best["v9"].macro_f05), "tau": float(b10.tau)}}
    (OUT / "eval.json").write_text(json.dumps(choice, indent=2, default=str))
    f[["s1_id", "cand_id"]].assign(q10=q10).to_parquet(OUT / "oof_q10.parquet", index=False)
    md = ["# v10a: noise-cleaned names on the pairs v9 rejects (OOF, same folds)", ""] + [f"- {x}" for x in lines] + ["", md_table(grid)]
    (OUT / "report.md").write_text("\n".join(md) + "\n")
    print("\n".join(md[:len(lines) + 2]), flush=True)


def submit() -> None:
    ch = json.loads((OUT / "eval.json").read_text())
    use = "stacker" if ch["stacker"]["gain"] >= ch["rule"]["gain"] else "rule"
    assert max(ch["stacker"]["gain"], ch["rule"]["gain"]) > 0, "no gain on OOF; not submitting"
    t = pd.read_parquet(ns["SCORES"]["test"])
    b = pd.read_parquet(ns["ns"]["RR"] / "test_band_scores.parquet")[["s1_id", "cand_id", "rr"]]
    f = ns["stack_features"](t.merge(b, on=["s1_id", "cand_id"], how="left"))
    del t
    R = Records("test")
    f = ns["add_new"](f, R, ns["comp_tables"]("test", R))
    for path, cols in [(RD / "cand_side_v9" / "feats_test.parquet", V9), (OUT / "feats_test.parquet", CN)]:
        v = pd.read_parquet(path, columns=cols)
        assert len(v) == len(f)
        for k in cols:
            f[k] = v[k].to_numpy()
        del v
    if use == "stacker":
        models = [lgb.Booster(model_file=str(p)) for p in sorted(OUT.glob("cs10_fold*.txt"))]
        p = np.mean([m.predict(f[BASE + NEW + V9 + CN], num_iteration=m.best_iteration) for m in models], axis=0)
        tau = ch["stacker"]["tau"]
    else:
        tau = json.loads((RD / "cand_side_v9" / "best.json").read_text())["tau"]
        q9 = pd.read_parquet(RD / "cand_side_v9" / "test_q9.parquet", columns=["p"]).p.to_numpy()
        t0, t1, t2 = ch["rule"]["params"]
        rule = (q9 < tau) & (q9 >= t0) & (f.a_crank.to_numpy() == 1) & (np.nan_to_num(f.cn_sim.to_numpy(), nan=-1) >= t1) \
            & (np.nan_to_num(f.cn_cmargin.to_numpy(), nan=100.0) >= t2)
        log(f"rule accepts {int(rule.sum()):,} extra test pairs")
        p = np.where(rule, np.maximum(q9, tau), q9)
    sc = pd.DataFrame({"s1_id": f.s1_id, "cand_id": f.cand_id, "p": p})
    pred = decide(sc, tau)
    rec = pd.read_parquet(records_path(cfg, "test"), columns=["entity_id", "source", "country"])
    s1_all = rec.entity_id[rec.source == 1].tolist()
    out = Path(f"artifacts/submissions/{SUB}")
    write_submission(s1_all, pred, f.groupby("s1_id").cand_id.agg(set).to_dict(), out)
    n = pd.Series({s: len(v) for s, v in pred.items()}).reindex(s1_all, fill_value=0)
    cc = rec.set_index("entity_id").country.reindex(n.index).to_numpy()
    log({"used": use, "tau": tau, "mean_matches": float(n.mean()), **{k: round(float(n[cc == k].mean()), 4) for k in ["US", "India", "France"]}})
    v = subprocess.run(["python3", cfg["paths"]["validator"], "--matching", str(out / "matching_results.tsv"),
                        "--candidate", str(out / "candidate_pairs.tsv"), "--test-dir",
                        str(Path(cfg["paths"]["data_dir"]) / "test")], capture_output=True, text=True)
    print(v.stdout[-800:], f"validator exit={v.returncode}", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "feats":
        feats(sys.argv[2])
    elif sys.argv[1] == "test_clean":
        for x in ["Asset 8uilding Institute Inc.", "Martinez & (Gallagher)", "XA Private Tax Limited (ID: 68674)", "Shri Alankar Co",
                  "Smt Elite Gifting Pvt-Ltd", "rmiestates.com", "Taylor &  Duan LLC #4453", "M/s Mumbai Sons", "Clínic 3378 Lincoln Street"]:
            print(f"{x!r:45} -> {clean_name(x)!r}")
    else:
        {"eval": eval_, "submit": submit}[sys.argv[1]]()
