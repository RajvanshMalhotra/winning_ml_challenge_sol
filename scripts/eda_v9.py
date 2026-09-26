"""EDA of the v9 stacker vs v8 (OOF): what v9 fixed, what it newly broke, and what is still left.
Reads cand_side/oof_q.parquet (v8 features + q), cand_side_v9/{oof_q9,feats_train}.parquet; writes
artifacts/v2/eda_v9/report.md.   usage: python -u scripts/eda_v9.py"""
import json
import time

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

from ber.config import load_config, run_dir
from ber.io import read_truth

cfg = load_config("configs/v2.yaml")
RD = run_dir(cfg)
OUT = RD / "eda_v9"
OUT.mkdir(exist_ok=True)
T0 = time.perf_counter()
log = lambda *a: print(f"[{time.perf_counter() - T0:6.0f}s]", *a, flush=True)
md = []
say = lambda s="": (md.append(s), print(s, flush=True))
V9 = ["a_crank", "a_cmargin", "a_ncomp", "a_nhi", "a_share", "s1_nconf", "sib_name_max", "addr_sim"]


def tbl(df: pd.DataFrame) -> str:
    df = df.reset_index() if not isinstance(df.index, pd.RangeIndex) else df
    rows = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    for r in df.itertuples(index=False):
        rows.append("| " + " | ".join(f"{v:,.3f}" if isinstance(v, float) else (f"{v:,}" if isinstance(v, (int, np.integer)) else str(v)) for v in r) + " |")
    return "\n".join(rows)


def mask(f: pd.DataFrame, col: str, tau: float) -> np.ndarray:
    s = f.loc[f[col] >= tau, ["cand_id", col]]
    m = np.zeros(len(f), bool)
    m[s.groupby("cand_id")[col].idxmax().to_numpy()] = True
    return m


def per_s1(f, pred, ntrue):
    g = pd.DataFrame({"s1_id": f.s1_id, "tp": pred & f.label.to_numpy().astype(bool), "np": pred}).groupby("s1_id")[["tp", "np"]].sum()
    g = g.reindex(ntrue.index, fill_value=0)
    nt = ntrue.to_numpy()
    return pd.Series(np.where(nt == 0, (g.np.to_numpy() == 0).astype(float), 1.25 * g.tp / np.maximum(0.25 * nt + g.np, 1e-9)), index=ntrue.index)


f = pd.read_parquet(RD / "cand_side" / "oof_q.parquet")
q9 = pd.read_parquet(RD / "cand_side_v9" / "oof_q9.parquet", columns=["s1_id", "cand_id", "q9"])
assert (q9.s1_id.to_numpy() == f.s1_id.to_numpy()).all()
f["q9"] = q9.q9.to_numpy()
v = pd.read_parquet(RD / "cand_side_v9" / "feats_train.parquet", columns=V9)
for c in V9:
    f[c] = v[c].to_numpy()
del v, q9
tau9 = json.loads((RD / "cand_side_v9" / "best.json").read_text())["tau"]
lab = f.label.to_numpy().astype(bool)
p8, p9 = mask(f, "q", 0.75), mask(f, "q9", tau9)
truth = read_truth(cfg["paths"]["data_dir"])
ntrue = pd.Series({s: len(truth[s]) for s in f.s1_id.unique()})
sp = pd.read_parquet(RD / "splits.parquet", columns=["s1_id", "country"]).set_index("s1_id").country
log(f"loaded; v9 tau {tau9}")

grp = np.select([lab & ~p8 & p9, lab & p8 & ~p9, lab & ~p8 & ~p9, ~lab & ~p8 & p9, ~lab & p8 & ~p9, ~lab & p8 & p9],
                ["fixed (v8 miss -> v9 found)", "newly lost (v8 found -> v9 miss)", "still missed", "new wrong accept",
                 "wrong accept removed", "still wrong"], "")
keep = grp != ""
d = f[keep].copy()
d["grp"] = grp[keep]
rec = pd.read_parquet(RD / "records_train.parquet", columns=["entity_id", "name_raw", "addr_raw", "name_norm", "name_core",
                                                              "addr_norm", "house_no"]).set_index("entity_id")
a, c = rec.reindex(d.s1_id.to_numpy()), rec.reindex(d.cand_id.to_numpy())
d["country"] = sp.reindex(d.s1_id).to_numpy()
d["native"] = c.name_raw.str.contains("[ऀ-෿]", regex=True).to_numpy()
d["name_sim"] = [fuzz.token_set_ratio(x, y) for x, y in zip(a.name_norm.fillna(""), c.name_norm.fillna(""))]
d["shared_words"] = [len(set(x.split()) & set(y.split())) for x, y in zip(a.name_core.fillna(""), c.name_core.fillna(""))]
ha, hc = a.house_no.fillna("").to_numpy(), c.house_no.fillna("").to_numpy()
typo = np.array([(x != "") & (y != "") & (x != y) and Levenshtein.distance(x, y) == 1 for x, y in zip(ha, hc)])
d["cause"] = np.select([d.cand_noaddr.to_numpy() == 1, d.native.to_numpy(), d.shared_words.to_numpy() == 0, typo],
                       ["no address", "Indian script name", "names share no word", "house no. 1-digit typo"], "other / similar text")
d["same_name_s1s"] = pd.cut(d.s1_name_cnt, [-1, 0, 1, 2, 5, 10_000], labels=["0", "1 (unique)", "2", "3-5", "6+"]).astype(str)
d["zone"] = pd.cut(d.p, [-1, 0.01, 0.99, 2], labels=["A<0.01", "band", "A>0.99"]).astype(str)
d["q9_bin"] = pd.cut(d.q9, [-0.001, 0.05, 0.2, 0.5, tau9, 1.0], labels=["<0.05", "0.05-0.2", "0.2-0.5", f"0.5-{tau9}", f">={tau9}"]).astype(str)
log("described")

say("# v9 vs v8: what changed and what is left (OOF, 150k S1)")
say()
sc8, sc9 = per_s1(f, p8, ntrue), per_s1(f, p9, ntrue)
say(f"- macro F0.5 v8 {sc8.mean():.4f} → v9 {sc9.mean():.4f}; S1s improved {int((sc9 > sc8 + 1e-9).sum()):,}, "
    f"worse {int((sc9 < sc8 - 1e-9).sum()):,}, S1s still not perfect {int((sc9 < 1).sum()):,}")
say()
say(tbl(d.grp.value_counts().rename("pairs").rename_axis("group")))
say()
order = ["fixed (v8 miss -> v9 found)", "newly lost (v8 found -> v9 miss)", "still missed", "new wrong accept", "wrong accept removed", "still wrong"]
say("## Cause x group (pairs)")
say()
say(tbl(pd.crosstab(d.cause, d.grp).reindex(columns=order, fill_value=0)))
say()
for col in ["country", "same_name_s1s", "zone", "q9_bin"]:
    say(f"## {col} x group (share within group)")
    say()
    say(tbl(pd.crosstab(d[col], d.grp, normalize="columns").reindex(columns=order, fill_value=0).round(3)))
    say()
say("## Medians per group")
say()
num = ["p", "rr", "q", "q9", "a_share", "a_cmargin", "a_crank", "a_nhi", "a_ncomp", "d_crank", "d_ncomp", "s1_name_cnt", "name_sim", "s1_nconf"]
say(tbl(d.groupby("grp")[num].median().reindex(order).T.rename_axis("feature")))
say()
sm = d[d.grp == "still missed"]
say("## Still missed: is this S1 the top choice for the record by Model A (a_crank)?")
say()
say(tbl(pd.crosstab(pd.cut(sm.a_crank, [0, 1, 2, 5, 1e6], labels=["1 (top)", "2", "3-5", "6+"]).astype(str), sm.cause, margins=True)))
say()
top = sm[sm.a_crank == 1]
say(f"- still missed but ranked #1 among all S1s by Model A: {len(top):,} (median a_share {top.a_share.median():.3f}, "
    f"median p {top.p.median():.3f}, median q9 {top.q9.median():.3f})")
say(f"- still missed and another S1 ranks higher: {int((sm.a_crank > 1).sum()):,} — the record looks more like a different business")
say()
# loss split
fnS = f[lab & ~p9].groupby("s1_id").size().reindex(ntrue.index, fill_value=0)
fpS = f[~lab & p9].groupby("s1_id").size().reindex(ntrue.index, fill_value=0)
kind = np.select([(fnS > 0) & (fpS > 0), fpS > 0, fnS > 0], ["missed + wrong", "wrong only", "missed only"], "perfect or blocking only")
L = pd.DataFrame({"kind": kind, "loss": 1 - sc9}).groupby("kind").agg(S1=("loss", "size"), loss=("loss", "sum"))
L["F0.5 pts"] = (L.loss / len(sc9)).round(4)
L["share"] = (L.loss / L.loss.sum()).round(3)
say("## Where v9's F0.5 loss sits")
say()
say(tbl(L.drop(columns="loss").rename_axis("S1 outcome")))
say()
say("## Examples")
for g in order:
    x = d[d.grp == g]
    say()
    say(f"### {g}")
    for r in x.sample(min(5, len(x)), random_state=2).itertuples():
        s, k = rec.loc[r.s1_id], rec.loc[r.cand_id]
        say(f"- [{r.cause}] p={r.p:.3f} q8={r.q:.3f} q9={r.q9:.3f} a_crank={r.a_crank} a_share={r.a_share:.2f} same-name S1s={r.s1_name_cnt}  \n"
            f"  S1 `{s.name_raw}` | `{s.addr_raw}`  \n  {r.cand_id[:2]} `{k.name_raw}` | `{k.addr_raw}`")
(OUT / "report.md").write_text("\n".join(md) + "\n")
d.drop(columns=["native"]).to_parquet(OUT / "v9_changes.parquet", index=False)
log(f"wrote {OUT}/report.md")
