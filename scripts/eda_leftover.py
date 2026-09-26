"""What is left after each stage? Traces every OOF true pair and every wrong accept through
Model A (tau 0.70) -> A + bge reranker (v5 stacker, tau 0.65) -> v8 candidate-side stacker (tau 0.75),
then profiles what v8 still gets wrong. Read-only on existing artifacts; writes artifacts/v2/eda_leftover/.
usage: python -u scripts/eda_leftover.py"""
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

from ber.config import load_config, run_dir
from ber.io import read_truth

cfg = load_config("configs/v2.yaml")
RD = run_dir(cfg)
OUT = RD / "eda_leftover"
OUT.mkdir(exist_ok=True)
T0 = time.perf_counter()
log = lambda *a: print(f"[{time.perf_counter() - T0:6.0f}s]", *a, flush=True)
BASE = ["p", "rr", "has_rr", "rr_rank", "rr_gap", "p_rank", "p_gap", "n_cands"]
STAGES = [("A", "p", 0.70), ("v5", "v5", 0.65), ("v8", "q", 0.75)]
md = []
say = lambda s="": (md.append(s), print(s, flush=True))


def decide_mask(f: pd.DataFrame, col: str, tau: float) -> np.ndarray:
    """Same rule as ber.matcher.decide (threshold, then each candidate to its best S1) as a row mask."""
    s = f.loc[f[col] >= tau, ["cand_id", col]]
    win = s.groupby("cand_id")[col].idxmax().to_numpy()
    m = np.zeros(len(f), bool)
    m[win] = True
    return m


def per_s1(f: pd.DataFrame, pred: np.ndarray, ntrue: pd.Series) -> pd.Series:
    g = pd.DataFrame({"s1_id": f.s1_id, "tp": pred & f.label.to_numpy(), "np": pred}).groupby("s1_id")[["tp", "np"]].sum()
    g = g.reindex(ntrue.index, fill_value=0)
    nt = ntrue.to_numpy()
    sc = np.where(nt == 0, (g.np.to_numpy() == 0).astype(float), 1.25 * g.tp / np.maximum(0.25 * nt + g.np, 1e-9))
    return pd.Series(sc, index=ntrue.index)


def tbl(df: pd.DataFrame) -> str:
    df = df.reset_index() if not isinstance(df.index, pd.RangeIndex) else df
    cols = [str(c) for c in df.columns]
    rows = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        rows.append("| " + " | ".join(f"{v:,.4f}" if isinstance(v, float) else (f"{v:,}" if isinstance(v, (int, np.integer)) else str(v)) for v in r) + " |")
    return "\n".join(rows)


def share(s: pd.Series) -> pd.DataFrame:
    c = s.value_counts()
    return pd.DataFrame({"pairs": c, "share": (c / c.sum()).round(3)})


# ---------------------------------------------------------------- load
f = pd.read_parquet(RD / "cand_side" / "oof_q.parquet")
log(f"OOF pairs {len(f):,}")
v5 = np.zeros(len(f), np.float32)
for k in range(5):
    m = lgb.Booster(model_file=str(RD / "reranker_v3" / f"stack_fold{k}.txt"))
    va = f.fold.to_numpy() == k
    v5[va] = m.predict(f.loc[va, BASE], num_iteration=m.best_iteration)
f["v5"] = v5
truth = read_truth(cfg["paths"]["data_dir"])
s1s = f.s1_id.unique()
ntrue = pd.Series({s: len(truth[s]) for s in s1s})
sp = pd.read_parquet(RD / "splits.parquet", columns=["s1_id", "country"]).set_index("s1_id").country
lab = f.label.to_numpy().astype(bool)
for name, col, tau in STAGES:
    f[f"pred_{name}"] = decide_mask(f, col, tau)
    f[f"above_{name}"] = f[col].to_numpy() >= tau
log("decisions done")

# ---------------------------------------------------------------- 1. what is left after each stage
pool = set(zip(f.s1_id[lab], f.cand_id[lab]))
n_block = int(ntrue.sum()) - int(lab.sum())
say("# What is left after each stage (OOF, B split, 150k S1)")
say()
say(f"- S1 businesses: {len(s1s):,} ({int((ntrue == 0).sum()):,} singletons); candidate pairs: {len(f):,}")
say(f"- True pairs: {int(ntrue.sum()):,}; in the shortlist: {int(lab.sum()):,}; **lost at blocking: {n_block:,}**")
say()
rows = []
for name, col, tau in STAGES:
    pr = f[f"pred_{name}"].to_numpy()
    sc = per_s1(f, pr, ntrue)
    fn = lab & ~pr
    rows.append({"stage": name, "tau": tau, "macro_F0.5": sc.mean(), "found": int((lab & pr).sum()),
                 "missed (shortlist)": int(fn.sum()), "  of which above tau but lost 1-owner": int((fn & f[f"above_{name}"].to_numpy()).sum()),
                 "wrong accepts": int((~lab & pr).sum()), "S1 not perfect": int((sc < 1).sum()),
                 "singletons wrong": int(((sc < 1) & (ntrue == 0)).sum())})
say(tbl(pd.DataFrame(rows)))
say()

# transitions: how each true pair / wrong accept moves through the stages
pa_, p5, p8 = f.pred_A.to_numpy(), f.pred_v5.to_numpy(), f.pred_v8.to_numpy()
path = lambda m: pd.Series(np.char.add(np.char.add(np.where(pa_[m], "Y", "n"), np.where(p5[m], "Y", "n")), np.where(p8[m], "Y", "n")))
say("## How pairs move through the stages (A / v5 / v8; Y = accepted)")
say()
tp_path = path(lab).value_counts().rename("true pairs")
fp_path = path(~lab & (pa_ | p5 | p8)).value_counts().rename("wrong pairs (accepted at least once)")
say(tbl(pd.concat([tp_path, fp_path], axis=1).fillna(0).astype(int).rename_axis("A/v5/v8")))
say()
say("Read: `nnY` = missed by A and v5, fixed by v8; `YYn` = found by A and v5, lost by v8.")
say()

# ---------------------------------------------------------------- 2. profile the v8 leftovers
rec = pd.read_parquet(RD / "records_train.parquet", columns=["entity_id", "name_raw", "addr_raw", "name_norm", "name_core",
                                                              "addr_norm", "house_no", "postal", "state"]).set_index("entity_id")
owner = {c: s for s, m in truth.items() for c in m}
fn = f[lab & ~p8].copy()
fp = f[~lab & p8].copy()
tp = f[lab & p8].sample(min(100_000, int((lab & p8).sum())), random_state=0).copy()
win = f.loc[p8, ["cand_id", "s1_id"]].set_index("cand_id").s1_id  # who got each accepted candidate


def describe(d: pd.DataFrame) -> pd.DataFrame:
    a, c = rec.reindex(d.s1_id.to_numpy()), rec.reindex(d.cand_id.to_numpy())
    d["country"] = sp.reindex(d.s1_id).to_numpy()
    d["src"] = d.cand_id.str[:2].to_numpy()
    d["native_script"] = c.name_raw.str.contains("[ऀ-෿]", regex=True).to_numpy()
    d["name_sim"] = [fuzz.token_set_ratio(x, y) for x, y in zip(a.name_norm.fillna(""), c.name_norm.fillna(""))]
    d["addr_sim"] = [fuzz.token_set_ratio(x, y) for x, y in zip(a.addr_norm.fillna(""), c.addr_norm.fillna(""))]
    na = set
    d["name_shared_words"] = [len(na(x.split()) & na(y.split())) for x, y in zip(a.name_core.fillna(""), c.name_core.fillna(""))]
    ha, hc = a.house_no.fillna("").to_numpy(), c.house_no.fillna("").to_numpy()
    both = (ha != "") & (hc != "")
    d["house"] = np.select([~both, ha == hc, np.array([Levenshtein.distance(x, y) == 1 for x, y in zip(ha, hc)])],
                           ["missing on one side", "same", "1-digit typo"], "different")
    d["postal_same"] = np.where((a.postal.fillna("").to_numpy() != "") & (c.postal.fillna("").to_numpy() != ""),
                                np.where(a.postal.to_numpy() == c.postal.to_numpy(), "same", "different"), "missing")
    d["zone"] = pd.cut(d.p, [-1, 0.01, 0.99, 2], labels=["A<0.01 (not reranked)", "band (reranked)", "A>0.99"]).astype(str)
    d["cluster_size"] = pd.cut(ntrue.reindex(d.s1_id).to_numpy(), [-1, 0, 1, 3, 6, 1000], labels=["0", "1", "2-3", "4-6", "7+"]).astype(str)
    return d


fn, fp, tp = describe(fn), describe(fp), describe(tp)
log("descriptions done")


def cause_fn(d: pd.DataFrame) -> np.ndarray:
    return np.select([d.above_v8.to_numpy(), d.cand_noaddr.to_numpy() == 1, d.native_script.to_numpy(),
                      d.name_shared_words.to_numpy() == 0, d.house.to_numpy() == "1-digit typo", d.addr_sim.to_numpy() < 50],
                     ["lost 1-owner to another S1", "no address", "Indian script name", "names share no word",
                      "house no. 1-digit typo", "address very different"], "similar text, still rejected")


fn["cause"] = cause_fn(fn)
fp["belongs_to"] = np.where(fp.cand_id.map(owner).isna(), "no S1 (unmatched record)", "another S1")
fp["s1_singleton"] = (ntrue.reindex(fp.s1_id).to_numpy() == 0)
fp["cause"] = np.select([fp.cand_noaddr.to_numpy() == 1, fp.name_shared_words.to_numpy() == 0, fp.native_script.to_numpy(),
                         fp.house.to_numpy() == "different", fp.house.to_numpy() == "1-digit typo"],
                        ["no address", "names share no word", "Indian script name", "house no. different", "house no. 1-digit off"],
                        "same/similar name + address")

say("# What v8 still gets wrong")
say()
say(f"Missed true pairs in the shortlist: **{len(fn):,}**; wrong accepts: **{len(fp):,}**; lost at blocking: **{n_block:,}**")
say()
say("## Missed true pairs, by main cause (first matching rule wins)")
say()
say(tbl(share(pd.Series(fn.cause)).rename_axis("cause")))
say()
say("## Missed pairs: cause x Model A zone")
say()
say(tbl(pd.crosstab(fn.cause, fn.zone, margins=True)))
say()
say("## Missed vs found (sampled 100k found), side by side")
say()
cmp = lambda col: pd.concat([share(fn[col]).share.rename("missed"), share(tp[col]).share.rename("found")], axis=1).fillna(0)
for col in ["country", "src", "cand_noaddr", "native_script", "house", "postal_same", "cluster_size", "zone"]:
    say(f"**{col}**")
    say()
    say(tbl(cmp(col).rename_axis(col)))
    say()
num = ["p", "rr", "q", "name_sim", "addr_sim", "name_shared_words", "s1_name_cnt", "d_crank", "d_ncomp", "n_cands", "p_rank"]
say("**Medians**")
say()
say(tbl(pd.DataFrame({"missed": fn[num].median(), "found": tp[num].median(), "wrong accept": fp[num].median()}).rename_axis("feature")))
say()
say("## Missed pairs: final score q")
say()
say(tbl(share(pd.cut(fn.q, [0, 0.05, 0.2, 0.5, 0.75, 1.0], include_lowest=True).astype(str)).sort_index().rename_axis("q bin")))
say()
fn["s1_found_any"] = fn.s1_id.map(f[lab & p8].groupby("s1_id").size()).fillna(0).gt(0)
say(f"- Missed pairs whose S1 found **none** of its matches: {int((~fn.s1_found_any).sum()):,}; S1 found some: {int(fn.s1_found_any.sum()):,}")
lost = fn[fn.cause == "lost 1-owner to another S1"]
if len(lost):
    w = win.reindex(lost.cand_id).to_numpy()
    same = [rec.name_core.get(x) == rec.name_core.get(y) for x, y in zip(lost.s1_id, w)]
    say(f"- Of {len(lost):,} lost to one-owner: the winning S1 has the SAME core name in {np.mean(same):.1%}")
say()
say("## Wrong accepts")
say()
say(tbl(pd.crosstab(fp.cause, fp.belongs_to, margins=True)))
say()
say(f"- On singleton S1s: {int(fp.s1_singleton.sum()):,} wrong accepts; by zone: " + ", ".join(f"{k} {v:,}" for k, v in fp.zone.value_counts().items()))
say()

# ---------------------------------------------------------------- 3. where the F0.5 loss sits
sc8 = per_s1(f, p8, ntrue)
fnS = f[lab & ~p8].groupby("s1_id").size().reindex(ntrue.index, fill_value=0)
fpS = f[~lab & p8].groupby("s1_id").size().reindex(ntrue.index, fill_value=0)
blk = (ntrue - f[lab].groupby("s1_id").size().reindex(ntrue.index, fill_value=0))
kind = np.select([(fnS > 0) & (fpS > 0), fpS > 0, fnS > 0, blk > 0], ["missed + wrong", "wrong only", "missed only", "blocking only"], "perfect")
loss = pd.DataFrame({"kind": kind, "loss": 1 - sc8, "country": sp.reindex(ntrue.index).to_numpy()})
g = loss.groupby("kind").agg(S1=("loss", "size"), loss_sum=("loss", "sum"))
g["share_of_loss"] = (g.loss_sum / g.loss_sum.sum()).round(3)
g["F0.5 pts"] = (g.loss_sum / len(loss)).round(4)
say("## Where the F0.5 loss sits (per S1)")
say()
say(tbl(g.drop(columns="loss_sum").rename_axis("S1 outcome")))
say()
say(tbl(loss.groupby("country").loss.agg(["size", "mean"]).rename(columns={"size": "S1", "mean": "loss"}).rename_axis("country")))
say()

# ---------------------------------------------------------------- 4. examples
def ex(d: pd.DataFrame, n: int = 6) -> None:
    for r in d.sample(min(n, len(d)), random_state=1).itertuples():
        a, c = rec.loc[r.s1_id], rec.loc[r.cand_id]
        say(f"- p={r.p:.3f} rr={r.rr if r.rr == r.rr else float('nan'):.3f} q={r.q:.3f}  \n"
            f"  S1 `{a.name_raw}` | `{a.addr_raw}`  \n  {r.cand_id[:2]} `{c.name_raw}` | `{c.addr_raw}`")
        if getattr(r, "cause", "") == "lost 1-owner to another S1" and r.cand_id in win.index:
            o = rec.loc[win[r.cand_id]]
            say(f"  won by S1 `{o.name_raw}` | `{o.addr_raw}`")


say("## Examples")
for c in fn.cause.value_counts().index:
    say()
    say(f"### missed: {c}")
    ex(fn[fn.cause == c])
for c in fp.cause.value_counts().index[:4]:
    say()
    say(f"### wrong accept: {c}")
    ex(fp[fp.cause == c], 4)

keep = ["s1_id", "cand_id", "label", "p", "rr", "v5", "q", "pred_A", "pred_v5", "pred_v8", "cause", "country", "src", "zone",
        "cand_noaddr", "native_script", "name_sim", "addr_sim", "name_shared_words", "house", "postal_same", "cluster_size",
        "s1_name_cnt", "d_crank", "d_ncomp"]
pd.concat([fn.assign(kind="missed"), fp.assign(kind="wrong")])[keep + ["kind"]].to_parquet(OUT / "v8_leftovers.parquet", index=False)
(OUT / "report.md").write_text("\n".join(md) + "\n")
log(f"wrote {OUT}/report.md and v8_leftovers.parquet")
