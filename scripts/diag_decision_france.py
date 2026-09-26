"""(1) Expected-F0.5-optimal list selection vs global cut-off on Model A v2 OOF (calibrated, cross-fitted).
(2) Test-set uncertainty by country (is France where the leaderboard gap comes from?)."""
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from ber.io import read_truth
from ber.matcher import decide, evaluate

rd = "artifacts/v2"
oof = pd.read_parquet(f"{rd}/matcher_emb/oof.parquet")
sp = pd.read_parquet(f"{rd}/splits.parquet", columns=["s1_id", "country", "fold"]).set_index("s1_id")
oof["fold"] = sp.fold.reindex(oof.s1_id).to_numpy()
truth = read_truth("data/student_resource/dataset")
s1 = sp.loc[oof.s1_id.unique()].reset_index()[["s1_id", "country"]]

# cross-fitted isotonic calibration
cal = np.zeros(len(oof))
for k in sorted(oof.fold.unique()):
    tr, va = oof.fold != k, oof.fold == k
    iso = IsotonicRegression(out_of_bounds="clip").fit(oof.p[tr], oof.label[tr])
    cal[va.to_numpy()] = iso.predict(oof.p[va])
oof["q"] = cal

def exp_f05_select(g: pd.DataFrame, n_mc: int = 0) -> set:
    """Choose the prefix (by q) maximising expected F0.5 under independent Bernoulli(q); empty list allowed.
    Expected F0.5 approximated with E[TP]/E[...] plug-in (fast, standard)."""
    q = np.sort(g.q.to_numpy())[::-1]
    ids = g.cand_id.to_numpy()[np.argsort(-g.q.to_numpy())]
    total = q.sum()  # expected number of true matches among candidates
    best_k, best = 0, (1.0 if total < 1e-9 else np.prod(1 - q))  # empty list: F=1 iff no true match
    cum = 0.0
    for k in range(1, min(len(q), 12) + 1):
        cum += q[k - 1]
        # plug-in expected F0.5: 1.25*TP / (1.25*TP + FP + 0.25*FN) with TP=cum, FP=k-cum, FN=total-cum
        f = 1.25 * cum / (1.25 * cum + (k - cum) + 0.25 * (total - cum))
        if f > best:
            best_k, best = k, f
    return set(ids[:best_k])

base = evaluate(decide(oof[["s1_id", "cand_id", "p"]], 0.70), truth, s1)
sel = {s: exp_f05_select(g) for s, g in oof.groupby("s1_id")}
# one-owner on the selected sets
pairs = pd.DataFrame([(s, c) for s, cs in sel.items() for c in cs], columns=["s1_id", "cand_id"]).merge(oof[["s1_id", "cand_id", "q"]])
pairs = pairs.loc[pairs.groupby("cand_id").q.idxmax()]
sel1 = pairs.groupby("s1_id").cand_id.agg(set).to_dict()
exp = evaluate(sel1, truth, s1)
print("global tau=0.70      :", {k: round(v, 4) for k, v in base.items()})
print("expected-F0.5 select :", {k: round(v, 4) for k, v in exp.items()})
for tau in (0.5, 0.6, 0.7, 0.8):
    print(f"calibrated q>= {tau}:", round(evaluate(decide(oof.rename(columns={'p': 'p_raw', 'q': 'p'})[['s1_id', 'cand_id', 'p']], tau), truth, s1)["macro_f05"], 4))

# (2) test uncertainty by country
t = pd.read_parquet(f"{rd}/test_scores.parquet")
rec = pd.read_parquet(f"{rd}/records_test.parquet", columns=["entity_id", "country"]).set_index("entity_id")
t["country"] = rec.country.reindex(t.s1_id).to_numpy()
g = t.groupby(["country", "s1_id"]).p
per = pd.DataFrame({"max_p": g.max(), "n_mid": g.apply(lambda x: ((x > 0.3) & (x < 0.9)).sum()),
                    "n_hi": g.apply(lambda x: (x >= 0.9).sum())}).reset_index()
o = oof.merge(s1, on="s1_id")
go = o.groupby(["country", "s1_id"]).p
per_o = pd.DataFrame({"max_p": go.max(), "n_mid": go.apply(lambda x: ((x > 0.3) & (x < 0.9)).sum())}).reset_index()
print("\n== uncertainty: share of S1s whose best candidate has p in (0.3, 0.9), and mean # of uncertain candidates")
for name, d in (("TEST", per), ("TRAIN-OOF", per_o)):
    for c, x in d.groupby("country"):
        print(f"  {name:<9} {c:<7} best-p uncertain {((x.max_p > 0.3) & (x.max_p < 0.9)).mean():.3f} | "
              f"mean uncertain cands {x.n_mid.mean():.3f} | best-p < 0.3 (likely empty) {(x.max_p < 0.3).mean():.3f}")
