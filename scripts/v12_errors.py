"""v12 OOF: cost of wrong accepts vs misses (what-if F0.5), and what the wrong accepts look like."""
import numpy as np, pandas as pd
from ber.config import load_config, run_dir
from ber.io import read_truth
from ber.matcher import decide, evaluate
cfg = load_config("configs/v2.yaml"); RD = run_dir(cfg)
truth = read_truth(cfg["paths"]["data_dir"])
f = pd.read_parquet(RD / "v12" / "oof_q12.parquet")
sp = pd.read_parquet(RD / "splits.parquet", columns=["s1_id", "country"]).set_index("s1_id")
s1 = sp.loc[f.s1_id.unique()].reset_index()[["s1_id", "country"]]
pred = decide(f[["s1_id", "cand_id"]].assign(p=f.q12), 0.75)
base = evaluate(pred, truth, s1)["macro_f05"]
no_fp = {s: pred.get(s, set()) & truth[s] for s in s1.s1_id}
pool = set(zip(f.s1_id, f.cand_id))
no_fn = {s: pred.get(s, set()) | {c for c in truth[s] if (s, c) in pool} for s in s1.s1_id}
print(f"v12 OOF {base:.4f} | no wrong accepts -> {evaluate(no_fp, truth, s1)['macro_f05']:.4f} | "
      f"all shortlisted matches found -> {evaluate(no_fn, truth, s1)['macro_f05']:.4f}")
owner = {c: s for s, m in truth.items() for c in m}
rows = []
for s in s1.s1_id:
    for c in pred.get(s, set()) - truth[s]:
        rows.append((s, c, not truth[s], owner.get(c)))
fp = pd.DataFrame(rows, columns=["s1_id", "cand_id", "singleton", "true_owner"])
fp = fp.merge(f[["s1_id", "cand_id", "q12", "cand_noaddr"]], on=["s1_id", "cand_id"])
fp["owner_in_sample"] = fp.true_owner.isin(set(s1.s1_id))
print(f"\nwrong accepts {len(fp):,}: on singletons {fp.singleton.sum():,} | record truly belongs to ANOTHER S1 "
      f"{fp.true_owner.notna().sum():,} (owner also in sample {fp.owner_in_sample.sum():,}) | record has NO owner {fp.true_owner.isna().sum():,}")
print("no-address share among wrong accepts:", f"{fp.cand_noaddr.mean():.1%}")
print("final score of wrong accepts:", fp.q12.quantile([.1, .25, .5, .75, .9]).round(3).to_dict())
# cost split of the remaining loss
def f05(P, T):
    if not T: return 1.0 if not P else 0.0
    tp = len(P & T)
    if not tp: return 0.0
    pr, rc = tp / len(P), tp / len(T); return 1.25 * pr * rc / (0.25 * pr + rc)
loss_fp = sum(f05(no_fp[s], truth[s]) - f05(pred.get(s, set()), truth[s]) for s in s1.s1_id) / len(s1)
loss_fn = sum(f05(no_fn[s], truth[s]) - f05(pred.get(s, set()), truth[s]) for s in s1.s1_id) / len(s1)
print(f"\nF0.5 lost to wrong accepts {loss_fp:.4f} | to missed shortlisted matches {loss_fn:.4f}")
rec = pd.read_parquet(RD / "records_train.parquet", columns=["entity_id", "name_raw", "addr_raw", "name_core", "addr_core", "house_no"]).set_index("entity_id")
a, c = rec.reindex(fp.s1_id), rec.reindex(fp.cand_id)
same_name = a.name_core.values == c.name_core.values
same_addr = (a.addr_core.values == c.addr_core.values) & (c.addr_core.fillna("").values != "")
hd = (a.house_no.values != "") & (c.house_no.values != "") & (a.house_no.values != c.house_no.values)
fp["kind"] = np.select([fp.cand_noaddr.values == 1, same_name & ~same_addr & hd, same_name, same_addr],
                       ["no-address record (wrong namesake)", "same name, different house no.", "same name, other address", "same address, other name"], "other")
print("\nwrong accepts by kind:\n" + fp.kind.value_counts().to_string())
for k in fp.kind.value_counts().index[:4]:
    print(f"\n== {k}")
    for r in fp[fp.kind == k].sample(min(3, (fp.kind == k).sum()), random_state=1).itertuples():
        o = rec.loc[r.true_owner] if isinstance(r.true_owner, str) else None
        print(f"  q={r.q12:.2f} S1: {rec.loc[r.s1_id].name_raw} | {str(rec.loc[r.s1_id].addr_raw)[:50]}")
        print(f"         cand: {rec.loc[r.cand_id].name_raw} | {str(rec.loc[r.cand_id].addr_raw)[:50]}")
        if o is not None: print(f"   true owner: {o.name_raw} | {str(o.addr_raw)[:50]}")
