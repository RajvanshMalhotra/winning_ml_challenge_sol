"""For no-address candidates in v8 OOF: how do true-found, true-missed and wrong pairs differ on the candidate-side features?"""
import numpy as np, pandas as pd
from ber.config import load_config, run_dir
from ber.matcher import decide
RD = run_dir(load_config("configs/v2.yaml"))
f = pd.read_parquet(RD / "cand_side" / "oof_q.parquet")
f = f[f.cand_noaddr == 1].copy()
full = pd.read_parquet(RD / "cand_side" / "oof_q.parquet", columns=["s1_id", "cand_id", "q"])
pred = decide(full.rename(columns={"q": "p"}), 0.75)
f["pred"] = [c in pred.get(s, set()) for s, c in zip(f.s1_id, f.cand_id)]
y = f.label.astype(bool)
grp = np.select([y & f.pred, y & ~f.pred, ~y & f.pred], ["TRUE found", "TRUE missed", "WRONG accepted"], "wrong rejected")
f["grp"] = grp
cols = ["p", "rr", "q", "name_exact", "s1_name_cnt", "cand_name_cnt", "d_crank", "d_cmargin", "d_ncomp", "t_crank", "n_crank", "n_cmargin", "p_rank", "n_cands"]
print(f.groupby("grp").size().to_string(), "\n")
print(f.groupby("grp")[cols].median().round(3).T.to_string(), "\n")
for c in ["name_exact"]:
    print(c, f.groupby("grp")[c].mean().round(3).to_dict())
print("d_crank==1 share:", f.assign(x=f.d_crank == 1).groupby("grp").x.mean().round(3).to_dict())
print("n_crank==1 share:", f.assign(x=f.n_crank == 1).groupby("grp").x.mean().round(3).to_dict())
print("in dense top-30 at all:", f.assign(x=f.d_ncomp > 0).groupby("grp").x.mean().round(3).to_dict())
print("s1_name_cnt==1 share:", f.assign(x=f.s1_name_cnt == 1).groupby("grp").x.mean().round(3).to_dict())
# how many no-address candidates does each S1 have, and how many true?
g = f.groupby("s1_id").agg(n=("cand_id", "size"), t=("label", "sum"))
print("no-address candidates per S1: median", g.n.median(), "mean", round(g.n.mean(), 1), "| true per S1 (when >0):", g.t[g.t > 0].value_counts().sort_index().head(5).to_dict())
# among missed: is S1's name core == cand core, and the candidate's best S1 by p within sample?
m = f[f.grp == "TRUE missed"]
print("missed: p_rank among this S1's candidates:", m.p_rank.value_counts().sort_index().head(8).to_dict())
