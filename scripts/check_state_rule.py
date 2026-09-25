"""Do true matches share a state? Measures same-state rate, conflict rate and missing-state rate on train pairs."""
import numpy as np
import pandas as pd

from ber.io import read_truth, truth_pairs
from ber.lexicons import load_lexicon
from ber.text import fold

lex = load_lexicon()
full_names = {c: set(m.values()) for c, m in lex.states.items()}


def state_of(raw: str, country: str) -> str:
    for native, english in lex.native.items():
        if native in raw:
            raw = raw.replace(native, english)
    abbr = lex.states.get(country, {})
    for comp in (c.strip() for c in fold(raw).split(",")):
        comp = abbr.get(comp, comp)
        if comp in full_names.get(country, ()):
            return comp
    return ""


rec = pd.read_parquet("artifacts/v1/records_train.parquet", columns=["entity_id", "country", "addr_raw"])
rec["state"] = [state_of(a, c) for a, c in zip(rec.addr_raw, rec.country)]
st = rec.set_index("entity_id").state
tp = truth_pairs(read_truth("data/student_resource/dataset"))
tp["s1_state"], tp["c_state"] = st.reindex(tp.s1_id).values, st.reindex(tp.cand_id).values
tp["country"] = rec.set_index("entity_id").country.reindex(tp.s1_id).values
for c, g in tp.groupby("country"):
    both = g[(g.s1_state != "") & (g.c_state != "")]
    print(f"{c}: pairs={len(g):,}  state known both sides={len(both)/len(g):.3f}  "
          f"same state={np.mean(both.s1_state == both.c_state):.5f}  "
          f"S1 missing={np.mean(g.s1_state == ''):.3f}  cand missing={np.mean(g.c_state == ''):.3f}")
    print("   records with a state:", round(float((rec[rec.country == c].state != "").mean()), 3),
          "| distinct states:", rec[rec.country == c].state.nunique())

print("\n== top cross-state pairs (share of all true pairs in that country)")
for c, g in tp.groupby("country"):
    x = g[(g.s1_state != "") & (g.c_state != "") & (g.s1_state != g.c_state)]
    top = x.groupby(["s1_state", "c_state"]).size().sort_values(ascending=False).head(12) / len(g)
    print(c); print(top.round(5).to_string())
