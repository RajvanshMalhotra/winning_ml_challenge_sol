import pandas as pd, numpy as np, collections, re
D = "data/student_resource/dataset/"
rd = lambda p: pd.read_csv(D+p, sep="\t", dtype=str, keep_default_na=False, quoting=3)
tr = {s: rd(f"train/train_source{s}.tsv") for s in (1,2,3)}
te = {s: rd(f"test/test_source{s}.tsv") for s in (1,2,3)}
gt = rd("train/train_ground_truth.tsv")
pd.set_option("display.width", 250, "display.max_colwidth", 90)
for nm, d in (("train", tr), ("test", te)):
    for s, df in d.items():
        print(f"{nm} s{s}: {len(df):,} rows, cols={list(df.columns)}, countries={df.country.value_counts().to_dict()}, "
              f"empty name={(df.business_name=='').mean():.2%} empty addr={(df.business_address=='').mean():.2%}, dup ids={df.entity_id.duplicated().sum()}")
cty = pd.concat([tr[s][["entity_id","country"]] for s in (1,2,3)]).set_index("entity_id").country
gt["ids"] = gt.matched_entity_ids.str.split(",").apply(lambda l: [x for x in l if x])
gt["n"] = gt.ids.str.len(); gt["c1"] = gt.source1_entity_id.map(cty)
print("\nGT rows", len(gt), "; S1 ids covered:", gt.source1_entity_id.isin(tr[1].entity_id).mean())
print("singleton rate overall %.3f" % (gt.n==0).mean(), "| by country:", gt.groupby("c1").n.apply(lambda x:(x==0).mean()).round(3).to_dict())
print("matches/entity dist:", gt.n.clip(upper=10).value_counts().sort_index().to_dict())
ex = gt.explode("ids").dropna(subset=["ids"])
ex["src"] = ex.ids.str[:2]
print("per-entity S2 count dist:", ex[ex.src=="S2"].groupby("source1_entity_id").size().clip(upper=6).value_counts().sort_index().to_dict())
print("per-entity S3 count dist:", ex[ex.src=="S3"].groupby("source1_entity_id").size().clip(upper=6).value_counts().sort_index().to_dict())
own = ex.ids.value_counts()
print("S2/S3 ids linked to >1 S1:", (own>1).sum(), "max", own.max())
print("cross-country links:", (ex.ids.map(cty) != ex.c1).sum())
for s in (2,3):
    m = tr[s].entity_id.isin(ex.ids)
    print(f"S{s}: matched {m.sum():,}/{len(m):,} ({m.mean():.1%}) -> unmatched S{s} records are distractors")
# postal code presence
pats = {"US": r"\b\d{5}(?:-\d{4})?\b", "India": r"\b\d{3}\s?\d{3}\b"}
for s in (1,2,3):
    df = tr[s]
    print(f"train s{s} postal present:", {c: round(df[df.country==c].business_address.str.contains(p).mean(),3) for c,p in pats.items()})
print("test France postal(5d) present:", {s: round(te[s][te[s].country=="France"].business_address.str.contains(r"\b\d{5}\b").mean(),3) for s in (1,2,3)})
# examples
rec = pd.concat([tr[s] for s in (1,2,3)]).set_index("entity_id")
rng = np.random.default_rng(0)
for c in ("US","India"):
    sub = gt[(gt.c1==c)&(gt.n>=2)].sample(4, random_state=1)
    print(f"\n=== {c} match examples")
    for _, r in sub.iterrows():
        print(" S1:", tuple(rec.loc[r.source1_entity_id, ["business_name","business_address"]]))
        for i in r.ids: print("   ", i[:2], tuple(rec.loc[i, ["business_name","business_address"]]))
print("\n=== France test samples")
for s in (1,2,3): print(te[s][te[s].country=="France"].head(4).to_string(index=False, header=False))
print("\n=== singleton S1 examples"); print(tr[1][tr[1].entity_id.isin(gt[gt.n==0].source1_entity_id)].head(5).to_string(index=False, header=False))
print("\nlength stats name/addr chars:", {s:(int(tr[s].business_name.str.len().median()), int(tr[s].business_address.str.len().median())) for s in (1,2,3)})
