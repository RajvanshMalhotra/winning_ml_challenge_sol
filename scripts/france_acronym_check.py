import re, numpy as np, pandas as pd, pyarrow.dataset as ds
from anyascii import anyascii
from ber.io import read_truth
RD = "artifacts/v2"
ACR = re.compile(r"^[A-Z]{2,6}$")
STOP = {"de","des","du","la","le","les","et","d","l","of","the","and","sarl","sas","sasu","eurl","sci","sa","snc","selarl","ei","cie",
        "inc","llc","ltd","pvt","private","limited","corp","co","pc","llp","lp"}
ini = lambda n: "".join(w[0] for w in re.findall(r"[A-Za-z]+", anyascii(n or "")) if w.lower() not in STOP).upper()
def acr_records(fam):
    r = pd.read_parquet(f"{RD}/records_{fam}.parquet", columns=["entity_id","source","country","name_raw"])
    r["clean"] = r.name_raw.fillna("").str.replace(".", "", regex=False).str.strip()
    a = r[(r.source != 1) & r.clean.str.match(ACR)]
    return r.set_index("entity_id"), a
# TRAIN: how many acronym records are true copies, and does the owner's initials match?
r, a = acr_records("train")
t = read_truth("data/student_resource/dataset"); owner = {c: s for s, m in t.items() for c in m}
a = a.assign(owner=a.entity_id.map(owner))
a["ini_match"] = [bool(o) and ini(r.name_raw[o]) == c for o, c in zip(a.owner.fillna(""), a.clean)] if len(a) else []
print("TRAIN acronym S2/S3 records by country:", a.groupby("country").size().to_dict(),
      "| owned:", a.groupby("country").owner.apply(lambda x: round(x.notna().mean(), 3)).to_dict(),
      "| owner initials == acronym:", a[a.owner.notna()].groupby("country").ini_match.mean().round(3).to_dict())
# TEST: acronym records, v14 acceptance, and whether a candidate S1 has matching initials
r, a = acr_records("test")
sub = pd.read_csv("artifacts/submissions/v14/matching_results.tsv", sep="\t", dtype=str, keep_default_na=False)
got = {c: s for s, v in zip(sub.source1_entity_id, sub.matched_entity_ids) for c in v.split(",") if c}
a["accepted"] = a.entity_id.isin(got)
sc = ds.dataset(f"{RD}/test_scores_matcher_v3.parquet").to_table(filter=ds.field("cand_id").isin(a.entity_id.tolist())).to_pandas()
sc["s1_ini"] = [ini(n) for n in r.name_raw.reindex(sc.s1_id).to_numpy()]
sc["clean"] = r.clean.reindex(sc.cand_id).to_numpy()
hit = sc[sc.s1_ini == sc.clean]
n_hit = hit.groupby("cand_id").size()
a["n_init_s1"] = a.entity_id.map(n_hit).fillna(0).astype(int)
best = hit.sort_values("p", ascending=False).drop_duplicates("cand_id").set_index("cand_id")
a["init_s1_p"] = a.entity_id.map(best.p)
a["got_init_s1"] = [got.get(c) == best.s1_id.get(c) if c in best.index else False for c in a.entity_id]
g = a.groupby("country")
n1 = r[r.source == 1].groupby("country").size()
print("\nTEST acronym S2/S3 records:")
print(pd.DataFrame({"records": g.size(), "per 1k S1": 1000 * g.size() / n1, "v14 accepted (any S1)": g.accepted.mean(),
      "a candidate S1 has these initials": g.n_init_s1.apply(lambda x: (x > 0).mean()), "exactly 1 such S1": g.n_init_s1.apply(lambda x: (x == 1).mean()),
      "accepted to that S1": g.got_init_s1.mean(), "median Model A p of that S1": g.init_s1_p.median()}).round(3).to_string())
fr = a[(a.country == "France") & (a.n_init_s1 == 1) & ~a.accepted]
print(f"\nFrance acronym records with exactly one initials-matching S1, NOT accepted: {len(fr):,}")
for x in fr.sample(0).itertuples():
    s = best.s1_id[x.entity_id]
    print(f"   {x.name_raw!r} | {r.addr_raw[x.entity_id] if 'addr_raw' in r else ''}  ->  S1 {r.name_raw[s]!r}  (Model A p {best.p[x.entity_id]:.3f})")

# ---- sizing: France acronym records not accepted, by Model A band and address agreement with the initials-matching S1
ra = pd.read_parquet(f"{RD}/records_testfr.parquet", columns=["entity_id","addr_norm","house_no"]).set_index("entity_id")
x = a[a.country == "France"].copy()
x["init_s1"] = x.entity_id.map(best.s1_id)
x["band"] = pd.cut(x.init_s1_p, [-1, 0.01, 0.99, 2], labels=["p<0.01 (Qwen never reads)", "band (Qwen reads)", "p>0.99"]).astype(str)
x.loc[x.init_s1.isna(), "band"] = "no initials-matching S1 in candidates"
ca, sa = ra.addr_norm.reindex(x.entity_id).fillna("").to_numpy(), ra.addr_norm.reindex(x.init_s1.fillna("")).fillna("").to_numpy()
ch, sh = ra.house_no.reindex(x.entity_id).fillna("").to_numpy(), ra.house_no.reindex(x.init_s1.fillna("")).fillna("").to_numpy()
x["addr"] = np.where(ca == "", "record has no address", np.where((ch != "") & (ch == sh), "same house no.", "house no. differs/missing"))
x["status"] = np.where(x.got_init_s1, "given to initials S1", np.where(x.accepted, "given to another S1", "not accepted"))
print("\nFrance acronym records: status x Model A band of the initials-matching S1")
print(pd.crosstab(x.band, x.status, margins=True).to_string())
print("\nnot accepted, with exactly one initials-matching S1: address agreement")
y = x[(x.status == "not accepted") & (x.n_init_s1 == 1)]
print(pd.crosstab(y.band, y.addr, margins=True).to_string())
