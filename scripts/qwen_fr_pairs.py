"""French pairs for a short reranker fine-tune, from the labelled French practice sets (proxy_fr = train, proxy_fr_holdout =
check). Positives: ground truth. Hard negatives (France-style confusers): records sharing the S1's accent-folded name key
(namesakes) or its house number + street key (same address), that are not true matches.
Writes artifacts/v2/reranker_qwen_fr/{fr_train,fr_check}.parquet with s1_text, cand_text, label, kind."""
import re
import unicodedata

import numpy as np
import pandas as pd

from ber.text import record_text

D = "data/student_resource/dataset"
OUT = "artifacts/v2/reranker_qwen_fr"
STOP = {"de", "du", "des", "la", "le", "les", "l", "d", "et", "a", "au", "aux", "en", "sur"}


def fold(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9 ]+", " ", s)


def name_key(s: str) -> str:
    return " ".join(sorted(t for t in fold(s).split() if t not in STOP))


def addr_key(s: str) -> str:
    t = fold(s).split()
    num = next((x for x in t if x.isdigit()), "")
    street = [x for x in t if not x.isdigit() and x not in STOP][:3]
    return f"{num}|{' '.join(street)}" if num and street else ""


def build(name: str, n_s1: int, seed: int) -> pd.DataFrame:
    base = f"{D}/{name}/{name}"
    s1 = pd.read_csv(f"{base}_source1.tsv", sep="\t", dtype=str, keep_default_na=False)
    oth = pd.concat([pd.read_csv(f"{base}_source{k}.tsv", sep="\t", dtype=str, keep_default_na=False) for k in (2, 3)])
    kind = pd.concat([pd.read_csv(f"{base}_source{k}_kind.tsv", sep="\t", dtype=str, keep_default_na=False) for k in (2, 3)]).set_index("entity_id").kind
    gt = pd.read_csv(f"{base}_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False)
    rng = np.random.default_rng(seed)
    gt = gt.iloc[rng.permutation(len(gt))[:n_s1]]
    truth = {s: set(m.split(",")) - {""} for s, m in zip(gt.source1_entity_id, gt.matched_entity_ids)}
    s1 = s1[s1.entity_id.isin(truth)].copy()
    txt = lambda d: [record_text(n, a, c) for n, a, c in zip(d.business_name, d.business_address, d.country)]
    s1["text"], oth["text"] = txt(s1), txt(oth)
    s1["nk"], s1["ak"] = s1.business_name.map(name_key), s1.business_address.map(addr_key)
    oth["nk"], oth["ak"] = oth.business_name.map(name_key), oth.business_address.map(addr_key)
    ot = oth.set_index("entity_id").text
    pos = [(s, c) for s, m in truth.items() for c in m]
    neg = []
    for key in ("nk", "ak"):
        j = s1[["entity_id", key]].merge(oth[["entity_id", key]].rename(columns={"entity_id": "c"}), on=key)
        j = j[j[key] != ""]
        j = j[[c not in truth[s] for s, c in zip(j.entity_id, j.c)]]
        j = j.assign(r=rng.random(len(j))).sort_values("r").groupby("entity_id").head(2)
        neg += list(zip(j.entity_id, j.c, [f"same {'name' if key == 'nk' else 'address'} ({kind.get(c, '?')})" for c in j.c]))
    st = s1.set_index("entity_id").text
    rows = [(st[s], ot[c], 1, "match") for s, c in pos if c in ot.index] + [(st[s], ot[c], 0, k) for s, c, k in neg]
    d = pd.DataFrame(rows, columns=["s1_text", "cand_text", "label", "kind"]).drop_duplicates(["s1_text", "cand_text"])
    print(f"{name}: {len(truth):,} S1s -> {len(d):,} pairs ({d.label.mean():.1%} positive)\n{d.kind.value_counts().to_string()}", flush=True)
    return d.sample(frac=1.0, random_state=seed).reset_index(drop=True)


if __name__ == "__main__":
    import os
    os.makedirs(OUT, exist_ok=True)
    build("proxy_fr", 26_000, 0).to_parquet(f"{OUT}/fr_train.parquet", index=False)
    build("proxy_fr_holdout", 6_000, 1).to_parquet(f"{OUT}/fr_check.parquet", index=False)
