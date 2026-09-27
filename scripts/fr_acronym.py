"""France acronym routing (docs/path_to_099.md step 1). French S2/S3 records whose name is an acronym (`MSJ`, `CF SAS`)
are copies of the S1 whose name initials match, but Model A scores many of them below 0.01, so the reranker never reads
them. Here: find unassigned France acronym records, pair each with the France S1s whose initials match (from the full
blocking pool, or sharing the record's house number + a street word when none is in the pool), let the fine-tuned Qwen
reranker read them, and add the best pair when Qwen is confident and clearly ahead of the runner-up.
usage: fr_acronym.py pairs | score | splice <base_sub> <out_sub> [prob=0.9]"""
import os
import re
import subprocess
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

RD = Path("artifacts/v2")
OUT = RD / "fr_acronym"
OUT.mkdir(exist_ok=True)
STOP = {"de", "du", "des", "la", "le", "les", "l", "d", "et", "a", "au", "aux", "en", "sur", "pour", "the", "of", "and"}
LEGAL = {"sas", "sarl", "sa", "eurl", "sasu", "sci", "snc", "scop", "gie", "ets", "cie", "ltd", "inc", "llc", "corp", "co"}


def fold(s: str) -> str:
    return unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()


def words(name: str) -> list:
    return [w for w in re.findall(r"[A-Za-z0-9]+", fold(name))]


def acronym(name: str) -> str:
    """'MSJ' / 'CF SAS' / 'M.S.J.' -> 'msj'; '' if the name is not an acronym."""
    w = [x for x in words(name) if x.lower() not in LEGAL]
    if len(w) == 1 and w[0].isupper() and w[0].isalpha() and 2 <= len(w[0]) <= 6:
        return w[0].lower()
    if 2 <= len(w) <= 6 and all(len(x) == 1 and x.isalpha() and x.isupper() for x in w):
        return "".join(w).lower()
    return ""


def initials(name: str) -> set:
    w = [x.lower() for x in words(name) if x.lower() not in LEGAL]
    a = "".join(x[0] for x in w if x not in STOP)
    b = "".join(x[0] for x in w)
    return {x for x in (a, b) if len(x) >= 2}


def pairs() -> None:
    rec = pd.read_parquet(RD / "records_test.parquet", columns=["entity_id", "source", "country", "name_raw", "addr_norm", "house_no"])
    fr = rec[rec.country == "France"]
    s1 = fr[fr.source == 1].copy()
    oth = fr[fr.source != 1].copy()
    oth["acr"] = oth.name_raw.map(acronym)
    acr = oth[oth.acr != ""]
    m = pd.read_csv("artifacts/submissions/v16/matching_results.tsv", sep="\t", dtype=str, keep_default_na=False)
    taken = {c for v in m.matched_entity_ids if v for c in v.split(",")}
    free = acr[~acr.entity_id.isin(taken)]
    print(f"France acronym records {len(acr):,}; not assigned by v16 {len(free):,}", flush=True)
    s1["ini"] = s1.name_raw.map(initials)
    ini = s1[["entity_id", "ini"]].explode("ini").dropna()
    ini = ini[ini.ini.str.len() >= 2]
    # (a) initials-matching S1s that are in the record's blocking pool
    pool = pq.read_table(RD / "test_scores_matcher_v3.parquet", columns=["s1_id", "cand_id", "p"],
                         filters=[("cand_id", "in", free.entity_id.tolist())]).to_pandas()
    a = pool.merge(free[["entity_id", "acr"]].rename(columns={"entity_id": "cand_id"}), on="cand_id") \
            .merge(ini.rename(columns={"entity_id": "s1_id", "ini": "acr"}), on=["s1_id", "acr"])
    a["how"] = "pool"
    # (b) records with no initials-matching S1 in the pool: same initials + same house number + a shared street word
    miss = free[~free.entity_id.isin(set(a.cand_id))]
    s1["st"] = s1.addr_norm.fillna("").str.split().map(lambda t: {x for x in t if len(x) > 3 and not x.isdigit()})
    b = miss[["entity_id", "acr", "house_no", "addr_norm"]].rename(columns={"entity_id": "cand_id"}) \
        .merge(ini.rename(columns={"entity_id": "s1_id", "ini": "acr"}), on="acr")
    b = b.merge(s1[["entity_id", "house_no", "st"]].rename(columns={"entity_id": "s1_id", "house_no": "s1_house"}), on="s1_id")
    b = b[(b.house_no.fillna("") != "") & (b.house_no == b.s1_house)]
    b = b[[bool(set(str(x).split()) & st) for x, st in zip(b.addr_norm, b.st)]]
    b = b[["s1_id", "cand_id", "acr"]].assign(p=np.nan, how="new key")
    out = pd.concat([a[["s1_id", "cand_id", "acr", "p", "how"]], b], ignore_index=True).drop_duplicates(["s1_id", "cand_id"])
    out.to_parquet(OUT / "pairs.parquet", index=False)
    print(f"pairs to read: {len(out):,} for {out.cand_id.nunique():,} records ({(out.how == 'pool').sum():,} from the pool, "
          f"{(out.how != 'pool').sum():,} from the new key); Model A p of pool pairs: {out.p.describe().round(3).to_dict()}", flush=True)


def score() -> None:
    sys.argv = ["qwen_reranker.py", "none"]
    os.environ.setdefault("QR_PROMPT", "short")
    ns = {"__name__": "acr"}
    exec(open("scripts/qwen_reranker.py").read().rsplit("\n{", 1)[0], ns)
    p = pd.read_parquet(OUT / "pairs.parquet")
    pp = p[["s1_id", "cand_id"]].copy()
    pp["prompt"] = ns["prompts"]("test", pp)
    pp.to_parquet(OUT / "prompts.parquet", index=False)
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONNOUSERSITE"] = "1"
    for f in OUT.glob("qwen.parquet*"):
        f.unlink()
    r = subprocess.run([os.environ.get("VLLM_PY", os.path.expanduser("~/vllm-sol/bin/python")), "-u", "scripts/vllm_score.py",
                        str(OUT / "prompts.parquet"), str(RD / "reranker_qwen3" / "merged"), str(OUT / "qwen.parquet"),
                        str(ns["vllm_mem"]())], env=env)
    assert r.returncode == 0
    q = pd.read_parquet(OUT / "qwen.parquet")
    p["prob"] = 1 / (1 + np.exp(-q.score.to_numpy()))
    p.to_parquet(OUT / "pairs_scored.parquet", index=False)
    print("Qwen prob quantiles:", p.prob.quantile([.1, .25, .5, .75, .9]).round(3).to_dict(), flush=True)
    print(f"prob >= 0.9: {(p.prob >= 0.9).sum():,} pairs", flush=True)


def splice(base: str, out: str, thr: float) -> None:
    p = pd.read_parquet(OUT / "pairs_scored.parquet").sort_values("prob", ascending=False)
    g = p.groupby("cand_id")
    best = g.head(1).set_index("cand_id")
    second = g.prob.apply(lambda x: x.iloc[1] if len(x) > 1 else 0.0)
    best["second"] = second.reindex(best.index).to_numpy()
    add = best[(best.prob >= thr) & (best.prob - best.second >= 0.3)].reset_index()
    base, out = Path(base), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    m = pd.read_csv(base / "matching_results.tsv", sep="\t", dtype=str, keep_default_na=False)
    c = pd.read_csv(base / "candidate_pairs.tsv", sep="\t", dtype=str, keep_default_na=False)
    pred = {s: set(v.split(",")) - {""} for s, v in zip(m.source1_entity_id, m.matched_entity_ids)}
    cand = {s: set(v.split(",")) - {""} for s, v in zip(c.source1_entity_id, c.candidate_entity_ids)}
    taken = {x for v in pred.values() for x in v}
    add = add[~add.cand_id.isin(taken)]
    for s, x in zip(add.s1_id, add.cand_id):
        pred[s].add(x)
    for s, x in zip(p.s1_id, p.cand_id):                                  # everything Qwen read is a candidate
        cand[s].add(x)
    pd.DataFrame({"source1_entity_id": m.source1_entity_id,
                  "matched_entity_ids": [",".join(sorted(pred[s])) for s in m.source1_entity_id]}).to_csv(out / "matching_results.tsv", sep="\t", index=False)
    pd.DataFrame({"source1_entity_id": c.source1_entity_id,
                  "candidate_entity_ids": [",".join(sorted(cand[s])) for s in c.source1_entity_id]}).to_csv(out / "candidate_pairs.tsv", sep="\t", index=False)
    print(f"added {len(add):,} France acronym matches ({add.how.value_counts().to_dict()}) to {add.s1_id.nunique():,} S1s; "
          f"candidates +{len(p):,} pairs", flush=True)
    v = subprocess.run(["python3", "data/student_resource/utils/validate_submission.py", "--matching", str(out / "matching_results.tsv"),
                        "--candidate", str(out / "candidate_pairs.tsv"), "--test-dir", "data/student_resource/dataset/test"],
                       capture_output=True, text=True)
    print(v.stdout[-120:], f"validator exit={v.returncode}", flush=True)


if __name__ == "__main__":
    a = sys.argv[1]
    if a == "pairs":
        pairs()
    elif a == "score":
        score()
    else:
        splice(sys.argv[2], sys.argv[3], float(sys.argv[4]) if len(sys.argv) > 4 else 0.9)
