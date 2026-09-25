import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ber.blocking.sparse import cand_sparse_path
from ber.config import run_dir
from ber.contrastive.encoders import ENCODERS, encode, load_encoder
from ber.contrastive.pairs import hard_negatives, render_triplets, sample_triplets
from ber.contrastive.retrieval import recall_at_k
from ber.contrastive.train import finetune
from ber.io import read_truth
from ber.normalize import records_path
from ber.split import splits_path
from ber.text import record_text

TIE_EPS = 0.005


def build_eval_set(records: pd.DataFrame, splits: pd.DataFrame, truth: dict, n_distractors: int,
                   seed: int) -> pd.DataFrame:
    """A_val queries (with >=1 match) + all their matches + n sampled same-country S2/S3 distractors.
    The pool is identical for every model, so recall numbers are comparable."""
    rng = np.random.default_rng(seed)
    q = splits[(splits.split == "A_val") & splits.s1_id.map(lambda s: len(truth[s]) > 0)]
    parts = [pd.DataFrame({"entity_id": q.s1_id, "role": "query", "country": q.country})]
    others = records[records.source != 1]
    for country, grp in q.groupby("country"):
        matches = sorted(set().union(*(truth[s] for s in grp.s1_id)))
        rest = others.entity_id[(others.country == country) & ~others.entity_id.isin(matches)].to_numpy()
        take = rng.choice(rest, size=min(n_distractors, len(rest)), replace=False)
        docs = np.concatenate([matches, np.sort(take)])
        parts.append(pd.DataFrame({"entity_id": docs, "role": "doc", "country": country}))
    return pd.concat(parts, ignore_index=True)


def result_path(cfg: dict, model_key: str, tag: str) -> Path:
    d = run_dir(cfg) / "bakeoff"
    d.mkdir(exist_ok=True)
    return d / f"{model_key}__{tag}.json"


def _mean_r50(r: dict) -> float:
    return float(np.mean([v["50"] for v in r["recall"].values()]))


def select_winner(results: list[dict]) -> str:
    ft = sorted((r for r in results if r["tag"] == "finetune"), key=_mean_r50, reverse=True)
    top = [r for r in ft if _mean_r50(ft[0]) - _mean_r50(r) <= TIE_EPS]
    if len(top) == 1:
        return top[0]["model"]
    loco = {r["model"]: r["recall"].get("India", {}).get("50", 0.0) for r in results if r["tag"].startswith("finetune_")}
    return max(top, key=lambda r: (round(loco.get(r["model"], 0.0), 3), r["throughput_rps"]))["model"]


def write_report(results: list[dict], out_md: Path) -> None:
    countries = sorted({c for r in results for c in r["recall"]})
    head = ["model", "tag"] + [f"{c} R@{k}" for c in countries for k in ("10", "50", "100")] + \
           ["rec/s", "test vec GB", "train peak GB"]
    lines = ["# Bi-encoder bake-off", "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in sorted(results, key=lambda r: (r["model"], r["tag"])):
        cells = [r["model"], r["tag"]] + [f"{r['recall'].get(c, {}).get(k, float('nan')):.4f}"
                                          for c in countries for k in ("10", "50", "100")]
        cells += [f"{r['throughput_rps']:.0f}", f"{r['test_vectors_gb']:.1f}", f"{r['peak_train_mem_gb']:.1f}"]
        lines.append("| " + " | ".join(cells) + " |")
    if any(r["tag"] == "finetune" for r in results):
        lines += ["", f"Winner: **{select_winner(results)}** (mean fine-tuned R@50; ties within {TIE_EPS} "
                  "broken by leave-one-country-out India R@50, then throughput)"]
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n")


def _eval_set(cfg: dict, records: pd.DataFrame, truth: dict) -> pd.DataFrame:
    path = run_dir(cfg) / "bakeoff" / "eval_set.parquet"
    if not path.exists():
        path.parent.mkdir(exist_ok=True)
        ev = build_eval_set(records, pd.read_parquet(splits_path(cfg)), truth,
                            cfg["bakeoff"]["eval_distractors_per_country"], cfg["seed"])
        ev.to_parquet(path, index=False)
    return pd.read_parquet(path)


def _run(cfg: dict, model_key: str, mode: str, train_countries: list[str] | None) -> None:
    spec = ENCODERS[model_key]
    tag = mode if not train_countries else f"{mode}_{'-'.join(train_countries)}"
    records = pd.read_parquet(records_path(cfg, "train"), columns=["entity_id", "source", "country", "name_raw", "addr_raw"])
    truth = read_truth(cfg["paths"]["data_dir"])
    ev = _eval_set(cfg, records, truth)
    rec = records.set_index("entity_id")
    stats = {"peak_train_mem_gb": 0.0, "train_seconds": 0.0}
    model_path = None
    if mode == "finetune":
        splits = pd.read_parquet(splits_path(cfg))
        a = splits[(splits.split == "A_train") & splits.s1_id.map(lambda s: len(truth[s]) > 0)]
        if train_countries:
            a = a[a.country.isin(train_countries)]
        rng = np.random.default_rng(cfg["seed"])
        s1_ids = sorted(rng.choice(a.s1_id.to_numpy(), size=min(cfg["train"]["n_entities"], len(a)), replace=False))
        cands = pd.read_parquet(cand_sparse_path(cfg, "train"), columns=["s1_id", "cand_id"])
        others = records[records.source != 1]
        pool = {c: g.entity_id.to_numpy() for c, g in others.groupby("country")}
        trip = sample_triplets(truth, s1_ids, hard_negatives(cands, truth, s1_ids), pool,
                               dict(zip(records.entity_id, records.country)),
                               cfg["train"]["n_rounds"], cfg["train"]["p_intra"], cfg["seed"])
        train_df = render_triplets(trip, rec, spec, cfg["train"]["aug_prob"], cfg["seed"])
        out_dir = run_dir(cfg) / "bakeoff" / f"{model_key}__{tag}"
        stats = finetune(spec, train_df, out_dir, cfg["train"], cfg["gpu"]["mem_fraction"], cfg["seed"])
        model_path = str(out_dir / "model")
    model = load_encoder(spec, path=model_path, mem_fraction=cfg["gpu"]["mem_fraction"])
    bs = cfg["bakeoff"]["encode_batch_size"]
    recall, n_docs, secs, dim = {}, 0, 0.0, 0
    for country, grp in ev.groupby("country"):
        q, d = grp[grp.role == "query"].entity_id.tolist(), grp[grp.role == "doc"].entity_id.tolist()
        texts = lambda ids: [record_text(rec.at[i, "name_raw"], rec.at[i, "addr_raw"], country) for i in ids]
        q_emb = encode(model, texts(q), spec.query_prefix, bs)
        t0 = time.perf_counter()
        d_emb = encode(model, texts(d), spec.doc_prefix, bs)
        secs += time.perf_counter() - t0
        n_docs += len(d)
        dim = d_emb.shape[1]
        r = recall_at_k(q, q_emb, d, d_emb, truth, cfg["bakeoff"]["ks"], n_threads=cfg["n_jobs"])
        recall[country] = {str(k): v for k, v in r.items()}
        print(model_key, tag, country, recall[country], flush=True)
    test_src = pd.read_parquet(records_path(cfg, "test"), columns=["source"]).source
    result = {"model": model_key, "tag": tag, "dim": dim, "recall": recall,
              "throughput_rps": n_docs / secs, "test_vectors_gb": int((test_src != 1).sum()) * dim * 2 / 1e9,
              "peak_train_mem_gb": stats["peak_train_mem_gb"], "train_seconds": stats["train_seconds"]}
    result_path(cfg, model_key, tag).write_text(json.dumps(result, indent=2))


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("action", choices=["zeroshot", "finetune", "report"])
    parser.add_argument("--model", choices=sorted(ENCODERS))
    parser.add_argument("--train-countries", nargs="+", default=None)


def run(cfg: dict, args: argparse.Namespace) -> None:
    if args.action == "report":
        results = [json.loads(p.read_text()) for p in sorted((run_dir(cfg) / "bakeoff").glob("*__*.json"))]
        write_report(results, Path("docs/results/bakeoff.md"))
        print(Path("docs/results/bakeoff.md").read_text())
        return
    if not args.model:
        raise SystemExit("--model is required for zeroshot/finetune")
    _run(cfg, args.model, args.action, args.train_countries)
