# Foundations + Bi-Encoder Bake-off Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the data foundations and run the bi-encoder bake-off. The foundations are IO, the metric, normalization, noise augmentation, splits and sparse blocking. The bake-off is spec §5, and it picks the encoder for the rest of the pipeline.

**Architecture:** A Python package `ber` (in `src/`), driven by one YAML config and a `python -m ber <command>` CLI. Each command reads and writes Parquet under `artifacts/<run_name>/`. Code is edited on the laptop, rsynced to the HPC and run there, with tests via `scripts/hpc.sh` and heavy jobs via SLURM.

**Tech Stack:** Python 3.11, pandas/pyarrow, scikit-learn, sparse_dot_topn, anyascii, faiss-cpu, torch, sentence-transformers, peft, pytest.

**Spec:** `docs/superpowers/specs/2026-09-25-er-pipeline-design.md` (read §2–§5 and §7–§8 before starting)

## Global Constraints

- Every model must be **MIT or Apache 2.0 licensed and at most 8B parameters**. Qwen3-Embedding-8B carries an open question (spec §10).
- No external data lookups: no geocoding, no business registries, no libpostal. Only regexes and lexicons.
- Country is an **open set of strings**. Never filter, hard-code or one-hot encode `{US, India}`. Always iterate over whatever country values appear in the data.
- Read every TSV with `sep="\t"`, `dtype=str`, `keep_default_na=False`, `quoting=csv.QUOTE_NONE`.
- Training S1 entities are split A-train 35% / A-val 5% / B 60%, grouped by S1 id. **The bi-encoder never trains on B entities.**
- All randomness is seeded from `cfg["seed"]`.
- Commit messages have **no `Co-Authored-By` or Claude attribution lines**.
- On the HPC always run with `PYTHONNOUSERSITE=1`. The account's `~/.local` has old pandas/numpy that would shadow the env.
- The GPU is shared. Every GPU process calls `torch.cuda.set_per_process_memory_fraction(cfg["gpu"]["mem_fraction"])`.

**HPC facts:**
- Host and user: `fcse_shwetasharma@172.16.0.77`
- SSH key: `~/.ssh/hpc_nopass`
- Repo checkout: `~/winning_ml_challenge_sol`
- Data: `~/winning_ml_challenge_sol/data/student_resource/`
- Conda: `/opt/anaconda3`
- SLURM partitions: `batch`, `gpu`, `interactive`, `student`
- Hardware: 256 cores, 503 GB RAM, 1× H100 NVL (shared)

## File map

| File | Responsibility |
|---|---|
| `pyproject.toml`, `requirements.in`, `requirements.txt` | package + deps (txt = pinned freeze from HPC) |
| `scripts/hpc.sh` | rsync laptop → HPC, then run a command in the `ber` env |
| `slurm/cpu.sbatch`, `slurm/gpu.sbatch` | generic SLURM wrappers: `sbatch slurm/x.sbatch <cmd...>` |
| `configs/base.yaml`, `configs/smoke.yaml` | all parameters; smoke = tiny overrides |
| `src/ber/config.py` | load YAML (with `base:` inheritance), run dir, metrics log |
| `src/ber/cli.py`, `src/ber/__main__.py` | command registry |
| `src/ber/io.py` | read TSVs and ground truth |
| `src/ber/evaluate.py` | official macro F0.5, pair completeness |
| `src/ber/lexicons/{en,in,fr}.yaml`, `src/ber/lexicons/__init__.py` | lexicon data + merged loader |
| `src/ber/text.py` | pure normalization functions (name, address, domain segmentation) |
| `src/ber/normalize.py` | build the `records` table in parallel; `normalize` command |
| `src/ber/noise.py` | augmentation operators |
| `src/ber/split.py` | A-train/A-val/B + folds; `split` command |
| `src/ber/blocking/sparse.py` | char TF-IDF top-k + key blocks; `block-sparse` command |
| `src/ber/contrastive/encoders.py` | model registry + load/encode |
| `src/ber/contrastive/retrieval.py` | FAISS recall@k |
| `src/ber/contrastive/pairs.py` | triplet sampling + rendering with augmentation |
| `src/ber/contrastive/train.py` | fine-tuning (CachedMNRL, LoRA for Qwen) |
| `src/ber/contrastive/bakeoff.py` | eval set, zero-shot / fine-tune runs, report; `bakeoff` command |
| `tests/...` | one test file per module |

---

### Task 1: Scaffold, HPC environment, config and CLI

**Files:**
- Create: `pyproject.toml`, `requirements.in`, `scripts/hpc.sh`, `slurm/cpu.sbatch`, `slurm/gpu.sbatch`, `configs/base.yaml`, `configs/smoke.yaml`, `src/ber/__init__.py`, `src/ber/__main__.py`, `src/ber/config.py`, `src/ber/cli.py`, `tests/test_config.py`
- Modify: `.gitignore`

**Interfaces:**
- Produces:
  - `load_config(path: str | Path) -> dict`
  - `run_dir(cfg: dict) -> Path` (creates `artifacts/<run_name>/`)
  - `log_metrics(cfg: dict, section: str, data: dict) -> None` (merges into `run_dir/metrics.json`)
  - CLI registry `COMMANDS: dict[str, tuple[str, str]]` in `ber.cli`. Each value is `(module_path, help)`, and each registered module must define `add_arguments(parser)` and `run(cfg, args)`.

- [ ] **Step 1: Pull the EDA script from the HPC into the repo** (the HPC copy would otherwise be deleted by `rsync --delete`)

```bash
mkdir -p scripts && scp -i ~/.ssh/hpc_nopass fcse_shwetasharma@172.16.0.77:~/winning_ml_challenge_sol/scripts/eda.py scripts/eda.py
```

It reads `data/student_resource/dataset/` relative to the repo root, which is where `scripts/hpc.sh` runs it.

- [ ] **Step 2: Write packaging files**

`pyproject.toml`:
```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "ber"
version = "0.1.0"
requires-python = ">=3.11"

[tool.setuptools.packages.find]
where = ["src"]

[tool.setuptools.package-data]
ber = ["lexicons/*.yaml"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
```

`requirements.in`:
```
pandas>=2.2
pyarrow>=16
numpy>=1.26,<2.3
scipy>=1.13
scikit-learn>=1.5
sparse_dot_topn>=1.1
rapidfuzz>=3.9
lightgbm>=4.5
faiss-cpu>=1.8
torch>=2.5
transformers>=4.51
sentence-transformers>=4.1
peft>=0.15
accelerate>=1.6
datasets>=3.0
einops>=0.8
anyascii>=0.3.2
pyyaml>=6
pytest>=8
```

Append to `.gitignore`:
```
*.egg-info/
.pytest_cache/
```

- [ ] **Step 3: Write `scripts/hpc.sh`**

```bash
#!/usr/bin/env bash
# Sync the working tree to the HPC and run a command there inside the `ber` conda env.
# Usage: scripts/hpc.sh <command...>     e.g. scripts/hpc.sh pytest tests -q
set -euo pipefail
HOST=fcse_shwetasharma@172.16.0.77
KEY="$HOME/.ssh/hpc_nopass"
REMOTE=winning_ml_challenge_sol
SSH="ssh -i $KEY -o BatchMode=yes -o LogLevel=ERROR"
cd "$(dirname "$0")/.."
rsync -az --delete -e "$SSH" \
  --exclude .git --exclude /data --exclude /artifacts --exclude /student_resource \
  --exclude __pycache__ --exclude '*.egg-info' --exclude .pytest_cache \
  ./ "$HOST:~/$REMOTE/"
$SSH "$HOST" "cd ~/$REMOTE && source /opt/anaconda3/etc/profile.d/conda.sh && conda activate ber && export PYTHONNOUSERSITE=1 && $*"
```

Run: `chmod +x scripts/hpc.sh`

- [ ] **Step 4: Create the conda env on the HPC** (one time only)

```bash
ssh -i ~/.ssh/hpc_nopass -o LogLevel=ERROR fcse_shwetasharma@172.16.0.77 \
  'source /opt/anaconda3/etc/profile.d/conda.sh && conda create -y -q -n ber python=3.11'
scripts/hpc.sh "pip install -q -r requirements.in && pip freeze > requirements.txt && python -c 'import torch;print(torch.__version__, torch.cuda.is_available())'"
scp -i ~/.ssh/hpc_nopass fcse_shwetasharma@172.16.0.77:~/winning_ml_challenge_sol/requirements.txt requirements.txt
```
Expected: the last command prints a torch version and `True`.

- [ ] **Step 5: Check the SLURM GPU resource name**

Run: `scripts/hpc.sh "scontrol show node AI-server | grep -iE 'gres|cfgtres'"`

If the output contains `gpu`, keep the `#SBATCH --gres=gpu:1` line in Step 6. If it doesn't, delete that line from `slurm/gpu.sbatch`.

- [ ] **Step 6: Write the SLURM wrappers**

`slurm/cpu.sbatch`:
```bash
#!/bin/bash
#SBATCH -J ber-cpu
#SBATCH -p batch
#SBATCH -c 64
#SBATCH --mem=400G
#SBATCH -o artifacts/logs/%x-%j.out
# Usage: sbatch slurm/cpu.sbatch python -m ber <command> [args]
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate ber
export PYTHONNOUSERSITE=1
cd ~/winning_ml_challenge_sol
srun "$@"
```

`slurm/gpu.sbatch`:
```bash
#!/bin/bash
#SBATCH -J ber-gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 32
#SBATCH --mem=200G
#SBATCH -o artifacts/logs/%x-%j.out
# Usage: sbatch slurm/gpu.sbatch python -m ber <command> [args]
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate ber
export PYTHONNOUSERSITE=1
cd ~/winning_ml_challenge_sol
srun "$@"
```

- [ ] **Step 7: Write configs**

`configs/base.yaml`:
```yaml
run_name: v1
seed: 42
n_jobs: 64
paths:
  data_dir: data/student_resource/dataset
  validator: data/student_resource/utils/validate_submission.py
  artifacts: artifacts
split:
  a_train: 0.35
  a_val: 0.05
  n_folds: 5
blocking:
  tfidf:
    ngram_range: [3, 5]
    min_df: 2
    top_k: 50
    min_sim: 0.1
    n_threads: 64
  keys:
    n_name_tokens: 2
    n_addr_tokens: 3
    min_token_len: 3
    max_block_size: 100
gpu:
  mem_fraction: 0.4
bakeoff:
  models: [gte, nomic, arctic, bgem3, qwen8b]
  eval_distractors_per_country: 1000000
  ks: [10, 50, 100]
  encode_batch_size: 512
train:
  n_entities: 300000
  n_rounds: 3
  p_intra: 0.3
  aug_prob: 0.5
  batch_size: 256
  cached_mini_batch_size: 32
  max_steps: 3000
  learning_rate: 2.0e-5
  lora_learning_rate: 1.0e-4
  warmup_ratio: 0.1
```

`configs/smoke.yaml`:
```yaml
base: configs/base.yaml
run_name: smoke
bakeoff:
  eval_distractors_per_country: 5000
train:
  n_entities: 2000
  n_rounds: 1
  batch_size: 32
  cached_mini_batch_size: 16
  max_steps: 20
```

- [ ] **Step 8: Write the failing config test**

`tests/test_config.py`:
```python
import json
from pathlib import Path

from ber.cli import main
from ber.config import load_config, log_metrics, run_dir


def test_base_inheritance_deep_merges(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text("run_name: a\ntrain:\n  batch_size: 256\n  max_steps: 3000\npaths:\n  artifacts: " + str(tmp_path / "art") + "\n")
    child = tmp_path / "child.yaml"
    child.write_text(f"base: {base}\nrun_name: b\ntrain:\n  max_steps: 20\n")
    cfg = load_config(child)
    assert cfg["run_name"] == "b"
    assert cfg["train"] == {"batch_size": 256, "max_steps": 20}
    assert "base" not in cfg


def test_run_dir_and_metrics(tmp_path):
    cfg = {"run_name": "r", "paths": {"artifacts": str(tmp_path)}}
    d = run_dir(cfg)
    assert d == tmp_path / "r" and d.is_dir()
    log_metrics(cfg, "blocking", {"pc": 0.99})
    log_metrics(cfg, "split", {"n": 3})
    assert json.loads((d / "metrics.json").read_text()) == {"blocking": {"pc": 0.99}, "split": {"n": 3}}


def test_cli_show_config(capsys):
    main(["--config", "configs/base.yaml", "show-config"])
    assert json.loads(capsys.readouterr().out)["run_name"] == "v1"
```

- [ ] **Step 9: Run it to verify it fails**

Run: `scripts/hpc.sh pytest tests/test_config.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'ber'`

- [ ] **Step 10: Implement config and CLI**

`src/ber/__init__.py`: empty file.

`src/ber/config.py`:
```python
import json
from pathlib import Path

import yaml


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str | Path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    if "base" in cfg:
        parent = load_config(cfg.pop("base"))
        cfg = _deep_merge(parent, cfg)
    return cfg


def run_dir(cfg: dict) -> Path:
    d = Path(cfg["paths"]["artifacts"]) / cfg["run_name"]
    d.mkdir(parents=True, exist_ok=True)
    return d


def log_metrics(cfg: dict, section: str, data: dict) -> None:
    f = run_dir(cfg) / "metrics.json"
    metrics = json.loads(f.read_text()) if f.exists() else {}
    metrics[section] = data
    f.write_text(json.dumps(metrics, indent=2, sort_keys=True))
```

`src/ber/cli.py`:
```python
import argparse
import importlib
import json

from ber.config import load_config

# name -> (module, help). Each module defines add_arguments(parser) and run(cfg, args).
COMMANDS: dict[str, tuple[str, str]] = {
    "show-config": ("ber.cli", "print the resolved config"),
}


def add_arguments(parser: argparse.ArgumentParser) -> None:
    pass


def run(cfg: dict, args: argparse.Namespace) -> None:
    print(json.dumps(cfg, indent=2))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ber")
    parser.add_argument("--config", default="configs/base.yaml")
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name, (module, help_) in COMMANDS.items():
        importlib.import_module(module).add_arguments(sub.add_parser(name, help=help_))
    args = parser.parse_args(argv)
    importlib.import_module(COMMANDS[args.cmd][0]).run(load_config(args.config), args)
```

`src/ber/__main__.py`:
```python
from ber.cli import main

main()
```

- [ ] **Step 11: Install the package (editable, so `python -m ber` works) and run the tests to verify they pass**

Run: `scripts/hpc.sh "pip install -q --no-deps -e . && pytest tests/test_config.py -q"`
Expected: `3 passed`

- [ ] **Step 12: Commit**

```bash
git add pyproject.toml requirements.in requirements.txt .gitignore scripts/ slurm/ configs/ src/ber/__init__.py src/ber/__main__.py src/ber/config.py src/ber/cli.py tests/test_config.py
git commit -m "Scaffold ber package, HPC sync script, SLURM wrappers, config and CLI"
```

---

### Task 2: Official metric (`evaluate`)

**Files:**
- Create: `src/ber/evaluate.py`, `tests/test_evaluate.py`

**Interfaces:**
- Produces:
  - `f05(pred: set[str], truth: set[str]) -> float`
  - `macro_f05(pred: dict[str, set[str]], truth: dict[str, set[str]]) -> float` (averages over **truth keys**; a missing pred counts as empty)
  - `f05_breakdown(pred, truth, groups: dict[str, str]) -> dict[str, float]` (mean F0.5 per group label)
  - `pair_completeness(cands: pd.DataFrame, truth_pairs: pd.DataFrame) -> float` (both frames have columns `s1_id`, `cand_id`; returns the fraction of `truth_pairs` rows present in `cands`)
  - `candidates_per_entity(cands: pd.DataFrame, s1_ids: list[str]) -> dict` (`{"mean":…, "p95":…, "max":…}`, counting entities with 0 candidates)

- [ ] **Step 1: Write the failing test**

`tests/test_evaluate.py`:
```python
import pandas as pd
import pytest

from ber.evaluate import candidates_per_entity, f05, f05_breakdown, macro_f05, pair_completeness


def test_readme_example():
    pred = {"S2-00047", "S2-00193", "S3-00812"}
    truth = {"S2-00047", "S3-00812"}
    assert f05(pred, truth) == pytest.approx(0.7142857, abs=1e-6)


def test_singleton_rules():
    assert f05(set(), set()) == 1.0
    assert f05({"S2-1"}, set()) == 0.0
    assert f05(set(), {"S2-1"}) == 0.0
    assert f05({"S2-9"}, {"S2-1"}) == 0.0


def test_macro_averages_over_truth_keys_and_missing_pred_is_empty():
    truth = {"S1-a": {"S2-1"}, "S1-b": set(), "S1-c": {"S3-1", "S3-2"}}
    pred = {"S1-a": {"S2-1"}, "S1-c": {"S3-1"}}  # S1-b missing -> empty -> 1.0
    expected = (1.0 + 1.0 + f05({"S3-1"}, {"S3-1", "S3-2"})) / 3
    assert macro_f05(pred, truth) == pytest.approx(expected)


def test_breakdown():
    truth = {"a": {"x"}, "b": set()}
    pred = {"a": set(), "b": set()}
    assert f05_breakdown(pred, truth, {"a": "US", "b": "India"}) == {"US": 0.0, "India": 1.0}


def test_pair_completeness_and_counts():
    truth = pd.DataFrame({"s1_id": ["a", "a", "b"], "cand_id": ["x", "y", "z"]})
    cands = pd.DataFrame({"s1_id": ["a", "a", "b"], "cand_id": ["x", "q", "z"]})
    assert pair_completeness(cands, truth) == pytest.approx(2 / 3)
    stats = candidates_per_entity(cands, ["a", "b", "c"])
    assert stats["mean"] == pytest.approx(1.0) and stats["max"] == 2
```

- [ ] **Step 2: Run it to verify it fails**

Run: `scripts/hpc.sh pytest tests/test_evaluate.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'ber.evaluate'`

- [ ] **Step 3: Implement**

`src/ber/evaluate.py`:
```python
from collections import defaultdict

import numpy as np
import pandas as pd


def f05(pred: set[str], truth: set[str]) -> float:
    """Per-entity F0.5 exactly as the challenge defines it (singletons included)."""
    if not truth:
        return 1.0 if not pred else 0.0
    tp = len(pred & truth)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(truth)
    return 1.25 * p * r / (0.25 * p + r)


def macro_f05(pred: dict[str, set[str]], truth: dict[str, set[str]]) -> float:
    return float(np.mean([f05(pred.get(k, set()), t) for k, t in truth.items()]))


def f05_breakdown(pred: dict[str, set[str]], truth: dict[str, set[str]], groups: dict[str, str]) -> dict[str, float]:
    scores: dict[str, list[float]] = defaultdict(list)
    for k, t in truth.items():
        scores[groups[k]].append(f05(pred.get(k, set()), t))
    return {g: float(np.mean(v)) for g, v in scores.items()}


def pair_completeness(cands: pd.DataFrame, truth_pairs: pd.DataFrame) -> float:
    if len(truth_pairs) == 0:
        return 1.0
    hit = truth_pairs[["s1_id", "cand_id"]].merge(
        cands[["s1_id", "cand_id"]].drop_duplicates(), how="left", indicator=True
    )
    return float((hit["_merge"] == "both").mean())


def candidates_per_entity(cands: pd.DataFrame, s1_ids: list[str]) -> dict:
    counts = cands.groupby("s1_id").size().reindex(s1_ids, fill_value=0)
    return {"mean": float(counts.mean()), "p95": float(counts.quantile(0.95)), "max": int(counts.max())}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `scripts/hpc.sh pytest tests/test_evaluate.py -q`
Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add src/ber/evaluate.py tests/test_evaluate.py
git commit -m "Add official macro F0.5 metric and blocking completeness"
```

---

### Task 3: Data IO

**Files:**
- Create: `src/ber/io.py`, `tests/test_io.py`

**Interfaces:**
- Produces:
  - `read_family(data_dir: str | Path, family: str) -> pd.DataFrame`. `family` is `"train"` or `"test"`. Returns columns `entity_id, business_name, business_address, country, source` (`source` is an int8 of 1/2/3), with all three sources concatenated.
  - `read_truth(data_dir: str | Path) -> dict[str, set[str]]`, keyed by every S1 id in the ground truth.
  - `truth_pairs(truth: dict[str, set[str]], s1_ids: list[str] | None = None) -> pd.DataFrame`, with columns `s1_id, cand_id`.

- [ ] **Step 1: Write the failing test**

`tests/test_io.py`:
```python
from ber.io import read_family, read_truth, truth_pairs

HDR = "entity_id\tbusiness_name\tbusiness_address\tcountry\n"


def _write(d, family):
    (d / family).mkdir(parents=True)
    (d / family / f"{family}_source1.tsv").write_text(HDR + 'S1-1\tA "quoted" & Co\t1 Main St, X, TX\tUS\nS1-2\tB\t\tUS\n')
    (d / family / f"{family}_source2.tsv").write_text(HDR + "S2-1\tA Co\t1 MAIN ST\tUS\n")
    (d / family / f"{family}_source3.tsv").write_text(HDR + "S3-1\tB\t2 Rd\tUS\n")


def test_read_family_concatenates_with_source(tmp_path):
    _write(tmp_path, "train")
    df = read_family(tmp_path, "train")
    assert list(df.columns) == ["entity_id", "business_name", "business_address", "country", "source"]
    assert df.source.tolist() == [1, 1, 2, 3]
    assert df.business_name.iloc[0] == 'A "quoted" & Co'  # QUOTE_NONE keeps quotes
    assert df.business_address.iloc[1] == ""  # empty stays "", not NaN


def test_truth(tmp_path):
    (tmp_path / "train").mkdir()
    (tmp_path / "train" / "train_ground_truth.tsv").write_text(
        "source1_entity_id\tmatched_entity_ids\nS1-1\tS2-1,S3-1\nS1-2\t\n"
    )
    t = read_truth(tmp_path)
    assert t == {"S1-1": {"S2-1", "S3-1"}, "S1-2": set()}
    tp = truth_pairs(t)
    assert sorted(map(tuple, tp.values.tolist())) == [("S1-1", "S2-1"), ("S1-1", "S3-1")]
    assert len(truth_pairs(t, ["S1-2"])) == 0
```

- [ ] **Step 2: Run it to verify it fails**

Run: `scripts/hpc.sh pytest tests/test_io.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'ber.io'`

- [ ] **Step 3: Implement**

`src/ber/io.py`:
```python
import csv
from pathlib import Path

import pandas as pd


def _read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)


def read_family(data_dir: str | Path, family: str) -> pd.DataFrame:
    parts = []
    for s in (1, 2, 3):
        df = _read_tsv(Path(data_dir) / family / f"{family}_source{s}.tsv")
        df["source"] = s
        parts.append(df)
    out = pd.concat(parts, ignore_index=True)
    out["source"] = out["source"].astype("int8")
    return out


def read_truth(data_dir: str | Path) -> dict[str, set[str]]:
    gt = _read_tsv(Path(data_dir) / "train" / "train_ground_truth.tsv")
    return {
        s1: {x for x in ids.split(",") if x}
        for s1, ids in zip(gt["source1_entity_id"], gt["matched_entity_ids"])
    }


def truth_pairs(truth: dict[str, set[str]], s1_ids: list[str] | None = None) -> pd.DataFrame:
    keys = truth.keys() if s1_ids is None else s1_ids
    rows = [(s1, c) for s1 in keys for c in truth.get(s1, ())]
    return pd.DataFrame(rows, columns=["s1_id", "cand_id"])
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `scripts/hpc.sh pytest tests/test_io.py -q`
Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add src/ber/io.py tests/test_io.py
git commit -m "Add TSV and ground-truth readers"
```

---

### Task 4: Lexicons and text normalization functions

**Files:**
- Create: `src/ber/lexicons/__init__.py`, `src/ber/lexicons/en.yaml`, `src/ber/lexicons/in.yaml`, `src/ber/lexicons/fr.yaml`, `src/ber/text.py`, `tests/test_text.py`

**Interfaces:**
- Produces:
  - `Lexicon` (frozen dataclass) with fields:
    - `fillers: frozenset[str]`
    - `stopwords: frozenset[str]`
    - `addr_fillers: frozenset[str]`
    - `abbreviations: dict[str, str]`
    - `states: dict[str, dict[str, str]]` (country → {abbr: full})
    - `native: dict[str, str]`
    - `legal_index: dict[str, list[tuple[tuple[str, ...], str]]]`
  - `load_lexicon() -> Lexicon` (cached)
  - In `ber.text`:
    - `fold(s: str) -> str`
    - `normalize_name(raw: str, lex: Lexicon) -> tuple[str, str, str]`, returning `(name_norm, name_core, legal_suffix)`
    - `normalize_address(raw: str, country: str, lex: Lexicon) -> tuple[str, str, str]`, returning `(addr_norm, house_no, num_tokens)`
    - `is_domain_name(raw: str) -> bool`
    - `domain_body(raw: str) -> str`
    - `segment(s: str, vocab: set[str], max_len: int = 20) -> list[str]`
    - `record_text(name: str, addr: str, country: str) -> str`

- [ ] **Step 1: Write the lexicon files**

`src/ber/lexicons/en.yaml`:
```yaml
fillers: [the, mr, mrs, ms]
stopwords: [and, of]
addr_fillers: [door, no, number, null]
legal_suffixes:
  llc: ["llc", "l l c"]
  inc: ["inc", "incorporated"]
  corp: ["corp", "corporation"]
  co: ["co", "company"]
  ltd: ["ltd", "limited"]
  pc: ["pc", "p c"]
  pllc: ["pllc"]
  llp: ["llp"]
  lp: ["lp"]
  plc: ["plc"]
  dba: ["dba", "d b a"]
abbreviations:
  rd: road
  st: street
  str: street
  ave: avenue
  av: avenue
  blvd: boulevard
  dr: drive
  ln: lane
  ct: court
  pl: place
  sq: square
  hwy: highway
  pkwy: parkway
  cir: circle
  fl: floor
  flr: floor
  bldg: building
  nr: near
  opp: opposite
  apt: apartment
  mkt: market
  ext: extension
  sec: sector
  sect: sector
states:
  US: {al: alabama, ak: alaska, az: arizona, ar: arkansas, ca: california, co: colorado, ct: connecticut,
       de: delaware, fl: florida, ga: georgia, hi: hawaii, id: idaho, il: illinois, in: indiana, ia: iowa,
       ks: kansas, ky: kentucky, la: louisiana, me: maine, md: maryland, ma: massachusetts, mi: michigan,
       mn: minnesota, ms: mississippi, mo: missouri, mt: montana, ne: nebraska, nv: nevada,
       nh: new hampshire, nj: new jersey, nm: new mexico, ny: new york, nc: north carolina,
       nd: north dakota, oh: ohio, ok: oklahoma, or: oregon, pa: pennsylvania, ri: rhode island,
       sc: south carolina, sd: south dakota, tn: tennessee, tx: texas, ut: utah, vt: vermont,
       va: virginia, wa: washington, wv: west virginia, wi: wisconsin, wy: wyoming, dc: district of columbia}
```

`src/ber/lexicons/in.yaml`:
```yaml
fillers: [shri, sri, smt]
stopwords: []
addr_fillers: []
legal_suffixes:
  pvt_ltd: ["private limited", "pvt ltd", "pvt limited", "private ltd", "pvt", "private"]
  opc: ["opc"]
abbreviations:
  bangalore: bengaluru
  bombay: mumbai
  madras: chennai
  calcutta: kolkata
  gurgaon: gurugram
  poona: pune
states:
  India: {ap: andhra pradesh, ar: arunachal pradesh, as: assam, br: bihar, cg: chhattisgarh,
          ct: chhattisgarh, ga: goa, gj: gujarat, hr: haryana, hp: himachal pradesh, jh: jharkhand,
          jk: jammu and kashmir, ka: karnataka, kl: kerala, mp: madhya pradesh, mh: maharashtra,
          mn: manipur, ml: meghalaya, mz: mizoram, nl: nagaland, od: odisha, or: odisha, pb: punjab,
          rj: rajasthan, sk: sikkim, tn: tamil nadu, ts: telangana, tg: telangana, tr: tripura,
          up: uttar pradesh, uk: uttarakhand, ut: uttarakhand, wb: west bengal, dl: delhi,
          ch: chandigarh, py: puducherry}
native:
  "ಕರ್ನಾಟಕ": Karnataka
  "தமிழ்நாடு": Tamil Nadu
  "महाराष्ट्र": Maharashtra
  "उत्तर प्रदेश": Uttar Pradesh
  "दिल्ली": Delhi
  "राजस्थान": Rajasthan
  "गुजरात": Gujarat
  "ગુજરાત": Gujarat
  "తెలంగాణ": Telangana
  "ఆంధ్రప్రదేశ్": Andhra Pradesh
  "পশ্চিমবঙ্গ": West Bengal
  "കേരളം": Kerala
```

`src/ber/lexicons/fr.yaml`:
```yaml
fillers: []
stopwords: [et, de, du, des, la, le, les, d, l]
addr_fillers: []
legal_suffixes:
  sarl: ["sarl", "s a r l"]
  sas: ["sas", "s a s"]
  sasu: ["sasu"]
  sa: ["sa", "s a"]
  sci: ["sci"]
  eurl: ["eurl"]
  snc: ["snc"]
  selarl: ["selarl"]
  cie: ["cie", "compagnie"]
abbreviations:
  r: rue
  bd: boulevard
  bld: boulevard
  ch: chemin
  rte: route
  imp: impasse
  fbg: faubourg
states: {}
```

- [ ] **Step 2: Write the failing tests**

`tests/test_text.py`:
```python
from ber.lexicons import load_lexicon
from ber.text import (domain_body, fold, is_domain_name, normalize_address, normalize_name,
                      record_text, segment)

LEX = load_lexicon()


def test_fold():
    assert fold("Tri-State Córnerstone Bérto") == "tri-state cornerstone berto"


def test_name_suffix_and_core():
    assert normalize_name("Tri-State Cornerstone Berto  [LLC]", LEX) == ("tri state cornerstone berto llc", "tri state cornerstone berto", "llc")
    assert normalize_name("Galaxy Solutions Pvt Ltd.", LEX)[1:] == ("galaxy solutions", "pvt_ltd")
    assert normalize_name("Solutions Jain Benefit of Noida Private", LEX)[1:] == ("solutions jain benefit noida", "pvt_ltd")
    assert normalize_name("Europ & Frères Distribution S.A.", LEX)[1:] == ("europ freres distribution", "sa")
    assert normalize_name("ZNB Club SARL", LEX)[2] == "sarl"


def test_name_fillers_dedupe_and_reorder():
    norm, core, suf = normalize_name("The The Morgan, Leonanie F., O.D., DDS PC", LEX)
    assert norm == "morgan leonanie f o d dds pc" and suf == "pc"
    assert normalize_name("M/s #southerneducational", LEX)[0] == "southerneducational"
    assert normalize_name("Co Nautical Center", LEX)[1:] == ("nautical center", "co")
    assert normalize_name("Nautical & Co", LEX)[1:] == ("nautical", "co")


def test_name_core_never_empty():
    assert normalize_name("Pvt Ltd", LEX)[1] == "pvt ltd"


def test_address_us():
    assert normalize_address("1500 JUPITER RD, PO BOX 8832, ALLEN, TX", "US", LEX) == (
        "1500 jupiter road po box 8832 allen texas", "1500", "1500 8832")
    assert normalize_address("Texas, # 609, Allen, 1500 Jupiter Road", "US", LEX)[1] == "1500"
    assert normalize_address("01130 REGENCY ROAD, ATL, GA", "US", LEX)[1] == "1130"
    assert normalize_address("1130- Regency Road, Atlanat, Georgia", "US", LEX)[1] == "1130"


def test_address_india():
    norm, house, nums = normalize_address("NULL, MH, 4-7/1 To 14 Plot No. 15 Airport Road, NULL, Kolhapur", "India", LEX)
    assert house == "47/1" and "maharashtra" in norm and "null" not in norm.split()
    assert normalize_address("DOOR NO 164, MOC SINGAPORE PLAZA", "India", LEX)[:2] == ("164 moc singapore plaza", "164")
    norm = normalize_address("Door No 236 Floor Salarapuria, Bengaluru, ಕರ್ನಾಟಕ, Bangalore", "India", LEX)[0]
    assert norm.split().count("bengaluru") == 2 and "karnataka" in norm


def test_state_map_is_per_country():
    # "GA" is Georgia in the US and Goa in India; unknown countries get no state expansion.
    assert normalize_address("X, GA", "US", LEX)[0] == "x georgia"
    assert normalize_address("X, GA", "India", LEX)[0] == "x goa"
    assert normalize_address("X, GA", "France", LEX)[0] == "x ga"


def test_address_france():
    assert normalize_address("63 R. DE DIEPPE, LILLE, Hauts-de-France", "France", LEX)[:2] == (
        "63 rue de dieppe lille hauts de france", "63")


def test_domain_detection_and_segmentation():
    assert is_domain_name("fafloonpetcare.com") and is_domain_name("M/s #southerneducational") is False
    assert is_domain_name("#southerneducational") and not is_domain_name("Fafloon Pet Care Inc")
    assert domain_body("www.2827art.com") == "2827art"
    assert segment("fafloonpetcare", {"fafloon", "pet", "care"}) == ["fafloon", "pet", "care"]
    assert segment("2827art", {"art"}) == ["2827", "art"]


def test_record_text():
    assert record_text("A Co", "1 Main St", "US") == "name: A Co | address: 1 Main St | country: US"
```

- [ ] **Step 3: Run it to verify it fails**

Run: `scripts/hpc.sh pytest tests/test_text.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'ber.lexicons'`

- [ ] **Step 4: Implement the lexicon loader**

`src/ber/lexicons/__init__.py`:
```python
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

FILES = ("en.yaml", "in.yaml", "fr.yaml")


@dataclass(frozen=True)
class Lexicon:
    fillers: frozenset[str]
    stopwords: frozenset[str]
    addr_fillers: frozenset[str]
    abbreviations: dict[str, str]
    states: dict[str, dict[str, str]]  # country label -> {abbr: full name}
    native: dict[str, str]  # native-script string -> English name
    # first token -> [(pattern tokens, canonical)], longest pattern first
    legal_index: dict[str, list[tuple[tuple[str, ...], str]]]


@lru_cache(maxsize=1)
def load_lexicon() -> Lexicon:
    fillers, stop, addr_f = set(), set(), set()
    abbr, states, native, legal = {}, {}, {}, {}
    for name in FILES:
        d = yaml.safe_load((Path(__file__).parent / name).read_text())
        fillers |= set(d.get("fillers", []))
        stop |= set(d.get("stopwords", []))
        addr_f |= set(d.get("addr_fillers", []))
        abbr.update(d.get("abbreviations", {}))
        states.update(d.get("states", {}) or {})
        native.update(d.get("native", {}) or {})
        for canon, variants in d.get("legal_suffixes", {}).items():
            for v in variants:
                toks = tuple(v.split())
                legal.setdefault(toks[0], []).append((toks, canon))
    for pats in legal.values():
        pats.sort(key=lambda p: -len(p[0]))
    states = {c: {str(k): v for k, v in m.items()} for c, m in states.items()}
    return Lexicon(frozenset(fillers), frozenset(stop), frozenset(addr_f), abbr, states, native, legal)
```

Note: `str(k)` guards against YAML 1.1 booleans (`yes/no/on/off/true/false`). None of the current state codes are booleans, but a future addition like `on` would otherwise silently become `True`.

- [ ] **Step 5: Implement the text functions**

`src/ber/text.py`:
```python
import re
import unicodedata

from anyascii import anyascii

from ber.lexicons import Lexicon

NUM_RE = re.compile(r"\d[\d/\-]*[a-z]?")
ADDR_TOKEN_RE = re.compile(r"#|[a-z0-9][a-z0-9/\-]*")
UNIT_WORDS = frozenset({"#", "unit", "suite", "apt", "apartment", "fl", "floor", "box", "lot",
                        "sector", "sec", "pin", "flat", "room", "ste"})
DOMAIN_RE = re.compile(r"#?(www\.)?[a-z0-9\-]+(\.[a-z]{2,4}){0,2}")


def fold(s: str) -> str:
    return anyascii(unicodedata.normalize("NFKC", s)).lower()


def _strip_legal(toks: list[str], lex: Lexicon) -> tuple[list[str], list[str]]:
    core, suffixes, i = [], [], 0
    while i < len(toks):
        for pat, canon in lex.legal_index.get(toks[i], ()):
            if tuple(toks[i:i + len(pat)]) == pat:
                suffixes.append(canon)
                i += len(pat)
                break
        else:
            core.append(toks[i])
            i += 1
    return core, suffixes


def normalize_name(raw: str, lex: Lexicon) -> tuple[str, str, str]:
    s = fold(raw)
    s = re.sub(r"\bm/s\b", " ", s).replace("&", " and ")
    toks = [t for t in re.sub(r"[^a-z0-9]+", " ", s).split() if t != "null" and t not in lex.fillers]
    toks = [t for i, t in enumerate(toks) if i == 0 or t != toks[i - 1]]
    core, suffixes = _strip_legal(toks, lex)
    core = [t for t in core if t not in lex.stopwords]
    name_norm = " ".join(toks)
    return name_norm, " ".join(core) or name_norm, "+".join(sorted(set(suffixes)))


def _canon_num(tok: str) -> str:
    return tok.replace("-", "").strip("/").lstrip("0") or "0"


def normalize_address(raw: str, country: str, lex: Lexicon) -> tuple[str, str, str]:
    for native, english in lex.native.items():
        if native in raw:
            raw = raw.replace(native, english)
    states = lex.states.get(country, {})
    comps = [c.strip() for c in fold(raw).split(",")]
    s = " , ".join(states.get(c, c) for c in comps if c and c != "null")
    out, nums, house_no, prev = [], [], "", ""
    for t in ADDR_TOKEN_RE.findall(s):
        if NUM_RE.fullmatch(t):
            c = _canon_num(t)
            nums.append(c)
            if not house_no and prev not in UNIT_WORDS and prev != "po" and not (c.isdigit() and len(c) == 6):
                house_no = c
            out.append(c)
        elif t != "#":
            out.extend(p for p in t.split("-") if p)
        prev = t
    out = [lex.abbreviations.get(t, t) for t in out]
    out = [t for t in out if t not in lex.addr_fillers]
    return " ".join(out), house_no, " ".join(dict.fromkeys(nums))


def is_domain_name(raw: str) -> bool:
    f = fold(raw).strip()
    return len(f) >= 6 and ("." in f or f.startswith("#")) and DOMAIN_RE.fullmatch(f) is not None


def domain_body(raw: str) -> str:
    f = fold(raw).strip()
    f = re.sub(r"^#|^www\.", "", f)
    f = re.sub(r"(\.[a-z]{2,4})+$", "", f)
    return f.replace("-", "")


def segment(s: str, vocab: set[str], max_len: int = 20) -> list[str]:
    """Min-cost split of s into vocab words; digits runs are free words, unknown chunks cost 1+len."""
    n = len(s)
    best: list[tuple[float, int] | None] = [(0.0, 0)] + [None] * n
    for i in range(1, n + 1):
        for j in range(max(0, i - max_len), i):
            if best[j] is None:
                continue
            w = s[j:i]
            cost = best[j][0] + (1 if (w in vocab or w.isdigit()) else 1 + len(w))
            if best[i] is None or cost < best[i][0]:
                best[i] = (cost, j)
    out, i = [], n
    while i > 0:
        j = best[i][1]
        out.append(s[j:i])
        i = j
    return out[::-1]


def record_text(name: str, addr: str, country: str) -> str:
    return f"name: {name} | address: {addr} | country: {country}"
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `scripts/hpc.sh pytest tests/test_text.py -q`
Expected: `10 passed`. If a normalization assertion fails, fix the function, not the test. The tests encode the EDA examples.

- [ ] **Step 7: Commit**

```bash
git add src/ber/lexicons src/ber/text.py tests/test_text.py
git commit -m "Add multilingual lexicons and name/address normalization"
```

---

### Task 5: Build the `records` table (`normalize` command)

**Files:**
- Create: `src/ber/normalize.py`, `tests/test_normalize.py`
- Modify: `src/ber/cli.py` (register the command)

**Interfaces:**
- Consumes:
  - `read_family`
  - `load_lexicon`
  - `normalize_name`, `normalize_address`, `is_domain_name`, `domain_body`, `segment` from Task 4
- Produces:
  - `build_records(raw: pd.DataFrame, n_jobs: int) -> pd.DataFrame` with columns `entity_id, source, country, name_raw, addr_raw, name_norm, name_core, legal_suffix, name_domain, addr_norm, house_no, num_tokens, block_text`
  - `records_path(cfg: dict, family: str) -> Path`, i.e. `run_dir/records_{family}.parquet`
  - The `normalize` command: `python -m ber normalize [--family train test]`

- [ ] **Step 1: Write the failing test**

`tests/test_normalize.py`:
```python
import pandas as pd

from ber.normalize import build_records


def _raw():
    return pd.DataFrame({
        "entity_id": ["S1-1", "S2-1", "S3-1", "S1-2", "S2-2"],
        "business_name": ["Fafloon Pet Care Inc", "FAFLOON CARE INC", "fafloonpetcare.com", "Galaxy Solutions Pvt Ltd", "Galaxy S0lutions Pvt Ltd."],
        "business_address": ["1130 Regency Road, Atlanta, GA", "01130 REGENCY ROAD, ATL, GA", "1130- Regency Road, Atlanat, Georgia",
                             "47/1 Airport Road, Kolhapur, Maharashtra", "NULL, MH, 4-7/1 Airport Road"],
        "country": ["US", "US", "US", "India", "India"],
        "source": pd.Series([1, 2, 3, 1, 2], dtype="int8"),
    })


def test_columns_and_domain_split():
    rec = build_records(_raw(), n_jobs=1)
    assert list(rec.columns) == ["entity_id", "source", "country", "name_raw", "addr_raw", "name_norm", "name_core",
                                 "legal_suffix", "name_domain", "addr_norm", "house_no", "num_tokens", "block_text"]
    r = rec.set_index("entity_id")
    assert r.loc["S3-1", "name_domain"] == "fafloon pet care"
    assert r.loc["S1-1", "name_domain"] == ""
    assert r.loc["S3-1", "block_text"].startswith("fafloon pet care ")
    assert r.loc["S2-2", "house_no"] == r.loc["S1-2", "house_no"] == "47/1"


def test_parallel_matches_serial():
    raw = pd.concat([_raw()] * 20, ignore_index=True)
    pd.testing.assert_frame_equal(build_records(raw, n_jobs=1), build_records(raw, n_jobs=4))
```

- [ ] **Step 2: Run it to verify it fails**

Run: `scripts/hpc.sh pytest tests/test_normalize.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'ber.normalize'`

- [ ] **Step 3: Implement**

`src/ber/normalize.py`:
```python
import argparse
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

from ber.config import run_dir
from ber.io import read_family
from ber.lexicons import load_lexicon
from ber.text import domain_body, is_domain_name, normalize_address, normalize_name, segment

COLUMNS = ["entity_id", "source", "country", "name_raw", "addr_raw", "name_norm", "name_core",
           "legal_suffix", "name_domain", "addr_norm", "house_no", "num_tokens", "block_text"]


def _normalize_chunk(raw: pd.DataFrame) -> pd.DataFrame:
    lex = load_lexicon()
    names = [normalize_name(n, lex) for n in raw["business_name"]]
    addrs = [normalize_address(a, c, lex) for a, c in zip(raw["business_address"], raw["country"])]
    out = pd.DataFrame({
        "entity_id": raw["entity_id"].values,
        "source": raw["source"].values,
        "country": raw["country"].values,
        "name_raw": raw["business_name"].values,
        "addr_raw": raw["business_address"].values,
    })
    out["name_norm"], out["name_core"], out["legal_suffix"] = map(list, zip(*names)) if names else ([], [], [])
    out["addr_norm"], out["house_no"], out["num_tokens"] = map(list, zip(*addrs)) if addrs else ([], [], [])
    out["is_domain"] = [is_domain_name(n) for n in raw["business_name"]]
    return out


def _segment_domains(rec: pd.DataFrame) -> pd.Series:
    """Split domain-style names using a per-country vocabulary of ordinary name tokens."""
    result = pd.Series("", index=rec.index)
    for country, grp in rec[rec["is_domain"]].groupby("country"):
        plain = rec.loc[(rec["country"] == country) & ~rec["is_domain"], "name_core"]
        vocab = {t for t in plain.str.split().explode().dropna() if len(t) >= 2}
        result.loc[grp.index] = [" ".join(segment(domain_body(n), vocab)) for n in grp["name_raw"]]
    return result


def build_records(raw: pd.DataFrame, n_jobs: int) -> pd.DataFrame:
    if n_jobs <= 1:
        rec = _normalize_chunk(raw)
    else:
        chunks = np.array_split(raw, n_jobs * 4)
        with ProcessPoolExecutor(n_jobs, mp_context=get_context("fork")) as ex:
            rec = pd.concat(list(ex.map(_normalize_chunk, chunks)), ignore_index=True)
    rec["name_domain"] = _segment_domains(rec)
    name_for_block = rec["name_domain"].where(rec["name_domain"] != "", rec["name_core"])
    rec["block_text"] = name_for_block + " " + rec["addr_norm"]
    return rec[COLUMNS].reset_index(drop=True)


def records_path(cfg: dict, family: str) -> Path:
    return run_dir(cfg) / f"records_{family}.parquet"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--family", nargs="+", default=["train", "test"], choices=["train", "test"])


def run(cfg: dict, args: argparse.Namespace) -> None:
    for family in args.family:
        rec = build_records(read_family(cfg["paths"]["data_dir"], family), cfg["n_jobs"])
        rec.to_parquet(records_path(cfg, family), index=False)
        print(family, len(rec), "records; domain names:", int((rec.name_domain != "").sum()),
              "; house_no present:", round(float((rec.house_no != "").mean()), 3))
```

Add to the `COMMANDS` dict in `src/ber/cli.py`:
```python
    "normalize": ("ber.normalize", "build records_{family}.parquet"),
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `scripts/hpc.sh pytest tests/test_normalize.py -q`
Expected: `2 passed`

- [ ] **Step 5: Run on the real data**

```bash
scripts/hpc.sh "mkdir -p artifacts/logs && sbatch slurm/cpu.sbatch python -m ber normalize"
scripts/hpc.sh "squeue -u fcse_shwetasharma; tail -n 5 artifacts/logs/ber-cpu-*.out"
```
Repeat the second command until the job leaves the queue.

Expected log: two lines, `train 12527040 records; …` and `test 11702133 records; …`

- [ ] **Step 6: Data-driven lexicon check**

Create `scripts/lexicon_gaps.py`:
```python
"""Print frequent 2-3 letter address components and non-ASCII components per country (lexicon gaps)."""
import pandas as pd

for family in ("train", "test"):
    df = pd.read_parquet(f"artifacts/v1/records_{family}.parquet", columns=["country", "addr_raw"])
    comps = df.assign(c=df.addr_raw.str.split(",")).explode("c")
    comps["c"] = comps.c.str.strip()
    short = comps[comps.c.str.fullmatch(r"[A-Za-z]{2,3}", na=False)]
    nonascii = comps[comps.c.str.contains(r"[^\x00-\x7F]", regex=True, na=False)]
    for label, sub in (("short", short), ("non-ascii", nonascii)):
        print(f"== {family} {label}")
        print(sub.groupby("country").c.value_counts().groupby(level=0).head(40).to_string())
```

Run: `scripts/hpc.sh python scripts/lexicon_gaps.py`

For each frequent component that isn't in the lexicons:
- a state abbreviation → add it under `states.<country>`
- a native-script state name → add it under `native`
- a city abbreviation such as `ATL` → add it under `abbreviations`

Then re-run Step 4 and Step 5.

- [ ] **Step 7: Commit**

```bash
git add src/ber/normalize.py src/ber/cli.py tests/test_normalize.py src/ber/lexicons scripts/lexicon_gaps.py
git commit -m "Add parallel records builder and normalize command"
```

---

### Task 6: Noise augmentation operators

**Files:**
- Create: `src/ber/noise.py`, `tests/test_noise.py`

**Interfaces:**
- Consumes: `load_lexicon()` (for per-country state maps)
- Produces:
  - `NAME_OPS: list[Callable[[str, np.random.Generator], str]]`
  - `ADDR_OPS: list[Callable[[str, str, np.random.Generator], str]]`, with args `(addr, country, rng)`
  - `augment(name: str, addr: str, country: str, rng: np.random.Generator, max_ops: int = 2) -> tuple[str, str]`

- [ ] **Step 1: Write the failing test**

`tests/test_noise.py`:
```python
import numpy as np

from ber.noise import ADDR_OPS, NAME_OPS, augment

NAME = "Galaxy Solutions Private Limited"
ADDR = "47 Airport Road, Kolhapur, Maharashtra"


def test_every_name_op_changes_text_for_some_seed():
    for op in NAME_OPS:
        outs = {op(NAME, np.random.default_rng(s)) for s in range(20)}
        assert outs - {NAME}, op.__name__
        assert all(o.strip() for o in outs), op.__name__


def test_every_addr_op_changes_text_for_some_seed():
    for op in ADDR_OPS:
        outs = {op(ADDR, "India", np.random.default_rng(s)) for s in range(20)}
        assert outs - {ADDR}, op.__name__


def test_augment_deterministic_and_name_never_empty():
    a = [augment(NAME, ADDR, "India", np.random.default_rng(7)) for _ in range(2)]
    assert a[0] == a[1]
    for s in range(300):
        n, _ = augment("X", "", "France", np.random.default_rng(s))
        assert n.strip()
```

- [ ] **Step 2: Run it to verify it fails**

Run: `scripts/hpc.sh pytest tests/test_noise.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'ber.noise'`

- [ ] **Step 3: Implement**

`src/ber/noise.py`:
```python
"""Stochastic noise operators mirroring the patterns observed in the data (approaches.md §8a)."""
import re

import numpy as np

from ber.lexicons import load_lexicon

ACCENTS = {"a": "á", "e": "é", "o": "ó", "i": "í", "u": "ú"}
LOOKALIKE = {"o": "0", "l": "1", "i": "1", "s": "5", "e": "3"}
PREFIX_FILLERS = ["The", "Mr", "M/s", "The The"]
SUFFIX_FILLERS = ["Center", "Group", "Services"]
SUFFIX_SWAPS = [("Private Limited", "Pvt Ltd"), ("Limited", "Ltd"), ("Private", "Pvt"), ("Incorporated", "Inc"),
                ("Corporation", "Corp"), ("Company", "Co"), ("SARL", "S.A.R.L."), ("SAS", "S.A.S")]
STREET_SWAPS = [("Road", "Rd"), ("Street", "St"), ("Avenue", "Ave"), ("Boulevard", "Blvd"), ("Rue", "R."),
                ("Floor", "Fl"), ("Near", "Nr"), ("Opposite", "Opp")]


def _pick(rng, seq):
    return seq[int(rng.integers(len(seq)))]


def _swap_pairs(text, pairs, rng):
    hits = [(a, b) for a, b in pairs if re.search(rf"\b{re.escape(a)}\b", text, re.I)]
    hits += [(b, a) for a, b in pairs if re.search(rf"(?<!\w){re.escape(b)}(?!\w)", text, re.I)]
    if not hits:
        return text
    a, b = _pick(rng, hits)
    return re.sub(rf"(?<!\w){re.escape(a)}(?!\w)", b, text, count=1, flags=re.I)


# ---- name operators: (name, rng) -> name ----
def upper_case(s, rng):
    return s.upper() if s != s.upper() else s.title()


def typo(s, rng):
    idx = [i for i, c in enumerate(s) if c.isalpha()]
    if not idx:
        return s
    i = _pick(rng, idx)
    kind = int(rng.integers(4))
    if kind == 0:
        return s[:i] + s[i + 1:] if len(s) > 1 else s
    if kind == 1 and i + 1 < len(s):
        return s[:i] + s[i + 1] + s[i] + s[i + 2:]
    if kind == 2 and s[i].lower() in LOOKALIKE:
        return s[:i] + LOOKALIKE[s[i].lower()] + s[i + 1:]
    return s[:i] + s[i] + s[i:]


def accent(s, rng):
    idx = [i for i, c in enumerate(s) if c.lower() in ACCENTS]
    if not idx:
        return s
    i = _pick(rng, idx)
    rep = ACCENTS[s[i].lower()]
    return s[:i] + (rep.upper() if s[i].isupper() else rep) + s[i + 1:]


def filler(s, rng):
    return f"{_pick(rng, PREFIX_FILLERS)} {s}" if rng.random() < 0.5 else f"{s} {_pick(rng, SUFFIX_FILLERS)}"


def drop_token(s, rng):
    toks = s.split()
    if len(toks) < 2:
        return s
    del toks[int(rng.integers(len(toks)))]
    return " ".join(toks)


def truncate(s, rng):
    toks = s.split()
    return " ".join(toks[: int(rng.integers(1, len(toks)))]) if len(toks) >= 2 else s


def domain_style(s, rng):
    body = re.sub(r"[^a-z0-9]", "", s.lower())
    return f"#{body}" if rng.random() < 0.3 else f"{body}.com"


def suffix_swap(s, rng):
    return _swap_pairs(s, SUFFIX_SWAPS, rng)


def reorder(s, rng):
    toks = s.split()
    return " ".join([toks[-1]] + toks[:-1]) if len(toks) >= 2 else s


NAME_OPS = [upper_case, typo, accent, filler, drop_token, truncate, domain_style, suffix_swap, reorder]


# ---- address operators: (addr, country, rng) -> addr ----
def _comps(a):
    return [c.strip() for c in a.split(",") if c.strip()]


def addr_upper(a, country, rng):
    return upper_case(a, rng)


def null_insert(a, country, rng):
    c = _comps(a)
    c.insert(int(rng.integers(len(c) + 1)), "NULL")
    return ", ".join(c)


def drop_component(a, country, rng):
    c = _comps(a)
    if len(c) < 2:
        return a
    del c[int(rng.integers(len(c)))]
    return ", ".join(c)


def shuffle_components(a, country, rng):
    c = _comps(a)
    if len(c) < 2:
        return a
    perm = rng.permutation(len(c))
    if (perm == np.arange(len(c))).all():
        perm = np.roll(perm, 1)
    return ", ".join(c[i] for i in perm)


def street_swap(a, country, rng):
    return _swap_pairs(a, STREET_SWAPS, rng)


def state_swap(a, country, rng):
    states = load_lexicon().states.get(country, {})
    c = _comps(a)
    for i, comp in enumerate(c):
        low = comp.lower()
        if low in states:
            c[i] = states[low].title()
            return ", ".join(c)
        for abbr, full in states.items():
            if low == full:
                c[i] = abbr.upper()
                return ", ".join(c)
    return a


def house_reformat(a, country, rng):
    m = re.search(r"\b\d+\b", a)
    if not m:
        return a
    n = m.group()
    new = _pick(rng, [f"0{n}", f"{n}-", f"{n[0]}-{n[1:]}" if len(n) > 1 else f"0{n}"])
    return a[: m.start()] + new + a[m.end():]


def unit_toggle(a, country, rng):
    c = _comps(a)
    units = [i for i, x in enumerate(c) if re.match(r"(?i)(unit|po box|#|suite|fl)\b", x)]
    if units:
        del c[units[0]]
    else:
        c.insert(min(1, len(c)), _pick(rng, [f"Unit {int(rng.integers(1, 999))}", f"PO BOX {int(rng.integers(100, 9999))}"]))
    return ", ".join(c)


ADDR_OPS = [addr_upper, null_insert, drop_component, shuffle_components, street_swap, state_swap, house_reformat, unit_toggle]
EMPTY_ADDR_P = 0.03


def augment(name: str, addr: str, country: str, rng: np.random.Generator, max_ops: int = 2) -> tuple[str, str]:
    for i in rng.choice(len(NAME_OPS), size=int(rng.integers(1, max_ops + 1)), replace=False):
        new = NAME_OPS[i](name, rng)
        name = new if new.strip() else name
    if rng.random() < EMPTY_ADDR_P:
        return name, ""
    if addr:
        for i in rng.choice(len(ADDR_OPS), size=int(rng.integers(1, max_ops + 1)), replace=False):
            addr = ADDR_OPS[i](addr, country, rng)
    return name, addr
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `scripts/hpc.sh pytest tests/test_noise.py -q`
Expected: `3 passed`. If an operator never changes the fixture text, change the fixture so the operator applies, not the operator. For example, `state_swap` needs a state component, which "Maharashtra" provides.

- [ ] **Step 5: Commit**

```bash
git add src/ber/noise.py tests/test_noise.py
git commit -m "Add noise augmentation operators mirroring observed data noise"
```

---

### Task 7: Train splits (`split` command)

**Files:**
- Create: `src/ber/split.py`, `tests/test_split.py`
- Modify: `src/ber/cli.py`

**Interfaces:**
- Consumes: `read_truth`, `records_path`, `run_dir`, `log_metrics`
- Produces:
  - `assign_splits(truth: dict[str, set[str]], s1_country: pd.Series, cfg_split: dict, seed: int) -> pd.DataFrame` with columns `s1_id, country, n_matches, split, fold`. `split` is one of `A_train`, `A_val`, `B`. `fold` is 0..k-1 for B and -1 otherwise.
  - `splits_path(cfg) -> Path`, i.e. `run_dir/splits.parquet`
  - The `split` command

- [ ] **Step 1: Write the failing test**

`tests/test_split.py`:
```python
import pandas as pd

from ber.split import _bucket, assign_splits

CFG = {"a_train": 0.35, "a_val": 0.05, "n_folds": 5}


def _data(n=4000):
    truth = {f"S1-{i}": {f"S2-{i}-{j}" for j in range(i % 7)} for i in range(n)}
    country = pd.Series({f"S1-{i}": ("US" if i % 3 else "India") for i in range(n)})
    return truth, country


def test_shares_folds_and_determinism():
    truth, country = _data()
    a = assign_splits(truth, country, CFG, seed=1)
    b = assign_splits(truth, country, CFG, seed=1)
    pd.testing.assert_frame_equal(a, b)
    share = a.split.value_counts(normalize=True)
    assert abs(share["A_train"] - 0.35) < 0.01 and abs(share["A_val"] - 0.05) < 0.01
    assert (a[a.split != "B"].fold == -1).all()
    folds = a[a.split == "B"].fold.value_counts()
    assert set(folds.index) == set(range(5)) and folds.max() - folds.min() <= 10


def test_stratified_by_country_and_bucket():
    truth, country = _data()
    a = assign_splits(truth, country, CFG, seed=1)
    for key, grp in a.groupby([a.country, a.n_matches.map(_bucket)]):
        assert abs((grp.split == "B").mean() - 0.60) < 0.01, key
```

- [ ] **Step 2: Run it to verify it fails**

Run: `scripts/hpc.sh pytest tests/test_split.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'ber.split'`

- [ ] **Step 3: Implement**

`src/ber/split.py`:
```python
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ber.config import log_metrics, run_dir
from ber.io import read_truth
from ber.normalize import records_path


def _bucket(n: int) -> str:
    return "0" if n == 0 else "1" if n == 1 else "2-3" if n <= 3 else "4-5" if n <= 5 else "6+"


def assign_splits(truth: dict[str, set[str]], s1_country: pd.Series, cfg_split: dict, seed: int) -> pd.DataFrame:
    df = pd.DataFrame({"s1_id": sorted(truth)})
    df["country"] = s1_country.reindex(df.s1_id).values
    df["n_matches"] = [len(truth[s]) for s in df.s1_id]
    df["split"], df["fold"] = "B", -1
    rng = np.random.default_rng(seed)
    for _, idx in df.groupby([df.country, df.n_matches.map(_bucket)]).groups.items():
        idx = rng.permutation(np.asarray(idx))
        n_tr, n_val = round(len(idx) * cfg_split["a_train"]), round(len(idx) * cfg_split["a_val"])
        df.loc[idx[:n_tr], "split"] = "A_train"
        df.loc[idx[n_tr:n_tr + n_val], "split"] = "A_val"
        b = idx[n_tr + n_val:]
        df.loc[b, "fold"] = (np.arange(len(b)) + int(rng.integers(cfg_split["n_folds"]))) % cfg_split["n_folds"]
    df["fold"] = df["fold"].astype("int8")
    return df


def splits_path(cfg: dict) -> Path:
    return run_dir(cfg) / "splits.parquet"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    pass


def run(cfg: dict, args: argparse.Namespace) -> None:
    rec = pd.read_parquet(records_path(cfg, "train"), columns=["entity_id", "source", "country"])
    s1 = rec[rec.source == 1].set_index("entity_id").country
    df = assign_splits(read_truth(cfg["paths"]["data_dir"]), s1, cfg["split"], cfg["seed"])
    df.to_parquet(splits_path(cfg), index=False)
    summary = df.groupby(["split", "country"]).size().unstack(fill_value=0)
    print(summary)
    log_metrics(cfg, "split", {f"{s}|{c}": int(v) for (s, c), v in summary.stack().items()})
```

Add to `COMMANDS` in `src/ber/cli.py`:
```python
    "split": ("ber.split", "assign A_train/A_val/B splits and B folds"),
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `scripts/hpc.sh pytest tests/test_split.py -q`
Expected: `2 passed`

- [ ] **Step 5: Run on the real data**

Run: `scripts/hpc.sh python -m ber split`
Expected: a table where each country splits roughly 35/5/60, e.g. US ≈ 463k / 66k / 794k.

- [ ] **Step 6: Commit**

```bash
git add src/ber/split.py src/ber/cli.py tests/test_split.py
git commit -m "Add stratified A/B split with B folds"
```

---

### Task 8: Sparse blocking (`block-sparse` command)

**Files:**
- Create: `src/ber/blocking/__init__.py`, `src/ber/blocking/sparse.py`, `tests/test_blocking_sparse.py`
- Modify: `src/ber/cli.py`

**Interfaces:**
- Consumes:
  - `build_records` (tests)
  - `records_path`, `splits_path`, `read_truth`, `truth_pairs`, `pair_completeness`, `candidates_per_entity`, `log_metrics`
- Produces:
  - `tfidf_topk(s1: pd.DataFrame, others: pd.DataFrame, cfg_tfidf: dict) -> pd.DataFrame`, with columns `s1_id, cand_id, tfidf_sim`
  - `key_blocks(recs: pd.DataFrame, cfg_keys: dict) -> pd.DataFrame`, with columns `s1_id, cand_id, key_hits`
  - `block_family(recs: pd.DataFrame, cfg_blocking: dict) -> pd.DataFrame`, with columns `s1_id, cand_id, tfidf_sim (float32, 0 if absent), tfidf_rank (int16, 999 if absent), key_hits (int16, 0 if absent)`
  - `cand_sparse_path(cfg, family) -> Path`, i.e. `run_dir/cand_sparse_{family}.parquet`
  - The `block-sparse` command

- [ ] **Step 1: Write the failing test**

`tests/test_blocking_sparse.py`:
```python
import pandas as pd

from ber.blocking.sparse import block_family, key_blocks, tfidf_topk
from ber.normalize import build_records

CFG = {
    "tfidf": {"ngram_range": [3, 5], "min_df": 1, "top_k": 3, "min_sim": 0.1, "n_threads": 1},
    "keys": {"n_name_tokens": 2, "n_addr_tokens": 3, "min_token_len": 3, "max_block_size": 100},
}


def _recs():
    rows = [
        ("S1-1", 1, "Galaxy Solutions Pvt Ltd", "47/1 Airport Road, Kolhapur, MH", "India"),
        ("S2-1", 2, "Galaxy Solutiocns Pvt Ltd", "4-7/1 AIRPORT ROAD, KOLHAPUR", "India"),
        ("S3-1", 3, "Pvt Ltd Galaxy", "Kolhapur, 47/1 Airport Rd", "India"),
        ("S2-2", 2, "Sunrise Traders", "12 MG Road, Pune", "India"),
        ("S2-9", 2, "Galaxy Solutions Pvt Ltd", "47/1 Airport Road, Kolhapur", "France"),  # other country
        ("S1-2", 1, "Morgan Dental PC", "412 Madrona Avenue, Salem, OR", "US"),
        ("S3-2", 3, "Morgan Dental (PC)", "412 Madrona Ave, Salem, Oregon", "US"),
    ]
    raw = pd.DataFrame(rows, columns=["entity_id", "source", "business_name", "business_address", "country"])
    raw["source"] = raw.source.astype("int8")
    return build_records(raw[["entity_id", "business_name", "business_address", "country", "source"]], n_jobs=1)


def test_tfidf_finds_noisy_match_within_country():
    rec = _recs()
    india = rec[rec.country == "India"]
    out = tfidf_topk(india[india.source == 1], india[india.source != 1], CFG["tfidf"])
    got = set(out.cand_id[out.s1_id == "S1-1"])
    assert {"S2-1", "S3-1"} <= got and "S2-9" not in got


def test_key_blocks_house_number_and_name():
    rec = _recs()
    out = key_blocks(rec[rec.country == "India"], CFG["keys"])
    assert {"S2-1", "S3-1"} <= set(out.cand_id[out.s1_id == "S1-1"])
    assert (out.key_hits >= 1).all()


def test_block_family_respects_country_and_fills_defaults():
    out = block_family(_recs(), CFG)
    assert list(out.columns) == ["s1_id", "cand_id", "tfidf_sim", "tfidf_rank", "key_hits"]
    assert not ((out.s1_id == "S1-1") & (out.cand_id == "S2-9")).any()
    assert not ((out.s1_id == "S1-2") & out.cand_id.isin(["S2-1", "S3-1"])).any()
    assert ((out.s1_id == "S1-2") & (out.cand_id == "S3-2")).any()
    assert out.tfidf_rank.min() >= 1
```

- [ ] **Step 2: Run it to verify it fails**

Run: `scripts/hpc.sh pytest tests/test_blocking_sparse.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'ber.blocking'`

- [ ] **Step 3: Implement**

`src/ber/blocking/__init__.py`: empty file.

`src/ber/blocking/sparse.py`:
```python
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

from ber.config import log_metrics, run_dir
from ber.evaluate import candidates_per_entity, pair_completeness
from ber.io import read_truth, truth_pairs
from ber.normalize import records_path
from ber.split import splits_path

EMPTY = pd.DataFrame({"s1_id": pd.Series(dtype=str), "cand_id": pd.Series(dtype=str)})


def tfidf_topk(s1: pd.DataFrame, others: pd.DataFrame, cfg_tfidf: dict) -> pd.DataFrame:
    if len(s1) == 0 or len(others) == 0:
        return EMPTY.assign(tfidf_sim=pd.Series(dtype="float32"))
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=tuple(cfg_tfidf["ngram_range"]),
                          min_df=cfg_tfidf["min_df"], sublinear_tf=True, dtype=np.float32)
    vec.fit(pd.concat([s1.block_text, others.block_text]))
    a, b = vec.transform(s1.block_text), vec.transform(others.block_text)
    c = sp_matmul_topn(a, b.T.tocsr(), top_n=cfg_tfidf["top_k"], threshold=cfg_tfidf["min_sim"],
                       sort=True, n_threads=cfg_tfidf["n_threads"]).tocoo()
    return pd.DataFrame({"s1_id": s1.entity_id.values[c.row], "cand_id": others.entity_id.values[c.col],
                         "tfidf_sim": c.data.astype(np.float32)})


def _rarest(tokens: pd.Series, n: int, min_len: int) -> pd.DataFrame:
    """The n lowest-document-frequency tokens per row. `tokens` holds token lists on a RangeIndex."""
    ex = tokens.explode().dropna().rename("tok").rename_axis("row").reset_index()
    ex = ex[(ex.tok.str.len() >= min_len) & ~ex.tok.str.isdigit()].drop_duplicates()
    ex["df"] = ex.groupby("tok").tok.transform("size")
    return ex.sort_values(["row", "df", "tok"]).groupby("row").head(n)[["row", "tok"]]


def key_blocks(recs: pd.DataFrame, cfg_keys: dict) -> pd.DataFrame:
    recs = recs.reset_index(drop=True)
    name = recs.name_domain.where(recs.name_domain != "", recs.name_core)
    name_t = _rarest(name.str.split(), cfg_keys["n_name_tokens"], cfg_keys["min_token_len"])
    addr_t = _rarest(recs.addr_norm.str.split(), cfg_keys["n_addr_tokens"], cfg_keys["min_token_len"])
    hn = recs.house_no.rename_axis("row").reset_index()
    hn = hn[hn.house_no != ""]
    k1 = addr_t.merge(hn, on="row")
    k1["key"] = "h|" + k1.house_no + "|" + k1.tok
    k2 = name_t.merge(addr_t, on="row", suffixes=("_n", "_a"))
    k2["key"] = "n|" + k2.tok_n + "|" + k2.tok_a
    keys = pd.concat([k1[["row", "key"]], k2[["row", "key"]]]).drop_duplicates()
    keys = keys[keys.groupby("key").row.transform("size") <= cfg_keys["max_block_size"]]
    is_s1 = recs.source.values[keys.row.values] == 1
    pairs = keys[is_s1].merge(keys[~is_s1], on="key", suffixes=("_s1", "_c"))
    pairs = pairs.groupby(["row_s1", "row_c"]).size().rename("key_hits").reset_index()
    return pd.DataFrame({"s1_id": recs.entity_id.values[pairs.row_s1.values],
                         "cand_id": recs.entity_id.values[pairs.row_c.values],
                         "key_hits": pairs.key_hits.values.astype(np.int16)})


def block_family(recs: pd.DataFrame, cfg_blocking: dict) -> pd.DataFrame:
    parts = []
    for country in recs.loc[recs.source == 1, "country"].unique():  # open set: whatever labels appear
        sub = recs[recs.country == country]
        s1, others = sub[sub.source == 1], sub[sub.source != 1]
        t = tfidf_topk(s1, others, cfg_blocking["tfidf"])
        t = t.sort_values(["s1_id", "tfidf_sim"], ascending=[True, False])
        t["tfidf_rank"] = t.groupby("s1_id").cumcount() + 1
        k = key_blocks(sub, cfg_blocking["keys"])
        parts.append(t.merge(k, on=["s1_id", "cand_id"], how="outer"))
    out = pd.concat(parts, ignore_index=True)
    out["tfidf_sim"] = out.tfidf_sim.fillna(0).astype(np.float32)
    out["tfidf_rank"] = out.tfidf_rank.fillna(999).astype(np.int16)
    out["key_hits"] = out.key_hits.fillna(0).astype(np.int16)
    return out[["s1_id", "cand_id", "tfidf_sim", "tfidf_rank", "key_hits"]]


def cand_sparse_path(cfg: dict, family: str) -> Path:
    return run_dir(cfg) / f"cand_sparse_{family}.parquet"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--family", nargs="+", default=["train", "test"], choices=["train", "test"])


def run(cfg: dict, args: argparse.Namespace) -> None:
    for family in args.family:
        recs = pd.read_parquet(records_path(cfg, family))
        cands = block_family(recs, cfg["blocking"])
        cands.to_parquet(cand_sparse_path(cfg, family), index=False)
        s1_ids = recs.entity_id[recs.source == 1].tolist()
        stats = {"n_pairs": len(cands), "per_entity": candidates_per_entity(cands, s1_ids)}
        if family == "train":
            truth, splits = read_truth(cfg["paths"]["data_dir"]), pd.read_parquet(splits_path(cfg))
            tp = truth_pairs(truth)
            stats["pc_all"] = pair_completeness(cands, tp)
            stats["pc_tfidf_only"] = pair_completeness(cands[cands.tfidf_rank < 999], tp)
            stats["pc_keys_only"] = pair_completeness(cands[cands.key_hits > 0], tp)
            for c, grp in splits.groupby("country"):
                stats[f"pc_{c}"] = pair_completeness(cands, truth_pairs(truth, grp.s1_id.tolist()))
        print(family, stats)
        log_metrics(cfg, f"block_sparse_{family}", stats)
```

Add to `COMMANDS` in `src/ber/cli.py`:
```python
    "block-sparse": ("ber.blocking.sparse", "char TF-IDF top-k + key blocks per country"),
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `scripts/hpc.sh pytest tests/test_blocking_sparse.py -q`
Expected: `3 passed`

- [ ] **Step 5: Run on the real data**

```bash
scripts/hpc.sh "sbatch slurm/cpu.sbatch python -m ber block-sparse"
scripts/hpc.sh "squeue -u fcse_shwetasharma; tail -n 3 artifacts/logs/ber-cpu-*.out"
```
Repeat the second command until the job finishes.

Expected: `train {...}` with `pc_all` printed. Record `pc_all`, `pc_tfidf_only`, `pc_keys_only`, per-country pair completeness and `per_entity` in the commit message. If the job runs out of memory in `key_blocks`, lower `max_block_size` to 50 in `configs/base.yaml` and rerun.

- [ ] **Step 6: Commit**

```bash
git add src/ber/blocking tests/test_blocking_sparse.py src/ber/cli.py
git commit -m "Add sparse blocking (char TF-IDF top-k + key blocks); train pc_all=<value>"
```

---

### Task 9: Encoder registry, encoding and recall@k

**Files:**
- Create: `src/ber/contrastive/__init__.py`, `src/ber/contrastive/encoders.py`, `src/ber/contrastive/retrieval.py`, `tests/test_retrieval.py`, `tests/test_encoders.py`

**Interfaces:**
- Produces:
  - `EncoderSpec` (frozen dataclass: `key, hf_id, query_prefix, doc_prefix, trust_remote_code, lora, truncate_dim, max_seq_length, padding_side`)
  - `ENCODERS: dict[str, EncoderSpec]`, with keys `gte, nomic, arctic, bgem3, qwen8b`
  - `load_encoder(spec: EncoderSpec, path: str | None = None, mem_fraction: float | None = None, for_training: bool = False)`, which returns a `SentenceTransformer`
  - `encode(model, texts: list[str], prefix: str, batch_size: int) -> np.ndarray` (float16, L2-normalized)
  - `recall_at_k(q_ids: list[str], q_emb: np.ndarray, d_ids: list[str], d_emb: np.ndarray, truth: dict[str, set[str]], ks: list[int], n_threads: int = 16) -> dict[int, float]` (micro recall over true pairs)

- [ ] **Step 1: Write the failing tests**

`tests/test_retrieval.py`:
```python
import numpy as np
import pytest

from ber.contrastive.retrieval import recall_at_k


def test_recall_at_k_micro():
    rng = np.random.default_rng(0)
    d = rng.normal(size=(50, 8)).astype(np.float32)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    d_ids = [f"S2-{i}" for i in range(50)]
    # q0 equals doc 0 exactly; its truth = {doc0, doc1}. q1 equals doc 5; truth = {doc5}.
    q = d[[0, 5]].copy()
    truth = {"S1-0": {"S2-0", "S2-1"}, "S1-1": {"S2-5"}}
    r = recall_at_k(["S1-0", "S1-1"], q, d_ids, d, truth, ks=[1, 50])
    assert r[1] == pytest.approx(2 / 3)
    assert r[50] == pytest.approx(1.0)
```

`tests/test_encoders.py`:
```python
from ber.contrastive.encoders import ENCODERS


def test_registry_has_bakeoff_models_with_permissive_licenses():
    assert set(ENCODERS) == {"gte", "nomic", "arctic", "bgem3", "qwen8b"}
    assert ENCODERS["nomic"].query_prefix == "search_query: " and ENCODERS["nomic"].doc_prefix == "search_document: "
    assert ENCODERS["qwen8b"].lora and ENCODERS["qwen8b"].truncate_dim == 1024
    assert not any(s.lora for k, s in ENCODERS.items() if k != "qwen8b")
```

- [ ] **Step 2: Run them to verify they fail**

Run: `scripts/hpc.sh pytest tests/test_retrieval.py tests/test_encoders.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'ber.contrastive'`

- [ ] **Step 3: Implement**

`src/ber/contrastive/__init__.py`: empty file.

`src/ber/contrastive/encoders.py`:
```python
from dataclasses import dataclass

import numpy as np

QWEN_INSTRUCT = "Instruct: Given a business name and address, retrieve records of the same business\nQuery: "


@dataclass(frozen=True)
class EncoderSpec:
    key: str
    hf_id: str
    query_prefix: str = ""
    doc_prefix: str = ""
    trust_remote_code: bool = False
    lora: bool = False
    truncate_dim: int | None = None
    max_seq_length: int = 128
    padding_side: str | None = None


ENCODERS: dict[str, EncoderSpec] = {
    "gte": EncoderSpec("gte", "Alibaba-NLP/gte-multilingual-base", trust_remote_code=True),
    "nomic": EncoderSpec("nomic", "nomic-ai/nomic-embed-text-v2-moe", "search_query: ", "search_document: ",
                         trust_remote_code=True),
    "arctic": EncoderSpec("arctic", "Snowflake/snowflake-arctic-embed-l-v2.0", "query: ", ""),
    "bgem3": EncoderSpec("bgem3", "BAAI/bge-m3"),
    "qwen8b": EncoderSpec("qwen8b", "Qwen/Qwen3-Embedding-8B", QWEN_INSTRUCT, "", lora=True,
                          truncate_dim=1024, padding_side="left"),
}


def load_encoder(spec: EncoderSpec, path: str | None = None, mem_fraction: float | None = None,
                 for_training: bool = False):
    import torch
    from sentence_transformers import SentenceTransformer

    if torch.cuda.is_available() and mem_fraction:
        torch.cuda.set_per_process_memory_fraction(mem_fraction)
    # bf16 weights for inference everywhere and for LoRA training; full fine-tunes keep fp32 master weights.
    dtype = torch.bfloat16 if (spec.lora or not for_training) else torch.float32
    model = SentenceTransformer(
        path or spec.hf_id,
        device="cuda" if torch.cuda.is_available() else "cpu",
        trust_remote_code=spec.trust_remote_code,
        truncate_dim=spec.truncate_dim,
        model_kwargs={"torch_dtype": dtype},
        tokenizer_kwargs={"padding_side": spec.padding_side} if spec.padding_side else None,
    )
    model.max_seq_length = spec.max_seq_length
    return model


def encode(model, texts: list[str], prefix: str, batch_size: int) -> np.ndarray:
    emb = model.encode(texts, prompt=prefix or None, batch_size=batch_size, convert_to_numpy=True,
                       normalize_embeddings=True, show_progress_bar=True)
    return emb.astype(np.float16)
```

`src/ber/contrastive/retrieval.py`:
```python
import faiss
import numpy as np


def recall_at_k(q_ids: list[str], q_emb: np.ndarray, d_ids: list[str], d_emb: np.ndarray,
                truth: dict[str, set[str]], ks: list[int], n_threads: int = 16) -> dict[int, float]:
    """Micro recall: found true pairs / all true pairs, over the top-k inner-product neighbours."""
    faiss.omp_set_num_threads(n_threads)
    index = faiss.IndexFlatIP(d_emb.shape[1])
    index.add(np.ascontiguousarray(d_emb, dtype=np.float32))
    _, nn = index.search(np.ascontiguousarray(q_emb, dtype=np.float32), max(ks))
    d_ids = np.asarray(d_ids)
    hits, total = {k: 0 for k in ks}, 0
    for qid, row in zip(q_ids, nn):
        t = truth[qid]
        total += len(t)
        found = np.cumsum([j >= 0 and d_ids[j] in t for j in row])
        for k in ks:
            hits[k] += int(found[k - 1])
    return {k: hits[k] / total for k in ks} if total else {k: 1.0 for k in ks}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `scripts/hpc.sh pytest tests/test_retrieval.py tests/test_encoders.py -q`
Expected: `2 passed`

- [ ] **Step 5: Smoke-load every model on the GPU**

This downloads the weights into the HF cache, confirms that `trust_remote_code` models import, and shows each model's embedding dimension.

Create `scripts/smoke_encoders.py`:
```python
from ber.contrastive.encoders import ENCODERS, encode, load_encoder

TEXT = "name: Galaxy Solutions Pvt Ltd | address: 47/1 Airport Road, Kolhapur | country: India"
for key, spec in ENCODERS.items():
    try:
        model = load_encoder(spec, mem_fraction=0.4)
        print("OK", key, encode(model, [TEXT], spec.query_prefix, 8).shape, flush=True)
        del model
    except Exception as ex:  # report every model, don't stop at the first failure
        print("FAIL", key, repr(ex), flush=True)
```

Run:
```bash
scripts/hpc.sh "mkdir -p artifacts/logs && sbatch -J smoke-enc slurm/gpu.sbatch python scripts/smoke_encoders.py"
scripts/hpc.sh "squeue -u fcse_shwetasharma; tail -n 20 artifacts/logs/smoke-enc-*.out"
```
Expected: five `OK` lines. If a model prints `FAIL` because of a missing package (for example `einops` or `megablocks` for nomic), add that package to `requirements.in`, run `scripts/hpc.sh "pip install -q <pkg> && pip freeze --exclude-editable > requirements.txt"`, copy `requirements.txt` back with `scp -i ~/.ssh/hpc_nopass fcse_shwetasharma@172.16.0.77:~/winning_ml_challenge_sol/requirements.txt requirements.txt`, and rerun.

- [ ] **Step 6: Commit**

```bash
git add src/ber/contrastive tests/test_retrieval.py tests/test_encoders.py scripts/smoke_encoders.py requirements.in requirements.txt
git commit -m "Add encoder registry, encoding and FAISS recall@k"
```

---

### Task 10: Training triplets (sampling + rendering)

**Files:**
- Create: `src/ber/contrastive/pairs.py`, `tests/test_pairs.py`

**Interfaces:**
- Consumes:
  - `augment` (Task 6)
  - `record_text` (Task 4)
  - `EncoderSpec` (Task 9)
  - The `cand_sparse` frame (Task 8)
- Produces:
  - `sample_triplets(truth: dict[str, set[str]], s1_ids: list[str], hard_negs: dict[str, list[str]], country_pool: dict[str, np.ndarray], id_country: dict[str, str], n_rounds: int, p_intra: float, seed: int) -> pd.DataFrame`, with columns `anchor_id, positive_id, negative_id`. Only S1 ids with at least one match contribute. Each round gives one row per entity.
  - `hard_negatives(cands: pd.DataFrame, truth: dict[str, set[str]], s1_ids: list[str]) -> dict[str, list[str]]`
  - `render_triplets(trip: pd.DataFrame, records: pd.DataFrame, spec: EncoderSpec, aug_prob: float, seed: int) -> pd.DataFrame`, with columns `anchor, positive, negative`. `records` is indexed by `entity_id` and has `name_raw, addr_raw, country`. The anchor gets `spec.query_prefix`; positive and negative get `spec.doc_prefix`. Positive and negative are each augmented with probability `aug_prob`.

- [ ] **Step 1: Write the failing test**

`tests/test_pairs.py`:
```python
import numpy as np
import pandas as pd

from ber.contrastive.encoders import EncoderSpec
from ber.contrastive.pairs import hard_negatives, render_triplets, sample_triplets

TRUTH = {"S1-a": {"S2-a1", "S2-a2", "S3-a1"}, "S1-b": {"S3-b1"}, "S1-z": set()}
IDC = {"S1-a": "US", "S2-a1": "US", "S2-a2": "US", "S3-a1": "US", "S1-b": "US", "S3-b1": "US",
       "S2-x": "US", "S3-y": "US", "S1-z": "US"}
POOL = {"US": np.array(["S2-a1", "S2-a2", "S3-a1", "S3-b1", "S2-x", "S3-y"])}


def test_hard_negatives_exclude_true_matches():
    cands = pd.DataFrame({"s1_id": ["S1-a", "S1-a", "S1-b"], "cand_id": ["S2-a1", "S2-x", "S3-y"]})
    assert hard_negatives(cands, TRUTH, ["S1-a", "S1-b"]) == {"S1-a": ["S2-x"], "S1-b": ["S3-y"]}


def test_sample_triplets_valid_and_deterministic():
    hn = {"S1-a": ["S2-x"], "S1-b": []}
    t = sample_triplets(TRUTH, ["S1-a", "S1-b", "S1-z"], hn, POOL, IDC, n_rounds=50, p_intra=0.5, seed=3)
    t2 = sample_triplets(TRUTH, ["S1-a", "S1-b", "S1-z"], hn, POOL, IDC, n_rounds=50, p_intra=0.5, seed=3)
    pd.testing.assert_frame_equal(t, t2)
    assert len(t) == 100  # S1-z (singleton) contributes nothing
    groups = {"S1-a": {"S1-a"} | TRUTH["S1-a"], "S1-b": {"S1-b"} | TRUTH["S1-b"]}
    for r in t.itertuples():
        g = next(v for v in groups.values() if r.anchor_id in v)
        assert r.positive_id in g and r.positive_id != r.anchor_id and r.negative_id not in g
    assert ((t.anchor_id != "S1-a") & (t.anchor_id != "S1-b")).any()  # intra S2/S3 pairs occur
    assert (t[t.anchor_id == "S1-a"].negative_id == "S2-x").all()


def test_render_prefixes():
    rec = pd.DataFrame({"entity_id": ["S1-a", "S2-a1", "S2-x"], "name_raw": ["A Co", "A CO", "B Inc"],
                        "addr_raw": ["1 Main St, X", "1 MAIN ST", "2 Elm"], "country": ["US"] * 3}).set_index("entity_id")
    trip = pd.DataFrame({"anchor_id": ["S1-a"], "positive_id": ["S2-a1"], "negative_id": ["S2-x"]})
    spec = EncoderSpec("t", "t", query_prefix="q: ", doc_prefix="d: ")
    out = render_triplets(trip, rec, spec, aug_prob=0.0, seed=0)
    assert out.iloc[0].tolist() == ["q: name: A Co | address: 1 Main St, X | country: US",
                                    "d: name: A CO | address: 1 MAIN ST | country: US",
                                    "d: name: B Inc | address: 2 Elm | country: US"]
```

- [ ] **Step 2: Run it to verify it fails**

Run: `scripts/hpc.sh pytest tests/test_pairs.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'ber.contrastive.pairs'`

- [ ] **Step 3: Implement**

`src/ber/contrastive/pairs.py`:
```python
import numpy as np
import pandas as pd

from ber.contrastive.encoders import EncoderSpec
from ber.noise import augment
from ber.text import record_text


def hard_negatives(cands: pd.DataFrame, truth: dict[str, set[str]], s1_ids: list[str]) -> dict[str, list[str]]:
    sub = cands[cands.s1_id.isin(set(s1_ids))]
    out: dict[str, list[str]] = {}
    for s1, c in zip(sub.s1_id, sub.cand_id):
        if c not in truth.get(s1, ()):
            out.setdefault(s1, []).append(c)
    return out


def sample_triplets(truth: dict[str, set[str]], s1_ids: list[str], hard_negs: dict[str, list[str]],
                    country_pool: dict[str, np.ndarray], id_country: dict[str, str],
                    n_rounds: int, p_intra: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_rounds):
        for s1 in s1_ids:
            matches = sorted(truth.get(s1, ()))
            if not matches:
                continue
            group = set(matches) | {s1}
            if len(matches) >= 2 and rng.random() < p_intra:
                i, j = rng.choice(len(matches), size=2, replace=False)
                anchor, pos = matches[i], matches[j]
            else:
                anchor, pos = s1, matches[int(rng.integers(len(matches)))]
            negs = hard_negs.get(s1) or []
            if negs:
                neg = negs[int(rng.integers(len(negs)))]
            else:
                pool = country_pool[id_country[s1]]
                neg = pool[int(rng.integers(len(pool)))]
                while neg in group:
                    neg = pool[int(rng.integers(len(pool)))]
            rows.append((anchor, pos, neg))
    return pd.DataFrame(rows, columns=["anchor_id", "positive_id", "negative_id"])


def render_triplets(trip: pd.DataFrame, records: pd.DataFrame, spec: EncoderSpec, aug_prob: float,
                    seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    name, addr, country = records["name_raw"], records["addr_raw"], records["country"]

    def text(eid: str, prefix: str, noisy: bool) -> str:
        n, a, c = name[eid], addr[eid], country[eid]
        if noisy and rng.random() < aug_prob:
            n, a = augment(n, a, c, rng)
        return prefix + record_text(n, a, c)

    return pd.DataFrame({
        "anchor": [text(e, spec.query_prefix, False) for e in trip.anchor_id],
        "positive": [text(e, spec.doc_prefix, True) for e in trip.positive_id],
        "negative": [text(e, spec.doc_prefix, True) for e in trip.negative_id],
    })
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `scripts/hpc.sh pytest tests/test_pairs.py -q`
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add src/ber/contrastive/pairs.py tests/test_pairs.py
git commit -m "Add contrastive triplet sampling with hard negatives and augmentation"
```

---

### Task 11: Fine-tuning

**Files:**
- Create: `src/ber/contrastive/train.py`, `tests/test_train.py`

**Interfaces:**
- Consumes: `load_encoder`, `EncoderSpec` (Task 9); a rendered triplet frame (Task 10)
- Produces: `finetune(spec: EncoderSpec, train_df: pd.DataFrame, out_dir: Path, cfg_train: dict, mem_fraction: float, seed: int, base_model: str | None = None) -> dict`. It saves the model to `out_dir/"model"` and returns `{"peak_train_mem_gb": float, "steps": int, "train_seconds": float}`.

Deviation from spec §4.3, deliberate: **every** model uses `CachedMultipleNegativesRankingLoss`, not only Qwen. It computes the same gradients as MNRL at the same batch size but with bounded memory, so all five models train under identical loss and batch conditions on the shared GPU.

- [ ] **Step 1: Write the failing test** (a GPU-free test using a tiny public model on CPU)

`tests/test_train.py`:
```python
import pandas as pd

from ber.contrastive.encoders import EncoderSpec
from ber.contrastive.train import finetune

TINY = "sentence-transformers-testing/stsb-bert-tiny-safetensors"


def test_finetune_runs_and_saves(tmp_path):
    df = pd.DataFrame({"anchor": [f"name: a{i}" for i in range(16)],
                       "positive": [f"name: A{i}" for i in range(16)],
                       "negative": [f"name: z{i}" for i in range(16)]})
    cfg = {"batch_size": 8, "cached_mini_batch_size": 4, "max_steps": 2, "learning_rate": 2e-5,
           "lora_learning_rate": 1e-4, "warmup_ratio": 0.0}
    stats = finetune(EncoderSpec("tiny", TINY), df, tmp_path, cfg, mem_fraction=None, seed=0)
    assert (tmp_path / "model" / "config.json").exists()
    assert stats["steps"] == 2
```

- [ ] **Step 2: Run it to verify it fails**

Run: `scripts/hpc.sh pytest tests/test_train.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'ber.contrastive.train'`

- [ ] **Step 3: Implement**

`src/ber/contrastive/train.py`:
```python
import time
from pathlib import Path

import pandas as pd

from ber.contrastive.encoders import EncoderSpec, load_encoder


def finetune(spec: EncoderSpec, train_df: pd.DataFrame, out_dir: Path, cfg_train: dict,
             mem_fraction: float | None, seed: int, base_model: str | None = None) -> dict:
    import torch
    from datasets import Dataset
    from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments, losses
    from sentence_transformers.training_args import BatchSamplers

    model = load_encoder(spec, path=base_model, mem_fraction=mem_fraction, for_training=True)
    lr = cfg_train["learning_rate"]
    if spec.lora:
        from peft import LoraConfig, TaskType

        model.add_adapter(LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, r=16, lora_alpha=32,
                                     lora_dropout=0.05, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
        hf = model[0].auto_model
        hf.gradient_checkpointing_enable()
        hf.enable_input_require_grads()
        lr = cfg_train["lora_learning_rate"]
    loss = losses.CachedMultipleNegativesRankingLoss(model, mini_batch_size=cfg_train["cached_mini_batch_size"])
    cuda = torch.cuda.is_available()
    args = SentenceTransformerTrainingArguments(
        output_dir=str(Path(out_dir) / "ckpt"),
        max_steps=cfg_train["max_steps"],
        per_device_train_batch_size=cfg_train["batch_size"],
        learning_rate=lr,
        warmup_ratio=cfg_train["warmup_ratio"],
        bf16=cuda,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        logging_steps=50,
        save_strategy="no",
        report_to="none",
        seed=seed,
    )
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    trainer = SentenceTransformerTrainer(
        model=model, args=args, loss=loss,
        train_dataset=Dataset.from_pandas(train_df[["anchor", "positive", "negative"]], preserve_index=False),
    )
    t0 = time.perf_counter()
    trainer.train()
    elapsed = time.perf_counter() - t0
    model.save(str(Path(out_dir) / "model"))
    return {"peak_train_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if cuda else 0.0,
            "steps": int(trainer.state.global_step), "train_seconds": elapsed}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `scripts/hpc.sh pytest tests/test_train.py -q`
Expected: `1 passed`

- [ ] **Step 5: Commit**

```bash
git add src/ber/contrastive/train.py tests/test_train.py
git commit -m "Add contrastive fine-tuning with CachedMNRL and LoRA for Qwen"
```

---

### Task 12: Bake-off orchestration, runs and report (`bakeoff` command)

**Files:**
- Create: `src/ber/contrastive/bakeoff.py`, `tests/test_bakeoff.py`, `docs/results/bakeoff.md` (generated)
- Modify: `src/ber/cli.py`, `docs/superpowers/specs/2026-09-25-er-pipeline-design.md` (§5, eval pool note)

**Interfaces:**
- Consumes:
  - `records_path`, `splits_path`, `cand_sparse_path`, `read_truth`
  - `ENCODERS`, `load_encoder`, `encode`
  - `recall_at_k`
  - `hard_negatives`, `sample_triplets`, `render_triplets`
  - `finetune`
- Produces:
  - `build_eval_set(records: pd.DataFrame, splits: pd.DataFrame, truth: dict, n_distractors: int, seed: int) -> pd.DataFrame`, with columns `entity_id, role ("query"|"doc"), country`
  - `result_path(cfg, model_key, tag) -> Path`, i.e. `run_dir/bakeoff/{model_key}__{tag}.json`
  - `select_winner(results: list[dict]) -> str`
  - `write_report(results: list[dict], out_md: Path) -> None`
  - The `bakeoff` command: `python -m ber bakeoff {zeroshot,finetune,report} [--model KEY] [--train-countries C ...]`

Result JSON schema (one file per run):
```json
{"model": "gte", "tag": "zeroshot|finetune|finetune_US", "dim": 768,
 "recall": {"US": {"10": 0.9, "50": 0.95, "100": 0.97}, "India": {...}},
 "throughput_rps": 5000.0, "test_vectors_gb": 15.3, "peak_train_mem_gb": 0.0, "train_seconds": 0.0}
```

- [ ] **Step 1: Write the failing test**

`tests/test_bakeoff.py`:
```python
import numpy as np
import pandas as pd

from ber.contrastive.bakeoff import build_eval_set, select_winner, write_report


def test_eval_set_contains_all_matches_and_sampled_distractors():
    rec = pd.DataFrame({
        "entity_id": ["S1-a", "S1-b", "S2-a", "S3-a", "S2-b", "S2-x1", "S2-x2", "S2-x3", "S2-f"],
        "source": [1, 1, 2, 3, 2, 2, 2, 2, 2],
        "country": ["US", "US", "US", "US", "US", "US", "US", "US", "France"],
    })
    splits = pd.DataFrame({"s1_id": ["S1-a", "S1-b"], "split": ["A_val", "B"], "country": ["US", "US"]})
    truth = {"S1-a": {"S2-a", "S3-a"}, "S1-b": {"S2-b"}}
    ev = build_eval_set(rec, splits, truth, n_distractors=2, seed=0)
    q = ev[ev.role == "query"]
    d = ev[ev.role == "doc"]
    assert q.entity_id.tolist() == ["S1-a"]
    assert {"S2-a", "S3-a"} <= set(d.entity_id)
    assert len(d) == 4 and "S2-f" not in set(d.entity_id)  # 2 matches + 2 same-country distractors


def _res(model, tag, us, india, rps=100.0):
    return {"model": model, "tag": tag, "dim": 768, "throughput_rps": rps, "test_vectors_gb": 1.0,
            "peak_train_mem_gb": 1.0, "train_seconds": 1.0,
            "recall": {"US": {"10": us, "50": us, "100": us}, "India": {"10": india, "50": india, "100": india}}}


def test_select_winner_rules():
    # clear winner on mean fine-tuned recall@50
    assert select_winner([_res("a", "finetune", 0.90, 0.90), _res("b", "finetune", 0.95, 0.95)]) == "b"
    # within 0.005 -> leave-one-country-out India recall decides
    res = [_res("a", "finetune", 0.950, 0.950), _res("b", "finetune", 0.952, 0.950),
           _res("a", "finetune_US", 0.9, 0.93), _res("b", "finetune_US", 0.9, 0.90)]
    assert select_winner(res) == "a"


def test_write_report(tmp_path):
    out = tmp_path / "r.md"
    write_report([_res("a", "zeroshot", 0.5, 0.4), _res("a", "finetune", 0.9, 0.8)], out)
    text = out.read_text()
    assert "| a | finetune |" in text and "Winner: **a**" in text
```

- [ ] **Step 2: Run it to verify it fails**

Run: `scripts/hpc.sh pytest tests/test_bakeoff.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'ber.contrastive.bakeoff'`

- [ ] **Step 3: Implement**

`src/ber/contrastive/bakeoff.py`:
```python
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
```

Add to `COMMANDS` in `src/ber/cli.py`:
```python
    "bakeoff": ("ber.contrastive.bakeoff", "bi-encoder bake-off: zeroshot | finetune | report"),
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `scripts/hpc.sh pytest -q`
Expected: every test in the suite passes.

- [ ] **Step 5: Record the eval-pool deviation in the spec**

In `docs/superpowers/specs/2026-09-25-er-pipeline-design.md` §5, replace protocol item 1 with:
```markdown
1. **Zero-shot:** embed the A-val S1 queries (those with ≥1 match) and a fixed eval pool per country. The pool is every true match of those queries plus `bakeoff.eval_distractors_per_country` randomly sampled same-country S2/S3 records. The pool is built once (`artifacts/<run>/bakeoff/eval_set.parquet`) and shared by all models. Report recall@{10, 50, 100} per country. The pool is sampled rather than "all records" so that the 8B model's cost stays tractable. Absolute recall is therefore optimistic relative to the full test pool, but the ranking between models is fair.
```
In §4.3, change the loss line to:
```markdown
- **Loss:** `CachedMultipleNegativesRankingLoss` for every model. It gives the same gradients as MNRL at the same batch size with bounded memory. Qwen3-Embedding-8B additionally uses LoRA + gradient checkpointing.
```

- [ ] **Step 6: Smoke run end to end on the tiny config**

Reuse the real v1 artifacts (records, splits, sparse candidates) in the smoke run directory. Then run a tiny zero-shot and fine-tune for one model:
```bash
scripts/hpc.sh "mkdir -p artifacts/smoke && cp artifacts/v1/records_*.parquet artifacts/v1/splits.parquet artifacts/v1/cand_sparse_train.parquet artifacts/smoke/ && sbatch -J smoke-bakeoff slurm/gpu.sbatch bash -c 'python -m ber --config configs/smoke.yaml bakeoff zeroshot --model gte && python -m ber --config configs/smoke.yaml bakeoff finetune --model gte'"
scripts/hpc.sh "squeue -u fcse_shwetasharma; tail -n 30 artifacts/logs/smoke-bakeoff-*.out"
```
Expected: two result JSONs under `artifacts/smoke/bakeoff/`, recall lines printed per country, and no exceptions.

- [ ] **Step 7: Launch the real bake-off**

Zero-shot runs for all five models, fine-tune runs for all five, and the leave-one-country-out fine-tune (US→India) for all five. The shared GPU runs them one after another. `--dependency=singleton` plus a single job name makes SLURM queue them serially:
```bash
scripts/hpc.sh "for m in gte nomic arctic bgem3 qwen8b; do
  sbatch -J bakeoff --dependency=singleton slurm/gpu.sbatch python -m ber bakeoff zeroshot --model \$m
  sbatch -J bakeoff --dependency=singleton slurm/gpu.sbatch python -m ber bakeoff finetune --model \$m
  sbatch -J bakeoff --dependency=singleton slurm/gpu.sbatch python -m ber bakeoff finetune --model \$m --train-countries US
done; squeue -u fcse_shwetasharma"
```
Monitor with `scripts/hpc.sh "squeue -u fcse_shwetasharma; ls artifacts/v1/bakeoff/*.json"`. If a run fails, for example with OOM on qwen8b, read its log. For qwen8b, lower `train.cached_mini_batch_size` to 8 in a new config `configs/qwen_lowmem.yaml` (with `base: configs/base.yaml`) and resubmit only that run with `--config configs/qwen_lowmem.yaml`. Keep `batch_size` unchanged so the comparison stays fair.

- [ ] **Step 8: Generate the report and commit**

```bash
scripts/hpc.sh python -m ber bakeoff report
scp -i ~/.ssh/hpc_nopass fcse_shwetasharma@172.16.0.77:~/winning_ml_challenge_sol/docs/results/bakeoff.md docs/results/bakeoff.md
git add src/ber/contrastive/bakeoff.py src/ber/cli.py tests/test_bakeoff.py docs/results/bakeoff.md docs/superpowers/specs/2026-09-25-er-pipeline-design.md
git commit -m "Add bi-encoder bake-off and results; winner: <model>"
git push origin main
```

---

## Next plan (not in scope here)

Plan 2 will be written once `docs/results/bakeoff.md` names a winner. It covers spec §4.4–§4.12:
- full fine-tune of the winner on A-train
- dense blocking + union
- features
- prefilter
- LightGBM matcher with GroupKFold on B
- has-match model
- decision + threshold tuning
- export + validator
- the end-to-end `all` command
