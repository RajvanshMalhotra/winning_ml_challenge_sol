# Business Entity Resolution: reproduction guide

This folder regenerates `output/matching_results.tsv` and `output/candidate_pairs.tsv` from the challenge data.

## Layout

```
src/
├── ber/                 core package (python -m ber <command>)
│   ├── normalize.py     record normalisation (names, addresses, states, PIN/postal, house numbers, name keys)
│   ├── split.py         A_train / A_val / B split of train S1 businesses, 5 GroupKFold folds on B
│   ├── blocking/        char TF-IDF top-k, key blocks, name-only search, learned state groups
│   ├── contrastive/     bi-encoder fine-tuning (bge-m3, cached MNRL)
│   ├── features.py      48 pair features (RapidFuzz, IDF, house/state/unit/legal comparisons, channel signals)
│   ├── matcher.py       Model A: LightGBM, out-of-fold, F0.5 cut-off tuning
│   ├── predict.py       scoring of all candidate pairs (test or train), submission writer
│   └── evaluate.py      official macro F0.5 (singletons included)
├── scripts/             pipeline stages (see run_all.sh for the order)
│   ├── run_all.sh       END-TO-END ENTRY POINT
│   ├── hard_name_retrieval.py, kg_expand.py, embed_family.py, dense_retrieval.py   extra blocking channels
│   ├── reranker.py      bge-reranker band scores (feature frame of the v8 stacker)
│   ├── cand_side.py, v8_errors.py, cand_side_v9.py   candidate-side competition features
│   ├── qwen_reranker.py, vllm_score.py              Qwen3-Reranker-4B LoRA fine-tuning + vLLM scoring
│   └── v12_combo.py     final stacker + submission
├── configs/             YAML configs (paths, blocking, training, matcher)
├── pyproject.toml
├── requirements.txt       (repo root) main environment, pinned
└── requirements-vllm.txt  (repo root) separate venv for vLLM scoring, pinned
```

## Environment

- Python 3.11, Linux, one NVIDIA GPU with ≥ 80 GB (an H100 was used). The GPU steps are marked `[GPU]` in `run_all.sh`. About 500 GB of RAM is needed for the full-scale scoring steps.
- Main environment:
  ```bash
  conda create -n ber python=3.11 -y && conda activate ber
  pip install -r requirements.txt
  cd src && pip install -e .
  ```
- vLLM is only used to score pairs with the fine-tuned reranker. It needs a different torch/transformers build, so it gets its own venv:
  ```bash
  python3.11 -m venv ~/vllm-sol && ~/vllm-sol/bin/pip install -r requirements-vllm.txt
  export VLLM_PY=~/vllm-sol/bin/python
  ```
- Pretrained models are downloaded from the Hugging Face Hub on first use:

  | Model | License | Size |
  |---|---|---|
  | `BAAI/bge-m3` | MIT | 568M |
  | `BAAI/bge-reranker-v2-m3` | Apache-2.0 | 568M |
  | `Qwen/Qwen3-Embedding-0.6B` | Apache-2.0 | 0.6B |
  | `Qwen/Qwen3-Reranker-4B` | Apache-2.0 | 4.0B |

  No external data or APIs are used.

## Data

Put the official `student_resource/` folder at `src/data/student_resource/`, so these paths exist:
- `src/data/student_resource/dataset/{train,test}/*.tsv`
- `src/data/student_resource/utils/validate_submission.py`

Both paths are set in `configs/base.yaml`.

## Run

```bash
cd src
bash scripts/run_all.sh            # all stages in order; artifacts in src/artifacts/v2/, final files in src/output/
python3 data/student_resource/utils/validate_submission.py \
    --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv \
    --test-dir data/student_resource/dataset/test
```

Rough stage times on one H100 (shared) with 96 CPU threads:
- normalisation + blocking: ~3 h
- bge-m3 fine-tuning: ~3 h
- embeddings + dense channel: ~3 h
- Model A (train + scoring): ~5 h
- Qwen3-Reranker-4B LoRA: ~5.5 h
- vLLM scoring: ~1.5 h
- stackers: ~1 h

Randomness is seeded from `configs/base.yaml`. Values that could differ slightly between GPUs:
- bf16 reranker scores: ±0.06 logit vs HF inference
- LightGBM thread-order effects

## Pipeline in one paragraph

1. Records are normalised.
2. Every S1 business gets ≈130 candidates from six blocking channels:
   - TF-IDF
   - key blocks
   - name-only
   - Qwen3-Embedding hard names
   - graph expansion
   - fine-tuned bge-m3 dense top-30

   Blocking recall on held-out training businesses is 99.9%.
3. **Model A** (LightGBM, 48 features) scores all pairs and acts as a **learned filter**: only pairs with p ≥ 0.01 go on (≈5 per S1; recall ceiling 99.5%). `candidate_pairs.tsv` is exactly this filtered set, the input of the final matcher.
4. Pairs with 0.01 ≤ p ≤ 0.99 are re-read by **Qwen3-Reranker-4B fine-tuned with LoRA** on training pairs with hard negatives.
5. A **LightGBM stacker** combines:
   - Model A
   - the reranker
   - **candidate-side competition features** (how this S1 ranks among *all* S1s wanting the same S2/S3 record, by several signals)
   - cluster support and address competition
6. A pair is accepted when its stacked probability is ≥ 0.75, and each S2/S3 record is kept only for its best S1 (one-owner rule).

Out-of-fold macro F0.5 is 0.9904; the public leaderboard score is 0.986.
