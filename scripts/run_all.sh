#!/usr/bin/env bash
# End-to-end reproduction of the final submission (v12): data -> blocking -> Model A -> Qwen reranker -> stacker -> output/.
# Run from the code root with the `ber` env active (see README.md). GPU steps are marked [GPU].
# Paths (data dir, run dir, validator) come from configs/base.yaml + configs/v2.yaml; artifacts land in artifacts/v2/.
set -euo pipefail
PY=${PY:-python}
VLLM_PY=${VLLM_PY:-$HOME/vllm-sol/bin/python}      # separate venv with vLLM (requirements-vllm.txt)
export PYTHONNOUSERSITE=1
step() { echo "[$(date '+%F %T')] $*"; "$@"; }

# 1. records: normalisation (names, addresses, states, PIN/postal, house numbers, name keys) for train and test
step $PY -m ber --config configs/v2.yaml normalize
# 2. A_train (35%) / A_val (5%) / B (60%, 5 GroupKFold folds) split of the train S1 businesses
step $PY -m ber --config configs/v2.yaml split
# 3. sparse blocking per country + learned state group: char TF-IDF top-50, key blocks, name-only search
step $PY -m ber --config configs/v2.yaml block-sparse
# 4. [GPU] fine-tune the bi-encoder bge-m3 (MIT) with cached MNRL on A_train pairs (+ report)
step $PY -m ber --config configs/v2.yaml bakeoff finetune --model bgem3
# 5. [GPU] hard-name channel (Qwen3-Embedding-0.6B, Apache-2.0) for native-script / empty-address records
step $PY scripts/hard_name_retrieval.py train
step $PY scripts/hard_name_retrieval.py test
# 6. record-record graph expansion channel
step $PY scripts/kg_expand.py configs/v2.yaml
# 7. [GPU] embed all records with the fine-tuned bge-m3, dense top-30 channel
step $PY scripts/embed_family.py train
step $PY scripts/embed_family.py test
step $PY scripts/dense_retrieval.py train 30
step $PY scripts/dense_retrieval.py test 30
# 8. Model A (LightGBM, 48 pair features, 5-fold OOF on a 150k B-split sample) and its scores for all pairs
step $PY -m ber --config configs/matcher_v3.yaml matcher
step $PY -m ber --config configs/matcher_v3.yaml predict --models matcher_v3 --out artifacts/submissions/model_a
step $PY -m ber --config configs/matcher_v3.yaml predict --models matcher_v3 --family train
# 9. [GPU] bge cross-encoder band scores (used by the v8 candidate-side feature frame)
export RR_DIR=reranker_v3 A_DIR=matcher_v3 TEST_SCORES=test_scores_matcher_v3.parquet
step $PY scripts/reranker.py train
mkdir -p artifacts/v2/reranker_v3 && ln -sfn ../reranker/model artifacts/v2/reranker_v3/model
step $PY scripts/reranker.py score_oof
step $PY scripts/reranker.py stack
step $PY scripts/reranker.py score_test
# 10. candidate-side features (v8) and the OOF feature frame (artifacts/v2/cand_side/oof_q.parquet)
step $PY scripts/cand_side.py train
step $PY scripts/v8_errors.py
# 11. Model-A competition / cluster-support / address-competition features over ALL S1s (v9)
step $PY scripts/cand_side_v9.py feats train
step $PY scripts/cand_side_v9.py feats test
step $PY scripts/cand_side_v9.py train
# 12. [GPU] fine-tune Qwen3-Reranker-4B (Apache-2.0) with LoRA on 1.04M A_train pairs (hard negatives/positives),
#     then score every uncertain pair (0.01 <= Model A p <= 0.99) of OOF and test with vLLM
export QR_DIR=reranker_qwen2 QR_PROMPT=short QR_GC=0 QR_BS=32 QR_MEM=0.82 QR_N=90000 QR_HOURS=40 QR_VLLM=1 \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
step $PY scripts/qwen_reranker.py train
step $PY scripts/qwen_reranker.py score_oof
step $PY scripts/qwen_reranker.py stack
step $PY scripts/qwen_reranker.py score_test
# 13. final stacker (v8 + v9 features, Qwen score in the reranker column), cut-off 0.75 + one-owner rule
step $PY scripts/v12_combo.py train
SUB=final step $PY scripts/v12_combo.py submit
mkdir -p output && cp artifacts/submissions/final/matching_results.tsv artifacts/submissions/final/candidate_pairs.tsv output/
echo "done: output/matching_results.tsv, output/candidate_pairs.tsv"
