#!/usr/bin/env bash
# End-to-end reproduction of the final submission (v16frmixacr, leaderboard 0.987427): data -> blocking -> Model A -> Qwen reranker -> stacker -> output/.
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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export QR_PROMPT=short QR_GC=0 QR_BS=32 QR_MEM=0.82 QR_VLLM=1
step env QR_DIR=reranker_qwen2 QR_N=90000 QR_HOURS=40 $PY scripts/qwen_reranker.py train
step env QR_DIR=reranker_qwen2 $PY scripts/qwen_reranker.py score_oof
step env QR_DIR=reranker_qwen2 $PY scripts/qwen_reranker.py stack
step env QR_DIR=reranker_qwen2 $PY scripts/qwen_reranker.py score_test
#     continue training the same adapter on 90k NEW A_train businesses (lower LR), score OOF + test again
step env QR_DIR=reranker_qwen3 QR_N=90000 QR_SKIP=90000 QR_LR=5e-5 QR_HOURS=4.5 \
  QR_INIT=artifacts/v2/reranker_qwen2/adapter $PY scripts/qwen_reranker.py train
step env QR_DIR=reranker_qwen3 $PY scripts/qwen_reranker.py score_oof
step env QR_DIR=reranker_qwen3 $PY scripts/qwen_reranker.py score_test
# 13. test-like "hidden owner" frames: in test only ~60% of S2/S3 records have their owner in S1 (74% in train), so the
#     competition features are recomputed with 20% of the non-OOF train S1s hidden (3 random samples)
step $PY scripts/v12_combo.py train            # reference stacker (v12) used in the hidden-owner comparison
step $PY scripts/hidden_sim.py 0.2 0
step $PY scripts/hidden_sim.py 0.2 1
step $PY scripts/hidden_sim.py 0.2 2
# 14. final stacker: v8 + v9 features, Qwen (v13) in the reranker column plus the earlier Qwen and bge scores and their
#     disagreement, trained on each hidden frame; mean of 3 x 5 models; learned filter p >= 0.01; cut-off + one-owner rule
step $PY scripts/v16.py ablate
step $PY scripts/v16.py final
step $PY scripts/v16.py submit
# 15. France: short French lesson for the reranker on the labelled French practice set (data/.../proxy_fr), then for
#     FRANCE ROWS ONLY the reranker score = average (logit) of the v13 Qwen and the French-tuned Qwen
#     The practice set is generated from the test France S1 records with train-measured noise (no labels, no external data)
step $PY -m ber --config configs/v2.yaml proxy measure
step $PY -m ber --config configs/v2.yaml proxy build
step $PY -m ber --config configs/v2.yaml proxy build --name proxy_fr_holdout --seed 7
step $PY scripts/qwen_fr_pairs.py
step env QR_MEM=0.85 QF_MIN=30 QF_BS=16 $PY scripts/qwen_fr.py train
step $PY scripts/qwen_fr.py merge
step $PY scripts/qwen_fr.py score_test
step $PY -c "import pandas as pd; n=pd.read_parquet('artifacts/v2/reranker_qwen_fr/test_qwen_fr.parquet'); o=pd.read_parquet('artifacts/v2/reranker_qwen3/test_qwen.parquet').rename(columns={'rr2':'rr_old'}); m=n.merge(o,on=['s1_id','cand_id']); m['rr_fr']=(m.rr_fr+m.rr_old)/2; m[['s1_id','cand_id','rr_fr']].to_parquet('artifacts/v2/reranker_qwen_fr/test_qwen_frmix.parquet',index=False)"
step env FR_RR=artifacts/v2/reranker_qwen_fr/test_qwen_frmix.parquet V16_SUB=v16frmix $PY scripts/v16.py submit
# 16. France acronym routing: unassigned French acronym records (e.g. "AJ") + S1s with matching initials -> Qwen reads
#     them -> added when Qwen >= 0.9 and clearly ahead (all pairs Qwen read are added to the candidate set)
step $PY scripts/fr_acronym.py pairs
step $PY scripts/fr_acronym.py score
step $PY scripts/fr_acronym.py splice artifacts/submissions/v16frmix artifacts/submissions/final 0.9
mkdir -p output && cp artifacts/submissions/final/matching_results.tsv artifacts/submissions/final/candidate_pairs.tsv output/
echo "done: output/matching_results.tsv, output/candidate_pairs.tsv"
