#!/usr/bin/env bash
# Full improvement chain (tmux). Logs: artifacts/logs/chain_v3/progress.log
cd ~/winning_ml_challenge_sol
export PYTHONNOUSERSITE=1
PY=~/.conda/envs/ber-sol/bin/python
L=artifacts/logs/chain_v3; mkdir -p $L
step() { local name=$1; shift
  echo "[$(date '+%F %T')] START $name" >> $L/progress.log
  if "$@" > $L/$name.log 2>&1; then
    echo "[$(date '+%F %T')] DONE  $name :: $(grep -aE 'Best cut-off|A \+ reranker|B2:|AUC|mean_matches|PASS|validator exit' $L/$name.log | tr '\n' ' ' | cut -c1-600)" >> $L/progress.log
  else echo "[$(date '+%F %T')] FAIL  $name" >> $L/progress.log; return 1; fi; }

# --- v2 branch: Model B v2 on A v2 + reranker (needs the v2 reranker chain to have finished) ---
while tmux has-session -t rerank 2>/dev/null; do sleep 60; done
if [ "${SKIP_V2_BRANCH:-0}" != "1" ]; then
  step b2_on_v2_train  $PY scripts/stage2_emb.py train
  step b2_on_v2_submit env SUB=v4 $PY scripts/stage2_emb.py submit
fi

# --- v3 branch (needs Model A v3 trained) ---
until grep -q "MATCHER_EXIT" artifacts/logs/matcher_v3.log 2>/dev/null; do sleep 60; done
grep -q "MATCHER_EXIT=0" artifacts/logs/matcher_v3.log || { echo "Model A v3 failed" >> $L/progress.log; exit 1; }
step predict_v3 $PY -m ber --config configs/matcher_v3.yaml predict --models matcher_v3 --out artifacts/submissions/v3a
mkdir -p artifacts/v2/reranker_v3 && ln -sfn ../reranker/model artifacts/v2/reranker_v3/model
export RR_DIR=reranker_v3 A_DIR=matcher_v3 TEST_SCORES=test_scores_matcher_v3.parquet
step rr_v3_score_oof  $PY scripts/reranker.py score_oof
step rr_v3_stack      $PY scripts/reranker.py stack
step rr_v3_score_test $PY scripts/reranker.py score_test
step rr_v3_submit     env SUB=v5 $PY scripts/reranker.py submit
export S2_DIR=stage2_emb_v3
step b2_on_v3_train   $PY scripts/stage2_emb.py train
step b2_on_v3_submit  env SUB=v6 $PY scripts/stage2_emb.py submit
echo "[$(date '+%F %T')] ALL DONE" >> $L/progress.log
