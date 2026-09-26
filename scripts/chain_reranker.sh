#!/usr/bin/env bash
# Full reranker chain: train (100k A_train S1s) -> score OOF band -> stack (report) -> score test band -> v3 submission.
cd ~/winning_ml_challenge_sol
export PYTHONNOUSERSITE=1
PY=~/.conda/envs/ber-sol/bin/python
L=artifacts/logs/reranker
mkdir -p $L
for step in train score_oof stack score_test submit; do
  echo "[$(date '+%F %T')] START $step" >> $L/progress.log
  if $PY scripts/reranker.py $step > $L/$step.log 2>&1; then
    echo "[$(date '+%F %T')] DONE  $step :: $(grep -aE 'AUC|A \+ reranker|train_loss|test band pairs|mean_matches|PASS|validator exit' $L/$step.log | tr '\n' ' ' | cut -c1-500)" >> $L/progress.log
  else
    echo "[$(date '+%F %T')] FAIL  $step" >> $L/progress.log; exit 1
  fi
done
echo "[$(date '+%F %T')] ALL DONE" >> $L/progress.log
