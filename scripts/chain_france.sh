#!/usr/bin/env bash
# France routing to Qwen3-Reranker-4B: waits for the model download, then OOF scoring -> India rehearsal ->
# France test scoring -> v7 submission.  Progress in artifacts/logs/chain_france/progress.log
set -u
L=artifacts/logs/chain_france
mkdir -p "$L"
P="$L/progress.log"
until grep -q "DL_EXIT" artifacts/logs/qwen_dl.log 2>/dev/null; do sleep 30; done
export RR_DIR=reranker_v3 A_DIR=matcher_v3 TEST_SCORES=test_scores_matcher_v3.parquet FR_DIR=reranker_fr SUB=v7
for step in score_oof rehearse score_test submit; do
  echo "[$(date '+%F %T')] START $step" >> "$P"
  if python -u scripts/france_qwen.py "$step" > "$L/$step.log" 2>&1; then
    echo "[$(date '+%F %T')] DONE  $step :: $(grep -hE 'AUC all|India F0.5|\| (bge|qwen|blend)|validator exit|tau' "$L/$step.log" | tr '\n' ' ' | cut -c1-600)" >> "$P"
  else
    echo "[$(date '+%F %T')] FAIL  $step :: $(tail -3 "$L/$step.log" | tr '\n' ' ' | cut -c1-400)" >> "$P"
    exit 1
  fi
done
echo "[$(date '+%F %T')] ALL DONE" >> "$P"
