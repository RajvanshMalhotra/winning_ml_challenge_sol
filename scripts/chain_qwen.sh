#!/usr/bin/env bash
# v10 overnight: Qwen3-Reranker-4B LoRA fine-tune -> OOF scoring -> stack vs v8 -> test scoring -> submission.
# test_sel (CPU) runs in parallel from the start.  Progress in artifacts/logs/chain_qwen/progress.log
set -u
L=artifacts/logs/chain_qwen
mkdir -p "$L"
P="$L/progress.log"
export QR_DIR=reranker_qwen SUB=v10
run() {
  echo "[$(date '+%F %T')] START $1" >> "$P"
  if python -u scripts/qwen_reranker.py "$1" > "$L/$1.log" 2>&1; then
    echo "[$(date '+%F %T')] DONE  $1 :: $(grep -hE 'train pairs|trainable|budget|saved adapter|AUC|no-address pairs|^- |selected|validator exit|tau' "$L/$1.log" | tr '\n' ' ' | cut -c1-700)" >> "$P"
  else
    echo "[$(date '+%F %T')] FAIL  $1 :: $(tail -3 "$L/$1.log" | tr '\n' ' ' | cut -c1-400)" >> "$P"
    return 1
  fi
}
( run test_sel ) &
SEL=$!
run train && run score_oof && run stack || exit 1
wait $SEL || { echo "[$(date '+%F %T')] FAIL  test_sel" >> "$P"; exit 1; }
run score_test && run submit && echo "[$(date '+%F %T')] ALL DONE" >> "$P"
