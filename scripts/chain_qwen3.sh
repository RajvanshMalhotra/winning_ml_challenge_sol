#!/usr/bin/env bash
# v13: continue training the v10 Qwen adapter on 90k NEW A_train businesses (lower LR), score OOF + test with vLLM.
set -u
L=artifacts/logs/chain_qwen3
mkdir -p "$L"
P="$L/progress.log"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export QR_DIR=reranker_qwen3 SUB=v13q QR_PROMPT=short QR_GC=0 QR_BS=32 QR_MEM=0.82 QR_N=90000 QR_SKIP=90000 QR_LR=5e-5 \
       QR_HOURS=4.5 QR_VLLM=1 QR_VLLM_MEM=0.6 QR_INIT=artifacts/v2/reranker_qwen2/adapter
run() {
  echo "[$(date '+%F %T')] START $1" >> "$P"
  if python -u scripts/qwen_reranker.py "$1" > "$L/$1.log" 2>&1; then
    echo "[$(date '+%F %T')] DONE  $1 :: $(grep -hE 'train pairs|continuing|budget|saved adapter|AUC|^- |validator exit|variant' "$L/$1.log" | tr '\n' ' ' | cut -c1-900)" >> "$P"
  else
    echo "[$(date '+%F %T')] FAIL  $1 :: $(tail -3 "$L/$1.log" | tr '\n' ' ' | cut -c1-400)" >> "$P"
    return 1
  fi
}
run train && run score_oof && run stack && run score_test && echo "[$(date '+%F %T')] ALL DONE" >> "$P"
