#!/usr/bin/env bash
# v10 (full GPU): Qwen3-Reranker-4B LoRA on ALL A_train pairs, short prompt, no grad checkpointing, bigger batch;
# vLLM scoring; Qwen replaces bge.  Progress in artifacts/logs/chain_qwen2/progress.log
set -u
L=artifacts/logs/chain_qwen2
mkdir -p "$L"
P="$L/progress.log"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export QR_DIR=reranker_qwen2 SUB=v10 QR_PROMPT=short QR_GC=0 QR_BS=32 QR_MEM=0.82 QR_N=90000 QR_HOURS=40 QR_VLLM=1 QR_VLLM_MEM=0.8
mkdir -p artifacts/v2/reranker_qwen2
cp -n artifacts/v2/reranker_qwen/test_sel.parquet artifacts/v2/reranker_qwen2/ 2>/dev/null
run() {
  echo "[$(date '+%F %T')] START $1" >> "$P"
  if python -u scripts/qwen_reranker.py "$1" > "$L/$1.log" 2>&1; then
    echo "[$(date '+%F %T')] DONE  $1 :: $(grep -hE 'train pairs|trainable|budget|saved adapter|AUC|^- |validator exit|variant' "$L/$1.log" | tr '\n' ' ' | cut -c1-900)" >> "$P"
  else
    echo "[$(date '+%F %T')] FAIL  $1 :: $(tail -3 "$L/$1.log" | tr '\n' ' ' | cut -c1-400)" >> "$P"
    return 1
  fi
}
run score_oof && run stack && run score_test && run submit && echo "[$(date '+%F %T')] ALL DONE" >> "$P"
