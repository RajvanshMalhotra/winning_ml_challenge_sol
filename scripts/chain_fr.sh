#!/usr/bin/env bash
# French reranker fine-tune -> check -> France test scoring -> v16fr (v16 with France re-scored). Waits for the GPU.
set -u
L=artifacts/logs/qwen_fr; mkdir -p $L; P=$L/progress.log
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True QR_MEM=0.85 QR_GC=0 QF_MIN=30 QF_BS=16
until grep -qE "ALL DONE|FAIL" artifacts/logs/chain_qwen3/progress.log; do sleep 30; done
for st in train check score_test; do
  echo "[$(date '+%F %T')] START $st" >> $P
  python -u scripts/qwen_fr.py $st > $L/$st.log 2>&1 || { echo "[$(date '+%F %T')] FAIL $st" >> $P; exit 1; }
  echo "[$(date '+%F %T')] DONE  $st :: $(grep -hE 'AUC|budget|train prompts|France uncertain' $L/$st.log | tr '\n' ' ' | cut -c1-500)" >> $P
done
until grep -q "EXIT=" artifacts/logs/v16/submit.log 2>/dev/null; do sleep 30; done
echo "[$(date '+%F %T')] START v16fr" >> $P
FR_RR=artifacts/v2/reranker_qwen_fr/test_qwen_fr.parquet V16_SUB=v16fr python -u scripts/v16.py submit > $L/v16fr.log 2>&1
echo "[$(date '+%F %T')] DONE  v16fr exit=$? :: $(grep -hE 'France rows|validator exit|mean' $L/v16fr.log | tr '\n' ' ' | cut -c1-400)" >> $P
