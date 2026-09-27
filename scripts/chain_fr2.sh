#!/usr/bin/env bash
# French Qwen: merge (own process) -> check -> France test scoring -> v16fr and v16fr + acronym routing.
set -u
L=artifacts/logs/qwen_fr; P=$L/progress.log
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True QR_MEM=0.85
for st in merge check score_test; do
  echo "[$(date '+%F %T')] START $st" >> $P
  python -u scripts/qwen_fr.py $st > $L/$st.log 2>&1 || { echo "[$(date '+%F %T')] FAIL $st :: $(tail -2 $L/$st.log | tr '\n' ' ' | cut -c1-300)" >> $P; exit 1; }
  echo "[$(date '+%F %T')] DONE  $st :: $(grep -hE 'AUC|France uncertain|merged' $L/$st.log | tr '\n' ' ' | cut -c1-500)" >> $P
done
echo "[$(date '+%F %T')] START v16fr" >> $P
FR_RR=artifacts/v2/reranker_qwen_fr/test_qwen_fr.parquet V16_SUB=v16fr python -u scripts/v16.py submit > $L/v16fr.log 2>&1
echo "[$(date '+%F %T')] DONE  v16fr :: $(grep -hE 'France rows|validator exit|mean' $L/v16fr.log | tr '\n' ' ' | cut -c1-400)" >> $P
python -u scripts/fr_acronym.py splice artifacts/submissions/v16fr artifacts/submissions/v16fracr 0.9 > $L/v16fracr.log 2>&1
echo "[$(date '+%F %T')] DONE  v16fracr :: $(grep -hE 'added|validator exit' $L/v16fracr.log | tr '\n' ' ' | cut -c1-300)" >> $P
