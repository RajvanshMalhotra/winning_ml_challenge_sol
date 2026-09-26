#!/usr/bin/env bash
# After the train embeddings exist and the reranker has finished TRAINING (GPU), run dense retrieval for train then test.
cd ~/winning_ml_challenge_sol
export PYTHONNOUSERSITE=1
until grep -q "EMB_EXIT=0" artifacts/logs/embed_train.log 2>/dev/null; do sleep 60; done
until grep -qE "DONE  train|FAIL" artifacts/logs/reranker/progress.log 2>/dev/null; do sleep 60; done
for fam in train test; do
  ~/.conda/envs/ber-sol/bin/python scripts/dense_retrieval.py $fam 30 > artifacts/logs/dense_$fam.log 2>&1
  echo "DENSE_EXIT=$?" >> artifacts/logs/dense_$fam.log
done
