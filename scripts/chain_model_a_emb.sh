#!/usr/bin/env bash
# Wait for the bake-off tmux session to finish, then run Model A v2 (+ fine-tuned bge-m3 cosine feature).
cd ~/winning_ml_challenge_sol
# start as soon as the fine-tuned model has been evaluated (the US-only run can keep the GPU busy meanwhile)
until [ -f artifacts/v2/bakeoff/bgem3__finetune.json ] || ! tmux has-session -t bakeoff 2>/dev/null; do sleep 60; done
while tmux has-session -t matcher 2>/dev/null; do sleep 60; done   # don't overlap with the CPU run
if [ ! -d artifacts/v2/bakeoff/bgem3__finetune/model ]; then echo "no fine-tuned model - skipping" > artifacts/logs/matcher_emb.log; exit 1; fi
PYTHONNOUSERSITE=1 ~/.conda/envs/ber-sol/bin/python -m ber --config configs/matcher_emb.yaml matcher > artifacts/logs/matcher_emb.log 2>&1
echo MATCHER_EXIT=$? >> artifacts/logs/matcher_emb.log
