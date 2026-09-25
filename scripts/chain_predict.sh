#!/usr/bin/env bash
# Wait for the test-side channels (Qwen hard names, KG edges), then score test with Model A and write the submission.
cd ~/winning_ml_challenge_sol
while tmux has-session -t hardtest 2>/dev/null || tmux has-session -t kgtest 2>/dev/null; do sleep 60; done
PYTHONNOUSERSITE=1 ~/.conda/envs/ber-sol/bin/python -m ber --config configs/v2.yaml predict --models matcher > artifacts/logs/predict_test.log 2>&1
echo PREDICT_EXIT=$? >> artifacts/logs/predict_test.log
