#!/usr/bin/env bash
# bge-m3 bake-off, run inside tmux on the HPC:  tmux new -d -s bakeoff 'bash scripts/run_bakeoff_bge.sh'
# smoke test first (stops on failure), then zero-shot -> fine-tune -> fine-tune on US only (India = France stand-in).
set -uo pipefail
cd ~/winning_ml_challenge_sol
export PYTHONNOUSERSITE=1
PY=~/.conda/envs/ber-sol/bin/python
LOG=artifacts/logs/bakeoff_bge
mkdir -p "$LOG"
step() {  # name, config, args...
  local name=$1 cfg=$2; shift 2
  echo "[$(date '+%F %T')] START $name" | tee -a "$LOG/progress.log"
  if $PY -m ber --config "$cfg" bakeoff "$@" > "$LOG/$name.log" 2>&1; then
    echo "[$(date '+%F %T')] DONE  $name :: $(grep -E '^bgem3' "$LOG/$name.log" | tr '\n' ' ' | cut -c1-400)" | tee -a "$LOG/progress.log"
  else
    echo "[$(date '+%F %T')] FAIL  $name (see $LOG/$name.log)" | tee -a "$LOG/progress.log"; return 1
  fi
}
step smoke_zeroshot configs/smoke.yaml zeroshot --model bgem3 && \
step smoke_finetune configs/smoke.yaml finetune --model bgem3 || { echo "smoke failed - stopping" | tee -a "$LOG/progress.log"; exit 1; }
step zeroshot    configs/v2.yaml zeroshot --model bgem3
step finetune    configs/v2.yaml finetune --model bgem3
step finetune_US configs/v2.yaml finetune --model bgem3 --train-countries US
$PY -m ber --config configs/v2.yaml bakeoff report >> "$LOG/progress.log" 2>&1
echo "[$(date '+%F %T')] ALL DONE" | tee -a "$LOG/progress.log"
