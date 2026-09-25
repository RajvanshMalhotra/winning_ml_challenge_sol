#!/usr/bin/env bash
# Sync the working tree to the HPC and run a command there inside the `ber` conda env.
# Usage: scripts/hpc.sh <command...>     e.g. scripts/hpc.sh pytest tests -q
set -euo pipefail
HOST=fcse_shwetasharma@172.16.0.77
KEY="$HOME/.ssh/hpc_nopass"
REMOTE=winning_ml_challenge_sol
SSH="ssh -i $KEY -o BatchMode=yes -o LogLevel=ERROR"
cd "$(dirname "$0")/.."
rsync -az --delete -e "$SSH" \
  --exclude .git --exclude /data --exclude /artifacts --exclude /student_resource \
  --exclude __pycache__ --exclude '*.egg-info' --exclude .pytest_cache \
  ./ "$HOST:~/$REMOTE/"
$SSH "$HOST" "cd ~/$REMOTE && source /opt/anaconda3/etc/profile.d/conda.sh && conda activate ber && export PYTHONNOUSERSITE=1 && $*"
