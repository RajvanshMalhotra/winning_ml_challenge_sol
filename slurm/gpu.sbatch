#!/bin/bash
#SBATCH -J ber-gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 32
#SBATCH --mem=200G
#SBATCH -o artifacts/logs/%x-%j.out
# Usage: sbatch slurm/gpu.sbatch python -m ber <command> [args]
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate ber
export PYTHONNOUSERSITE=1
cd ~/winning_ml_challenge_sol
srun "$@"
