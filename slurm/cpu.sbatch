#!/bin/bash
#SBATCH -J ber-cpu
#SBATCH -p batch
#SBATCH -c 64
#SBATCH --mem=400G
#SBATCH -o artifacts/logs/%x-%j.out
# Usage: sbatch slurm/cpu.sbatch python -m ber <command> [args]
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate ber
export PYTHONNOUSERSITE=1
cd ~/winning_ml_challenge_sol
srun "$@"
