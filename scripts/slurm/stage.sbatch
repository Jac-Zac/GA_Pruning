#!/bin/bash
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=GPU
#SBATCH --gres=gpu:V100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --export=ALL

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

case "${1:-}" in
  core)
    python main.py train --force
    python main.py structured
    ;;
  structured)
    python main.py structured
    ;;
  unstructured)
    python main.py unstructured
    ;;
  ablation)
    python main.py ablation
    ;;
  *)
    echo "Usage: sbatch $0 {core|structured|unstructured|ablation}" >&2
    exit 2
    ;;
esac
