#!/bin/bash
##################################################################
# Simple sbatch for MAP_slurm.py (one task = one dataset x split)
# Activate conda env 'laplace' before running.
##################################################################
#SBATCH --job-name=MAP_array
#SBATCH --output=slurm_map_%A_%a.out
#SBATCH --error=slurm_map_%A_%a.err
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
# Use --array when launching, e.g. --array=1-20%20

# Project dir (edit if needed)
PROJECT_DIR="/home/pedrojgl/qla/QuadraticLaplaceMethod"

# Configurable via --export on sbatch
DATASET=${DATASET:-boston}       # boston|energy|yacht|concrete|redwine
INBETWEEN=${INBETWEEN:-0}       # 0: get_splits (kfold). 1: get_in_between_splits
SEED=${SEED:-2147483647}
RESULTS_ROOT=${RESULTS_ROOT:-results}

# Path to conda installation and env name (adjust CONDA_PATH if your conda is elsewhere)
CONDA_PATH=${CONDA_PATH:-"/home/pedrojgl/miniconda3"}
CONDA_ENV=${CONDA_ENV:-"laplace"}

cd $PROJECT_DIR

# Source and activate conda env
source "${CONDA_PATH}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# reduce BLAS/OMP threads
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

SPLIT_IDX=${SLURM_ARRAY_TASK_ID}
echo "[MAP] host=$(hostname) dataset=${DATASET} split=${SPLIT_IDX} in_between=${INBETWEEN}"

if [ "$INBETWEEN" -eq 1 ]; then
    python MAP_slurm.py --dataset "$DATASET" --split "$SPLIT_IDX" --in_between_splits --seed "$SEED" --results_root "$RESULTS_ROOT"
else
    python MAP_slurm.py --dataset "$DATASET" --split "$SPLIT_IDX" --seed "$SEED" --results_root "$RESULTS_ROOT"
fi
