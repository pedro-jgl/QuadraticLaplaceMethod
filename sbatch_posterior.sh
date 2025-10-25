#!/bin/bash
##################################################################
# Simple sbatch for Posterior_slurm.py (one task = one dataset x split x method)
# Activate conda env 'laplace' before running.
##################################################################
#SBATCH --job-name=Posterior_array
#SBATCH --output=slurm_post_%A_%a.out
#SBATCH --error=slurm_post_%A_%a.err
#SBATCH --time=1-01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
# Use --array when launching, e.g. --array=1-20%40

# Project dir (edit if needed)
PROJECT_DIR="/home/pedrojgl/qla/QuadraticLaplaceMethod"

# Configurable via --export on sbatch
DATASET=${DATASET:-boston}
METHOD=${METHOD:-lla}            # lla|qla|llla|kron
INBETWEEN=${INBETWEEN:-0}
SEED=${SEED:-2147483647}
RESULTS_ROOT=${RESULTS_ROOT:-results}

# Path to conda installation and env name (adjust if needed)
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
echo "[Posterior] host=$(hostname) dataset=${DATASET} split=${SPLIT_IDX} method=${METHOD} in_between=${INBETWEEN}"

if [ "$INBETWEEN" -eq 1 ]; then
    python Posterior.py --dataset "$DATASET" --split "$SPLIT_IDX" --name "$METHOD" --in_between_splits --seed "$SEED" --results_root "$RESULTS_ROOT"
else
    python Posterior.py --dataset "$DATASET" --split "$SPLIT_IDX" --name "$METHOD" --seed "$SEED" --results_root "$RESULTS_ROOT"
fi
