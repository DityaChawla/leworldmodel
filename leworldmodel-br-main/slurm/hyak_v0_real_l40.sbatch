#!/bin/bash
#SBATCH --job-name=lewm_real_life
#SBATCH --partition=gpu-l40
#SBATCH --account=stf
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/gscratch/stf/dc245/leworldmodel-br/logs/%x_%j.out
#SBATCH --error=/gscratch/stf/dc245/leworldmodel-br/logs/%x_%j.err

set -euo pipefail
set -x

REPO_DIR="/gscratch/stf/dc245/leworldmodel-br/repo"
DATA_ROOT="/gscratch/stf/dc245/leworldmodel-br/data"
WORK_DIR="/gscratch/stf/dc245/leworldmodel-br/work_real_v0_l40"
ENV_DIR="/gscratch/scrubbed/dc245/conda-envs/lewm-br"
TMP_BASE="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"
JOB_TMP="${TMP_BASE}/${USER}/lewm_v0_${SLURM_JOB_ID}"
mkdir -p "$JOB_TMP" "$WORK_DIR"
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export PIP_DISABLE_PIP_VERSION_CHECK=1

source /mmfs1/home/dc245/miniconda3/etc/profile.d/conda.sh

if [ ! -x "$ENV_DIR/bin/python" ]; then
  rm -rf "$ENV_DIR"
  conda create -y -p "$ENV_DIR" python=3.11
fi

conda activate "$ENV_DIR"
python -m pip install --upgrade pip
python -m pip install torch torchvision pyyaml h5py
python -m pip install -e "$REPO_DIR"

cd "$REPO_DIR"
nvidia-smi || true

python -u -m slow_state_wm.run_suite algonauts \
  --dataset-root "$DATA_ROOT" \
  --work-dir "$WORK_DIR/suite" \
  --clip-len 24 \
  --tr-frames 4 \
  --frame-size 112 \
  --window-stride-tr 4 \
  --subjects sub-01 \
  --tasks movie10 \
  --movies life \
  --max-windows-per-split 256 \
  --max-train-steps 50 \
  --max-val-steps 10
