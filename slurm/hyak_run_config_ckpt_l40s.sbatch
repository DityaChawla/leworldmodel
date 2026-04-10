#!/bin/bash
#SBATCH --job-name=lewm_cfg
#SBATCH --partition=ckpt-all
#SBATCH --account=stf-ckpt
#SBATCH --qos=ckpt
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/gscratch/stf/dc245/leworldmodel-br/logs/%x_%j.out
#SBATCH --error=/gscratch/stf/dc245/leworldmodel-br/logs/%x_%j.err

set -euo pipefail
set -x

REPO_DIR="/gscratch/stf/dc245/leworldmodel-br/repo"
ENV_DIR="/gscratch/scrubbed/dc245/conda-envs/lewm-br"
CONFIG_PATH="${CONFIG_PATH:?CONFIG_PATH must be set}"
EXP_NAME="${EXP_NAME:-lewm_cfg}"
TMP_BASE="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"
JOB_TMP="${TMP_BASE}/${USER}/${EXP_NAME}_${SLURM_JOB_ID}"
mkdir -p "$JOB_TMP"

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
python -m pip install torch torchvision pyyaml h5py av
python -m pip install -e "$REPO_DIR"

cd "$REPO_DIR"
nvidia-smi || true
python -u -m slow_state_wm.cli --config "$CONFIG_PATH"
