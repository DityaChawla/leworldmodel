#!/bin/bash
#SBATCH --job-name=lewm_smoke
#SBATCH --partition=ckpt-all
#SBATCH --account=stf-ckpt
#SBATCH --qos=ckpt
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/gscratch/stf/dc245/leworldmodel-br/logs/%x_%j.out
#SBATCH --error=/gscratch/stf/dc245/leworldmodel-br/logs/%x_%j.err

set -euo pipefail
set -x

REPO_DIR="/gscratch/stf/dc245/leworldmodel-br/repo"
ENV_DIR="/gscratch/scrubbed/dc245/conda-envs/lewm-br"
TMP_BASE="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"
JOB_TMP="${TMP_BASE}/${USER}/lewm_smoke_${SLURM_JOB_ID}"
mkdir -p "$JOB_TMP"

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
python -m slow_state_wm.run_suite synthetic \
  --output-root /gscratch/stf/dc245/leworldmodel-br/runs/suite_synth_hyak \
  --max-train-steps 2 \
  --max-val-steps 1
