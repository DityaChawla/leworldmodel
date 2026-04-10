#!/bin/bash
#SBATCH --job-name=lewm_phase
#SBATCH --partition=ckpt-all
#SBATCH --account=stf-ckpt
#SBATCH --qos=ckpt
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/gscratch/stf/dc245/leworldmodel-br/logs/%x_%j.out
#SBATCH --error=/gscratch/stf/dc245/leworldmodel-br/logs/%x_%j.err

set -euo pipefail
set -x

REPO_DIR="${REPO_DIR:-/gscratch/stf/dc245/leworldmodel-br/repo}"
ENV_DIR="${ENV_DIR:-/gscratch/scrubbed/dc245/conda-envs/lewm-br}"
CONFIG_PATH="${CONFIG_PATH:?CONFIG_PATH must be set}"
EXP_NAME="${EXP_NAME:-lewm_phase}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-}"
MAX_VAL_STEPS="${MAX_VAL_STEPS:-}"

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export PIP_DISABLE_PIP_VERSION_CHECK=1

source /mmfs1/home/dc245/miniconda3/etc/profile.d/conda.sh

if [ ! -x "${ENV_DIR}/bin/python" ]; then
  rm -rf "${ENV_DIR}"
  conda create -y -p "${ENV_DIR}" python=3.11
fi

conda activate "${ENV_DIR}"
python -m pip install --upgrade pip
python -m pip install torch torchvision pyyaml h5py av
python -m pip install -e "${REPO_DIR}"

OUTPUT_DIR="$(python - "${CONFIG_PATH}" <<'PY'
import json
import sys
import yaml

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    data = yaml.safe_load(handle)
print(data["output_dir"])
PY
)"

if [ -s "${OUTPUT_DIR}/metrics.json" ] && python - "${OUTPUT_DIR}/metrics.json" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    json.load(handle)
PY
then
  echo "Metrics already exist for ${CONFIG_PATH}; skipping."
  exit 0
fi

cd "${REPO_DIR}"
nvidia-smi || true

EXTRA_ARGS=()
if [ -n "${MAX_TRAIN_STEPS}" ]; then
  EXTRA_ARGS+=(--max-train-steps "${MAX_TRAIN_STEPS}")
fi
if [ -n "${MAX_VAL_STEPS}" ]; then
  EXTRA_ARGS+=(--max-val-steps "${MAX_VAL_STEPS}")
fi

python -u -m slow_state_wm.cli --config "${CONFIG_PATH}" "${EXTRA_ARGS[@]}"
