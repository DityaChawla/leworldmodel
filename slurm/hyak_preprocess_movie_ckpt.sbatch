#!/bin/bash
#SBATCH --job-name=lewm_prep
#SBATCH --partition=ckpt-all
#SBATCH --account=stf-ckpt
#SBATCH --qos=ckpt
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/gscratch/stf/dc245/leworldmodel-br/logs/%x_%j.out
#SBATCH --error=/gscratch/stf/dc245/leworldmodel-br/logs/%x_%j.err

set -euo pipefail
set -x

MOVIE="${MOVIE:?MOVIE must be set}"
WORK_DIR="${WORK_DIR:?WORK_DIR must be set}"
REPO_DIR="${REPO_DIR:-/gscratch/stf/dc245/leworldmodel-br/repo}"
DATA_ROOT="${DATA_ROOT:-/gscratch/stf/dc245/leworldmodel-br/data}"
ENV_DIR="${ENV_DIR:-/gscratch/scrubbed/dc245/conda-envs/lewm-br}"
TRAIN_MANIFEST="${WORK_DIR}/suite/algonauts_windowed/train_manifest.jsonl"
VAL_MANIFEST="${WORK_DIR}/suite/algonauts_windowed/val_manifest.jsonl"

source /mmfs1/home/dc245/miniconda3/etc/profile.d/conda.sh

if [ ! -x "${ENV_DIR}/bin/python" ]; then
  rm -rf "${ENV_DIR}"
  conda create -y -p "${ENV_DIR}" python=3.11
fi

conda activate "${ENV_DIR}"
python -m pip install --upgrade pip
python -m pip install torch torchvision pyyaml h5py av
python -m pip install -e "${REPO_DIR}"

cd "${REPO_DIR}"
python -u -m slow_state_wm.prepare_algonauts preflight \
  --dataset-root "${DATA_ROOT}" \
  --movie "${MOVIE}" \
  --work-dir "${WORK_DIR}"

if [ -s "${TRAIN_MANIFEST}" ] && [ -s "${VAL_MANIFEST}" ]; then
  echo "Manifests already exist; regenerating configs only."
  python -u -m slow_state_wm.prepare_algonauts write-configs --work-dir "${WORK_DIR}"
  exit 0
fi

mkdir -p "${WORK_DIR}/suite"

python -u -m slow_state_wm.preprocess_algonauts scan \
  --dataset-root "${DATA_ROOT}" \
  --output-jsonl "${WORK_DIR}/suite/algonauts_source_index.jsonl" \
  --subjects sub-01 \
  --tasks movie10 \
  --movies "${MOVIE}"

python -u -m slow_state_wm.preprocess_algonauts window \
  --index-jsonl "${WORK_DIR}/suite/algonauts_source_index.jsonl" \
  --output-dir "${WORK_DIR}/suite/algonauts_windowed" \
  --clip-len 24 \
  --tr-frames 4 \
  --frame-size 112 \
  --window-stride-tr 4 \
  --max-windows-per-split 256

python -u -m slow_state_wm.prepare_algonauts write-configs --work-dir "${WORK_DIR}"
