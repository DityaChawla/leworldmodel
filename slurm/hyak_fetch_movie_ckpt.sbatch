#!/bin/bash
#SBATCH --job-name=lewm_fetch
#SBATCH --partition=ckpt-all
#SBATCH --account=stf-ckpt
#SBATCH --qos=ckpt
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/gscratch/stf/dc245/leworldmodel-br/logs/%x_%j.out
#SBATCH --error=/gscratch/stf/dc245/leworldmodel-br/logs/%x_%j.err

set -euo pipefail
set -x

MOVIE="${MOVIE:?MOVIE must be set}"
DATASET_DIR="${DATASET_DIR:-/gscratch/stf/dc245/leworldmodel-br/data/algonauts_2025.competitors}"
FETCH_ENV="${FETCH_ENV:-/gscratch/scrubbed/dc245/conda-envs/lewm-fetch}"
MOVIE_DIR="${DATASET_DIR}/stimuli/movies/movie10/${MOVIE}"

source /mmfs1/home/dc245/miniconda3/etc/profile.d/conda.sh

have_fetched_movie() {
  local found=0
  shopt -s nullglob
  for clip in "${MOVIE_DIR}"/*.mkv; do
    found=1
    if [ ! -e "${clip}" ]; then
      return 1
    fi
  done
  shopt -u nullglob
  [ "${found}" -eq 1 ]
}

if have_fetched_movie; then
  echo "Movie ${MOVIE} is already fetched; skipping."
  exit 0
fi

if [ ! -x "${FETCH_ENV}/bin/python" ]; then
  rm -rf "${FETCH_ENV}"
  conda create -y -p "${FETCH_ENV}" -c conda-forge python=3.11 datalad git-annex
fi

conda activate "${FETCH_ENV}"
cd "${DATASET_DIR}"
datalad get "stimuli/movies/movie10/${MOVIE}"

have_fetched_movie
echo "Fetch completed for movie ${MOVIE}."
