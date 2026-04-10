#!/bin/bash
#SBATCH --job-name=lewm_fetch_life
#SBATCH --partition=compute
#SBATCH --account=stf
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=/gscratch/stf/dc245/leworldmodel-br/logs/%x_%j.out
#SBATCH --error=/gscratch/stf/dc245/leworldmodel-br/logs/%x_%j.err

set -euo pipefail
set -x

BASE_DIR="/gscratch/stf/dc245/leworldmodel-br"
DATA_ROOT="$BASE_DIR/data/algonauts_2025.competitors"
MOVIES_ROOT="$DATA_ROOT/stimuli/movies"
MOVIE10_DIR="$MOVIES_ROOT/movie10"
TOOLS_ENV_DIR="/gscratch/scrubbed/dc245/conda-envs/lewm-tools"

source /mmfs1/home/dc245/miniconda3/etc/profile.d/conda.sh

if [ ! -x "$TOOLS_ENV_DIR/bin/git-annex" ]; then
  rm -rf "$TOOLS_ENV_DIR"
  conda create -y -p "$TOOLS_ENV_DIR" -c conda-forge python=3.11 git git-annex
fi

conda activate "$TOOLS_ENV_DIR"

mkdir -p "$MOVIES_ROOT"

if [ ! -d "$MOVIE10_DIR/.git" ]; then
  rm -rf "$MOVIE10_DIR"
  git clone https://github.com/courtois-neuromod/movie10.stimuli.git "$MOVIE10_DIR"
fi

cd "$MOVIE10_DIR"
git annex get -J4 --from=conp-ria-storage-http \
  life/life01.mkv \
  life/life02.mkv \
  life/life03.mkv \
  life/life04.mkv \
  life/life05.mkv

find "$MOVIE10_DIR/life" -maxdepth 1 -type f -name 'life*.mkv' -ls
