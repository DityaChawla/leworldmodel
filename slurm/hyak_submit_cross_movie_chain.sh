#!/bin/bash
set -euo pipefail

ROOT="/gscratch/stf/dc245/leworldmodel-br"
REPO_DIR="${ROOT}/repo"
DATA_ROOT="${ROOT}/data"
ENV_DIR="/gscratch/scrubbed/dc245/conda-envs/lewm-br"
PREP_TIME="${PREP_TIME:-04:00:00}"

source /mmfs1/home/dc245/miniconda3/etc/profile.d/conda.sh
if [ ! -x "${ENV_DIR}/bin/python" ]; then
  rm -rf "${ENV_DIR}"
  conda create -y -p "${ENV_DIR}" python=3.11
fi
conda activate "${ENV_DIR}"
python -m pip install --upgrade pip
python -m pip install torch torchvision pyyaml h5py av
python -m pip install -e "${REPO_DIR}"

submit_movie_block() {
  local movie="$1"
  local work_dir="$2"
  local dependency="${3:-}"

  local dep_args=()
  if [ -n "${dependency}" ]; then
    dep_args=(--dependency="afterok:${dependency}")
  fi

  python -u -m slow_state_wm.prepare_algonauts preflight \
    --dataset-root "${DATA_ROOT}" \
    --movie "${movie}" \
    --work-dir "${work_dir}" >/dev/null

  local fetch_job
  fetch_job="$(sbatch --parsable "${dep_args[@]}" --export=ALL,MOVIE="${movie}",DATASET_DIR="${DATA_ROOT}/algonauts_2025.competitors" \
    "${REPO_DIR}/slurm/hyak_fetch_movie_ckpt.sbatch")"
  local prep_job
  prep_job="$(sbatch --parsable --dependency="afterok:${fetch_job}" --time="${PREP_TIME}" --export=ALL,MOVIE="${movie}",WORK_DIR="${work_dir}",DATA_ROOT="${DATA_ROOT}" \
    "${REPO_DIR}/slurm/hyak_preprocess_movie_ckpt.sbatch")"

  local phase_job="${prep_job}"
  local config_name
  for config_name in \
    v0_algonauts \
    v1_algonauts \
    v2_algonauts \
    v2_algonauts_fast_latent \
    v2_algonauts_no_temporal \
    v2_algonauts_no_pred
  do
    local config_dir="${work_dir}/suite/generated_configs"
    if [[ "${config_name}" == v2_algonauts_fast_latent || "${config_name}" == v2_algonauts_no_temporal || "${config_name}" == v2_algonauts_no_pred ]]; then
      config_dir="${config_dir}/shortlist"
    fi
    phase_job="$(sbatch --parsable --dependency="afterok:${phase_job}" \
      --export=ALL,CONFIG_PATH="${config_dir}/${config_name}.yaml",EXP_NAME="${movie}_${config_name}" \
      "${REPO_DIR}/slurm/hyak_run_phase_ckpt_l40s.sbatch")"
  done

  echo "${phase_job}"
}

tail_job=""
tail_job="$(submit_movie_block figures "${ROOT}/work_real_figures_ckpt_l40s" "${tail_job}")"
tail_job="$(submit_movie_block wolf "${ROOT}/work_real_wolf_ckpt_l40s" "${tail_job}")"
tail_job="$(submit_movie_block bourne "${ROOT}/work_real_bourne_ckpt_l40s" "${tail_job}")"

echo "Final tail job: ${tail_job}"
