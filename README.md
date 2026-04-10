# Slow-State Cortical Regularization for Predictive World Models

This repository implements a validation-first version of the research idea:

**use cortical supervision as a selective regularizer on a compact predictive world model, applied only through a slow pooled latent state.**

The stack is built around four pieces:

- a compact passive-video JEPA-style world model
- a slow-state temporal aligner that maps fast latent trajectories to fMRI-rate states
- an explicit causal HRF/FIR aligner that converts pooled TR states into BOLD-aligned states
- a subject-aware parcel prediction head
- phase-aware training for `V0`, `V1`, and `V2`

The code is designed to be useful before CNeuroMod / Algonauts preprocessing is finished:

- `synthetic` mode gives you an end-to-end movie-to-fMRI smoke test
- `manifest` mode expects pre-windowed tensors for real experiments

## What is implemented

- `V0`: world-model-only validation
- `V1`: frozen-latent brain readout (`G + B` only)
- `V2`: selective joint training with slow-state brain regularization
- Ablation toggles for:
  - no temporal alignment
  - brain loss on fast latents instead of slow states
  - no predictive loss during joint phase
  - encoded vs predictive trajectory as the brain supervision source

## Project layout

- [`src/slow_state_wm/models/backbone.py`](/Users/dityachawla/Documents/important docs/leWorldModel-br?/src/slow_state_wm/models/backbone.py): compact passive-video JEPA backbone
- [`src/slow_state_wm/models/brain.py`](/Users/dityachawla/Documents/important docs/leWorldModel-br?/src/slow_state_wm/models/brain.py): temporal aligner, HRF/FIR aligner, and parcel readout head
- [`src/slow_state_wm/data.py`](/Users/dityachawla/Documents/important docs/leWorldModel-br?/src/slow_state_wm/data.py): synthetic dataset and manifest loader
- [`src/slow_state_wm/trainer.py`](/Users/dityachawla/Documents/important docs/leWorldModel-br?/src/slow_state_wm/trainer.py): training loop and phase logic
- [`configs/`](/Users/dityachawla/Documents/important docs/leWorldModel-br?/configs): ready-to-run configs

## Quickstart

Run a synthetic smoke test:

```bash
PYTHONPATH=src python3 -m slow_state_wm.cli --config configs/v0_synthetic.yaml --max-train-steps 2 --max-val-steps 1
PYTHONPATH=src python3 -m slow_state_wm.cli --config configs/v1_synthetic.yaml --max-train-steps 2 --max-val-steps 1
PYTHONPATH=src python3 -m slow_state_wm.cli --config configs/v2_synthetic.yaml --max-train-steps 2 --max-val-steps 1
```

Summarize runs consistently:

```bash
PYTHONPATH=src python3 -m slow_state_wm.report_runs --runs-dir runs --metric val/parcel_corr
PYTHONPATH=src python3 -m slow_state_wm.report_matrix \
  --row life=/path/to/life/suite \
  --row figures=/path/to/figures/suite \
  --row wolf=/path/to/wolf/suite \
  --row bourne=/path/to/bourne/suite
```

Run the whole suite with one command:

```bash
PYTHONPATH=src python3 -m slow_state_wm.run_suite synthetic --output-root runs/suite_synthetic
PYTHONPATH=src python3 -m slow_state_wm.run_suite algonauts --dataset-root /path/to/algonauts_root --work-dir runs/algonauts_suite
```

For real Hyak production runs, prefer the staged flow:

```bash
PYTHONPATH=src python3 -m slow_state_wm.prepare_algonauts preflight \
  --dataset-root /path/to/algonauts_root \
  --movie figures \
  --work-dir runs/work_real_figures

PYTHONPATH=src python3 -m slow_state_wm.prepare_algonauts write-configs \
  --work-dir runs/work_real_figures
```

## Real-data mode

For real data, use the `manifest` dataset kind and point the config at a JSONL manifest where each row contains:

```json
{
  "frames_path": "/abs/path/to/frames.pt",
  "fmri_path": "/abs/path/to/fmri.pt",
  "subject_id": 0,
  "sample_id": "friends_s01e01_clip000"
}
```

Expected tensor shapes:

- `frames`: `[T, C, H, W]`
- `fmri`: `[T_tr, n_parcels]`

The template config is at [`configs/algonauts_template.yaml`](/Users/dityachawla/Documents/important docs/leWorldModel-br?/configs/algonauts_template.yaml).

If you have the official Algonauts 2025 folder layout locally, this repo now includes a preprocessing path in [`src/slow_state_wm/preprocess_algonauts.py`](/Users/dityachawla/Documents/important docs/leWorldModel-br?/src/slow_state_wm/preprocess_algonauts.py):

```bash
PYTHONPATH=src python3 -m slow_state_wm.preprocess_algonauts scan \
  --dataset-root /path/to/algonauts_root \
  --output-jsonl data/algonauts_source_index.jsonl

PYTHONPATH=src python3 -m slow_state_wm.preprocess_algonauts window \
  --index-jsonl data/algonauts_source_index.jsonl \
  --output-dir data/algonauts_windowed \
  --clip-len 24 \
  --tr-frames 4 \
  --frame-size 224 \
  --window-stride-tr 1
```

This preprocessing uses the official dataset naming conventions, keeps train/val splits deterministic at the stimulus level, and aligns each video window to the fMRI target using the standard `4.47s` hemodynamic delay.

## Timeout-proof Hyak flow

The production Hyak workflow is intentionally split into independent stages:

1. [`slurm/hyak_fetch_movie_ckpt.sbatch`](/Users/dityachawla/Documents/important docs/leWorldModel-br?/slurm/hyak_fetch_movie_ckpt.sbatch)
2. [`slurm/hyak_preprocess_movie_ckpt.sbatch`](/Users/dityachawla/Documents/important docs/leWorldModel-br?/slurm/hyak_preprocess_movie_ckpt.sbatch)
3. [`slurm/hyak_run_phase_ckpt_l40s.sbatch`](/Users/dityachawla/Documents/important docs/leWorldModel-br?/slurm/hyak_run_phase_ckpt_l40s.sbatch)

Each stage is resumable:

- fetch skips if the movie clips already resolve to real annex content
- preprocess skips if train and val manifests already exist
- phase runs skip if `metrics.json` already exists and parses

The helper submitter [`slurm/hyak_submit_cross_movie_chain.sh`](/Users/dityachawla/Documents/important docs/leWorldModel-br?/slurm/hyak_submit_cross_movie_chain.sh) queues the full `figures -> wolf -> bourne` rigor batch with `afterok` dependencies.

## Training phases

### V0

Trains only the world model objective:

- predictive latent loss
- SIGReg latent regularization

Use this to validate pixel-to-latent dynamics and data interfaces.

### V1

Freezes the world-model backbone and trains only:

- temporal aligner `G`
- HRF/FIR aligner
- brain head `B`

This answers whether LeWorldModel-style latent trajectories already carry cortical signal.

### V2

Keeps the visual encoder mostly frozen and jointly trains:

- predictor
- prediction projector
- temporal aligner
- HRF/FIR aligner
- brain head
- optional top encoder blocks

This is the main experiment for testing whether cortical supervision improves latent quality without breaking predictive modeling.

## Metrics

- predictive MSE
- parcel-wise Pearson correlation
- latent isotropy diagnostics
- optional OOD loader support

## Timescale alignment

The brain target is treated as a delayed, temporally smeared function of the latent trajectory rather than a direct readout from one fast state. The implemented brain path is:

`latent sequence -> temporal encoder -> TR pooling -> causal HRF/FIR mixing -> parcel head`

This makes the code match the intended hypothesis: if the temporal alignment is wrong, the auxiliary cortical signal is weakened before it can regularize the world-model state.

## Notes

- This repo does **not** reproduce the official Algonauts feature pipeline.
- It gives you the world-model side, the slow-state interface, the parcel prediction head, and a clean training/evaluation harness for rapid falsification.
- The synthetic dataset is intentionally structured so that slow-state supervision is meaningful and smoke-testable on one GPU or CPU.

