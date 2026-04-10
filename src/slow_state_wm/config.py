from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DatasetConfig:
    kind: str = "synthetic"
    batch_size: int = 8
    num_workers: int = 0
    train_size: int = 96
    val_size: int = 32
    ood_size: int = 32
    clip_len: int = 24
    tr_frames: int = 4
    frame_size: int = 32
    frame_channels: int = 3
    fmri_dim: int = 128
    num_subjects: int = 4
    synthetic_hidden_dim: int = 12
    synthetic_noise_std: float = 0.08
    synthetic_style_shift: float = 0.45
    manifest_path: str | None = None
    val_manifest_path: str | None = None
    ood_manifest_path: str | None = None


@dataclass
class ModelConfig:
    patch_size: int = 4
    encoder_dim: int = 128
    latent_dim: int = 96
    encoder_layers: int = 4
    encoder_heads: int = 4
    predictor_layers: int = 4
    predictor_heads: int = 4
    predictor_mlp_ratio: float = 4.0
    predictor_dropout: float = 0.1
    projector_hidden_dim: int = 192
    aligner_dim: int = 128
    aligner_layers: int = 2
    aligner_heads: int = 4
    aligner_mlp_ratio: float = 4.0
    aligner_dropout: float = 0.1
    hrf_kernel_size: int = 4
    hrf_learnable: bool = True
    brain_hidden_dim: int = 192
    subject_embedding_dim: int = 32
    trajectory_source: str = "predictive"


@dataclass
class LossConfig:
    lambda_sig: float = 0.05
    lambda_brain: float = 0.2
    lambda_temporal: float = 0.01
    brain_mse_weight: float = 1.0
    brain_corr_weight: float = 0.25


@dataclass
class OptimConfig:
    epochs: int = 3
    lr: float = 3e-4
    weight_decay: float = 1e-2
    device: str = "cpu"


@dataclass
class AblationConfig:
    use_temporal_alignment: bool = True
    brain_on_slow_state: bool = True
    include_pred_loss: bool = True
    freeze_backbone: bool = False
    unfreeze_top_encoder_blocks: int = 0
    use_subject_embedding: bool = True


@dataclass
class ExperimentConfig:
    name: str = "default"
    phase: str = "v0"
    seed: int = 7
    output_dir: str = "runs/default"
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    ablation: AblationConfig = field(default_factory=AblationConfig)


def _construct(dataclass_type: type[Any], values: dict[str, Any] | None) -> Any:
    values = values or {}
    return dataclass_type(**values)


def load_config(path: str | Path) -> ExperimentConfig:
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return ExperimentConfig(
        name=raw.get("name", "default"),
        phase=raw.get("phase", "v0"),
        seed=raw.get("seed", 7),
        output_dir=raw.get("output_dir", "runs/default"),
        dataset=_construct(DatasetConfig, raw.get("dataset")),
        model=_construct(ModelConfig, raw.get("model")),
        loss=_construct(LossConfig, raw.get("loss")),
        optim=_construct(OptimConfig, raw.get("optim")),
        ablation=_construct(AblationConfig, raw.get("ablation")),
    )
