from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ..config import AblationConfig, DatasetConfig, ModelConfig
from .backbone import PassiveLeWorldModel, WorldModelOutput
from .blocks import MLP, PositionalEncoding


@dataclass
class ModelOutputs:
    world: WorldModelOutput
    tr_states: torch.Tensor
    aligned_states: torch.Tensor
    pred_fmri: torch.Tensor


class TemporalAligner(nn.Module):
    def __init__(self, model_cfg: ModelConfig, clip_len: int, tr_count: int) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=model_cfg.aligner_dim,
            nhead=model_cfg.aligner_heads,
            dim_feedforward=int(model_cfg.aligner_dim * model_cfg.aligner_mlp_ratio),
            dropout=model_cfg.aligner_dropout,
            batch_first=True,
            activation="gelu",
        )
        self.input_proj = nn.Linear(model_cfg.latent_dim, model_cfg.aligner_dim)
        self.positional = PositionalEncoding(max_len=clip_len, dim=model_cfg.aligner_dim)
        self.encoder = nn.TransformerEncoder(layer, num_layers=model_cfg.aligner_layers)
        self.pool = nn.AdaptiveAvgPool1d(tr_count)

    def forward(self, latents: torch.Tensor, use_temporal_alignment: bool) -> torch.Tensor:
        x = self.input_proj(latents)
        if use_temporal_alignment:
            x = self.positional(x)
            x = self.encoder(x)
        x = self.pool(x.transpose(1, 2)).transpose(1, 2)
        return x


class HemodynamicAligner(nn.Module):
    def __init__(self, hidden_dim: int, kernel_size: int, learnable: bool) -> None:
        super().__init__()
        if kernel_size < 1:
            raise ValueError("hrf_kernel_size must be >= 1")
        init_kernel = torch.linspace(1, kernel_size, steps=kernel_size, dtype=torch.float32)
        init_kernel = torch.minimum(init_kernel, torch.flip(init_kernel, dims=[0]))
        init_kernel = init_kernel / init_kernel.sum()
        self.kernel_size = kernel_size
        self.learnable = learnable
        if learnable:
            self.kernel_logits = nn.Parameter(init_kernel.log())
        else:
            self.register_buffer("kernel", init_kernel)
        self.channel_mixer = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def current_kernel(self) -> torch.Tensor:
        if self.learnable:
            return torch.softmax(self.kernel_logits, dim=0)
        return self.kernel

    def forward(self, tr_states: torch.Tensor) -> torch.Tensor:
        kernel = self.current_kernel()
        batch, steps, dim = tr_states.shape
        padded = tr_states.new_zeros(batch, steps + self.kernel_size - 1, dim)
        for lag, weight in enumerate(kernel):
            padded[:, lag : lag + steps] += weight * tr_states
        aligned = padded[:, :steps]
        return self.norm(self.channel_mixer(aligned))


class SubjectConditionedBrainHead(nn.Module):
    def __init__(
        self,
        model_cfg: ModelConfig,
        dataset_cfg: DatasetConfig,
        use_subject_embedding: bool,
    ) -> None:
        super().__init__()
        self.use_subject_embedding = use_subject_embedding
        hidden_dim = model_cfg.brain_hidden_dim
        input_dim = model_cfg.aligner_dim
        if use_subject_embedding:
            self.subject_embedding = nn.Embedding(dataset_cfg.num_subjects, model_cfg.subject_embedding_dim)
            input_dim += model_cfg.subject_embedding_dim
        self.mlp = MLP(input_dim, hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, dataset_cfg.fmri_dim)

    def forward(self, slow_states: torch.Tensor, subject_ids: torch.Tensor) -> torch.Tensor:
        if self.use_subject_embedding:
            subject_embed = self.subject_embedding(subject_ids)
            subject_embed = subject_embed.unsqueeze(1).expand(-1, slow_states.size(1), -1)
            slow_states = torch.cat([slow_states, subject_embed], dim=-1)
        hidden = self.mlp(slow_states)
        return self.output(hidden)


class BrainRegularizedWorldModel(nn.Module):
    def __init__(
        self,
        model_cfg: ModelConfig,
        dataset_cfg: DatasetConfig,
        ablation_cfg: AblationConfig,
    ) -> None:
        super().__init__()
        self.model_cfg = model_cfg
        self.dataset_cfg = dataset_cfg
        self.ablation_cfg = ablation_cfg
        self.backbone = PassiveLeWorldModel(
            cfg=model_cfg,
            frame_channels=dataset_cfg.frame_channels,
            frame_size=dataset_cfg.frame_size,
            clip_len=dataset_cfg.clip_len,
        )
        tr_count = dataset_cfg.clip_len // dataset_cfg.tr_frames
        self.aligner = TemporalAligner(model_cfg, clip_len=dataset_cfg.clip_len, tr_count=tr_count)
        self.hemodynamic_aligner = HemodynamicAligner(
            hidden_dim=model_cfg.aligner_dim,
            kernel_size=model_cfg.hrf_kernel_size,
            learnable=model_cfg.hrf_learnable,
        )
        self.brain_head = SubjectConditionedBrainHead(
            model_cfg=model_cfg,
            dataset_cfg=dataset_cfg,
            use_subject_embedding=ablation_cfg.use_subject_embedding,
        )

    def forward(self, frames: torch.Tensor, subject_ids: torch.Tensor) -> ModelOutputs:
        world = self.backbone(frames)
        if self.model_cfg.trajectory_source == "encoded":
            brain_source = world.embeddings
        else:
            brain_source = world.trajectory_latents

        if not self.ablation_cfg.brain_on_slow_state:
            tr_states = self.aligner(brain_source, use_temporal_alignment=False)
            aligned_states = tr_states
        else:
            tr_states = self.aligner(
                brain_source,
                use_temporal_alignment=self.ablation_cfg.use_temporal_alignment,
            )
            aligned_states = self.hemodynamic_aligner(tr_states)
        pred_fmri = self.brain_head(aligned_states, subject_ids)
        return ModelOutputs(
            world=world,
            tr_states=tr_states,
            aligned_states=aligned_states,
            pred_fmri=pred_fmri,
        )

    def freeze_all_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_predictive_stack(self) -> None:
        for module in [self.backbone.predictor, self.backbone.pred_projector]:
            for param in module.parameters():
                param.requires_grad = True

    def unfreeze_top_encoder_blocks(self, count: int) -> None:
        if count <= 0:
            return
        blocks = list(self.backbone.encoder.transformer.layers)
        for block in blocks[-count:]:
            for param in block.parameters():
                param.requires_grad = True
        for param in self.backbone.encoder.norm.parameters():
            param.requires_grad = True
