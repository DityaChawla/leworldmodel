from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ..config import ModelConfig
from .blocks import MLP, PositionalEncoding


@dataclass
class WorldModelOutput:
    embeddings: torch.Tensor
    trajectory_latents: torch.Tensor
    pred_latents: torch.Tensor
    target_latents: torch.Tensor


class FrameEncoder(nn.Module):
    def __init__(self, channels: int, frame_size: int, cfg: ModelConfig) -> None:
        super().__init__()
        if frame_size % cfg.patch_size != 0:
            raise ValueError("frame_size must be divisible by patch_size")
        self.patch_embed = nn.Conv2d(
            channels,
            cfg.encoder_dim,
            kernel_size=cfg.patch_size,
            stride=cfg.patch_size,
        )
        n_patches = (frame_size // cfg.patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.encoder_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, cfg.encoder_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.encoder_dim,
            nhead=cfg.encoder_heads,
            dim_feedforward=int(cfg.encoder_dim * cfg.predictor_mlp_ratio),
            dropout=cfg.predictor_dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg.encoder_layers)
        self.norm = nn.LayerNorm(cfg.encoder_dim)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        patches = self.patch_embed(frames)  # [B, D, H', W']
        patches = patches.flatten(2).transpose(1, 2)  # [B, P, D]
        cls = self.cls_token.expand(frames.size(0), -1, -1)
        tokens = torch.cat([cls, patches], dim=1)
        tokens = tokens + self.pos_embed[:, : tokens.size(1)]
        encoded = self.transformer(tokens)
        return self.norm(encoded[:, 0])


class CausalPredictor(nn.Module):
    def __init__(self, cfg: ModelConfig, max_len: int) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.latent_dim,
            nhead=cfg.predictor_heads,
            dim_feedforward=int(cfg.latent_dim * cfg.predictor_mlp_ratio),
            dropout=cfg.predictor_dropout,
            batch_first=True,
            activation="gelu",
        )
        self.positional = PositionalEncoding(max_len=max_len, dim=cfg.latent_dim)
        self.transformer = nn.TransformerEncoder(layer, num_layers=cfg.predictor_layers)
        self.norm = nn.LayerNorm(cfg.latent_dim)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        latents = self.positional(latents)
        seq_len = latents.size(1)
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=latents.device),
            diagonal=1,
        )
        return self.norm(self.transformer(latents, mask=causal_mask))


class PassiveLeWorldModel(nn.Module):
    def __init__(self, cfg: ModelConfig, frame_channels: int, frame_size: int, clip_len: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.clip_len = clip_len
        self.encoder = FrameEncoder(frame_channels, frame_size, cfg)
        self.projector = MLP(cfg.encoder_dim, cfg.projector_hidden_dim, cfg.latent_dim)
        self.predictor = CausalPredictor(cfg, max_len=clip_len - 1)
        self.pred_projector = MLP(cfg.latent_dim, cfg.projector_hidden_dim, cfg.latent_dim)

    def encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        batch, steps = frames.shape[:2]
        flat_frames = frames.reshape(batch * steps, *frames.shape[2:])
        encoded = self.encoder(flat_frames)
        latents = self.projector(encoded).reshape(batch, steps, self.cfg.latent_dim)
        return latents

    def predict_latents(self, latents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        context = latents[:, :-1]
        target = latents[:, 1:]
        predicted = self.pred_projector(self.predictor(context))
        return predicted, target

    def forward(self, frames: torch.Tensor) -> WorldModelOutput:
        embeddings = self.encode_frames(frames)
        pred_latents, target_latents = self.predict_latents(embeddings)
        trajectory_latents = torch.cat([embeddings[:, :1], pred_latents], dim=1)
        return WorldModelOutput(
            embeddings=embeddings,
            trajectory_latents=trajectory_latents,
            pred_latents=pred_latents,
            target_latents=target_latents,
        )

