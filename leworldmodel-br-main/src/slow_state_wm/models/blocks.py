from __future__ import annotations

import math

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int | None = None) -> None:
        super().__init__()
        output_dim = output_dim or input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SIGReg(nn.Module):
    """Inspired by the public MIT-licensed LeWorldModel implementation."""

    def __init__(self, knots: int = 17, num_proj: int = 256) -> None:
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3.0 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-(t.square()) / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        # latents: [B, T, D]
        proj_source = latents.transpose(0, 1)  # [T, B, D]
        proj_matrix = torch.randn(
            proj_source.size(-1),
            self.num_proj,
            device=proj_source.device,
            dtype=proj_source.dtype,
        )
        proj_matrix = proj_matrix / (proj_matrix.norm(p=2, dim=0, keepdim=True) + 1e-6)
        x_t = (proj_source @ proj_matrix).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj_source.size(-2)
        return statistic.mean()


class PositionalEncoding(nn.Module):
    def __init__(self, max_len: int, dim: int) -> None:
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]

