from __future__ import annotations

import torch
import torch.nn.functional as F


def predictive_loss(pred_latents: torch.Tensor, target_latents: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred_latents, target_latents)


def pearson_corr(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    numerator = (x * y).sum(dim=0)
    denominator = torch.sqrt((x.square().sum(dim=0) + eps) * (y.square().sum(dim=0) + eps))
    return numerator / denominator


def brain_loss(
    pred_fmri: torch.Tensor,
    target_fmri: torch.Tensor,
    mse_weight: float,
    corr_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    flat_pred = pred_fmri.reshape(-1, pred_fmri.size(-1))
    flat_target = target_fmri.reshape(-1, target_fmri.size(-1))
    mse = F.mse_loss(flat_pred, flat_target)
    corr = pearson_corr(flat_pred, flat_target).mean()
    total = mse_weight * mse + corr_weight * (1.0 - corr)
    return total, {"brain_mse": mse, "brain_corr": corr}


def temporal_smoothness(states: torch.Tensor) -> torch.Tensor:
    if states.size(1) < 2:
        return states.new_tensor(0.0)
    return (states[:, 1:] - states[:, :-1]).square().mean()

