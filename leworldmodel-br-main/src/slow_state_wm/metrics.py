from __future__ import annotations

import torch

from .losses import pearson_corr


def mean_parcel_correlation(pred_fmri: torch.Tensor, target_fmri: torch.Tensor) -> torch.Tensor:
    flat_pred = pred_fmri.reshape(-1, pred_fmri.size(-1))
    flat_target = target_fmri.reshape(-1, target_fmri.size(-1))
    return pearson_corr(flat_pred, flat_target).mean()


def latent_isotropy_stats(latents: torch.Tensor) -> dict[str, float]:
    flat = latents.reshape(-1, latents.size(-1))
    centered = flat - flat.mean(dim=0, keepdim=True)
    std = centered.std(dim=0)
    cov = centered.T @ centered / max(centered.size(0) - 1, 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    return {
        "latent_std_mean": std.mean().item(),
        "latent_std_min": std.min().item(),
        "latent_offdiag_abs_mean": off_diag.abs().mean().item(),
    }

