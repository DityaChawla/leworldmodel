from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from .config import DatasetConfig


@dataclass
class ClipBatch:
    frames: torch.Tensor
    fmri: torch.Tensor
    subject_ids: torch.Tensor
    sample_ids: list[str]
    split: str

    def to(self, device: str) -> "ClipBatch":
        return ClipBatch(
            frames=self.frames.to(device),
            fmri=self.fmri.to(device),
            subject_ids=self.subject_ids.to(device),
            sample_ids=self.sample_ids,
            split=self.split,
        )


def _collate(samples: list[dict]) -> ClipBatch:
    return ClipBatch(
        frames=torch.stack([sample["frames"] for sample in samples], dim=0),
        fmri=torch.stack([sample["fmri"] for sample in samples], dim=0),
        subject_ids=torch.tensor([sample["subject_id"] for sample in samples], dtype=torch.long),
        sample_ids=[sample["sample_id"] for sample in samples],
        split=samples[0]["split"],
    )


class SyntheticMovieFmriDataset(Dataset):
    def __init__(self, cfg: DatasetConfig, split: str, size: int, seed: int) -> None:
        self.cfg = cfg
        self.split = split
        self.size = size
        self.seed = seed
        self.tr_count = cfg.clip_len // cfg.tr_frames
        self.split_offset = {"train": 0, "val": 10_000, "ood": 20_000}[split]
        if cfg.clip_len % cfg.tr_frames != 0:
            raise ValueError("clip_len must be divisible by tr_frames for the synthetic dataset")

        basis_gen = torch.Generator().manual_seed(seed)
        hidden_dim = cfg.synthetic_hidden_dim
        frame_shape = (cfg.frame_channels, cfg.frame_size, cfg.frame_size)
        self.frame_basis = torch.randn(hidden_dim, *frame_shape, generator=basis_gen) / math.sqrt(hidden_dim)
        self.transition = self._build_transition(hidden_dim, basis_gen)
        self.subject_weights = torch.randn(
            cfg.num_subjects,
            hidden_dim,
            cfg.fmri_dim,
            generator=basis_gen,
        ) / math.sqrt(hidden_dim)
        self.hrf_kernel = torch.tensor([0.1, 0.2, 0.4, 0.3], dtype=torch.float32)
        self.ood_color_shift = torch.randn(cfg.frame_channels, generator=basis_gen) * cfg.synthetic_style_shift

    def _build_transition(self, hidden_dim: int, generator: torch.Generator) -> torch.Tensor:
        diagonal = 0.72 + 0.18 * torch.rand(hidden_dim, generator=generator)
        low_rank = torch.randn(hidden_dim, 3, generator=generator)
        low_rank = low_rank @ low_rank.transpose(0, 1)
        low_rank = low_rank / low_rank.norm()
        return torch.diag(diagonal) + 0.08 * low_rank

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> dict:
        generator = torch.Generator().manual_seed(self.seed + self.split_offset + index * 17)
        hidden = self._sample_hidden(generator)
        subject_id = index % self.cfg.num_subjects
        frames = self._render_frames(hidden, generator)
        fmri = self._render_fmri(hidden, subject_id, generator)
        return {
            "frames": frames,
            "fmri": fmri,
            "subject_id": subject_id,
            "sample_id": f"{self.split}-{index:05d}",
            "split": self.split,
        }

    def _sample_hidden(self, generator: torch.Generator) -> torch.Tensor:
        hidden_dim = self.cfg.synthetic_hidden_dim
        hidden = torch.zeros(self.cfg.clip_len, hidden_dim)
        hidden[0] = torch.randn(hidden_dim, generator=generator)
        drift = torch.randn(hidden_dim, generator=generator) * 0.03
        if self.split == "ood":
            drift = drift + 0.07
        for step in range(1, self.cfg.clip_len):
            noise = torch.randn(hidden_dim, generator=generator) * self.cfg.synthetic_noise_std
            hidden[step] = hidden[step - 1] @ self.transition.T + drift + noise
        return hidden

    def _render_frames(self, hidden: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        frames = torch.einsum("td,dchw->tchw", hidden, self.frame_basis)
        if self.split == "ood":
            frames = frames.clone()
            frames[:, :, :, :] = frames + self.ood_color_shift.view(1, -1, 1, 1)
            frames = torch.roll(frames, shifts=2, dims=-1)
        frames = frames + 0.05 * torch.randn(frames.shape, generator=generator)
        frames = torch.sigmoid(frames)
        return frames

    def _render_fmri(self, hidden: torch.Tensor, subject_id: int, generator: torch.Generator) -> torch.Tensor:
        slow_hidden = hidden.view(self.tr_count, self.cfg.tr_frames, -1).mean(dim=1)
        padded = torch.zeros(self.tr_count + self.hrf_kernel.numel() - 1, slow_hidden.size(-1))
        for lag, weight in enumerate(self.hrf_kernel):
            padded[lag : lag + self.tr_count] += weight * slow_hidden
        hemodynamic_hidden = padded[: self.tr_count]
        fmri = hemodynamic_hidden @ self.subject_weights[subject_id]
        fmri = fmri + self.cfg.synthetic_noise_std * torch.randn(fmri.shape, generator=generator)
        return fmri


class ManifestDataset(Dataset):
    def __init__(self, manifest_path: str, split: str) -> None:
        self.manifest_path = Path(manifest_path)
        self.split = split
        self.records = []
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                self.records.append(record)
        if not self.records:
            raise ValueError(f"Manifest at {manifest_path} did not contain any samples")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        frames = torch.load(record["frames_path"], map_location="cpu")
        fmri = torch.load(record["fmri_path"], map_location="cpu")
        if frames.ndim != 4:
            raise ValueError(f"Expected frames to have shape [T, C, H, W], got {frames.shape}")
        if fmri.ndim != 2:
            raise ValueError(f"Expected fMRI to have shape [T_tr, n_parcels], got {fmri.shape}")
        return {
            "frames": frames.float(),
            "fmri": fmri.float(),
            "subject_id": int(record["subject_id"]),
            "sample_id": record.get("sample_id", f"{self.split}-{index:05d}"),
            "split": self.split,
        }


def build_dataloaders(cfg: DatasetConfig, seed: int) -> dict[str, DataLoader]:
    if cfg.kind == "synthetic":
        train_dataset = SyntheticMovieFmriDataset(cfg, split="train", size=cfg.train_size, seed=seed)
        val_dataset = SyntheticMovieFmriDataset(cfg, split="val", size=cfg.val_size, seed=seed + 1)
        ood_dataset = SyntheticMovieFmriDataset(cfg, split="ood", size=cfg.ood_size, seed=seed + 2)
    elif cfg.kind == "manifest":
        if not cfg.manifest_path or not cfg.val_manifest_path:
            raise ValueError("manifest dataset requires manifest_path and val_manifest_path")
        train_dataset = ManifestDataset(cfg.manifest_path, split="train")
        val_dataset = ManifestDataset(cfg.val_manifest_path, split="val")
        ood_dataset = ManifestDataset(cfg.ood_manifest_path, split="ood") if cfg.ood_manifest_path else None
    else:
        raise ValueError(f"Unsupported dataset kind: {cfg.kind}")

    loaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            collate_fn=_collate,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            collate_fn=_collate,
        ),
    }
    if ood_dataset is not None:
        loaders["ood"] = DataLoader(
            ood_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            collate_fn=_collate,
        )
    return loaders
