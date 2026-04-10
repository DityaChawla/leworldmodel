from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path

import torch

from .config import ExperimentConfig
from .data import ClipBatch, build_dataloaders
from .losses import brain_loss, predictive_loss, temporal_smoothness
from .metrics import latent_isotropy_stats, mean_parcel_correlation
from .models import BrainRegularizedWorldModel
from .models.blocks import SIGReg


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ExperimentRunner:
    def __init__(self, cfg: ExperimentConfig) -> None:
        self.cfg = cfg
        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(cfg.optim.device if torch.cuda.is_available() or cfg.optim.device == "cpu" else "cpu")
        self.model = BrainRegularizedWorldModel(cfg.model, cfg.dataset, cfg.ablation).to(self.device)
        self.sigreg = SIGReg().to(self.device)
        self.loaders = build_dataloaders(cfg.dataset, seed=cfg.seed)
        self._configure_phase()
        trainable_params = [param for param in self.model.parameters() if param.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=cfg.optim.lr,
            weight_decay=cfg.optim.weight_decay,
        )

    def _configure_phase(self) -> None:
        phase = self.cfg.phase.lower()
        if phase == "v0":
            return
        if phase == "v1" or self.cfg.ablation.freeze_backbone:
            self.model.freeze_all_backbone()
        if phase == "v2":
            self.model.unfreeze_predictive_stack()
            self.model.unfreeze_top_encoder_blocks(self.cfg.ablation.unfreeze_top_encoder_blocks)

    def _compute_losses(self, outputs, batch: ClipBatch) -> tuple[torch.Tensor, dict[str, float]]:
        metrics = {}
        pred_loss = predictive_loss(outputs.world.pred_latents, outputs.world.target_latents)
        sig_loss = self.sigreg(outputs.world.embeddings)
        total = pred_loss.new_tensor(0.0)

        if self.cfg.phase.lower() == "v0":
            total = pred_loss + self.cfg.loss.lambda_sig * sig_loss
        else:
            if self.cfg.ablation.include_pred_loss:
                total = total + pred_loss
            total = total + self.cfg.loss.lambda_sig * sig_loss
            if self.cfg.loss.lambda_brain > 0:
                brain_total, brain_terms = brain_loss(
                    outputs.pred_fmri,
                    batch.fmri,
                    mse_weight=self.cfg.loss.brain_mse_weight,
                    corr_weight=self.cfg.loss.brain_corr_weight,
                )
                total = total + self.cfg.loss.lambda_brain * brain_total
                metrics.update({name: value.item() for name, value in brain_terms.items()})
            if self.cfg.loss.lambda_temporal > 0:
                smooth = temporal_smoothness(outputs.aligned_states)
                total = total + self.cfg.loss.lambda_temporal * smooth
                metrics["temporal_smoothness"] = smooth.item()

        metrics["pred_loss"] = pred_loss.item()
        metrics["sigreg_loss"] = sig_loss.item()
        metrics["loss"] = total.item()
        metrics["parcel_corr"] = mean_parcel_correlation(outputs.pred_fmri, batch.fmri).item()
        metrics.update(latent_isotropy_stats(outputs.world.embeddings))
        return total, metrics

    def _run_epoch(
        self,
        split: str,
        training: bool,
        max_steps: int | None,
    ) -> dict[str, float]:
        loader = self.loaders[split]
        running: dict[str, float] = {}
        steps = 0
        self.model.train(training)

        for batch in loader:
            batch = batch.to(str(self.device))
            if training:
                self.optimizer.zero_grad(set_to_none=True)
            outputs = self.model(batch.frames, batch.subject_ids)
            loss, metrics = self._compute_losses(outputs, batch)
            if training:
                loss.backward()
                self.optimizer.step()

            for name, value in metrics.items():
                running[name] = running.get(name, 0.0) + float(value)
            steps += 1
            if max_steps is not None and steps >= max_steps:
                break

        return {name: value / max(steps, 1) for name, value in running.items()}

    def train(self, max_train_steps: int | None = None, max_val_steps: int | None = None) -> dict[str, dict[str, float]]:
        history: dict[str, dict[str, float]] = {}
        for epoch in range(1, self.cfg.optim.epochs + 1):
            train_metrics = self._run_epoch("train", training=True, max_steps=max_train_steps)
            val_metrics = self._run_epoch("val", training=False, max_steps=max_val_steps)
            ood_metrics = {}
            if "ood" in self.loaders:
                ood_metrics = self._run_epoch("ood", training=False, max_steps=max_val_steps)
            history[f"epoch_{epoch}"] = {
                **{f"train/{k}": v for k, v in train_metrics.items()},
                **{f"val/{k}": v for k, v in val_metrics.items()},
                **{f"ood/{k}": v for k, v in ood_metrics.items()},
            }
            summary = ", ".join(
                [
                    f"train_loss={train_metrics.get('loss', 0.0):.4f}",
                    f"val_loss={val_metrics.get('loss', 0.0):.4f}",
                    f"val_corr={val_metrics.get('parcel_corr', 0.0):.4f}",
                    f"ood_corr={ood_metrics.get('parcel_corr', 0.0):.4f}" if ood_metrics else "ood_corr=n/a",
                ]
            )
            print(f"[epoch {epoch}] {summary}")

        self._save(history)
        return history

    def _save(self, history: dict[str, dict[str, float]]) -> None:
        checkpoint = {
            "config": asdict(self.cfg),
            "model_state": self.model.state_dict(),
            "history": history,
        }
        torch.save(checkpoint, self.output_dir / "checkpoint.pt")
        with (self.output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2)


def run_experiment(cfg: ExperimentConfig, max_train_steps: int | None = None, max_val_steps: int | None = None) -> dict[str, dict[str, float]]:
    set_seed(cfg.seed)
    runner = ExperimentRunner(cfg)
    return runner.train(max_train_steps=max_train_steps, max_val_steps=max_val_steps)
