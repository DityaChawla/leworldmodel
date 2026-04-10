from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from .preprocess_algonauts import _require_h5py, _resolve_dataset_root, _resolve_fmri_path


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _merge(dst: dict[str, Any], src: dict[str, Any]) -> None:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _merge(dst[key], value)
        else:
            dst[key] = value


def _phase_template(phase_name: str) -> Path:
    if phase_name == "v0":
        return Path("configs") / "v0_synthetic.yaml"
    if phase_name == "v1":
        return Path("configs") / "v1_synthetic.yaml"
    return Path("configs") / "v2_synthetic.yaml"


def _base_updates(
    work_dir: Path,
    manifest_dir: Path,
    clip_len: int,
    tr_frames: int,
    frame_size: int,
    epochs: int,
    lr: float,
) -> dict[str, Any]:
    return {
        "dataset": {
            "kind": "manifest",
            "manifest_path": str((manifest_dir / "train_manifest.jsonl").resolve()),
            "val_manifest_path": str((manifest_dir / "val_manifest.jsonl").resolve()),
            "ood_manifest_path": str((manifest_dir / "test_manifest.jsonl").resolve())
            if (manifest_dir / "test_manifest.jsonl").exists()
            else None,
            "clip_len": clip_len,
            "tr_frames": tr_frames,
            "frame_size": frame_size,
            "frame_channels": 3,
            "fmri_dim": 1000,
            "num_subjects": 4,
            "train_size": 0,
            "val_size": 0,
            "ood_size": 0,
        },
        "model": {
            "patch_size": 14 if frame_size >= 112 else 4,
            "encoder_dim": 192,
            "latent_dim": 192,
            "encoder_layers": 8,
            "encoder_heads": 6,
            "predictor_layers": 6,
            "predictor_heads": 6,
            "projector_hidden_dim": 384,
            "aligner_dim": 192,
            "aligner_layers": 2,
            "aligner_heads": 6,
            "brain_hidden_dim": 384,
        },
        "optim": {"device": "cuda", "epochs": epochs, "lr": lr},
    }


def _write_config(base_config_path: Path, target_path: Path, updates: dict[str, Any]) -> Path:
    data = _load_yaml(base_config_path)
    _merge(data, updates)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)
    return target_path


def generate_movie_configs(
    work_dir: Path,
    clip_len: int = 24,
    tr_frames: int = 4,
    frame_size: int = 112,
    epochs: int = 20,
    lr: float = 1e-4,
) -> dict[str, Path]:
    manifest_dir = work_dir / "suite" / "algonauts_windowed"
    config_dir = work_dir / "suite" / "generated_configs"
    shortlist_dir = config_dir / "shortlist"
    shortlist_dir.mkdir(parents=True, exist_ok=True)

    base_updates = _base_updates(
        work_dir=work_dir,
        manifest_dir=manifest_dir,
        clip_len=clip_len,
        tr_frames=tr_frames,
        frame_size=frame_size,
        epochs=epochs,
        lr=lr,
    )

    generated: dict[str, Path] = {}
    for phase_name in ["v0", "v1", "v2"]:
        updates = {
            **base_updates,
            "name": f"{phase_name}_algonauts",
            "phase": phase_name,
            "output_dir": str((work_dir / "suite" / f"runs_{phase_name}_algonauts").resolve()),
        }
        target = config_dir / f"{phase_name}_algonauts.yaml"
        generated[phase_name] = _write_config(_phase_template(phase_name), target, updates)

    ablations = {
        "v2_algonauts_fast_latent": {
            "name": "v2_algonauts_fast_latent",
            "phase": "v2",
            "output_dir": str((work_dir / "suite" / "runs_v2_algonauts_fast_latent").resolve()),
            "ablation": {
                "brain_on_slow_state": False,
                "use_temporal_alignment": True,
                "include_pred_loss": True,
                "freeze_backbone": False,
                "unfreeze_top_encoder_blocks": 1,
                "use_subject_embedding": True,
            },
        },
        "v2_algonauts_no_temporal": {
            "name": "v2_algonauts_no_temporal",
            "phase": "v2",
            "output_dir": str((work_dir / "suite" / "runs_v2_algonauts_no_temporal").resolve()),
            "ablation": {
                "brain_on_slow_state": True,
                "use_temporal_alignment": False,
                "include_pred_loss": True,
                "freeze_backbone": False,
                "unfreeze_top_encoder_blocks": 1,
                "use_subject_embedding": True,
            },
        },
        "v2_algonauts_no_pred": {
            "name": "v2_algonauts_no_pred",
            "phase": "v2",
            "output_dir": str((work_dir / "suite" / "runs_v2_algonauts_no_pred").resolve()),
            "ablation": {
                "brain_on_slow_state": True,
                "use_temporal_alignment": True,
                "include_pred_loss": False,
                "freeze_backbone": False,
                "unfreeze_top_encoder_blocks": 1,
                "use_subject_embedding": True,
            },
        },
    }

    for name, updates in ablations.items():
        generated[name] = _write_config(_phase_template("v2"), shortlist_dir / f"{name}.yaml", {**base_updates, **updates})

    return generated


def preflight_movie(
    dataset_root: Path,
    movie: str,
    work_dir: Path,
    subject: str = "sub-01",
    task: str = "movie10",
) -> dict[str, Any]:
    root = _resolve_dataset_root(dataset_root)
    h5py = _require_h5py()
    fmri_path = _resolve_fmri_path(root, subject, task)
    if not fmri_path.exists():
        raise FileNotFoundError(f"Missing fMRI file: {fmri_path}")

    with h5py.File(fmri_path, "r") as handle:
        keys = list(handle.keys())
    matching_keys = [key for key in keys if movie in key]
    if not matching_keys:
        raise RuntimeError(f"No HDF5 keys found for movie '{movie}' in {fmri_path}")

    movie_dir = root / "stimuli" / "movies" / task / movie
    if not movie_dir.exists():
        raise FileNotFoundError(f"Missing movie directory: {movie_dir}")

    clips = sorted(movie_dir.glob("*.mkv"))
    if not clips:
        raise RuntimeError(f"No MKV clips found in {movie_dir}")

    work_dir.mkdir(parents=True, exist_ok=True)
    probe = work_dir / ".write_probe"
    probe.write_text("ok", encoding="utf-8")
    probe.unlink()

    return {
        "dataset_root": str(root),
        "fmri_path": str(fmri_path),
        "matching_key_count": len(matching_keys),
        "sample_keys": matching_keys[:8],
        "movie_dir": str(movie_dir),
        "clip_count": len(clips),
        "work_dir": str(work_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Preflight and config helpers for staged Algonauts Hyak runs.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight", help="Validate that a movie slice can be queued safely")
    preflight.add_argument("--dataset-root", required=True)
    preflight.add_argument("--movie", required=True)
    preflight.add_argument("--work-dir", required=True)
    preflight.add_argument("--subject", default="sub-01")
    preflight.add_argument("--task", default="movie10")

    write_configs = subparsers.add_parser("write-configs", help="Generate per-movie phase and ablation configs")
    write_configs.add_argument("--work-dir", required=True)
    write_configs.add_argument("--clip-len", type=int, default=24)
    write_configs.add_argument("--tr-frames", type=int, default=4)
    write_configs.add_argument("--frame-size", type=int, default=112)
    write_configs.add_argument("--epochs", type=int, default=20)
    write_configs.add_argument("--lr", type=float, default=1e-4)

    args = parser.parse_args()
    if args.command == "preflight":
        result = preflight_movie(
            dataset_root=Path(args.dataset_root),
            movie=args.movie,
            work_dir=Path(args.work_dir),
            subject=args.subject,
            task=args.task,
        )
        print(json.dumps(result, indent=2))
    else:
        generated = generate_movie_configs(
            work_dir=Path(args.work_dir),
            clip_len=args.clip_len,
            tr_frames=args.tr_frames,
            frame_size=args.frame_size,
            epochs=args.epochs,
            lr=args.lr,
        )
        print(json.dumps({name: str(path) for name, path in generated.items()}, indent=2))


if __name__ == "__main__":
    main()
