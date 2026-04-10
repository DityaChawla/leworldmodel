from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .prepare_algonauts import generate_movie_configs
from .preprocess_algonauts import WindowConfig, build_source_index, build_windowed_manifest
from .trainer import run_experiment


def run_synthetic_suite(output_root: Path, max_train_steps: int | None, max_val_steps: int | None) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    for config_name in ["v0_synthetic.yaml", "v1_synthetic.yaml", "v2_synthetic.yaml"]:
        cfg = load_config(Path("configs") / config_name)
        cfg.output_dir = str((output_root / Path(cfg.output_dir).name).resolve())
        run_experiment(cfg, max_train_steps=max_train_steps, max_val_steps=max_val_steps)


def run_algonauts_suite(
    dataset_root: Path,
    work_dir: Path,
    clip_len: int,
    tr_frames: int,
    frame_size: int,
    window_stride_tr: int,
    subjects: set[str] | None,
    tasks: set[str] | None,
    movies: set[str] | None,
    max_windows_per_split: int | None,
    max_train_steps: int | None,
    max_val_steps: int | None,
) -> None:
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    work_dir.mkdir(parents=True, exist_ok=True)
    index_path = work_dir / "algonauts_source_index.jsonl"
    manifest_dir = work_dir / "algonauts_windowed"

    count = build_source_index(
        dataset_root=dataset_root,
        output_jsonl=index_path,
        subjects=subjects,
        tasks=tasks,
        movies=movies,
    )
    print(f"Indexed {count} source rows")

    counts = build_windowed_manifest(
        index_jsonl=index_path,
        output_dir=manifest_dir,
        cfg=WindowConfig(
            clip_len=clip_len,
            tr_frames=tr_frames,
            frame_size=frame_size,
            window_stride_tr=window_stride_tr,
            max_windows_per_split=max_windows_per_split,
        ),
    )
    print("Window counts:", json.dumps(counts, indent=2))

    generated = generate_movie_configs(
        work_dir=work_dir.parent if work_dir.name == "suite" else work_dir,
        clip_len=clip_len,
        tr_frames=tr_frames,
        frame_size=frame_size,
        epochs=20,
        lr=1e-4,
    )

    for phase_name in ["v0", "v1", "v2"]:
        cfg = load_config(generated[phase_name])
        run_experiment(cfg, max_train_steps=max_train_steps, max_val_steps=max_val_steps)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run synthetic or Algonauts experiment suites end to end.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    synthetic = subparsers.add_parser("synthetic", help="Run V0/V1/V2 on the synthetic benchmark")
    synthetic.add_argument("--output-root", default="runs/suite_synthetic", help="Where to store the suite outputs")
    synthetic.add_argument("--max-train-steps", type=int, default=None)
    synthetic.add_argument("--max-val-steps", type=int, default=None)

    algonauts = subparsers.add_parser("algonauts", help="Run the full Algonauts pipeline from indexing to V2")
    algonauts.add_argument("--dataset-root", required=True)
    algonauts.add_argument("--work-dir", default="runs/algonauts_suite")
    algonauts.add_argument("--clip-len", type=int, default=24)
    algonauts.add_argument("--tr-frames", type=int, default=4)
    algonauts.add_argument("--frame-size", type=int, default=224)
    algonauts.add_argument("--window-stride-tr", type=int, default=1)
    algonauts.add_argument("--subjects", default=None, help="Comma-separated subject filter, e.g. sub-01,sub-02")
    algonauts.add_argument("--tasks", default=None, help="Comma-separated task filter, e.g. movie10,friends")
    algonauts.add_argument("--movies", default=None, help="Comma-separated movie filter within a task, e.g. life,wolf or 1,2")
    algonauts.add_argument("--max-windows-per-split", type=int, default=None)
    algonauts.add_argument("--max-train-steps", type=int, default=None)
    algonauts.add_argument("--max-val-steps", type=int, default=None)

    args = parser.parse_args()
    if args.command == "synthetic":
        run_synthetic_suite(
            output_root=Path(args.output_root),
            max_train_steps=args.max_train_steps,
            max_val_steps=args.max_val_steps,
        )
    else:
        run_algonauts_suite(
            dataset_root=Path(args.dataset_root),
            work_dir=Path(args.work_dir),
            clip_len=args.clip_len,
            tr_frames=args.tr_frames,
            frame_size=args.frame_size,
            window_stride_tr=args.window_stride_tr,
            subjects=set(args.subjects.split(",")) if args.subjects else None,
            tasks=set(args.tasks.split(",")) if args.tasks else None,
            movies=set(args.movies.split(",")) if args.movies else None,
            max_windows_per_split=args.max_windows_per_split,
            max_train_steps=args.max_train_steps,
            max_val_steps=args.max_val_steps,
        )


if __name__ == "__main__":
    main()
