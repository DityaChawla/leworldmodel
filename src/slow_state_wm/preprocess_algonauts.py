from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ALGONAUTS_SUBJECTS = ["sub-01", "sub-02", "sub-03", "sub-05"]
SUBJECT_TO_ID = {subject: idx for idx, subject in enumerate(ALGONAUTS_SUBJECTS)}
TR_SECONDS = 1.49
HRF_DELAY_SECONDS = 4.47


def _require_h5py():
    try:
        import h5py  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("h5py is required for Algonauts preprocessing") from exc
    return h5py


def _read_video_with_torchvision(video_path: str, start_sec: float, end_sec: float) -> torch.Tensor:
    try:
        from torchvision.io import read_video  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("torchvision video decoding is unavailable") from exc
    frames, _, _ = read_video(video_path, start_pts=start_sec, end_pts=end_sec, pts_unit="sec")
    return frames


def _read_video_with_pyav(video_path: str, start_sec: float, end_sec: float) -> torch.Tensor:
    try:
        import av  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyAV video decoding is unavailable") from exc

    container = av.open(video_path)
    try:
        video_stream = container.streams.video[0]
        frames = []
        for frame in container.decode(video=0):
            if frame.time is None:
                continue
            if frame.time < start_sec:
                continue
            if frame.time > end_sec:
                break
            frames.append(torch.from_numpy(frame.to_ndarray(format="rgb24")))
    finally:
        container.close()

    if not frames:
        return torch.empty(0, dtype=torch.uint8)
    return torch.stack(frames, dim=0)


def _stable_bucket(name: str) -> float:
    digest = hashlib.md5(name.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def _stimulus_id(task: str, movie: str, chunk: str, run: int) -> str:
    if task == "friends":
        return f"friends-s{int(movie):02d}-{chunk}"
    suffix = f"-run{run}" if run else ""
    return f"movie10-{movie}-{int(chunk):02d}{suffix}"


def _resolve_dataset_root(dataset_root: Path) -> Path:
    candidates = [
        dataset_root / "download" / "algonauts_2025.competitors",
        dataset_root / "algonauts_2025.competitors",
        dataset_root,
    ]
    for candidate in candidates:
        if (candidate / "fmri").exists() and (candidate / "stimuli").exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find an Algonauts dataset under {dataset_root}. "
        "Expected either fmri/ and stimuli/ directly, "
        "or download/algonauts_2025.competitors/."
    )


def _resolve_movie_path(root: Path, task: str, movie: str, chunk: str, run: int) -> Path:
    if task == "friends":
        return root / "stimuli" / "movies" / task / f"s{movie}" / f"friends_s{int(movie):02d}{chunk}.mkv"
    return root / "stimuli" / "movies" / task / movie / f"{movie}{int(chunk):02d}.mkv"


def _resolve_fmri_path(root: Path, subject: str, task: str) -> Path:
    subj_dir = root / "fmri" / subject / "func"
    stem = f"{subject}_task-{task}_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net"
    if task == "friends":
        return subj_dir / f"{stem}_desc-s123456_bold.h5"
    return subj_dir / f"{stem}_bold.h5"


def _resolve_fmri_key(task: str, movie: str, chunk: str, run: int, keys: list[str]) -> str:
    if task == "friends":
        needle = f"{int(movie):02d}{chunk}"
    else:
        needle = f"{movie}{int(chunk):02d}"
        if movie in {"life", "figures"}:
            needle += f"_run-{run}"
    matches = [key for key in keys if needle in key]
    if len(matches) != 1:
        raise ValueError(f"Could not resolve unique fMRI key for {task=} {movie=} {chunk=} {run=}; matches={matches}")
    return matches[0]


def _iter_friends() -> list[tuple[str, str, int]]:
    items = []
    skipped = {
        ("5", "e20a"),
        ("4", "e01a"),
        ("6", "e03a"),
        ("4", "e13b"),
        ("4", "e01b"),
    }
    for season in range(1, 8):
        for episode in range(1, 26):
            for suffix in "abcd":
                chunk = f"e{episode:02d}{suffix}"
                if (str(season), chunk) in skipped:
                    continue
                items.append((str(season), chunk, 0))
    return items


def _iter_movie10() -> list[tuple[str, str, int]]:
    items = []
    for movie in ["bourne", "wolf", "life", "figures"]:
        for chunk in range(1, 18):
            for run in [1, 2]:
                if movie in {"bourne", "wolf"} and run == 2:
                    continue
                items.append((movie, str(chunk), run))
    return items


def _parse_csv_list(value: str | None) -> set[str] | None:
    if value is None:
        return None
    items = {item.strip() for item in value.split(",") if item.strip()}
    return items or None


def build_source_index(
    dataset_root: Path,
    output_jsonl: Path,
    val_ratio: float = 0.1,
    include_test: bool = False,
    subjects: set[str] | None = None,
    tasks: set[str] | None = None,
    movies: set[str] | None = None,
) -> int:
    root = _resolve_dataset_root(dataset_root)
    h5py = _require_h5py()
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    for subject in ALGONAUTS_SUBJECTS:
        if subjects is not None and subject not in subjects:
            continue
        for task, iterator in [("friends", _iter_friends()), ("movie10", _iter_movie10())]:
            if tasks is not None and task not in tasks:
                continue
            fmri_path = _resolve_fmri_path(root, subject, task)
            if task == "friends" and not fmri_path.exists():
                continue
            available_keys: list[str] = []
            if fmri_path.exists():
                with h5py.File(fmri_path, "r") as handle:
                    available_keys = list(handle.keys())

            for movie, chunk, run in iterator:
                if movies is not None and movie not in movies:
                    continue
                video_path = _resolve_movie_path(root, task, movie, chunk, run)
                if not video_path.exists():
                    continue
                stimulus_id = _stimulus_id(task, movie, chunk, run)
                split = "test" if task == "friends" and movie == "7" else "train"
                if split == "test" and not include_test:
                    continue
                experiment_split = split
                if split == "train":
                    experiment_split = "val" if _stable_bucket(stimulus_id) < val_ratio else "train"

                fmri_key = None
                if split != "test":
                    fmri_key = _resolve_fmri_key(task, movie, chunk, run, available_keys)

                rows.append(
                    {
                        "subject": subject,
                        "subject_id": SUBJECT_TO_ID[subject],
                        "task": task,
                        "movie": movie,
                        "chunk": chunk,
                        "run": run,
                        "stimulus_id": stimulus_id,
                        "split": experiment_split,
                        "source_split": split,
                        "video_path": str(video_path),
                        "fmri_path": str(fmri_path) if fmri_key is not None else None,
                        "fmri_key": fmri_key,
                    }
                )

    with output_jsonl.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return len(rows)


def _load_fmri_matrix(record: dict[str, Any], zscore: bool) -> torch.Tensor:
    h5py = _require_h5py()
    if record["fmri_path"] is None or record["fmri_key"] is None:
        raise ValueError("This record does not have an fMRI target")
    with h5py.File(record["fmri_path"], "r") as handle:
        matrix = torch.from_numpy(handle[record["fmri_key"]][:]).float()
    if matrix.ndim != 2:
        raise ValueError(f"Expected raw fMRI matrix to be [T_tr, parcels], got {matrix.shape}")
    if zscore:
        matrix = matrix - matrix.mean(dim=0, keepdim=True)
        matrix = matrix / (matrix.std(dim=0, keepdim=True) + 1e-6)
    return matrix


def _sample_video_clip(video_path: str, start_sec: float, end_sec: float, clip_len: int, frame_size: int) -> torch.Tensor:
    last_error: Exception | None = None
    frames = None
    for reader in (_read_video_with_torchvision, _read_video_with_pyav):
        try:
            frames = reader(video_path, start_sec, end_sec)
            break
        except Exception as exc:
            last_error = exc
    if frames is None:
        raise RuntimeError(f"Could not decode video clip from {video_path}") from last_error
    if frames.numel() == 0:
        raise ValueError(f"No frames decoded for {video_path} between {start_sec:.2f}s and {end_sec:.2f}s")
    frames = frames.float() / 255.0
    index = torch.linspace(0, frames.size(0) - 1, steps=clip_len).round().long()
    frames = frames[index]
    frames = frames.permute(0, 3, 1, 2)  # [T, C, H, W]
    frames = F.interpolate(frames, size=(frame_size, frame_size), mode="bilinear", align_corners=False)
    return frames.contiguous()


@dataclass
class WindowConfig:
    clip_len: int
    tr_frames: int
    frame_size: int
    window_stride_tr: int
    hrf_delay_seconds: float = HRF_DELAY_SECONDS
    tr_seconds: float = TR_SECONDS
    zscore_fmri: bool = True
    max_windows_per_split: int | None = None


def build_windowed_manifest(index_jsonl: Path, output_dir: Path, cfg: WindowConfig) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tensors_dir = output_dir / "tensors"
    tensors_dir.mkdir(parents=True, exist_ok=True)
    split_handles: dict[str, Any] = {}
    counts: dict[str, int] = {}
    decode_failures = 0
    negative_start_skips = 0
    tr_window = cfg.clip_len // cfg.tr_frames
    if cfg.clip_len % cfg.tr_frames != 0:
        raise ValueError("clip_len must be divisible by tr_frames")

    def manifest_handle(split: str):
        if split not in split_handles:
            split_path = output_dir / f"{split}_manifest.jsonl"
            split_handles[split] = split_path.open("w", encoding="utf-8")
        return split_handles[split]

    with index_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            split = record["split"]
            if record["fmri_path"] is None:
                continue
            fmri = _load_fmri_matrix(record, zscore=cfg.zscore_fmri)
            max_start = fmri.size(0) - tr_window + 1
            for fmri_start in range(0, max_start, cfg.window_stride_tr):
                if cfg.max_windows_per_split is not None and counts.get(split, 0) >= cfg.max_windows_per_split:
                    break
                start_sec = fmri_start * cfg.tr_seconds - cfg.hrf_delay_seconds
                if start_sec < 0:
                    negative_start_skips += 1
                    continue
                end_sec = start_sec + tr_window * cfg.tr_seconds
                try:
                    frames = _sample_video_clip(
                        video_path=record["video_path"],
                        start_sec=start_sec,
                        end_sec=end_sec,
                        clip_len=cfg.clip_len,
                        frame_size=cfg.frame_size,
                    )
                except Exception:
                    decode_failures += 1
                    continue

                fmri_window = fmri[fmri_start : fmri_start + tr_window]
                sample_id = (
                    f"{record['stimulus_id']}__{record['subject']}__tr{fmri_start:04d}"
                )
                frame_path = tensors_dir / f"{sample_id}_frames.pt"
                fmri_path = tensors_dir / f"{sample_id}_fmri.pt"
                torch.save(frames, frame_path)
                torch.save(fmri_window, fmri_path)

                manifest_row = {
                    "frames_path": str(frame_path.resolve()),
                    "fmri_path": str(fmri_path.resolve()),
                    "subject_id": record["subject_id"],
                    "sample_id": sample_id,
                    "split": split,
                    "stimulus_id": record["stimulus_id"],
                    "subject": record["subject"],
                    "task": record["task"],
                    "movie": record["movie"],
                    "chunk": record["chunk"],
                    "run": record["run"],
                    "fmri_start_tr": fmri_start,
                    "hrf_delay_seconds": cfg.hrf_delay_seconds,
                    "tr_seconds": cfg.tr_seconds,
                }
                manifest_handle(split).write(json.dumps(manifest_row) + "\n")
                counts[split] = counts.get(split, 0) + 1

    for handle in split_handles.values():
        handle.close()
    if not counts:
        raise RuntimeError(
            "No windowed samples were created. "
            f"negative_start_skips={negative_start_skips}, decode_failures={decode_failures}. "
            "This usually means the video decoder backend is missing or the HRF/window settings eliminate all windows."
        )
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Algonauts source indices and windowed manifests.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("scan", help="Scan the official Algonauts folder into a source index")
    index_parser.add_argument(
        "--dataset-root",
        required=True,
        help="Path containing the Algonauts dataset, either directly or under download/algonauts_2025.competitors",
    )
    index_parser.add_argument("--output-jsonl", required=True, help="Where to write the source index")
    index_parser.add_argument("--val-ratio", type=float, default=0.1)
    index_parser.add_argument("--include-test", action="store_true")
    index_parser.add_argument("--subjects", default=None, help="Comma-separated subject filter, e.g. sub-01,sub-02")
    index_parser.add_argument("--tasks", default=None, help="Comma-separated task filter, e.g. movie10,friends")
    index_parser.add_argument("--movies", default=None, help="Comma-separated movie filter within a task, e.g. life,wolf or 1,2")

    build_parser = subparsers.add_parser("window", help="Convert the source index into windowed manifests")
    build_parser.add_argument("--index-jsonl", required=True)
    build_parser.add_argument("--output-dir", required=True)
    build_parser.add_argument("--clip-len", type=int, required=True)
    build_parser.add_argument("--tr-frames", type=int, required=True)
    build_parser.add_argument("--frame-size", type=int, default=224)
    build_parser.add_argument("--window-stride-tr", type=int, default=1)
    build_parser.add_argument("--hrf-delay-seconds", type=float, default=HRF_DELAY_SECONDS)
    build_parser.add_argument("--tr-seconds", type=float, default=TR_SECONDS)
    build_parser.add_argument("--no-zscore-fmri", action="store_true")
    build_parser.add_argument("--max-windows-per-split", type=int, default=None)

    args = parser.parse_args()
    if args.command == "scan":
        count = build_source_index(
            dataset_root=Path(args.dataset_root),
            output_jsonl=Path(args.output_jsonl),
            val_ratio=args.val_ratio,
            include_test=args.include_test,
            subjects=_parse_csv_list(args.subjects),
            tasks=_parse_csv_list(args.tasks),
            movies=_parse_csv_list(args.movies),
        )
        print(f"Wrote {count} source rows to {args.output_jsonl}")
    else:
        counts = build_windowed_manifest(
            index_jsonl=Path(args.index_jsonl),
            output_dir=Path(args.output_dir),
            cfg=WindowConfig(
                clip_len=args.clip_len,
                tr_frames=args.tr_frames,
                frame_size=args.frame_size,
                window_stride_tr=args.window_stride_tr,
                hrf_delay_seconds=args.hrf_delay_seconds,
                tr_seconds=args.tr_seconds,
                zscore_fmri=not args.no_zscore_fmri,
                max_windows_per_split=args.max_windows_per_split,
            ),
        )
        print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
