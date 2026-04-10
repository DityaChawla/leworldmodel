from __future__ import annotations

import argparse
import json
from pathlib import Path


def _best_epoch(metrics: dict[str, dict[str, float]], metric_name: str) -> tuple[str, dict[str, float]]:
    reverse = "corr" in metric_name.lower()
    epoch_name, values = sorted(
        metrics.items(),
        key=lambda item: item[1].get(metric_name, float("-inf") if reverse else float("inf")),
        reverse=reverse,
    )[0]
    return epoch_name, values


def summarize_runs(runs_dir: Path, metric_name: str) -> list[dict[str, object]]:
    rows = []
    for metrics_path in sorted(runs_dir.glob("*/metrics.json")):
        with metrics_path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        epoch_name, values = _best_epoch(metrics, metric_name)
        rows.append(
            {
                "run": metrics_path.parent.name,
                "best_epoch": epoch_name,
                metric_name: values.get(metric_name),
                "val_loss": values.get("val/loss"),
                "val_corr": values.get("val/parcel_corr"),
                "ood_corr": values.get("ood/parcel_corr"),
                "train_loss": values.get("train/loss"),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize experiment runs by best validation metric.")
    parser.add_argument("--runs-dir", default="runs", help="Directory containing run subfolders")
    parser.add_argument("--metric", default="val/parcel_corr", help="Metric used to choose the best epoch")
    args = parser.parse_args()

    rows = summarize_runs(Path(args.runs_dir), args.metric)
    if not rows:
        print("No runs found.")
        return

    header = ["run", "best_epoch", args.metric, "val_loss", "val_corr", "ood_corr", "train_loss"]
    print("\t".join(header))
    for row in rows:
        print("\t".join("" if row.get(col) is None else f"{row[col]}" for col in header))


if __name__ == "__main__":
    main()
