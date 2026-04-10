from __future__ import annotations

import argparse
import json
from pathlib import Path


RUN_COLUMNS = [
    ("V0", "runs_v0_algonauts"),
    ("V1", "runs_v1_algonauts"),
    ("V2", "runs_v2_algonauts"),
    ("fast_latent", "runs_v2_algonauts_fast_latent"),
    ("no_temporal", "runs_v2_algonauts_no_temporal"),
    ("no_pred", "runs_v2_algonauts_no_pred"),
]


def _best_epoch(metrics: dict[str, dict[str, float]], metric_name: str = "val/parcel_corr") -> dict[str, float]:
    return sorted(metrics.values(), key=lambda values: values.get(metric_name, float("-inf")), reverse=True)[0]


def _read_metrics(run_dir: Path) -> dict[str, float] | None:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return None
    with metrics_path.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    return _best_epoch(metrics)


def build_matrix(rows: dict[str, Path]) -> list[dict[str, str]]:
    table: list[dict[str, str]] = []
    for slice_name, root in rows.items():
        row: dict[str, str] = {"slice": slice_name}
        for label, run_name in RUN_COLUMNS:
            metrics = _read_metrics(root / run_name)
            if metrics is None:
                row[label] = ""
            else:
                row[label] = f"{metrics.get('val/parcel_corr', float('nan')):.6f}"
        table.append(row)
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a compact cross-slice comparison table.")
    parser.add_argument(
        "--row",
        action="append",
        default=[],
        help="Row definition in the form name=/absolute/path/to/suite",
    )
    args = parser.parse_args()

    rows: dict[str, Path] = {}
    for item in args.row:
        name, raw_path = item.split("=", 1)
        rows[name] = Path(raw_path)

    if not rows:
        raise SystemExit("No rows provided")

    table = build_matrix(rows)
    headers = ["slice", *[label for label, _ in RUN_COLUMNS]]
    print("\t".join(headers))
    for row in table:
        print("\t".join(row.get(header, "") for header in headers))


if __name__ == "__main__":
    main()
