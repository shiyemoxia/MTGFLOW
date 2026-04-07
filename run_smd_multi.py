#!/usr/bin/env python
"""Batch runner for MTGFLOW on SMD machines."""

from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List


def parse_machine_list(data_dir: Path, value: str) -> List[str]:
    raw = (value or "").strip()
    if raw and raw.lower() != "all":
        return [item.strip() for item in raw.split(",") if item.strip()]
    return sorted(
        path.name
        for path in data_dir.iterdir()
        if path.is_dir() and path.name.startswith("machine-")
    )


def summarize_metric(values: Iterable[float]) -> str:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return "no valid values"
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    return (
        f"mean={mean:.4f} "
        f"std={var ** 0.5:.4f} "
        f"min={min(vals):.4f} "
        f"max={max(vals):.4f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MTGFLOW across SMD machines.")
    parser.add_argument("--data-dir", required=True, type=str)
    parser.add_argument("--machines", default="all", type=str)
    parser.add_argument("--output-dir", default="checkpoint/prl_mtgflow_smd_final", type=str)
    parser.add_argument("--results-csv", default="checkpoint/prl_mtgflow_smd_final/smd_seed_results.csv", type=str)
    parser.add_argument("--seed-start", default=42, type=int)
    parser.add_argument("--seed-end", default=42, type=int)
    parser.add_argument("--setting", default="unsupervised", choices=["unsupervised", "occ"])
    parser.add_argument("--window-size", default=100, type=int)
    parser.add_argument("--stride-size", default=10, type=int)
    parser.add_argument("--train-split", default=0.8, type=float)
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("--epochs", default=40, type=int)
    parser.add_argument("--lr", default=2e-3, type=float)
    parser.add_argument("--weight-decay", default=5e-4, type=float)
    parser.add_argument("--n-blocks", default=1, type=int)
    parser.add_argument("--hidden-size", default=32, type=int)
    parser.add_argument("--n-hidden", default=1, type=int)
    return parser.parse_args()


def run_one_machine(args: argparse.Namespace, root: Path, machine: str) -> dict:
    machine_output_dir = Path(args.output_dir).resolve()
    log_dir = machine_output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{machine}.log"
    cmd = [
        sys.executable,
        "main.py",
        "--data_dir",
        str(Path(args.data_dir).resolve()),
        "--output_dir",
        str(machine_output_dir),
        "--name",
        machine,
        "--window_size",
        str(args.window_size),
        "--stride_size",
        str(args.stride_size),
        "--train_split",
        str(args.train_split),
        "--batch_size",
        str(args.batch_size),
        "--epochs",
        str(args.epochs),
        "--lr",
        str(args.lr),
        "--weight_decay",
        str(args.weight_decay),
        "--n_blocks",
        str(args.n_blocks),
        "--hidden_size",
        str(args.hidden_size),
        "--n_hidden",
        str(args.n_hidden),
        "--seed_start",
        str(args.seed_start),
        "--seed_end",
        str(args.seed_end),
        "--setting",
        args.setting,
    ]
    proc = subprocess.run(
        cmd,
        cwd=root,
        capture_output=True,
        text=True,
    )
    log_path.write_text(proc.stdout + "\n" + proc.stderr, encoding="utf-8")

    seed_results = machine_output_dir / machine / "seed_results.csv"
    if proc.returncode != 0:
        combined = proc.stdout + "\n" + proc.stderr
        status = "invalid_test_split" if "Only one class present" in combined else f"error({proc.returncode})"
        return {
            "machine": machine,
            "seed": "",
            "best_roc_auc": "",
            "checkpoint_path": "",
            "status": status,
            "log_path": str(log_path),
        }

    if not seed_results.exists():
        return {
            "machine": machine,
            "seed": "",
            "best_roc_auc": "",
            "checkpoint_path": "",
            "status": "missing_seed_results",
            "log_path": str(log_path),
        }

    rows = []
    with seed_results.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "machine": machine,
                    "seed": row.get("seed", ""),
                    "best_roc_auc": row.get("best_roc_auc", ""),
                    "checkpoint_path": row.get("checkpoint_path", ""),
                    "status": "ok",
                    "log_path": str(log_path),
                }
            )
    return {"rows": rows}


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results_csv = Path(args.results_csv).resolve()
    results_csv.parent.mkdir(parents=True, exist_ok=True)

    machines = parse_machine_list(data_dir, args.machines)
    if not machines:
        raise SystemExit(f"No SMD machines found under {data_dir}")

    print("=" * 60)
    print("MTGFLOW SMD Multi-Machine Runner")
    print("=" * 60)
    print(f"Machines: {len(machines)}")
    print(f"Seeds: {args.seed_start}..{args.seed_end}")
    print(f"Window size: {args.window_size}")
    print(f"Train split: {args.train_split}")
    print(f"Batch size: {args.batch_size}")
    print(f"Output dir: {output_dir}")

    all_rows = []
    for idx, machine in enumerate(machines, start=1):
        print(f"\n[{idx}/{len(machines)}] {machine}")
        result = run_one_machine(args, root, machine)
        if "rows" in result:
            all_rows.extend(result["rows"])
            best_auc = max(float(row["best_roc_auc"]) for row in result["rows"])
            print(f"  best_roc_auc={best_auc:.4f}")
        else:
            all_rows.append(result)
            print(f"  {result['status']}")

    with results_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["machine", "seed", "best_roc_auc", "checkpoint_path", "status", "log_path"],
        )
        writer.writeheader()
        writer.writerows(all_rows)

    ok_rows = [row for row in all_rows if row["status"] == "ok" and row["best_roc_auc"] not in {"", None}]
    invalid_rows = [row for row in all_rows if row["status"] == "invalid_test_split"]

    print("\n" + "=" * 60)
    print("AGGREGATE RESULTS")
    print("=" * 60)
    if ok_rows:
        print(summarize_metric(float(row["best_roc_auc"]) for row in ok_rows))
        print(f"Completed: {len(ok_rows)}/{len(all_rows)}")
        best_row = max(ok_rows, key=lambda row: float(row["best_roc_auc"]))
        worst_row = min(ok_rows, key=lambda row: float(row["best_roc_auc"]))
        print(
            f"Best machine: {best_row['machine']} seed={best_row['seed']} "
            f"auc={float(best_row['best_roc_auc']):.4f}"
        )
        print(
            f"Worst machine: {worst_row['machine']} seed={worst_row['seed']} "
            f"auc={float(worst_row['best_roc_auc']):.4f}"
        )
    else:
        print("No successful runs.")

    if invalid_rows:
        invalid_names = ", ".join(sorted({row["machine"] for row in invalid_rows}))
        print(f"Invalid test splits ({len(sorted({row['machine'] for row in invalid_rows}))}): {invalid_names}")
    print(f"Results CSV saved to: {results_csv}")


if __name__ == "__main__":
    main()
