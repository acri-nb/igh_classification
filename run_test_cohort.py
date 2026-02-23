#!/usr/bin/env python3
"""
Batch testing of all available transformer models.
Results are saved following the same structure as individual unit tests.
"""

import argparse
import csv
import glob
import json
import os
import sys
from datetime import datetime
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch testing for IGH transformer models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--base-dir",
        required=True,
        help="Directory containing the trained model results (e.g. /path/to/deep_pipeline)",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to the test CSV file (e.g. df_patient_test.csv)",
    )
    parser.add_argument(
        "--output-root",
        default="./test_results",
        help="Root directory for test output results",
    )
    parser.add_argument(
        "--pattern",
        default="results_transformer_real*",
        help="Glob pattern for model directories to test",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="GPU ID to use (via CUDA_VISIBLE_DEVICES)",
    )
    return parser.parse_args()


def discover_model_dirs(base_dir: str, pattern: str) -> List[str]:
    return sorted(glob.glob(os.path.join(base_dir, pattern)))


def model_name_from_dir(model_dir: str) -> str:
    base = os.path.basename(os.path.normpath(model_dir))
    return base.replace("results_", "")


def build_model_path(model_dir: str) -> str:
    return os.path.join(model_dir, "checkpoints", "best_model.pt")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_summary(output_root: str, rows: List[Dict]) -> None:
    if not rows:
        return

    ensure_dir(output_root)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_csv = os.path.join(output_root, f"transformer_summary_{timestamp}.csv")
    summary_json = os.path.join(output_root, f"transformer_summary_{timestamp}.json")

    fixed_keys = ["model_name", "model_path", "output_dir"]
    metric_keys = sorted({k for row in rows for k in row.keys() if k not in fixed_keys})
    fieldnames = fixed_keys + metric_keys

    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    print(f"Summary CSV: {summary_csv}")
    print(f"Summary JSON: {summary_json}")


def main() -> int:
    args = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    print(f"GPU visible (CUDA_VISIBLE_DEVICES): {os.environ['CUDA_VISIBLE_DEVICES']}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    try:
        from test_deep_bio_classifier import ModelTester
    except ImportError as exc:
        print(f"Import error: {exc}")
        return 1

    model_dirs = discover_model_dirs(args.base_dir, args.pattern)
    if not model_dirs:
        print("No transformer model directories found.")
        return 1

    summary_rows = []

    for model_dir in model_dirs:
        model_path = build_model_path(model_dir)
        if not os.path.exists(model_path):
            print(f"Checkpoint not found, skipping: {model_path}")
            continue

        model_name = model_name_from_dir(model_dir)
        output_dir = os.path.join(args.output_root, model_name)

        print("=" * 80)
        print(f"Testing model: {model_name}")
        print(f"Checkpoint: {model_path}")
        print(f"Output: {output_dir}")

        tester = ModelTester(
            model_path=model_path,
            dataset_path=args.dataset,
            cohort_name=model_name,
            output_dir=output_dir,
            batch_size=args.batch_size,
        )

        metrics = tester.run()
        row = {
            "model_name": model_name,
            "model_path": model_path,
            "output_dir": output_dir,
        }
        row.update(metrics)
        summary_rows.append(row)

    save_summary(args.output_root, summary_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
