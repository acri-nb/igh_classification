#!/usr/bin/env python3
"""
Run progressive training cohorts with fixed real sizes
and increasing synthetic percentages.
"""

import argparse
import json
import os
import sys
from contextlib import contextmanager
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd


class _TeeStream:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)
        return len(data)

    def flush(self):
        for s in self._streams:
            s.flush()

    def isatty(self):
        return any(getattr(s, "isatty", lambda: False)() for s in self._streams)


@contextmanager
def _tee_stdout_stderr(log_path: str):
    old_out, old_err = sys.stdout, sys.stderr
    with open(log_path, "a", encoding="utf-8") as f:
        sys.stdout = _TeeStream(old_out, f)
        sys.stderr = _TeeStream(old_err, f)
        try:
            yield
        finally:
            try:
                sys.stdout.flush()
                sys.stderr.flush()
            finally:
                sys.stdout, sys.stderr = old_out, old_err


def _parse_int_list(value: str) -> List[int]:
    parts = [p.strip() for p in value.replace(";", ",").split(",") if p.strip()]
    return [int(p) for p in parts]


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _load_dataframe(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.columns[0] == "Unnamed: 0" or str(df.columns[0]).isdigit():
        df = df.iloc[:, 1:]
    return df


def _format_real_dir(real_size: int) -> str:
    return f"real_{real_size:06d}"


def _format_synth_dir(synth_pct: int, synth_n: int) -> str:
    return f"synth_{synth_pct:03d}pct_{synth_n:06d}"


def _distribution(df: pd.DataFrame, column: str) -> dict:
    if column not in df.columns:
        return {}
    return df[column].value_counts().to_dict()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Progressive training cohorts for DeepBioClassifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to the combined training CSV file (e.g. df_combined.csv)",
    )
    parser.add_argument(
        "--output-root",
        default="./training_cohort",
        help="Root directory for training cohort outputs",
    )
    parser.add_argument(
        "--model",
        default="transformer",
        choices=["mlp", "tabnet", "transformer", "dann"],
        help="Model architecture to train",
    )
    parser.add_argument(
        "--real-sizes",
        default="50000,100000,150000,200000,213100",
        help="Comma-separated list of real data sizes to evaluate",
    )
    parser.add_argument(
        "--synthetic-pcts",
        default="10,20,30,40,50,60,70,80,90,100",
        help="Comma-separated list of synthetic data percentages to evaluate",
    )
    parser.add_argument(
        "--always-include-source",
        default="cll_dna",
        help="Value of the 'source' column to always include in each cohort",
    )
    parser.add_argument(
        "--always-include-n",
        type=int,
        default=20000,
        help="Fixed number of rows from always-include-source to include in each cohort",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Base random seed for reproducibility",
    )
    parser.add_argument(
        "--no-stratify",
        action="store_true",
        help="Disable stratification by label when sampling",
    )
    parser.add_argument(
        "--save-dataset",
        action="store_true",
        help="Save the cohort dataset CSV in each output directory",
    )
    parser.add_argument("--epochs", type=int, default=150, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--patience", type=int, default=50, help="Early stopping patience (epochs)")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--loss", type=str, default="focal", help="Loss function type")
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[512, 256, 128, 64],
                        help="Hidden layer dimensions")
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="GPU ID to use (via CUDA_VISIBLE_DEVICES)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # IMPORTANT: must be set before importing torch / deep_bio_classifier
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    print(f"GPU visible (CUDA_VISIBLE_DEVICES): {os.environ['CUDA_VISIBLE_DEVICES']}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    from deep_bio_classifier import DeepBioClassifier, build_fixed_cohort_dataset

    real_sizes = _parse_int_list(args.real_sizes)
    synth_pcts = _parse_int_list(args.synthetic_pcts)

    df = _load_dataframe(args.data)

    for real_size in real_sizes:
        real_dir = os.path.join(args.output_root, _format_real_dir(real_size))
        _ensure_dir(real_dir)

        for synth_pct in synth_pcts:
            synth_n = int(round(real_size * (synth_pct / 100.0)))
            synth_dir_name = _format_synth_dir(synth_pct, synth_n)
            output_dir = os.path.join(real_dir, synth_dir_name)
            _ensure_dir(output_dir)

            cohort_seed = args.random_state + real_size + synth_pct

            df_cohort = build_fixed_cohort_dataset(
                df=df,
                n_real_total=real_size,
                synthetic_pct=synth_pct,
                random_state=cohort_seed,
                always_include_source=args.always_include_source,
                always_include_n=args.always_include_n,
                stratify=not args.no_stratify,
            )

            manifest = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "data_path": args.data,
                "output_dir": output_dir,
                "model_type": args.model,
                "real_size": real_size,
                "synthetic_pct": synth_pct,
                "synthetic_size": synth_n,
                "total_rows": int(len(df_cohort)),
                "random_state": cohort_seed,
                "always_include_source": args.always_include_source,
                "always_include_n": args.always_include_n,
                "label_distribution": _distribution(df_cohort, "label"),
                "data_type_distribution": _distribution(df_cohort, "data_type"),
                "source_distribution": _distribution(df_cohort, "source"),
            }

            manifest_path = os.path.join(output_dir, "cohort_manifest.json")
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)

            if args.save_dataset:
                df_cohort.to_csv(os.path.join(output_dir, "cohort_dataset.csv"), index=False)

            log_path = os.path.join(output_dir, "cohort_training.log")
            with _tee_stdout_stderr(log_path):
                print("=" * 80)
                print(f"timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"output_dir: {output_dir}")
                print(f"real_size: {real_size}")
                print(f"synthetic_pct: {synth_pct}")
                print(f"synthetic_size: {synth_n}")
                print(f"random_state: {cohort_seed}")
                print(f"GPU visible (CUDA_VISIBLE_DEVICES): {os.environ.get('CUDA_VISIBLE_DEVICES', '')}")
                print(f"manifest_path: {manifest_path}")
                print("=" * 80)

                classifier = DeepBioClassifier(
                    model_type=args.model,
                    hidden_dims=args.hidden_dims,
                    dropout=args.dropout,
                    learning_rate=args.learning_rate,
                    batch_size=args.batch_size,
                    epochs=args.epochs,
                    patience=args.patience,
                    loss_type=args.loss,
                    output_dir=output_dir,
                )

                classifier.run(data_df=df_cohort, random_state=cohort_seed)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
