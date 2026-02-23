#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_ID="${1:-1}"

cd "${SCRIPT_DIR}"
python3 run_test_cohort.py --gpu-id "${GPU_ID}"
