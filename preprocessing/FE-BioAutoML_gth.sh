#!/bin/bash
# Example: feature extraction for a single patient (TP and TN samples).
# Adapt the paths below to match your data and project layout.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/FE-BioAutoMl_gth.py"

PATIENT="CLL-001-BCells"
TP_FASTA="/path/to/sampled_reads/TP/${PATIENT}_DNA.igh.fasta"
TN_FASTA="/path/to/sampled_reads/TN/${PATIENT}_DNA.nonigh.fasta"
OUTPUT_BASE="/path/to/output/${PATIENT}"

python "${PYTHON_SCRIPT}" \
    --fasta_to_extract "${TP_FASTA}" \
    --fasta_label TP \
    --output "${OUTPUT_BASE}/TP"

python "${PYTHON_SCRIPT}" \
    --fasta_to_extract "${TN_FASTA}" \
    --fasta_label TN \
    --output "${OUTPUT_BASE}/TN"
