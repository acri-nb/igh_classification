#!/bin/bash
# Batch feature extraction for all patients in TP and TN directories.
# Automatically iterates over all FASTA files and runs feature extraction.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/FE-BioAutoMl_gth.py"

# --- Configure these paths ---
TP_DIR="/path/to/sampled_reads/TP"
TN_DIR="/path/to/sampled_reads/TN"
OUTPUT_BASE="/path/to/output/all_patients"
# -----------------------------

echo "###################################################################################"
echo "##########         BioAutoML - Batch Feature Extraction                  ###########"
echo "##########              Processing all patients in TP and TN             ###########"
echo "###################################################################################"
echo ""

if [ ! -d "$TP_DIR" ]; then
    echo "Error: TP directory not found: $TP_DIR"
    exit 1
fi

if [ ! -d "$TN_DIR" ]; then
    echo "Error: TN directory not found: $TN_DIR"
    exit 1
fi

mkdir -p "$OUTPUT_BASE"

echo "Source directories verified:"
echo "  TP: $TP_DIR"
echo "  TN: $TN_DIR"
echo "  Output: $OUTPUT_BASE"
echo ""

# Extract patient name (part before _DNA)
extract_patient_name() {
    local filename
    filename=$(basename "$1")
    echo "$filename" | sed 's/_DNA.*//'
}

# Process a single FASTA file
process_fasta() {
    local fasta_file="$1"
    local label="$2"
    local patient_name
    patient_name=$(extract_patient_name "$fasta_file")
    local output_dir="$OUTPUT_BASE/$patient_name/$label"

    echo "Processing: $fasta_file"
    echo "  Patient: $patient_name"
    echo "  Label: $label"
    echo "  Output: $output_dir"

    mkdir -p "$output_dir"

    python "${PYTHON_SCRIPT}" \
        --fasta_to_extract "$fasta_file" \
        --fasta_label "$label" \
        --output "$output_dir"

    if [ $? -eq 0 ]; then
        echo "  Done"
    else
        echo "  ERROR during processing"
    fi
    echo ""
}

echo "=== Processing TP files ==="
tp_count=0
for fasta in "$TP_DIR"/*.igh.fasta; do
    if [ -f "$fasta" ]; then
        process_fasta "$fasta" "TP"
        ((tp_count++))
    fi
done
echo "TP files processed: $tp_count"
echo ""

echo "=== Processing TN files ==="
tn_count=0
for fasta in "$TN_DIR"/*.nonigh.fasta; do
    if [ -f "$fasta" ]; then
        process_fasta "$fasta" "TN"
        ((tn_count++))
    fi
done
echo "TN files processed: $tn_count"
echo ""

echo "###################################################################################"
echo "##########         Processing complete                                   ###########"
echo "##########         Total: $((tp_count + tn_count)) files processed                ###########"
echo "###################################################################################"
