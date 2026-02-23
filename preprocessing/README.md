# Preprocessing — IGH Feature Extraction

This directory contains a lightweight adaptation of the **BioAutoML** feature extraction pipeline for IGH read classification.

> The full BioAutoML framework, documentation, and original code are available at:
> **https://github.com/Bonidia/BioAutoML**
>
> Please refer to that repository for installation instructions, MathFeature setup, and the complete list of supported descriptors.

---

## What was modified

Three files were added on top of the original BioAutoML codebase:

| File | Description |
|------|-------------|
| `FE-BioAutoMl_gth.py` | Adapted feature extraction script. Removes the `nameseq` and `label` columns from the output, so the resulting CSV contains **only the 464 feature values** (suitable for direct ML input). Uses feature groups 1–10 (NAC, DNC, TNC, kGap di/tri, ORF, Fickett, Shannon entropy, Fourier binary, Fourier complex, Tsallis entropy). The Chaos game representation (group 11) is excluded. |
| `FE-BioAutoML_gth.sh` | Shell script example for processing a single patient's TP (`.igh.fasta`) and TN (`.nonigh.fasta`) files. |
| `FE-BioAutoML_gth_dir.sh` | Shell script for batch processing all patients in a directory tree. Automatically derives patient names from filenames and organises outputs as `OUTPUT_BASE/<patient>/<label>/`. |

---

## Prerequisites

1. Clone the original BioAutoML repository (with submodules) **alongside this repository**, or set `SCRIPT_DIR` to point to your BioAutoML installation:

   ```bash
   git clone --recurse-submodules https://github.com/Bonidia/BioAutoML.git
   ```

2. Create and activate the conda environment:

   ```bash
   conda env create -f BioAutoML-env.yml
   conda activate bioautoml
   ```

---

## Usage

### Single patient

Edit the paths at the top of `FE-BioAutoML_gth.sh`, then run:

```bash
bash preprocessing/FE-BioAutoML_gth.sh
```

### All patients (batch)

Edit the three path variables at the top of `FE-BioAutoML_gth_dir.sh`:

```bash
TP_DIR="/path/to/sampled_reads/TP"
TN_DIR="/path/to/sampled_reads/TN"
OUTPUT_BASE="/path/to/output/all_patients"
```

Then run from the repository root (so that MathFeature submodule paths resolve correctly):

```bash
bash preprocessing/FE-BioAutoML_gth_dir.sh
```

### Direct Python call

```bash
cd /path/to/BioAutoML
python FE-BioAutoMl_gth.py \
    --fasta_to_extract /path/to/reads.igh.fasta \
    --fasta_label TP \
    --output /path/to/output/patient_001/TP
```

**Output:** `<output>/feat_extraction/features_extracted.csv` — one row per read, 464 columns.
