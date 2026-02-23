# IGH Classification

Deep learning pipeline for classifying sequencing reads as IGH (immunoglobulin heavy chain) or non-IGH using a Feature Tokenizer Transformer (FT-Transformer) trained on a combination of real CLL patient data and synthetic V(D)J recombination sequences.

This repository accompanies the manuscript:

> Darmendre J. *et al.* **Machine learning-based classification of IGHV mutation status in CLL from whole-genome sequencing data.** *(submitted)*

Pre-trained model weights are available on Hugging Face: [gth-ai/igh_classification_fttransformer](https://huggingface.co/gthai/igh_classification_fttransformer)

---

## Overview

The pipeline consists of three stages:

```
                      IMGT alleles (V, D, J)
                            │
                            ▼
[0] Synthetic data generation (V(D)J recombination simulator)
    │  synthetic IGH reads (.fasta)         real WGS reads (.fasta)
    │                                            │
    └──────────────────┬─────────────────────────┘
                       ▼
[1] Preprocessing (BioAutoML-based feature extraction)
    │  464 numerical descriptors per read
    ▼
[2] Training (FT-Transformer with progressive real/synthetic cohorts)
    │  best_model.pt checkpoints
    ▼
[3] Testing (batch evaluation on patient test set)
       metrics, ROC/PR curves, confusion matrix
```

### Key results

| Configuration | Balanced Acc. | F1-score | ROC-AUC | PR-AUC |
|---|---|---|---|---|
| 150 K real + 100% synthetic | **97.5%** | **95.6%** | **99.7%** | **99.3%** |

---

## Repository structure

```
igh_classification/
├── README.md
├── requirements.txt
├── combinatorics/                   # Synthetic IGH V(D)J recombination simulator
│   ├── combinatorics.py
│   ├── input/                       # IMGT V, D, J allele FASTA files
│   └── true_neg/                    # True negative read generator
├── preprocessing/                   # Feature extraction (BioAutoML-based)
│   ├── README.md
│   ├── FE-BioAutoMl_gth.py          # Feature extraction script
│   ├── FE-BioAutoML_gth.sh          # Single-patient wrapper
│   └── FE-BioAutoML_gth_dir.sh      # Batch processing wrapper
├── models.py                        # Model architectures (MLP, TabNet, FT-Transformer, DANN)
├── losses.py                        # Loss functions (Focal, Label Smoothing, DANN, ...)
├── deep_bio_classifier.py           # Training pipeline (DeepBioClassifier class)
├── test_deep_bio_classifier.py      # Evaluation pipeline (ModelTester class)
├── run_progressive_training.py      # Progressive training script (N_real_fixe approach)
├── run_progressive_training.sh      # Shell wrapper for training
├── run_test_cohort.py               # Batch testing script (N_global_fixe approach)
├── run_test.sh                      # Shell wrapper for testing
├── analyse_result.py                # Result visualization (per-metric plots)
└── regroup_result.py                # Metric aggregation from multiple cohort JSON files
```

---

## Installation

### Prerequisites

- Python >= 3.9
- CUDA-capable GPU (recommended)
- [MathFeature](https://github.com/Bonidia/MathFeature) (required for preprocessing only — see `preprocessing/README.md`)

### Setup

```bash
git clone https://github.com/acri-nb/igh_classification.git
cd igh_classification
pip install -r requirements.txt
```

---

## Usage

### 0. Synthetic data generation (optional)

Generate labeled synthetic IGH reads to augment the training set. The [`combinatorics/`](combinatorics/) module simulates V(D)J recombination (exonuclease trimming, P/N nucleotide addition, somatic hypermutation) and produces fixed-length reads centered on the junction. A companion script generates true negative reads from a reference genome. See [`combinatorics/README.md`](combinatorics/README.md) for full details and parameters.

```bash
# True positives — synthetic IGH reads
python combinatorics/combinatorics.py \
    -vf combinatorics/input/all/IGHV_all.txt \
    -df combinatorics/input/all/IGHD_all.txt \
    -jf combinatorics/input/all/IGHJ_all.txt \
    -rl 100 -pid 0.9 -o synthetic_tp/

# True negatives — random genomic reads (IGH locus masked)
python combinatorics/true_neg/true_negatives.py \
    -i /path/to/genome.fna -rl 100 -nc 1000 -g hg38 -o synthetic_tn/
```

### 1. Preprocessing

Extract 464 numerical descriptors from FASTA reads using the BioAutoML-based pipeline. See [`preprocessing/README.md`](preprocessing/README.md) for full instructions.

```bash
# Single patient
cd preprocessing
bash FE-BioAutoML_gth.sh

# All patients (batch)
bash FE-BioAutoML_gth_dir.sh
```

The output is a CSV file (`features_extracted.csv`) containing one row per read and 464 feature columns.

### 2. Training

#### Progressive training (N_real_fixe — recommended)

Trains one FT-Transformer model per combination of real data size and synthetic augmentation percentage. This is the approach reported in the paper.

```bash
python run_progressive_training.py \
    --data /path/to/df_combined.csv \
    --output-root ./training_cohort \
    --model transformer \
    --real-sizes 50000,100000,150000,200000,213100 \
    --synthetic-pcts 10,20,30,40,50,60,70,80,90,100 \
    --epochs 150 \
    --gpu-id 0
```

Or using the shell wrapper:

```bash
bash run_progressive_training.sh \
    --data /path/to/df_combined.csv \
    --output-root ./training_cohort \
    --gpu-id 0
```

**Output structure:**

```
training_cohort/
├── real_050000/
│   ├── synth_010pct_005000/
│   │   ├── checkpoints/best_model.pt
│   │   ├── cohort_manifest.json
│   │   └── cohort_training.log
│   └── ...
└── real_150000/
    └── synth_100pct_150000/   ← best configuration
        └── checkpoints/best_model.pt
```

**Input CSV format:** `df_combined.csv` must contain feature columns plus the following metadata columns: `label` (TP/TN), `data_type` (real/synthetic), `source` (e.g. `cll_dna`).

### 3. Testing

#### Batch testing of trained models

```bash
python run_test_cohort.py \
    --base-dir ./training_cohort \
    --dataset /path/to/df_patient_test.csv \
    --output-root ./test_results \
    --pattern "real_*/synth_*" \
    --gpu-id 0
```

Or using the shell wrapper:

```bash
bash run_test.sh 0   # argument: GPU ID
```

**Output per model:** ROC curve, PR curve, confusion matrix, full report (TXT + JSON), probability distributions, and a summary CSV across all models.

### 4. Analysis

Aggregate metrics across all tested cohorts and generate per-metric plots:

```bash
# Aggregate JSON reports from test_results/
python regroup_result.py \
    --base-dir ./test_results \
    --output-dir ./analysis

# Generate plots
python analyse_result.py \
    --csv ./analysis/regrouped_transformer_metrics.csv \
    --output-dir ./analysis/plots
```

---

## Pre-trained weights

Pre-trained FT-Transformer weights for all 61 configurations (11 fixed-total-size + 50 progressive training) are available on Hugging Face:

**[gth-ai/igh_classification_fttransformer](https://huggingface.co/gthai/igh_classification_fttransformer)**

### Loading a checkpoint

```python
import torch
from models import FTTransformer

# Load checkpoint
checkpoint = torch.load("best_model.pt", map_location="cpu")
model = FTTransformer(
    input_dim=checkpoint["input_dim"],
    hidden_dims=checkpoint["hidden_dims"],
    dropout=checkpoint["dropout"],
)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
```

The recommended checkpoint is `progressive_training/real_150000/synth_100pct_150000/best_model.pt`
(F1 = 0.956, ROC-AUC = 0.997 on the patient-derived test set).

---

## Data

- **Synthetic sequences**: generated with the V(D)J recombination simulator in [`combinatorics/`](combinatorics/) using alleles from [IMGT/GENE-DB](https://www.imgt.org/genedb/).
- **Real sequences**: ICGC-CLL Genome cohort (accessed via [ICGC Data Portal](https://dcc.icgc.org/)) and CLL patient samples from Georges-L.-Dumont University Hospital Centre (Moncton, NB, Canada).

---

## Citation

If you use this code or the pre-trained weights, please cite:

```bibtex
@article{darmendre2025igh,
  title   = {Machine learning-based classification of IGHV mutation status
             in CLL from whole-genome sequencing data},
  author  = {Darmendre, Jessica and others},
  journal = {(submitted)},
  year    = {2026}
}
```

---

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
