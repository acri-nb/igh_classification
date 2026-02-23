# Combinatorics — Synthetic IGH V(D)J Recombination Simulator

This module generates synthetic immunoglobulin heavy chain (IGH) sequences by simulating V(D)J recombination. It is designed to produce labeled training data for deep learning classifiers that identify IGH recombination events in next-generation sequencing (NGS) reads.

## Overview

[`combinatorics.py`](combinatorics.py) enumerates V–D–J gene segment combinations from IMGT-formatted FASTA inputs and applies biologically grounded modifications to each recombined sequence:

- **Exonuclease trimming** — Poisson-distributed truncations at the 5′ and 3′ ends of V, D, and J segments
- **P and N nucleotide addition** — palindromic and non-templated bases inserted at V–D and D–J junctions
- **Somatic hypermutation** — point mutations introduced in the V region at a configurable percent identity
- **NGS read simulation** — final sequences are cut into fixed-length reads centered on the junction

[`true_neg/true_negatives.py`](true_neg/true_negatives.py) independently generates true negative control sequences by randomly sampling fixed-length reads from a reference genome, providing non-IGH background sequences for binary classification.

## Repository structure

```
combinatorics/
├── combinatorics.py          # Main recombination simulator
├── input/
│   ├── all/                  # Full IGHV, IGHD, IGHJ FASTA files
│   │   ├── IGHV_all.txt
│   │   ├── IGHD_all.txt
│   │   └── IGHJ_all.txt
│   └── three/                # Minimal 3-gene test set
├── true_neg/
│   ├── true_negatives.py     # True negative read generator
│   └── masked_regions/
│       └── ighv_locus.bed    # BED file of IGH locus (for masking)
└── output-all-fixed/         # Example output (reads + per-parameter logs)
```

## Dependencies

- [Biopython](https://biopython.org/) — FASTA parsing
- [NumPy](https://numpy.org/) — random sampling and distributions
- [Matplotlib](https://matplotlib.org/) — log histograms

Install with:

```bash
pip install biopython numpy matplotlib
```

## Usage

**Generate recombined IGH reads (positive examples) with [`combinatorics.py`](combinatorics.py):**

```bash
python combinatorics.py \
    -vf input/all/IGHV_all.txt \
    -df input/all/IGHD_all.txt \
    -jf input/all/IGHJ_all.txt \
    -rl 100 \
    -pid 0.9 \
    -o output/
```

**Generate true negative reads from a reference genome with [`true_neg/true_negatives.py`](true_neg/true_negatives.py):**

```bash
python true_neg/true_negatives.py \
    -i path/to/genome.fna \
    -rl 100 \
    -nc 1000 \
    -g hg38 \
    -o output/
```

## Key parameters — [`combinatorics.py`](combinatorics.py)

| Argument | Default | Description |
|---|---|---|
| `-vf / -df / -jf` | `input/all/*.txt` | Input FASTA files for V, D, J segments |
| `-vtf / -vtt` | `0 / 1.2` | Poisson λ for V 5′ / 3′ truncation |
| `-dtf / -dtt` | `4.4 / 5.0` | Poisson λ for D 5′ / 3′ truncation |
| `-jtf / -jtt` | `4.7 / 0` | Poisson λ for J 5′ / 3′ truncation |
| `-pid` | `0.9` | V-region percent identity after hypermutation |
| `-pp / -np` | `1 / 10` | Poisson λ for P / N nucleotide addition |
| `-rl` | `100` | Output read length (bp) |
| `-rb` | `20` | Minimum bp retained on each side of the junction |
| `-n` | `all` | Number of V–D–J permutations to sample |
| `-o` | `output` | Output directory |
