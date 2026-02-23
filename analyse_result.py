#!/usr/bin/env python3
"""
Génère un graphique par métrique à partir du CSV regroupé des cohorts
transformer_real0 à transformer_real100.

Chaque graphique présente :
- Des barres empilées Synth % / Real % (composition du jeu d'entraînement)
- Une courbe de la métrique (axe secondaire droit)
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Constantes ────────────────────────────────────────────────────────────────

TOTAL_TRAIN = 598_709
TOTAL_TEST = 173_100

METRICS = [
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "specificity",
    "f1_score",
    "mcc",
    "cohen_kappa",
    "roc_auc",
    "pr_auc",
    "log_loss",
    "total_samples",
    "positive_samples",
    "negative_samples",
    "predicted_positive",
    "predicted_negative",
    "true_positives",
    "true_negatives",
    "false_positives",
    "false_negatives",
    "fpr",
    "fnr",
]

# Métriques dont la valeur est entre 0 et 1 (affichage axe droit 0‑1)
RATIO_METRICS = {
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "specificity",
    "f1_score",
    "roc_auc",
    "pr_auc",
    "fpr",
    "fnr",
}

# Noms lisibles pour les titres et l'axe droit
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "balanced_accuracy": "Balanced Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "specificity": "Specificity",
    "f1_score": "F1 Score",
    "mcc": "MCC (Matthews Corr. Coeff.)",
    "cohen_kappa": "Cohen's Kappa",
    "roc_auc": "ROC AUC",
    "pr_auc": "PR AUC",
    "log_loss": "Log Loss",
    "total_samples": "Total Samples",
    "positive_samples": "Positive Samples",
    "negative_samples": "Negative Samples",
    "predicted_positive": "Predicted Positive",
    "predicted_negative": "Predicted Negative",
    "true_positives": "True Positives",
    "true_negatives": "True Negatives",
    "false_positives": "False Positives",
    "false_negatives": "False Negatives",
    "fpr": "False Positive Rate",
    "fnr": "False Negative Rate",
}

# Palette inspirée du graphique de référence
COLOR_SYNTH = "#1B6B7D"   # bleu-teal foncé
COLOR_REAL = "#E07B39"    # orange
COLOR_LINE = "#3D8B37"    # vert
COLOR_MARKER = "#2D6B2D"  # vert foncé pour les points


def default_csv_path() -> str:
    root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
    )
    return os.path.join(root, "train_data", "test_dl", "regrouped_transformer_metrics.csv")


def default_output_dir() -> str:
    root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
    )
    return os.path.join(root, "train_data", "test_dl", "transformer", "all", "plots")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Génère un graphique par métrique (barres Synth/Real + courbe).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--csv",
        default=default_csv_path(),
        help="Chemin vers le CSV regroupé des métriques",
    )
    parser.add_argument(
        "--output-dir",
        default=default_output_dir(),
        help="Répertoire de sortie pour les images",
    )
    parser.add_argument(
        "--format",
        default="png",
        choices=["png", "pdf", "svg"],
        help="Format de sortie des images",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Résolution des images",
    )
    return parser.parse_args()


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Identifier la colonne du % Real
    if "cohort" in df.columns:
        df["real_pct"] = df["cohort"].astype(int)
    elif "cohort_name" in df.columns:
        df["real_pct"] = (
            df["cohort_name"]
            .str.replace("transformer_real", "", regex=False)
            .astype(int)
        )
    else:
        raise ValueError("Colonne 'cohort' ou 'cohort_name' introuvable dans le CSV.")
    df["synth_pct"] = 100 - df["real_pct"]
    df = df.sort_values("real_pct").reset_index(drop=True)
    return df


def plot_metric(df: pd.DataFrame, metric: str, output_dir: str,
                fmt: str, dpi: int) -> str:
    """Génère un graphique pour une métrique donnée et retourne le chemin du fichier."""

    real_pct = df["real_pct"].values
    synth_pct = df["synth_pct"].values
    values = df[metric].values

    label = METRIC_LABELS.get(metric, metric)
    x = np.arange(len(real_pct))
    bar_width = 0.55

    fig, ax_bar = plt.subplots(figsize=(10, 5.5))

    # ── Barres empilées (axe gauche) ──────────────────────────────────────
    bars_synth = ax_bar.bar(
        x, synth_pct, bar_width,
        label="Synth (%)", color=COLOR_SYNTH, edgecolor="white", linewidth=0.4,
    )
    bars_real = ax_bar.bar(
        x, real_pct, bar_width, bottom=synth_pct,
        label="Real (%)", color=COLOR_REAL, edgecolor="white", linewidth=0.4,
    )
    ax_bar.set_ylim(0, 115)
    ax_bar.set_ylabel("Training composition (%)", fontsize=10)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([f"{v}%" for v in real_pct], fontsize=9)
    ax_bar.set_xlabel("Percentage of real data (Real %)", fontsize=10)
    ax_bar.tick_params(axis="y", labelsize=9)

    # ── Courbe de la métrique (axe droit) ─────────────────────────────────
    ax_line = ax_bar.twinx()

    # Ajuster l'échelle de l'axe droit selon le type de métrique
    is_ratio = metric in RATIO_METRICS
    if is_ratio:
        # Afficher l'axe en pourcentage (0 – 100 %)
        ax_line.set_ylim(-2, 108)
        ax_line.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f"{v:.0f}%")
        )
        plot_values = values * 100  # convertir 0‑1 → 0‑100 pour le tracé
    else:
        plot_values = values
        vmin, vmax = values.min(), values.max()
        margin = (vmax - vmin) * 0.15 if vmax != vmin else vmax * 0.1
        ax_line.set_ylim(vmin - margin, vmax + margin)

    ax_line.set_ylabel(f"{label} (%)" if is_ratio else label,
                       fontsize=10, color=COLOR_LINE)
    ax_line.tick_params(axis="y", labelsize=9, colors=COLOR_LINE)

    # Tracer la courbe avec les valeurs (éventuellement en %)
    ax_line.plot(
        x, plot_values,
        color=COLOR_LINE, linewidth=2.2, marker="o",
        markersize=6, markerfacecolor=COLOR_MARKER, markeredgecolor="white",
        markeredgewidth=1.0, zorder=5, label=label,
    )

    # Annoter la valeur sur chaque point
    for xi, v_raw, v_plot in zip(x, values, plot_values):
        if is_ratio:
            txt = f"{v_raw * 100:.2f}"
        elif abs(v_raw) < 10:
            txt = f"{v_raw:.3f}"
        else:
            txt = f"{v_raw:,.0f}"
        ax_line.annotate(
            txt, (xi, v_plot),
            textcoords="offset points", xytext=(0, 10),
            ha="center", fontsize=7, color=COLOR_LINE, fontweight="bold",
        )

    # ── Titre ─────────────────────────────────────────────────────────────
    ax_bar.set_title(
        f"{label} per Synth/Real cohort\n"
        f"(training: {TOTAL_TRAIN:,} seq. | test: {TOTAL_TEST:,} seq.)",
        fontsize=12, fontweight="bold", pad=14,
    )

    # ── Légende combinée ─────────────────────────────────────────────────
    handles_bar, labels_bar = ax_bar.get_legend_handles_labels()
    handles_line, labels_line = ax_line.get_legend_handles_labels()
    ax_bar.legend(
        handles_bar + handles_line,
        labels_bar + labels_line,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=3,
        frameon=False,
        fontsize=9,
    )

    fig.tight_layout()

    filepath = os.path.join(output_dir, f"{metric}.{fmt}")
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return filepath


def main() -> int:
    args = parse_args()

    if not os.path.isfile(args.csv):
        print(f"ERREUR: fichier CSV introuvable: {args.csv}", file=sys.stderr)
        return 1

    os.makedirs(args.output_dir, exist_ok=True)
    df = load_data(args.csv)

    generated = []
    for metric in METRICS:
        if metric not in df.columns:
            print(f"WARNING: colonne '{metric}' absente du CSV, ignorée.", file=sys.stderr)
            continue
        path = plot_metric(df, metric, args.output_dir, args.format, args.dpi)
        generated.append(path)
        print(f"  OK  {path}")

    print(f"\n{len(generated)} graphiques générés dans {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
