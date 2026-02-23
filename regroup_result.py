#!/usr/bin/env python3
"""
Regroupe les métriques des cohorts transformer_real0 à transformer_real100
depuis les JSON dans reports/ de chaque répertoire et écrit un JSON et un CSV
dans le répertoire de base (train_data/test_dl).
"""

import argparse
import csv
import glob
import json
import logging
import os
import sys

COHORT_VALUES = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
JSON_OUT = "regrouped_transformer_metrics.json"
CSV_OUT = "regrouped_transformer_metrics.csv"

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger(__name__)


def default_base_dir() -> str:
    """Répertoire test_dl par défaut (depuis la racine du projet)."""
    root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
    )
    return os.path.join(root, "train_data", "test_dl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regroupe les métriques transformer_real0..real100 depuis reports/ en JSON et CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--base-dir",
        default=default_base_dir(),
        help="Répertoire contenant les dossiers transformer_real* et où écrire les sorties",
    )
    return parser.parse_args()


def latest_metrics_path(reports_dir: str) -> str | None:
    """
    Retourne le chemin du fichier metrics_*.json le plus récent (par timestamp dans le nom),
    ou None si aucun trouvé.
    """
    pattern = os.path.join(reports_dir, "metrics_*.json")
    files = glob.glob(pattern)
    if not files:
        return None
    # Tri décroissant par nom de fichier (timestamp YYYYMMDD_HHMMSS) → le dernier est le plus récent
    files.sort(reverse=True)
    return files[0]


def load_cohort_metrics(base_dir: str) -> list[dict]:
    rows = []
    for i in COHORT_VALUES:
        cohort_name = f"transformer_real{i}"
        reports_dir = os.path.join(base_dir, cohort_name, "reports")
        if not os.path.isdir(reports_dir):
            log.warning("Répertoire absent ou invalide: %s", reports_dir)
            continue
        path = latest_metrics_path(reports_dir)
        if not path:
            log.warning("Aucun metrics_*.json dans: %s", reports_dir)
            continue
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                log.warning("JSON invalide (pas un objet): %s", path)
                continue
            rows.append(data)
        except (OSError, json.JSONDecodeError) as e:
            log.warning("Impossible de charger %s: %s", path, e)
            continue
    # Tri par valeur numérique du cohort (real0, real10, real20, ..., real100)
    def _cohort_sort_key(r: dict) -> int:
        name = r.get("cohort_name") or ""
        if name.startswith("transformer_real"):
            try:
                return int(name.replace("transformer_real", ""))
            except ValueError:
                pass
        return -1

    rows.sort(key=_cohort_sort_key)
    return rows


def write_outputs(base_dir: str, rows: list[dict]) -> None:
    os.makedirs(base_dir, exist_ok=True)
    json_path = os.path.join(base_dir, JSON_OUT)
    csv_path = os.path.join(base_dir, CSV_OUT)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    if not rows:
        log.warning("Aucune donnée à écrire en CSV.")
        return
    # Union des clés pour que toutes les colonnes soient présentes
    all_keys: list[str] = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                all_keys.append(k)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    log.info("Écrit: %s", json_path)
    log.info("Écrit: %s", csv_path)


def main() -> int:
    args = parse_args()
    base_dir = os.path.abspath(args.base_dir)
    if not os.path.isdir(base_dir):
        log.error("Le répertoire de base n'existe pas: %s", base_dir)
        return 1
    rows = load_cohort_metrics(base_dir)
    if not rows:
        log.error("Aucune métrique chargée.")
        return 1
    write_outputs(base_dir, rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
