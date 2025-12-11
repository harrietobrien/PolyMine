#!/usr/bin/env python
"""
For each chemical formula that appears multiple times, this script:

- Treats COD entries with that formula as polymorph candidates
- Clusters them in structural feature space
     (volume, density, lattice parameters, site counts)
- Computes energy statistics (formation energy and energy above hull)
     per (formula, cluster)
- Ranks clusters within each formula by energy and computes energy gaps

Outputs:
  - A cluster summary CSV with one row per (formula, cluster_id)
  - A per-entry CSV with cluster labels for each structure

Usage (from repo root):

  python src/scripts/polymorph_clustering.py \
      --data data/processed/master_with_energies.csv \
      --out-summary data/processed/polymorph_clusters_summary.csv \
      --out-assignments data/processed/polymorph_clusters_assignments.csv \
      --min-entries 3 \
      --min-energy 2 \
      --distance-threshold 2.0
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Cluster polymorph candidates within each composition and rank clusters by energy."
    )
    p.add_argument(
        "--data",
        type=str,
        required=True,
        help="CSV file with structural features + energies (e.g. master_with_energies.csv).",
    )
    p.add_argument(
        "--out-summary",
        type=str,
        required=True,
        help="Output CSV for cluster-level summary (one row per formula+cluster).",
    )
    p.add_argument(
        "--out-assignments",
        type=str,
        required=True,
        help="Output CSV for per-entry cluster assignments (one row per structure).",
    )
    p.add_argument(
        "--min-entries",
        type=int,
        default=3,
        help="Minimum number of entries per formula to be considered.",
    )
    p.add_argument(
        "--min-energy",
        type=int,
        default=2,
        help="Minimum number of entries with non-missing energy per formula.",
    )
    p.add_argument(
        "--distance-threshold",
        type=float,
        default=2.0,
        help=(
            "Distance threshold (in standardized feature space) for "
            "AgglomerativeClustering. Smaller values -> more clusters."
        ),
    )
    p.add_argument(
        "--form-energy-col",
        type=str,
        default="mp_e_form_per_atom",
        help="Column name for formation energy per atom.",
    )
    p.add_argument(
        "--hull-energy-col",
        type=str,
        default="mp_e_above_hull",
        help="Column name for energy above hull.",
    )
    return p.parse_args()


STRUCTURAL_FEATURES_DEFAULT: List[str] = [
    "volume",
    "density",
    "a",
    "b",
    "c",
    "alpha",
    "beta",
    "gamma",
    "n_sites",
    "n_species",
]


def cluster_one_formula(
    df_f: pd.DataFrame,
    struct_feats: List[str],
    distance_threshold: float,
) -> np.ndarray:
    """
    Given a subset of the df for a single formula, cluster entries in
    standardized structural feature space using agglomerative clustering
    """
    X = df_f[struct_feats].values.astype(float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Agglomerative clustering w/ automatic no. of clusters determined
    # by the distance threshold in standardized space
    model = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
    )
    labels = model.fit_predict(X_scaled)
    return labels


def main() -> None:
    args = parse_args()

    data_path = Path(args.data)
    if not data_path.is_file():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path)
    print(f"[INFO] Loaded data with shape {df.shape} from {data_path}")

    # Check required columns
    for col in ["formula", "cod_id"] + STRUCTURAL_FEATURES_DEFAULT:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in data.")

    if args.form_energy_col not in df.columns:
        raise ValueError(f"Formation energy column '{args.form_energy_col}' not found.")
    if args.hull_energy_col not in df.columns:
        raise ValueError(f"Hull energy column '{args.hull_energy_col}' not found.")

    struct_feats = STRUCTURAL_FEATURES_DEFAULT
    e_form_col = args.form_energy_col
    e_hull_col = args.hull_energy_col

    # Filter out rows with missing structural features
    df = df.dropna(subset=struct_feats)
    print(f"[INFO] After dropping rows with missing structural features: {df.shape[0]} rows")

    # Group by formula and filter by min entries + min energy labels
    group = df.groupby("formula")
    formulas = []
    for formula, g in group:
        n_total = len(g)
        n_with_energy = g[e_form_col].notna().sum()
        if n_total >= args.min_entries and n_with_energy >= args.min_energy:
            formulas.append((formula, n_total, n_with_energy))

    if not formulas:
        print("[WARN] No formulas met the min-entries/min-energy criteria.")
        return

    print("[INFO] Formulas selected for polymorph clustering:")
    for f, n_tot, n_en in formulas:
        print(f"  {f}: n_total={n_tot}, n_with_energy={n_en}")

    # Containers for outputs
    summary_records: List[Dict[str, Any]] = []
    assignment_records: List[Dict[str, Any]] = []

    # Process each formula
    for formula, _, _ in formulas:
        df_f = df[df["formula"] == formula].copy()
        n_f = len(df_f)

        print(f"\n[INFO] Processing formula '{formula}' with {n_f} entries")

        # Cluster in structural feature space
        try:
            labels = cluster_one_formula(
                df_f=df_f,
                struct_feats=struct_feats,
                distance_threshold=args.distance_threshold,
            )
        except Exception as e:
            print(f"[WARN] Clustering failed for {formula}: {e}")
            continue

        df_f["cluster_id"] = labels

        # Attach to assignment records
        for _, row in df_f.iterrows():
            assignment_records.append(
                {
                    "cod_id": row["cod_id"],
                    "formula": row["formula"],
                    "cluster_id": int(row["cluster_id"]),
                    e_form_col: row.get(e_form_col, np.nan),
                    e_hull_col: row.get(e_hull_col, np.nan),
                }
            )

        # Compute energy stats per cluster
        clusters = df_f.groupby("cluster_id")

        # Compute raw stats and track min cluster energies
        cluster_stats: List[Dict[str, Any]] = []
        min_mean_e_form = None
        min_mean_e_hull = None

        for cid, g in clusters:
            n_members = len(g)
            n_with_e_form = g[e_form_col].notna().sum()
            n_with_e_hull = g[e_hull_col].notna().sum()

            mean_e_form = g[e_form_col].mean()
            median_e_form = g[e_form_col].median()
            min_e_form = g[e_form_col].min()
            max_e_form = g[e_form_col].max()

            mean_e_hull = g[e_hull_col].mean()
            median_e_hull = g[e_hull_col].median()
            min_e_hull = g[e_hull_col].min()
            max_e_hull = g[e_hull_col].max()

            stat = {
                "formula": formula,
                "cluster_id": int(cid),
                "n_members": n_members,
                "n_with_e_form": int(n_with_e_form),
                "n_with_e_hull": int(n_with_e_hull),
                "mean_e_form": mean_e_form,
                "median_e_form": median_e_form,
                "min_e_form": min_e_form,
                "max_e_form": max_e_form,
                "mean_e_hull": mean_e_hull,
                "median_e_hull": median_e_hull,
                "min_e_hull": min_e_hull,
                "max_e_hull": max_e_hull,
            }
            cluster_stats.append(stat)

            if n_with_e_form > 0:
                if (min_mean_e_form is None) or (mean_e_form < min_mean_e_form):
                    min_mean_e_form = mean_e_form
            if n_with_e_hull > 0:
                if (min_mean_e_hull is None) or (mean_e_hull < min_mean_e_hull):
                    min_mean_e_hull = mean_e_hull

        # Add energy gaps relative to lowest-energy cluster
        for stat in cluster_stats:
            if (min_mean_e_form is not None) and \
                (stat["mean_e_form"] is not None):
                stat["delta_mean_e_form_to_min"] = (
                    stat["mean_e_form"] - min_mean_e_form
                )
            else:
                stat["delta_mean_e_form_to_min"] = np.nan

            if (min_mean_e_hull is not None) and \
                (stat["mean_e_hull"] is not None):
                stat["delta_mean_e_hull_to_min"] = (
                    stat["mean_e_hull"] - min_mean_e_hull
                )
            else:
                stat["delta_mean_e_hull_to_min"] = np.nan

            summary_records.append(stat)

    out_summary_path = Path(args.out_summary)
    out_assign_path = Path(args.out_assignments)
    out_summary_path.parent.mkdir(parents=True, exist_ok=True)
    out_assign_path.parent.mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame(summary_records)
    assign_df = pd.DataFrame(assignment_records)

    summary_df.to_csv(out_summary_path, index=False)
    assign_df.to_csv(out_assign_path, index=False)

    print(f"\n[INFO] Wrote cluster summary to: {out_summary_path}")
    print(f"[INFO] Wrote per-entry cluster assignments to: {out_assign_path}")
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
