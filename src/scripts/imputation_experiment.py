#!/usr/bin/env python
"""
Imputation experiment on crystal energy labels using structural features

Usage (from repo root):

python src/scripts/imputation_experiment.py \
    --data data/processed/master_with_energies.csv \
    --targets mp_e_form_per_atom mp_e_above_hull \
    --features volume density a b c alpha beta gamma n_sites n_species \
    --test-size 0.2 \
    --n-neighbors 5
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mean vs KNN imputation experiment on energy labels.")
    p.add_argument(
        "--data",
        type=str,
        required=True,
        help="CSV file with structural features + energy labels.",
    )
    p.add_argument(
        "--targets",
        nargs="+",
        required=True,
        help="One or more target energy columns to treat as 'missing' labels, e.g. mp_e_form_per_atom.",
    )
    p.add_argument(
        "--features",
        nargs="+",
        required=True,
        help="Feature columns to define the similarity space, e.g. volume density a b c alpha beta gamma n_sites n_species.",
    )
    p.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of observed rows to hold out for evaluation (masked labels).",
    )
    p.add_argument(
        "--n-neighbors",
        type=int,
        default=5,
        help="Number of neighbors for KNN imputation.",
    )
    p.add_argument(
        "--random-state",
        type=int,
        default=0,
        help="Random seed for train/test split.",
    )
    return p.parse_args()


def describe_missingness(df: pd.DataFrame, targets: List[str]) -> None:
    print("\n=== Missingness summary ===")
    n = len(df)
    print(f"Total rows: {n}")
    for col in targets:
        if col not in df.columns:
            print(f"  [WARN] Target {col} not found in data.")
            continue
        n_notnull = df[col].notna().sum()
        n_null = n - n_notnull
        frac_missing = n_null / n if n > 0 else np.nan
        print(f"  {col}: observed={n_notnull}, missing={n_null}, missing_frac={frac_missing:.3f}")


def eval_one_target(
    df: pd.DataFrame,
    target: str,
    features: List[str],
    test_size: float,
    n_neighbors: int,
    random_state: int,
) -> None:
    print(f"\n=== Target: {target} ===")

    # Only rows where target is observed (so we have ground truth)
    mask_obs = df[target].notna()
    df_obs = df.loc[mask_obs, features + [target]].dropna(subset=features)

    n_obs = len(df_obs)
    if n_obs < 20:
        print(f"  [WARN] Only {n_obs} rows with observed {target} and complete features; skipping.")
        return

    print(f"  Using {n_obs} rows with observed {target} for evaluation.")

    X = df_obs[features].values.astype(float)
    y = df_obs[target].values.astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    # Baseline: mean imputation (global mean from train)
    global_mean = y_train.mean()
    y_pred_mean = np.full_like(y_test, fill_value=global_mean)

    mae_mean = mean_absolute_error(y_test, y_pred_mean)
    rmse_mean = np.sqrt(mean_squared_error(y_test, y_pred_mean))


    # KNN-based imputation: KNN regressor in feature space
    knn_model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("knn", KNeighborsRegressor(n_neighbors=n_neighbors, \
                                        weights="distance")),
        ]
    )
    knn_model.fit(X_train, y_train)
    y_pred_knn = knn_model.predict(X_test)

    mae_knn = mean_absolute_error(y_test, y_pred_knn)
    rmse_knn = np.sqrt(mean_squared_error(y_test, y_pred_knn))

    print("  Mean imputation:")
    print(f"    MAE  = {mae_mean:.4f}")
    print(f"    RMSE = {rmse_mean:.4f}")

    print("  KNN imputation (k = {0}):".format(n_neighbors))
    print(f"    MAE  = {mae_knn:.4f}")
    print(f"    RMSE = {rmse_knn:.4f}")


def main() -> None:
    args = parse_args()

    data_path = Path(args.data)
    if not data_path.is_file():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path)
    print(f"[INFO] Loaded data with shape {df.shape} from {data_path}")

    # Missingness summary
    describe_missingness(df, args.targets)

    # For each target, run imputation comparison
    for target in args.targets:
        if target not in df.columns:
            print(f"[WARN] Target {target} missing; skipping.")
            continue
        eval_one_target(
            df=df,
            target=target,
            features=args.features,
            test_size=args.test_size,
            n_neighbors=args.n_neighbors,
            random_state=args.random_state,
        )


if __name__ == "__main__":
    main()
