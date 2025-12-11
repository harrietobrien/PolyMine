#!/usr/bin/env python
"""
Compare KNN imputation across different feature spaces:

  A) Structural-only features
  B) Composition-only features (element fractions)
  C) Combined structural + composition

Usage (from repo root):

python src/scripts/imputation_feature_spaces.py \
    --data data/processed/master_with_energies.csv \
    --targets mp_e_form_per_atom mp_e_above_hull \
    --test-size 0.2 \
    --n-neighbors 5
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
from pymatgen.core import Composition
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="KNN imputation comparison across structural vs composition feature spaces."
    )
    p.add_argument(
        "--data",
        type=str,
        required=True,
        help="CSV file with structural features + energy labels + formula (e.g. master_with_energies.csv).",
    )
    p.add_argument(
        "--targets",
        nargs="+",
        required=True,
        help="Target energy columns to evaluate, e.g. mp_e_form_per_atom mp_e_above_hull.",
    )
    p.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of observed rows to hold out for evaluation.",
    )
    p.add_argument(
        "--n-neighbors",
        type=int,
        default=5,
        help="Number of neighbors for KNN.",
    )
    p.add_argument(
        "--random-state",
        type=int,
        default=0,
        help="Random seed for train/test splitting.",
    )
    p.add_argument(
        "--out-csv",
        type=str,
        default=None,
        help="Optional: path to save a CSV with all MAE/RMSE results.",
    )
    return p.parse_args()


def infer_element_list(formulas: pd.Series) -> List[str]:
    """Infer sorted list of element symbols from formula strings."""
    elems = set()
    for f in formulas.dropna().unique():
        try:
            comp = Composition(f)
            elems.update(str(el) for el in comp.elements)
        except Exception:
            continue
    return sorted(elems)


def add_composition_features(df: pd.DataFrame, elem_list: List[str]) -> pd.DataFrame:
    """
    Add columns el_<Element>_frac giving atomic fraction of each
    element from the 'formula' column
    """
    def frac_vector(formula: str) -> dict:
        if not isinstance(formula, str):
            return {f"el_{el}_frac": 0.0 for el in elem_list}
        try:
            comp = Composition(formula)
        except Exception:
            return {f"el_{el}_frac": 0.0 for el in elem_list}
        return {f"el_{el}_frac": comp.get_atomic_fraction(el) for el in elem_list}

    records = df["formula"].apply(frac_vector)
    comp_df = pd.DataFrame(list(records))
    return pd.concat([df.reset_index(drop=True), comp_df.reset_index(drop=True)], axis=1)


def eval_feature_spaces_for_target(
    df: pd.DataFrame,
    target: str,
    struct_feats: List[str],
    comp_feats: List[str],
    test_size: float,
    n_neighbors: int,
    random_state: int,
) -> List[dict]:
    """
    Evaluate KNN imputation for a single target across three feature spaces:
      - structural
      - composition
      - combined
    Returns a list of dicts with metrics
    """
    results = []

    # Keep only rows where target is observed and all features exist
    mask_obs = df[target].notna()
    df_obs = df.loc[mask_obs].copy()

    # Drop rows with any missing features in either space
    feat_all = list(set(struct_feats) | set(comp_feats))
    df_obs = df_obs.dropna(subset=feat_all)

    n_obs = len(df_obs)
    if n_obs < 20:
        print(f"[WARN] Target {target}: only {n_obs} usable rows; skipping.")
        return results

    print(f"\n=== Target: {target} ===")
    print(f"  Usable rows (observed target + complete features): {n_obs}")

    # Build feature matrices
    X_struct = df_obs[struct_feats].values.astype(float)
    X_comp = df_obs[comp_feats].values.astype(float)
    X_comb = df_obs[struct_feats + comp_feats].values.astype(float)
    y = df_obs[target].values.astype(float)

    # Use a single train/test split (by index) shared across feature spaces
    idx = np.arange(n_obs)
    idx_train, idx_test = train_test_split(
        idx,
        test_size=test_size,
        random_state=random_state,
    )

    def eval_space(name: str, X: np.ndarray) -> dict:
        X_train, X_test = X[idx_train], X[idx_test]
        y_train, y_test = y[idx_train], y[idx_test]

        model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("knn", KNeighborsRegressor(n_neighbors=n_neighbors, 
                                            weights="distance")),
            ]
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        print(f"  {name:12s} -> MAE = {mae:.4f}, RMSE = {rmse:.4f}")
        return {
            "target": target,
            "feature_space": name,
            "n_neighbors": n_neighbors,
            "n_obs": n_obs,
            "mae": mae,
            "rmse": rmse,
        }

    results.append(eval_space("structural", X_struct))
    results.append(eval_space("composition", X_comp))
    results.append(eval_space("combined", X_comb))

    return results


def main() -> None:
    args = parse_args()

    data_path = Path(args.data)
    if not data_path.is_file():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path)
    print(f"[INFO] Loaded data with shape {df.shape} from {data_path}")

    if "formula" not in df.columns:
        raise ValueError("Data must contain a 'formula' column for composition features.")

    # Structural feature set
    structural_features = [
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
    for f in structural_features:
        if f not in df.columns:
            raise ValueError(f"Required structural feature '{f}' not found in data.")

    # Build composition features
    elem_list = infer_element_list(df["formula"])
    print(f"[INFO] Inferred {len(elem_list)} elements from formulas: {elem_list}")
    df_with_comp = add_composition_features(df, elem_list)
    comp_features = [f"el_{el}_frac" for el in elem_list]

    # Evaluate for each target
    all_results: List[dict] = []
    for target in args.targets:
        if target not in df_with_comp.columns:
            print(f"[WARN] Target '{target}' not found in data; skipping.")
            continue
        res = eval_feature_spaces_for_target(
            df=df_with_comp,
            target=target,
            struct_feats=structural_features,
            comp_feats=comp_features,
            test_size=args.test_size,
            n_neighbors=args.n_neighbors,
            random_state=args.random_state,
        )
        all_results.extend(res)

    if not all_results:
        print("[WARN] No results were computed.")
        return

    results_df = pd.DataFrame(all_results)
    print("\n=== Summary ===")
    print(results_df)

    if args.out_csv is not None:
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(out_path, index=False)
        print(f"[INFO] Saved summary results to {out_path}")


if __name__ == "__main__":
    main()
