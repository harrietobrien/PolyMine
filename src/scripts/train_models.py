"""Train simple regression or classification models on a feature CSV.

This module exposes a callable `ModelTrainer` (accepts **kwargs) plus a thin
CLI wrapper. It supports regression (Linear, Ridge, RandomForestRegressor)
and classification (LogisticRegression, RidgeClassifier, RandomForestClassifier).

Example:
    python src/scripts/train_models.py \
        --csv data/clean/extra_features.csv \
        --meta-csv data/clean/polymorph_clusters.csv \
        --target density \
        --task regression \
        --drop-cols entry_id
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeCV, RidgeClassifier
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train linear/ridge/random-forest models on a feature CSV.")
    ap.add_argument("--csv", type=Path, required=True, help="Input feature CSV with `entry_id` and target column.")
    ap.add_argument("--meta-csv", type=Path, default=None, help="Optional metadata CSV to join on entry_id.")
    ap.add_argument("--target", type=str, required=True, help="Target column name.")
    ap.add_argument("--task", choices=["regression", "classification"], default="regression", help="Learning task type.")
    ap.add_argument(
        "--drop-cols",
        type=str,
        default="entry_id",
        help="Comma-separated columns to drop from features (default: entry_id).",
    )
    ap.add_argument("--test-size", type=float, default=0.2, help="Test split fraction.")
    ap.add_argument("--random-state", type=int, default=42, help="Random seed.")
    return ap.parse_args()


def load_data(path: Path, target: str, drop_cols: List[str], \
              meta: Path | None) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Make entry_id a string if present
    if "entry_id" in df.columns:
        df["entry_id"] = df["entry_id"].astype(str)

    if meta is not None:
        meta_df = pd.read_csv(meta)
        if "entry_id" not in meta_df.columns:
            raise SystemExit("meta CSV must include `entry_id`.")
        # Make entry_id a string here too
        meta_df["entry_id"] = meta_df["entry_id"].astype(str)

        df = df.merge(meta_df, on="entry_id", how="left", suffixes=("", "_meta"))

    if target not in df.columns:
        raise SystemExit(f"Target column `{target}` not found. Available: {list(df.columns)}")

    return df


def evaluate_reg(model, X_train, X_test, y_train, y_test, name: str):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"{name}: MAE={mae:.4f}, R2={r2:.4f}")


def evaluate_clf(model, X_train, X_test, y_train, y_test, name: str, encoder: LabelEncoder):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="weighted")
    print(f"{name}: acc={acc:.4f}, f1_weighted={f1:.4f}, classes={len(encoder.classes_)}")


class ModelTrainer:
    """Callable training wrapper (regression or classification)."""

    def __init__(self, **kwargs):
        self.csv = Path(kwargs["csv"]) if kwargs.get("csv") else None
        self.meta_csv = Path(kwargs["meta_csv"]) \
            if kwargs.get("meta_csv") else None
        self.target = kwargs.get("target")
        self.task = kwargs.get("task", "regression")
        self.drop_cols = [c.strip() \
                          for c in kwargs.get("drop_cols", "entry_id").split(",") \
                            if c.strip()]
        self.test_size = float(kwargs.get("test_size", 0.2))
        self.random_state = int(kwargs.get("random_state", 42))
        self.label_encoder = LabelEncoder()

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "ModelTrainer":
        return cls(
            csv=args.csv,
            meta_csv=args.meta_csv,
            target=args.target,
            task=args.task,
            drop_cols=args.drop_cols,
            test_size=args.test_size,
            random_state=args.random_state,
        )

    def _prepare(self):
        if self.csv is None:
            raise SystemExit("CSV path is required.")
        df = load_data(self.csv, self.target, self.drop_cols, self.meta_csv)

        # Drop target + any explicitly requested columns (e.g., entry_id)
        feature_df = df.drop(columns=[self.target] + self.drop_cols, errors="ignore")

        numeric_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
        X = feature_df[numeric_cols]

        # Target vector
        y_raw = df[self.target]
        if self.task == "regression":
            y = y_raw.to_numpy(dtype=float)
        else:
            # encode labels for classification
            y = self.label_encoder.fit_transform(y_raw.astype(str))

        print(f"[INFO] Using {len(numeric_cols)} numeric feature columns.")
        return X, y

    def run(self):
        X, y = self._prepare()

        stratify_vec = None
        if self.task == "classification":
            unique, counts = np.unique(y, return_counts=True)
            if counts.min() >= 2:
                stratify_vec = y
            else:
                print(
                    "[WARN] Some classes have < 2 samples; "
                    "disabling stratified split."
                )

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify_vec,
        )

        print(
            f"Training on {X_train.shape[0]} samples; "
            f"testing on {X_test.shape[0]} samples; d={X_train.shape[1]}"
        )

        if self.task == "regression":
            linear = make_pipeline(StandardScaler(), LinearRegression())
            ridge = make_pipeline(
                StandardScaler(),
                RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1, 10, 100]),
            )
            rf = RandomForestRegressor(
                n_estimators=300,
                max_depth=None,
                n_jobs=-1,
                random_state=self.random_state,
            )
            evaluate_reg(linear, X_train, X_test, y_train, y_test, "LinearRegression")
            evaluate_reg(ridge, X_train, X_test, y_train, y_test, "RidgeCV")
            evaluate_reg(rf, X_train, X_test, y_train, y_test, "RandomForestRegressor")
        else:
            logreg = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=200, multi_class="auto"),
            )
            ridge_clf = make_pipeline(StandardScaler(), RidgeClassifier())
            rf_clf = RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                n_jobs=-1,
                random_state=self.random_state,
            )
            evaluate_clf(
                logreg, X_train, X_test, y_train, y_test,
                "LogisticRegression", self.label_encoder
            )
            evaluate_clf(
                ridge_clf, X_train, X_test, y_train, y_test,
                "RidgeClassifier", self.label_encoder
            )
            evaluate_clf(
                rf_clf, X_train, X_test, y_train, y_test,
                "RandomForestClassifier", self.label_encoder
            )


if __name__ == "__main__":
    args = parse_args()
    trainer = ModelTrainer.from_args(args)
    trainer.run()
