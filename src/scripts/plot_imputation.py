#!/usr/bin/env python
"""
Input CSV is expected from `imputation_experiment.py` and contains:
  target, method, mae, rmse, n_eval, mask_frac, n_neighbors

Example:
    python src/scripts/plot_imputation.py \
        --csv data/processed/imputation_results.csv \
        --out data/processed/imputation_results_plot.png \
        --title "Imputation baselines (density)"
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Plot imputation metrics grouped by target and method.")
    ap.add_argument("--csv", type=Path, required=True, help="Imputation results CSV.")
    ap.add_argument("--out", type=Path, default=Path("data/processed/imputation_results_plot.png"), help="Output image.")
    ap.add_argument(
        "--metric",
        choices=["mae", "rmse", "both"],
        default="both",
        help="Metric(s) to plot. 'both' makes a 1x2 figure.",
    )
    ap.add_argument("--title", type=str, default=None, help="Figure title.")
    ap.add_argument("--annotate", action="store_true", help="Annotate bars with metric values.")
    return ap.parse_args()


def _load(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"target", "method", "mae", "rmse", "n_eval"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"CSV missing required columns: {missing}")
    return df


def _plot_metric(ax, df: pd.DataFrame, metric: str, annotate: bool) -> None:
    methods_order: List[str] = ["mean", "group_median", "knn"]
    palette = {"mean": "#6baed6", "group_median": "#74c476", "knn": "#fd8d3c"}
    targets = df["target"].unique().tolist()

    x = np.arange(len(targets))
    width = 0.23 if len(targets) > 1 else 0.3
    offsets = np.linspace(-width, width, num=len(methods_order))

    for i, m in enumerate(methods_order):
        sub = df[df["method"] == m]
        vals = [sub[sub["target"] == t][metric].iloc[0] 
                if not sub[sub["target"] == t].empty 
                else np.nan for t in targets]
        bars = ax.bar(x + offsets[i], vals, width, 
                      label=m, color=palette.get(m, None))
        if annotate:
            for bar, v in zip(bars, vals):
                if np.isnan(v):
                    continue
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height,
                    f"{v:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=90,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(targets, rotation=0)
    ax.set_ylabel(metric.upper())
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.legend(
        frameon=False,
        title="method",
        loc="lower center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=len(methods_order),
    )


def plot_imputation(csv_path: Path, out_path: Path, metric: str = "both", title: str | None = None, annotate: bool = False) -> None:
    df = _load(csv_path)
    targets = df["target"].unique().tolist()
    base_width = max(6, 1.5 * len(targets))
    if metric == "both":
        fig, axes = plt.subplots(1, 2, figsize=(base_width * 1.6, 4), 
                                 dpi=150, constrained_layout=True)
        _plot_metric(axes[0], df, "mae", annotate)
        _plot_metric(axes[1], df, "rmse", annotate)
    else:
        fig, ax = plt.subplots(figsize=(base_width, 4), 
                               dpi=150, constrained_layout=True)
        _plot_metric(ax, df, metric, annotate)

    if title:
        fig.suptitle(title, fontsize=12, y=1.02)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved plot to {out_path}")


def main() -> None:
    args = parse_args()
    plot_imputation(args.csv, args.out, metric=args.metric, 
                    title=args.title, annotate=args.annotate)


if __name__ == "__main__":
    main()
