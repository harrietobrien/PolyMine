#!/usr/bin/env python
"""
Plot KNN imputation MAE by feature space.

Usage example:

  python scripts/make_summary_plots.py \
      --mae-form 0.3798 0.2376 0.2553 \
      --mae-hull 0.0336 0.0416 0.0399 \
      --out figures/feature_space_mae_1000.png
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot KNN MAE by feature space.")
    p.add_argument(
        "--mae-form",
        type=float,
        nargs=3,
        metavar=("STRUCT", "COMP", "COMB"),
        required=True,
        help="MAE for E_form in structural, composition, combined spaces.",
    )
    p.add_argument(
        "--mae-hull",
        type=float,
        nargs=3,
        metavar=("STRUCT", "COMP", "COMB"),
        required=True,
        help="MAE for E_hull in structural, composition, combined spaces.",
    )
    p.add_argument(
        "--out",
        type=str,
        default="figures/feature_space_mae.png",
        help="Output PNG path.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    feature_spaces = ["Structural", "Composition", "Combined"]
    x = np.arange(len(feature_spaces))
    width = 0.35

    mae_form = np.array(args.mae_form)
    mae_hull = np.array(args.mae_hull)

    cmap = plt.get_cmap("rainbow")
    color_form = cmap(0.15) 
    color_hull = cmap(0.65)

    fig, ax = plt.subplots(figsize=(6, 5))

    bars_form = ax.bar(
        x - width / 2, mae_form, width,
        label=r"$E_\mathrm{form}$",
        color=color_form,
    )
    bars_hull = ax.bar(
        x + width / 2, mae_hull, width,
        label=r"$E_\mathrm{hull}$",
        color=color_hull,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(feature_spaces)
    ax.set_ylabel("MAE (eV/atom)")
    ax.set_title("KNN Imputation MAE by Feature Space")
    ax.legend()

    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.005,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    add_labels(bars_form)
    add_labels(bars_hull)

    fig.tight_layout()
    fig.savefig(args.out, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
