#!/usr/bin/env python
"""
Plot mean vs KNN imputation errors (MAE / RMSE) for each target

Numbers are taken from the 1000-structure COD–MP subset:
- E_form: Mean vs KNN
- E_hull: Mean vs KNN
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUTDIR = Path("figures")
OUTDIR.mkdir(parents=True, exist_ok=True)

def plot_imputation_bar():
    # Magic no. metrics from 1000-structure experiment
    data = {
        ("$E_\\mathrm{form}$", "MAE"):  (0.7082, 0.3798),
        ("$E_\\mathrm{form}$", "RMSE"): (0.9028, 0.6207),
        ("$E_\\mathrm{hull}$", "MAE"):  (0.0506, 0.0336),
        ("$E_\\mathrm{hull}$", "RMSE"): (0.1178, 0.0908),
    }

    groups = [
        ("$E_\\mathrm{form}$", "MAE"),
        ("$E_\\mathrm{form}$", "RMSE"),
        ("$E_\\mathrm{hull}$", "MAE"),
        ("$E_\\mathrm{hull}$", "RMSE"),
    ]

    mean_vals = [data[g][0] for g in groups]
    knn_vals  = [data[g][1] for g in groups]

    x = np.arange(len(groups))
    width = 0.35

    # Slightly taller figure
    fig, ax = plt.subplots(figsize=(6.5, 5.0))

    # Nicer colors
    bars_mean = ax.bar(
        x - width/2,
        mean_vals,
        width,
        label="Mean",
        color="slateblue"
    )
    bars_knn = ax.bar(
        x + width/2,
        knn_vals,
        width,
        label="KNN",
        color="lawngreen"
    )

    xticklabels = [
        r"$E_\mathrm{form}$ (MAE)",
        r"$E_\mathrm{form}$ (RMSE)",
        r"$E_\mathrm{hull}$ (MAE)",
        r"$E_\mathrm{hull}$ (RMSE)",
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(xticklabels, rotation=20, ha="right")

    ax.set_ylabel("Error (eV/atom)")
    ax.set_title("Imputation Performance on 1000-Structure COD–MP Subset")
    ax.legend()

    # Add a bit of headroom so labels don't touch the top
    y_max = max(max(mean_vals), max(knn_vals))
    ax.set_ylim(0, y_max * 1.15)

    # Add value labels on top of each bar
    def add_labels(bar_container):
        for bar in bar_container:
            height = bar.get_height()
            ax.annotate(
                f"{height:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    add_labels(bars_mean)
    add_labels(bars_knn)

    fig.tight_layout()
    out_path = OUTDIR / "imputation_bar_1000.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[INFO] Saved {out_path}")
