#!/usr/bin/env python
"""
Global volume–density scatter 

Usage example (from repo root):

    python src/scripts/plot_global_vol_density.py \
        --master data/processed/master_with_energies_1000.csv \
        --out figures/scatter_global_vol_density.png
"""

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot global volume vs density for all structures."
    )
    p.add_argument(
        "--master",
        type=str,
        required=True,
        help="Path to master_with_energies CSV (e.g., 1000-row dataset).",
    )
    p.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output PNG path for the scatter plot.",
    )
    p.add_argument(
        "--point-size",
        type=float,
        default=25.0,
        help="Marker size for scatter points (default: 25).",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=0.7,
        help="Point transparency (default: 0.7).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    master_path = Path(args.master)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(master_path)
    print(f"[INFO] Loaded master table with shape {df.shape} from {master_path}")

    if not {"volume", "density"} <= set(df.columns):
        raise ValueError("Input CSV must contain 'volume' and 'density' columns.")

    sub = df.dropna(subset=["volume", "density"])
    print(f"[INFO] Plotting {len(sub)} rows with non-missing volume & density")

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.scatter(
        sub["volume"],
        sub["density"],
        s=args.point_size,
        alpha=args.alpha,
        edgecolor="none",
    )

    ax.set_xlabel(r"Volume ($\mathrm{\AA^3}$)")
    ax.set_ylabel(r"Density (g/cm$^3$)")
    ax.set_title("COD–MP Subset: Volume vs Density")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[INFO] Saved global volume–density scatter to {out_path}")


if __name__ == "__main__":
    main()
