#!/usr/bin/env python

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pymatgen.core import Structure
from pymatgen.analysis.diffraction.xrd import XRDCalculator


def load_df(master_path: str | Path,
           assignments_path: str | Path) -> pd.DataFrame:
    """
    Load master table + cluster assignments and merge on cod_id
    """
    master = pd.read_csv(master_path)
    assign = pd.read_csv(assignments_path)
    df = master.merge(assign[["cod_id", "cluster_id"]],
                      on="cod_id",
                      how="inner")
    return df


def formula_tex(formula: str) -> str:
    """
    Return a LaTeX version of a formula
    """
    if formula == "CaMg(SiO3)2":
        return r"CaMg(SiO_3)_2"
    elif formula == "PrNiO3":
        return r"PrNiO_3"
    elif formula == "Si3N4":
        return r"Si_3N_4"
    else:
        return rf"\mathrm{{{formula}}}"


def xrd_overlay_for_formula(df: pd.DataFrame,
                            formula: str,
                            outdir: Path,
                            wavelength: str = "CuKa",
                            two_theta_min: float = 10.0,
                            two_theta_max: float = 80.0,
                            two_theta_step: float = 0.1) -> None:
    """
    For a reduced formula, compute mean powder XRD for each cluster
    and overlay them on a single plot

    Saves a PNG named f"xrd_{formula}_by_cluster.png" in outdir
    """
    sub = df[df["formula"] == formula].dropna(subset=["cluster_id"])
    if sub.empty:
        print(f"[WARN] No clustered entries for formula {formula}")
        return

    calc = XRDCalculator(wavelength=wavelength)
    # common 2θ grid
    two_theta = np.arange(two_theta_min, two_theta_max + two_theta_step,
                          two_theta_step)

    cluster_ids = sorted(sub["cluster_id"].unique())

    cmap_name = "rainbow"
    colors = plt.cm.get_cmap(cmap_name)(np.linspace(0, 1, len(cluster_ids)))

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, cid in enumerate(cluster_ids):
        sub_c = sub[sub["cluster_id"] == cid]

        patterns = []
        for _, row in sub_c.iterrows():
            struct = Structure.from_file(row["path"])
            pattern = calc.get_pattern(
                struct,
                two_theta_range=(two_theta_min, two_theta_max),
            )
            tt_raw = np.array(pattern.x)
            yy_raw = np.array(pattern.y)

            # interpolate onto common grid
            yy = np.interp(two_theta, tt_raw, yy_raw, left=0.0, right=0.0)
            if yy.max() > 0:
                yy = yy / yy.max()
            patterns.append(yy)

        mean_intensity = np.mean(patterns, axis=0)
        ax.plot(
            two_theta,
            mean_intensity,
            color=colors[i],
            linewidth=1.5,
            label=f"Cluster {cid}",
        )

    ax.set_xlabel(r"$2\theta$ (deg)")
    ax.set_ylabel("Normalized intensity")

    chem_tex = formula_tex(formula)
    ax.set_title(rf"$\mathbf{{{chem_tex}}}$: Powder XRD by Cluster")

    leg = ax.legend(
        title="Cluster",
        loc="upper right",
        frameon=True,
        framealpha=0.9,
    )
    plt.setp(leg.get_title(), fontweight="bold")
    fig.tight_layout()

    outname = f"xrd_{formula.replace('(', '').replace(')', '').replace(' ', '')}_by_cluster.png"
    outpath = outdir / outname
    fig.savefig(outpath, dpi=600)
    plt.close(fig)
    print(f"[INFO] Saved {outpath}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Overlay powder XRD patterns by cluster for selected formulas."
    )
    p.add_argument(
        "--master",
        type=str,
        default="data/processed/master_with_energies_1000.csv",
        help="CSV with structural features and CIF paths.",
    )
    p.add_argument(
        "--assignments",
        type=str,
        default="data/processed/polymorph_clusters_assignments_1000.csv",
        help="CSV with per-entry cluster assignments.",
    )
    p.add_argument(
        "--formulas",
        nargs="+",
        required=True,
        help="One or more reduced formulas to plot (e.g. 'CaMg(SiO3)2').",
    )
    p.add_argument(
        "--outdir",
        type=str,
        default="figures/xrd",
        help="Directory to write PNG figures.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_df(args.master, args.assignments)
    print(f"[INFO] Loaded dataframe with shape {df.shape}")

    for formula in args.formulas:
        xrd_overlay_for_formula(df, formula, outdir)


if __name__ == "__main__":
    main()
