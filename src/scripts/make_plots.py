#!/usr/bin/env python
"""
- Histograms of MP energies (E_form, E_hull).
- Observed vs missing energy bar chart.
- Volume vs density by cluster for selected formulas
   (CaMg(SiO3)2, PrNiO3, Si3N4).

Assumes the 1000-structure subset and clustering outputs:
  - data/processed/master_with_energies_1000.csv
  - data/processed/polymorph_clusters_assignments_1000.csv
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


MASTER = Path("data/processed/master_with_energies_1000.csv")
ASSIGN = Path("data/processed/polymorph_clusters_assignments_1000.csv")
OUTDIR = Path("figures")
OUTDIR.mkdir(parents=True, exist_ok=True)


def load_data():
    master = pd.read_csv(MASTER)
    assign = pd.read_csv(ASSIGN)
    df = master.merge(assign[["cod_id", "cluster_id"]], 
                      on="cod_id", how="left")
    return df


def plot_energy_histograms(df: pd.DataFrame):
    """Make separate histograms for E_form and E_hull
       E_hull uses a broken y-axis so small bins are visible"""
    e_form = df["mp_e_form_per_atom"].dropna()
    e_hull = df["mp_e_above_hull"].dropna()

    # Formation energy histogram
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(e_form, bins=40, color="m")
    ax.set_title(r"Histogram of $E_\mathrm{form}$")
    ax.set_xlabel(r"$E_\mathrm{form}$ (eV/atom)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(OUTDIR / "hist_E_form.png", dpi=600)
    plt.close(fig)

    bins = np.linspace(e_hull.min(), e_hull.max(), 40)

    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1,
        sharex=True,
        figsize=(6, 5),
        gridspec_kw={"height_ratios": [1, 2]},
    )

    ax_top.hist(e_hull, bins=bins, color="c")
    ax_bottom.hist(e_hull, bins=bins, color="c")


    max_count = ax_top.get_ylim()[1]
    ax_top.set_ylim(max_count * 0.6, max_count)

    # Bottom: zoom in near zero counts
    ax_bottom.set_ylim(0, max_count * 0.15)

    ax_top.spines["bottom"].set_visible(False)
    ax_bottom.spines["top"].set_visible(False)
    ax_top.tick_params(labelbottom=False)

    d = 0.5 # size of break mark
    kwargs = dict(
        marker=[(-1, -d), (1, -d), (0, d)],
        markersize=4,
        linestyle="none",
        color="k",
        mec="k",
        mew=1,
        clip_on=False,
    )
    # Top plot break
    ax_top.plot([0, 1], [0, 0], transform=ax_top.transAxes, **kwargs)
    # Bottom plot break
    ax_bottom.plot([0, 1], [1, 1], transform=ax_bottom.transAxes, **kwargs)

    # Labels / title on bottom axis
    ax_bottom.set_xlabel(r"$E_\mathrm{hull}$ (eV/atom)")
    ax_top.set_ylabel("Count")
    ax_bottom.set_ylabel("Count")
    fig.suptitle(r"Histogram of $E_\mathrm{hull}$", y=0.97)

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.05)
    fig.savefig(OUTDIR / "hist_E_hull_broken.png", dpi=600)
    plt.close(fig)



def plot_missingness_bar(df: pd.DataFrame):
    """Bar chart of observed vs missing MP energies"""
    totals = len(df)
    n_form_obs = df["mp_e_form_per_atom"].notna().sum()
    n_hull_obs = df["mp_e_above_hull"].notna().sum()

    observed = [n_form_obs, n_hull_obs]
    missing = [totals - n_form_obs, totals - n_hull_obs]

    labels = [r"$E_\mathrm{form}$", r"$E_\mathrm{hull}$"]
    x = range(len(labels))

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.bar([i - 0.15 for i in x], observed, width=0.3, label="Observed")
    ax.bar([i + 0.15 for i in x], missing, width=0.3, label="Missing")

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Count")
    ax.set_title("Observed vs Missing MP Energies")
    ax.legend()

    fig.tight_layout()
    fig.savefig(OUTDIR / "mp_energy_missingness.png", dpi=300)
    plt.close(fig)


def scatter_volume_density_by_cluster(df: pd.DataFrame, formula: str, fname: str):
    """Scatter plot of volume vs density colored by cluster for one formula"""
    sub = df[df["formula"] == formula].copy()
    sub = sub.dropna(subset=["cluster_id"])

    if sub.empty:
        print(f"[WARN] No clustered entries for formula {formula}")
        return

    fig, ax = plt.subplots(figsize=(6, 5))

    # Pretty LaTeX for formula
    if formula == "CaMg(SiO3)2":
        chem = r"CaMg(SiO_3)_2"
    elif formula == "PrNiO3":
        chem = r"PrNiO_3"
    elif formula == "Si3N4":
        chem = r"Si_3N_4"
    else:
        chem = formula.replace("_", r"\_")

    ax.set_title(rf"$\mathbf{{{chem}}}$: Volume vs Density by Cluster")

    ax.set_xlabel(r"Volume ($\mathrm{\AA^3}$)")
    ax.set_ylabel(r"Density (g/cm$^3$)")

    cluster_ids = sorted(sub["cluster_id"].unique())
    cmap = plt.cm.get_cmap("gist_rainbow", len(cluster_ids))

    for i, cid in enumerate(cluster_ids):
        mask = sub["cluster_id"] == cid
        ax.scatter(
            sub.loc[mask, "volume"],
            sub.loc[mask, "density"],
            color=cmap(i),
            label=f"Cluster {int(cid)}",
        )

    # Legend placement: special-case CaMg(SiO3)2
    if formula == "CaMg(SiO3)2":
        leg = ax.legend(
            title="Cluster",
            loc="upper right",
            framealpha=0.9,
            borderaxespad=0.5,
        )
    else:
        leg = ax.legend(
            title="Cluster",
            loc="center",
            bbox_to_anchor=(0.5, 0.5),
            framealpha=0.9,
            borderaxespad=0.0,
        )

    leg.get_title().set_fontweight("bold")

    fig.tight_layout()
    fig.savefig(OUTDIR / fname, dpi=600)
    plt.close(fig)


def main():
    df = load_data()
    print(f"[INFO] Loaded dataframe with shape {df.shape}")

    plot_energy_histograms(df)
    print("[INFO] Saved hist_mp_energies.png")

    # plot_missingness_bar(df)
    # print("[INFO] Saved mp_energy_missingness.png")

    # scatter_volume_density_by_cluster(df, "CaMg(SiO3)2", "scatter_CaMgSiO32_vol_density.png")
    # scatter_volume_density_by_cluster(df, "PrNiO3", "scatter_PrNiO3_vol_density.png")
    # scatter_volume_density_by_cluster(df, "Si3N4", "scatter_Si3N4_vol_density.png")
    # print("[INFO] Saved volume–density scatter plots for selected formulas")


if __name__ == "__main__":
    main()
