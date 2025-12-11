import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--master", default="data/processed/master_with_energies_1000.csv")
    p.add_argument("--out", default="figures/embed_1000_tsne_eform.png")
    args = p.parse_args()

    df = pd.read_csv(args.master)

    struct_feats = ["volume","density","a","b","c","alpha","beta","gamma","n_sites","n_species"]
    comp_feats = [c for c in df.columns if c.startswith("el_") and c.endswith("_frac")]
    feats = struct_feats + comp_feats

    mask = df["mp_e_form_per_atom"].notna()
    df_use = df[mask].reset_index(drop=True)

    X = df_use[feats].values
    X = StandardScaler().fit_transform(X)

    emb = TSNE(
        n_components=2,
        perplexity=30,
        random_state=0,
        init="random",
        learning_rate="auto",
    ).fit_transform(X)

    e_form = df_use["mp_e_form_per_atom"].values

    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(
        emb[:, 0],
        emb[:, 1],
        s=18,
        c=e_form,
        cmap="viridis",  # change to any colormap you like
        edgecolors="none",
        alpha=0.85,
    )
    ax.set_xlabel(r"embed$_x$")
    ax.set_ylabel(r"embed$_y$")
    ax.set_title("COD–MP subset: t-SNE colored by $E_\\mathrm{form}$")

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(r"$E_\mathrm{form}$ (eV/atom)")

    fig.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    fig.savefig(args.out, dpi=300)

if __name__ == "__main__":
    main()
