"""Q
Usage:
    python src/scripts/plot_features.py \
        --csv data/clean/extra_features.csv \
            --out data/clean/extra_features_plot.png

Expects a CSV with an `entry_id` column and the remaining columns
as numeric features (e.g., output of `gen_features.py`); reduces data
to 2D with PCA or t-SNE and saves a scatter plot.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _maybe_import_pandas():
    try:
        import pandas as pd  # type: ignore

        return pd
    except Exception:
        return None


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Plot feature CSV (e.g., SOAP/XRD) using PCA or t-SNE.")
    ap.add_argument("--csv", type=Path, required=True, help="Path to feature CSV (must contain `entry_id`).")
    ap.add_argument("--out", type=Path, default=Path("data/clean/extra_features_plot.png"), help="Output plot path.")
    ap.add_argument("--method", choices=["pca", "tsne"], default="pca", help="Dimensionality reduction method.")
    ap.add_argument("--label-col", type=str, default=None, help="Optional column name to color points.")
    ap.add_argument("--standardize", action="store_true", help="Standardize features before embedding.")
    ap.add_argument("--sample", type=int, default=None, help="Optional random subsample size for plotting.")
    ap.add_argument("--top-k", type=int, default=8, help="Top contributing features to print (PCA only).")
    return ap.parse_args()


def load_table(
    path: Path, label_col: Optional[str]
) -> Tuple[List[str], np.ndarray, Optional[Iterable], List[str]]:
    """
    Load feature table. Tries pandas if available; falls back to csv module
    Returns: entry_ids, feature matrix (n x d), labels (or None), feature names
    """
    pd = _maybe_import_pandas()
    if pd:
        df = pd.read_csv(path)
        if "entry_id" not in df.columns:
            raise SystemExit("CSV must contain an `entry_id` column.")
        labels = None
        if label_col:
            if label_col not in df.columns:
                raise SystemExit(f"label column `{label_col}` not found in CSV.")
            labels = df[label_col].tolist()
        feats = df.drop(columns=["entry_id"] + ([label_col] \
                                                if label_col else []))
        return df["entry_id"].tolist(), feats.to_numpy(dtype=float), labels, feats.columns.tolist()

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        raise SystemExit("CSV is empty.")
    header = rows[0]
    if header[0] != "entry_id":
        raise SystemExit("CSV must have `entry_id` as the first column.")
    label_idx = header.index(label_col) if label_col else None
    feats: List[List[float]] = []
    entry_ids: List[str] = []
    labels: List = []
    for row in rows[1:]:
        if not row:
            continue
        entry_ids.append(row[0])
        if label_idx is not None:
            labels.append(row[label_idx])
        feature_cols = [c for i, c in enumerate(row[1:])
                        if (label_idx is None or i + 1 != label_idx)]
        feats.append([float(x) for x in feature_cols])
    feature_names = [c for i, c in enumerate(header[1:]) 
                     if (label_idx is None or i + 1 != label_idx)]
    return entry_ids, np.asarray(feats, dtype=float), labels \
    if label_idx is not None else None, feature_names


def embed(features: np.ndarray, method: str) -> Tuple[np.ndarray, Dict]:
    if method == "pca":
        from sklearn.decomposition import PCA

        pca = PCA(n_components=2)
        emb = pca.fit_transform(features)
        info = {
            "method": "pca",
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "components": pca.components_,
        }
        return emb, info
    if method == "tsne":
        from sklearn.manifold import TSNE

        emb = TSNE(n_components=2, init="pca", 
                   learning_rate="auto", n_iter=1000, 
                   perplexity=30).fit_transform(features)
        return emb, {"method": "tsne"}
    raise ValueError(f"Unknown method: {method}")


def make_plot(
    emb: np.ndarray, labels: Optional[Iterable], 
    out_path: Path, title: str, 
    axis_labels: Tuple[str, str] = ("Component 1", "Component 2")
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    if labels is None:
        ax.scatter(emb[:, 0], emb[:, 1], s=12, alpha=0.8, 
                   linewidths=0, c="#1f77b4")
    else:
        labels_list = list(labels)
        try:
            lbl = np.asarray(labels_list, dtype=float)
            sc = ax.scatter(emb[:, 0], emb[:, 1], c=lbl, 
                            s=12, alpha=0.85, cmap="viridis", 
                            linewidths=0)
            fig.colorbar(sc, ax=ax, label="label")
        except Exception:
            uniq = {v: i for i, v in enumerate(sorted(set(labels_list)))}
            colors = [uniq[v] for v in labels_list]
            sc = ax.scatter(emb[:, 0], emb[:, 1], c=colors, 
                            s=12, alpha=0.85, cmap="tab20", linewidths=0)
            cbar = fig.colorbar(sc, ax=ax, ticks=range(len(uniq)))
            cbar.ax.set_yticklabels(list(uniq.keys()))

    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.set_title(title)
    ax.grid(alpha=0.2, linestyle="--", linewidth=0.5)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    entry_ids, feats, labels, feature_names = \
        load_table(args.csv, args.label_col)
    if feats.size == 0:
        raise SystemExit("No feature columns found.")
    if args.sample and feats.shape[0] > args.sample:
        idx = np.random.choice(feats.shape[0], 
                               size=args.sample, replace=False)
        feats = feats[idx]
        if labels is not None:
            labels = [labels[i] for i in idx]
        entry_ids = [entry_ids[i] for i in idx]

    if args.standardize:
        from sklearn.preprocessing import StandardScaler

        feats = StandardScaler().fit_transform(feats)

    emb, info = embed(feats, args.method)
    if info.get("method") == "pca":
        ratio = info["explained_variance_ratio"]
        title = f"PCA of {args.csv.name} (n={len(entry_ids)}) • var=({ratio[0]:.2%}, {ratio[1]:.2%})"
        axis_labels = (f"PC1 ({ratio[0]:.1%})", f"PC2 ({ratio[1]:.1%})")
    else:
        title = f"{args.method.upper()} of {args.csv.name} (n={len(entry_ids)})"
        axis_labels = ("Component 1", "Component 2")
    make_plot(emb, labels, args.out, title, axis_labels=axis_labels)
    print(f"Saved plot to {args.out}")

    if info.get("method") == "pca":
        ratio = info["explained_variance_ratio"]
        comps = info["components"]
        top_k = max(1, min(args.top_k, len(feature_names)))
        print(f"PCA explained variance ratio: PC1={ratio[0]:.4f}, PC2={ratio[1]:.4f}")
        for i, comp in enumerate(comps):
            weights = list(zip(feature_names, comp))
            weights.sort(key=lambda x: abs(x[1]), reverse=True)
            top = weights[:top_k]
            desc = ", ".join(f"{name} ({w:+.3f})" for name, w in top)
            print(f"Top {top_k} loadings for PC{i+1}: {desc}")


if __name__ == "__main__":
    main()
