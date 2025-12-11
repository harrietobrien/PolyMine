"""PolyMine CLI: orchestrates parsing/clustering using modular components."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Optional

from clustering.dbscan import cluster_family, dbscan
from clustering.advanced import spectral_cluster, try_hdbscan, try_tsne, try_umap
from clustering.multiview import build_feature_matrix, pca_2d
from features.extract import (
    Entry,
    cell_volume,
    comp_signature,
    discover_cifs,
    load_atomic_masses,
    parse_all,
    feature_vector,
    feature_vector_lengths,
    set_atomic_masses,
)
from features.extra_view import load_extra_view


def write_header(writer: csv.writer) -> None:
    writer.writerow(
        [
            "entry_id",
            "formula",
            "formula_norm",
            "cluster",
            "spacegroup",
            "a",
            "b",
            "c",
            "alpha",
            "beta",
            "gamma",
            "cell_volume",
            "density",
            "z_value",
            "volume_per_fu",
            "atom_count",
            "formula_mass",
            "comp_signature",
            "elements",
            "cif_path",
        ]
    )


def write_entry(writer: csv.writer, ent: Entry, comp_sig: str) -> None:
    vol = cell_volume(ent.cell) if ent.cell else None
    vol_per_fu = None
    if vol is not None and ent.z_value and ent.z_value > 0:
        vol_per_fu = vol / ent.z_value
    writer.writerow(
        [
            ent.entry_id,
            ent.formula,
            comp_sig,
            ent.cluster_id if ent.cluster_id is not None else -1,
            ent.sgnum if ent.sgnum is not None else "",
            *(ent.cell if ent.cell else ("",) * 6),
            vol if vol is not None else "",
            ent.density if ent.density is not None else "",
            ent.z_value if ent.z_value is not None else "",
            vol_per_fu if vol_per_fu is not None else "",
            ent.atom_count if ent.atom_count is not None else "",
            ent.formula_mass if ent.formula_mass is not None else "",
            comp_sig,
            ",".join(ent.elements),
            ent.cif_path,
        ]
    )


def run_pipeline(
    root: Path,
    out_csv: Path,
    limit: Optional[int],
    tol: float,
    workers: int,
    stride: int,
    eps: float,
    min_samples: int,
    extra_view_csv: Optional[Path],
    primary_weight: float,
    extra_weight: float,
    embed_out: Optional[Path],
    min_id: Optional[int],
    max_id: Optional[int],
    cluster_method: str,
    embed_method: str,
    knn: int,
    n_components: Optional[int],
    tsne_perplexity: float,
    umap_neighbors: int,
    umap_min_dist: float,
    hdbscan_min_cluster: int,
    feature_mode: str,
) -> None:
    cif_paths = discover_cifs(root, limit, stride=stride, min_id=min_id, max_id=max_id)
    if not cif_paths:
        raise SystemExit(f"No CIF files found under {root}")

    entries: List[Entry] = list(parse_all(cif_paths, workers=workers))
    feature_fn = feature_vector_lengths if feature_mode == "lengths" else feature_vector
    by_family: Dict[str, List[Entry]] = {}
    for ent in entries:
        comp_sig = comp_signature(ent.comp)
        key = comp_sig or ent.formula
        by_family.setdefault(key, []).append(ent)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        write_header(writer)

        extra_map: Dict[str, np.ndarray] = {}
        if extra_view_csv:
            extra_map = load_extra_view(extra_view_csv)
            if extra_map:
                print(f"Loaded extra view for {len(extra_map)} entries from {extra_view_csv}")
            else:
                print(f"[warn] Extra view file {extra_view_csv} could not be read or was empty; using primary features only.")

        embed_rows: List[List[object]] = []

        for key, ents in by_family.items():
            mv_matrix, mv_idx = build_feature_matrix(
                ents, extra_map, primary_weight=primary_weight, extra_weight=extra_weight, feature_fn=feature_fn
            )
            labels: List[int]
            if mv_matrix.shape[0] == len(ents) and mv_matrix.size > 0:
                if cluster_method == "hdbscan":
                    lab_arr = try_hdbscan(mv_matrix, min_cluster_size=hdbscan_min_cluster)
                    if lab_arr is None:
                        print("[warn] hdbscan not available; falling back to DBSCAN")
                        lab_arr = dbscan(mv_matrix, eps=eps, min_samples=min_samples)
                elif cluster_method == "spectral":
                    lab_arr = spectral_cluster(mv_matrix, knn=knn, k=n_components)
                else:
                    lab_arr = dbscan(mv_matrix, eps=eps, min_samples=min_samples)
                labels = lab_arr.tolist()
                for ent, cid in zip(ents, labels):
                    ent.cluster_id = cid
                if embed_out:
                    if embed_method == "tsne":
                        emb = try_tsne(mv_matrix, perplexity=tsne_perplexity)
                    elif embed_method == "umap":
                        emb = try_umap(mv_matrix, n_neighbors=umap_neighbors, min_dist=umap_min_dist)
                    else:
                        emb = pca_2d(mv_matrix)
                    for row_idx, ent_idx in enumerate(mv_idx):
                        ent = ents[ent_idx]
                        embed_rows.append(
                            [
                                ent.entry_id,
                                ent.formula,
                                comp_signature(ent.comp),
                                ent.cluster_id if ent.cluster_id is not None else -1,
                                emb[row_idx, 0],
                                emb[row_idx, 1],
                            ]
                        )
            else:
                labels = cluster_family(ents, eps=eps, min_samples=min_samples, feature_fn=feature_fn)
                for ent, cid in zip(ents, labels):
                    ent.cluster_id = cid

            for ent in ents:
                comp_sig = comp_signature(ent.comp)
                write_entry(writer, ent, comp_sig)

    print(
        f"Wrote {len(entries)} entries across {len(by_family)} formula families to {out_csv} "
        f"(eps={eps}, min_samples={min_samples}, limit={limit or 'all'}, stride={stride}, workers={workers})."
    )

    if embed_out:
        embed_out.parent.mkdir(parents=True, exist_ok=True)
        with open(embed_out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["entry_id", "formula", "formula_norm", "cluster", "embed_x", "embed_y", "embed_method"])
            for row in embed_rows:
                w.writerow(row + [embed_method])
        print(f"Wrote embedding to {embed_out}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Minimal PolyMine prototype (CIF → polymorph clustering).")
    ap.add_argument(
        "--cif-root",
        type=Path,
        default=Path("data/raw/cif"),
        help="Root directory containing CIF files (scanned recursively).",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Optional cap on the number of CIFs to process (0/None = all).",
    )
    ap.add_argument(
        "--tol",
        type=float,
        default=0.08,
        help="Clustering tolerance in normalized feature space (lower = tighter groups).",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("data/clean/polymorph_clusters.csv"),
        help="Output CSV path for clustered metadata.",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel workers for CIF parsing (use 1 to disable multiprocessing/threads).",
    )
    ap.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Take every Nth CIF for quick sampling of large datasets.",
    )
    ap.add_argument(
        "--atomic-mass-csv",
        type=Path,
        default=None,
        help="Optional CSV/TSV with element symbol and atomic mass (first two columns). Overrides defaults.",
    )
    ap.add_argument(
        "--eps",
        type=float,
        default=0.15,
        help="DBSCAN epsilon (distance threshold) for clustering within a formula family.",
    )
    ap.add_argument(
        "--min-samples",
        type=int,
        default=2,
        help="DBSCAN min_samples for clustering within a formula family.",
    )
    ap.add_argument(
        "--extra-view-csv",
        type=Path,
        default=None,
        help="Optional CSV/TSV with extra features per entry_id (first column entry_id, rest numeric features).",
    )
    ap.add_argument(
        "--primary-weight",
        type=float,
        default=1.0,
        help="Weight applied to primary (cell/sg/density) features when combining views.",
    )
    ap.add_argument(
        "--extra-weight",
        type=float,
        default=1.0,
        help="Weight applied to extra-view features when combining views.",
    )
    ap.add_argument(
        "--embed-out",
        type=Path,
        default=None,
        help="Optional CSV path to write a 2D PCA embedding (for plotting) of the combined feature space.",
    )
    ap.add_argument(
        "--min-id",
        type=int,
        default=1_000_000,
        help="Minimum CIF numeric stem to include (inclusive).",
    )
    ap.add_argument(
        "--max-id",
        type=int,
        default=1_999_999,
        help="Maximum CIF numeric stem to include (inclusive).",
    )
    ap.add_argument(
        "--cluster-method",
        choices=["dbscan", "hdbscan", "spectral"],
        default="dbscan",
        help="Clustering method per family.",
    )
    ap.add_argument(
        "--embed-method",
        choices=["pca", "tsne", "umap"],
        default="pca",
        help="Embedding method for --embed-out.",
    )
    ap.add_argument(
        "--knn",
        type=int,
        default=5,
        help="kNN size for spectral clustering.",
    )
    ap.add_argument(
        "--n-components",
        type=int,
        default=None,
        help="Target cluster count for spectral clustering (optional).",
    )
    ap.add_argument(
        "--tsne-perplexity",
        type=float,
        default=30.0,
        help="t-SNE perplexity (only if embed-method=tsne).",
    )
    ap.add_argument(
        "--umap-neighbors",
        type=int,
        default=15,
        help="UMAP n_neighbors (only if embed-method=umap).",
    )
    ap.add_argument(
        "--umap-min-dist",
        type=float,
        default=0.1,
        help="UMAP min_dist (only if embed-method=umap).",
    )
    ap.add_argument(
        "--hdbscan-min-cluster",
        type=int,
        default=2,
        help="Min cluster size for HDBSCAN (if available).",
    )
    ap.add_argument(
        "--feature-mode",
        choices=["full", "lengths"],
        default="full",
        help="Use full feature vector or length-only (a,b,c normalized).",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    limit = args.limit if args.limit and args.limit > 0 else None

    if args.atomic_mass_csv:
        masses = load_atomic_masses(args.atomic_mass_csv)
        if masses:
            set_atomic_masses(masses)
            print(f"Loaded {len(masses)} atomic masses from {args.atomic_mass_csv}")
        else:
            print(f"[warn] Failed to load atomic masses from {args.atomic_mass_csv}; using built-in defaults.")

    run_pipeline(
        root=args.cif_root,
        out_csv=args.out,
        limit=limit,
        tol=args.tol,  # retained for backwards compat (unused in DBSCAN)
        workers=max(1, args.workers),
        stride=max(1, args.stride),
        eps=args.eps,
        min_samples=max(1, args.min_samples),
        extra_view_csv=args.extra_view_csv,
        primary_weight=args.primary_weight,
        extra_weight=args.extra_weight,
        embed_out=args.embed_out,
        min_id=args.min_id,
        max_id=args.max_id,
        cluster_method=args.cluster_method,
        embed_method=args.embed_method,
        knn=args.knn,
        n_components=args.n_components,
        tsne_perplexity=args.tsne_perplexity,
        umap_neighbors=args.umap_neighbors,
        umap_min_dist=args.umap_min_dist,
        hdbscan_min_cluster=args.hdbscan_min_cluster,
        feature_mode=args.feature_mode,
    )


if __name__ == "__main__":
    main()
