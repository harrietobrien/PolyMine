"""Plot a 2D embedding CSV produced by src/main.py --embed-out"""

from __future__ import annotations
import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")  # ensure non-GUI backend
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors


def read_embedding(path: Path) -> \
    Tuple[List[float], List[float], List[int]]:
    xs: List[float] = []
    ys: List[float] = []
    clusters: List[int] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                x = float(row.get("embed_x", ""))
                y = float(row.get("embed_y", ""))
            except Exception:
                continue
            try:
                cid = int(row.get("cluster", "-1"))
            except Exception:
                cid = -1
            xs.append(x)
            ys.append(y)
            clusters.append(cid)
    return xs, ys, clusters


def plot_embedding(xs: List[float], ys: List[float], 
                   clusters: List[int], title: str, 
                   out: Path, dpi: int) -> None:
    plt.figure(figsize=(8, 6))
    cmap = plt.get_cmap("tab10")
    unique = sorted(set(clusters))
    for cid in unique:
        cx = [x for x, c in zip(xs, clusters) if c == cid]
        cy = [y for y, c in zip(ys, clusters) if c == cid]
        label = f"cluster {cid}" if cid >= 0 else "noise (-1)"
        color = cmap(cid % 10) if cid >= 0 else "#999999"
        plt.scatter(cx, cy, s=12, alpha=0.7, label=label, 
                    color=color, edgecolor="none")
    plt.xlabel("embed_x")
    plt.ylabel("embed_y")
    plt.title(title)
    plt.legend(markerscale=2, fontsize=8, frameon=False)
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=dpi)
    plt.close()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Plot PolyMine embedding CSV.")
    ap.add_argument("--embed-csv", type=Path, required=True, help="CSV from main.py --embed-out")
    ap.add_argument("--out", type=Path, default=Path("data/clean/embed_plot.png"), 
                    help="Output image path.")
    ap.add_argument("--dpi", type=int, default=150, help="Output image DPI.")
    ap.add_argument("--title", type=str, default="PolyMine embedding", 
                    help="Plot title (method appended automatically if present).")
    ap.add_argument(
        "--facet-by-formula",
        action="store_true",
        help="Facet by formula (top N formulas by count).",
    )
    ap.add_argument(
        "--facet-top",
        type=int,
        default=6,
        help="Number of formulas to facet (only if --facet-by-formula).",
    )
    ap.add_argument(
        "--split-by-formula",
        action="store_true",
        help="Write one plot per formula (top N) into a directory instead of one faceted figure.",
    )
    ap.add_argument(
        "--split-top",
        type=int,
        default=20,
        help="Number of formulas to split out (only if --split-by-formula).",
    )
    ap.add_argument(
        "--split-dir",
        type=Path,
        default=Path("data/clean/embed_per_formula"),
        help="Output directory for per-formula plots (only if --split-by-formula).",
    )
    ap.add_argument(
        "--cluster-hist",
        action="store_true",
        help="Also write a cluster-size histogram image next to the main plot.",
    )
    ap.add_argument(
        "--hist-out",
        type=Path,
        default=Path("data/clean/embed_cluster_hist.png"),
        help="Output path for cluster-size histogram (if --cluster-hist).",
    )
    ap.add_argument(
        "--nn-graph-formula",
        type=str,
        default=None,
        help="Optional formula/comp_signature to plot a kNN neighbor graph overlay (single facet).",
    )
    ap.add_argument(
        "--nn-formula-field",
        type=str,
        default="formula",
        help="Field to match for nn-graph selection (e.g., formula or formula_norm).",
    )
    ap.add_argument(
        "--nn-k",
        type=int,
        default=8,
        help="k for kNN graph overlay.",
    )
    return ap.parse_args()


def sanitize(name: str) -> str:
    keep = []
    for ch in name:
        if ch.isalnum() or ch in ("-", "_"):
            keep.append(ch)
        elif ch.isspace():
            keep.append("_")
    return "".join(keep) or "unknown"


def main() -> None:
    args = parse_args()
    rows = []
    with open(args.embed_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if not rows:
        raise SystemExit(f"No embedding rows found in {args.embed_csv}")

    method = rows[0].get("embed_method") or ""
    full_title = args.title
    if method:
        full_title = f"{full_title} ({method})"

    if args.split_by_formula:
        by_formula = defaultdict(list)
        for r in rows:
            by_formula[r.get("formula") or \
                       r.get("formula_norm") or \
                        "UNKNOWN"].append(r)
        top = sorted(by_formula.items(), \
                     key=lambda kv: len(kv[1]), \
                        reverse=True)[: args.split_top]
        out_dir = args.split_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        cmap = plt.get_cmap("tab10")
        for form, rlist in top:
            xs = [float(r.get("embed_x", 0)) for r in rlist]
            ys = [float(r.get("embed_y", 0)) for r in rlist]
            clusters = [int(r.get("cluster", -1)) for r in rlist]
            file_name = out_dir / f"{sanitize(form)}.png"
            plot_embedding(xs, ys, clusters, 
                           title=f"{full_title} [{form}]", 
                           out=file_name, dpi=args.dpi)
        print(f"Wrote per-formula plots to {out_dir}")

    elif args.facet_by_formula:
        by_formula = defaultdict(list)
        for r in rows:
            try:
                by_formula[r["formula"]].append(r)
            except Exception:
                continue
        top = sorted(by_formula.items(), 
                     key=lambda kv: len(kv[1]), 
                     reverse=True)[: args.facet_top]
        n = len(top)
        cols = min(3, n)
        rows_n = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows_n, cols, 
                                 figsize=(4 * cols, 3.5 * rows_n), 
                                 squeeze=False)
        cmap = plt.get_cmap("tab10")
        for ax, (form, rlist) in zip(axes.flat, top):
            xs = [float(r.get("embed_x", 0)) for r in rlist]
            ys = [float(r.get("embed_y", 0)) for r in rlist]
            clusters = [int(r.get("cluster", -1)) for r in rlist]
            unique = sorted(set(clusters))
            for cid in unique:
                cx = [x for x, c in zip(xs, clusters) if c == cid]
                cy = [y for y, c in zip(ys, clusters) if c == cid]
                label = f"{cid}" if cid >= 0 else "noise"
                color = cmap(cid % 10) if cid >= 0 else "#999999"
                ax.scatter(cx, cy, s=12, alpha=0.7, label=label, 
                           color=color, edgecolor="none")
            ax.set_title(form)
            ax.set_xlabel("embed_x")
            ax.set_ylabel("embed_y")
            ax.legend(markerscale=2, fontsize=7, frameon=False)
        for ax in axes.flat[n:]:
            ax.axis("off")
        plt.tight_layout()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.out, dpi=args.dpi)
        plt.close()
        print(f"Wrote faceted plot to {args.out}")
    else:
        xs = [float(r.get("embed_x", 0)) for r in rows]
        ys = [float(r.get("embed_y", 0)) for r in rows]
        clusters = [int(r.get("cluster", -1)) for r in rows]
        if args.nn_graph_formula:
            # filter for the requested formula
            field = args.nn_formula_field
            filt = [r for r in rows if r.get(field) == args.nn_graph_formula]
            if not filt and field != "formula_norm":
                filt = [r for r in rows if r.get("formula_norm") == args.nn_graph_formula]
            if not filt:
                formulas = sorted(set(r.get(field, "")
                                      for r in rows if r.get(field)))
                sample = ", ".join(formulas[:5])
                print(
                    f"[warn] No rows found for {field}={args.nn_graph_formula}; "
                    f"available examples: {sample} (+{max(0, len(formulas)-5)} more). Skipping nn-graph overlay."
                )
                plot_embedding(xs, ys, clusters, title=full_title, 
                               out=args.out, dpi=args.dpi)
            else:
                fx = np.array([float(r.get("embed_x", 0)) for r in filt])
                fy = np.array([float(r.get("embed_y", 0)) for r in filt])
                fcl = np.array([int(r.get("cluster", -1)) for r in filt])
                pts = np.column_stack([fx, fy])
                k = min(args.nn_k, max(1, pts.shape[0] - 1))
                nbrs = NearestNeighbors(n_neighbors=k).fit(pts)
                _, idx = nbrs.kneighbors(pts)
                cmap = plt.get_cmap("tab10")
                plt.figure(figsize=(8, 6))
                for i, neigh in enumerate(idx):
                    for j in neigh:
                        plt.plot([pts[i, 0], pts[j, 0]], [pts[i, 1], pts[j, 1]], 
                                 color="#cccccc", alpha=0.3, linewidth=0.5)
                for cid in sorted(set(fcl)):
                    mask = fcl == cid
                    color = cmap(cid % 10) if cid >= 0 else "#999999"
                    label = f"{cid}" if cid >= 0 else "noise"
                    plt.scatter(pts[mask, 0], pts[mask, 1], 
                                s=20, alpha=0.8, color=color, 
                                edgecolor="none", label=label)
                plt.xlabel("embed_x")
                plt.ylabel("embed_y")
                plt.title(f"{full_title} (kNN={k}, {field}={args.nn_graph_formula})")
                plt.legend(markerscale=2, fontsize=8, frameon=False)
                plt.tight_layout()
                args.out.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(args.out, dpi=args.dpi)
                plt.close()
        else:
            plot_embedding(xs, ys, clusters, title=full_title, 
                           out=args.out, dpi=args.dpi)
        print(f"Wrote plot to {args.out}")

    if args.cluster_hist:
        hist_out = args.hist_out
        counts = defaultdict(int)
        for r in rows:
            try:
                cid = int(r.get("cluster", -1))
            except Exception:
                cid = -1
            counts[cid] += 1
        labels = sorted(counts.keys())
        sizes = [counts[c] for c in labels]
        plt.figure(figsize=(8, 4))
        colors = ["#999999" if cid == -1 else plt.get_cmap("tab10")(cid % 10) \
                  for cid in labels]
        plt.bar([str(c) for c in labels], sizes, color=colors)
        plt.xlabel("cluster id (-1 = noise)")
        plt.ylabel("size")
        plt.title("Cluster sizes")
        plt.tight_layout()
        hist_out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(hist_out, dpi=args.dpi)
        plt.close()
        print(f"Wrote cluster histogram to {hist_out}")


if __name__ == "__main__":
    main()
