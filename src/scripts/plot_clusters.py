"""Plot and summarize precomputed embeddings with cluster labels

Intended for files like data/clean/polymorph_clusters_embed.csv that already
contain `embed_x`, `embed_y`, and a `cluster` column (plus optional metadata
like `formula` / `formula_norm`)

Usage:
    python src/scripts/plot_clusters.py \
        --csv data/clean/polymorph_clusters_embed.csv \
        --out data/clean/polymorph_clusters_embed_annotated.png
"""

from __future__ import annotations
import argparse
import csv
import re
import textwrap
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _maybe_import_pandas():
    try:
        import pandas as pd  # type: ignore

        return pd
    except Exception:
        return None


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Plot an embedding CSV with cluster labels and print summaries.")
    ap.add_argument("--csv", type=Path, required=True, help="Input CSV with embed_x, embed_y, cluster columns.")
    ap.add_argument("--out", type=Path, default=Path("data/clean/cluster_plot.png"), help="Output plot path.")
    ap.add_argument("--annotate", action="store_true", help="Annotate plot with cluster IDs at centroids.")
    ap.add_argument("--sample", type=int, default=None, help="Optional random subsample for plotting.")
    ap.add_argument("--top-forms", type=int, default=3, help="Top formulas to show per cluster in the summary.")
    ap.add_argument("--top-spacegroups", type=int, default=3, help="Top spacegroups to show per cluster if present.")
    ap.add_argument(
        "--meta-csv",
        type=Path,
        default=None,
        help="Optional CSV with additional columns (e.g., spacegroup, density) to join by entry_id.",
    )
    return ap.parse_args()


def load_table(path: Path, meta_path: Path | None = None):
    """Load embedding table; prefer pandas for convenience"""
    pd = _maybe_import_pandas()
    if pd:
        df = pd.read_csv(path)
        for col in ("embed_x", "embed_y", "cluster"):
            if col not in df.columns:
                raise SystemExit(f"CSV must contain column `{col}`.")
        if meta_path:
            meta = pd.read_csv(meta_path)
            if "entry_id" not in meta.columns:
                raise SystemExit("meta CSV must contain `entry_id`.")
            df = df.merge(meta, on="entry_id", how="left", 
                          suffixes=("", "_meta"))
        return df

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise SystemExit("CSV is empty.")
    for col in ("embed_x", "embed_y", "cluster"):
        if col not in reader.fieldnames:
            raise SystemExit(f"CSV must contain column `{col}`.")
    if meta_path:
        with open(meta_path, newline="", encoding="utf-8") as fmeta:
            meta_reader = csv.DictReader(fmeta)
            meta_rows = list(meta_reader)
        meta_map = {r["entry_id"]: r for r in meta_rows if "entry_id" in r}
        for r in rows:
            if r["entry_id"] in meta_map:
                r.update(meta_map[r["entry_id"]])
    return rows


def summarize_clusters(records, top_forms: int, top_sgs: int) -> Dict[str, Dict]:
    """Return summary stats keyed by cluster label. Includes counts, centroids, top formulas, spacegroups, density stats."""
    clusters: Dict[str, Dict] = defaultdict(
        lambda: {"count": 0, "xs": [], "ys": [], 
                 "formulas": Counter(), "spacegroups": Counter(),
                   "densities": []}
    )
    for r in records:
        cluster = str(r["cluster"])
        clusters[cluster]["count"] += 1
        clusters[cluster]["xs"].append(float(r["embed_x"]))
        clusters[cluster]["ys"].append(float(r["embed_y"]))
        if "formula_norm" in r:
            val = r["formula_norm"]
        elif "formula" in r:
            val = r["formula"]
        else:
            val = None
        if val is not None:
            sval = str(val)
            if sval and sval.lower() != "nan":
                clusters[cluster]["formulas"][sval] += 1
        if "spacegroup" in r and r["spacegroup"] not in (None, "", "nan"):
            clusters[cluster]["spacegroups"][r["spacegroup"]] += 1
        if "density" in r:
            try:
                clusters[cluster]["densities"].append(float(r["density"]))
            except Exception:
                pass
    summaries: Dict[str, Dict] = {}
    for c, data in clusters.items():
        cx = float(np.mean(data["xs"])) if data["xs"] else 0.0
        cy = float(np.mean(data["ys"])) if data["ys"] else 0.0
        top = data["formulas"].most_common(top_forms) \
            if data["formulas"] else []
        top_sg = data["spacegroups"].most_common(top_sgs) \
            if data["spacegroups"] else []
        dens = data["densities"]
        dens_stats = None
        if dens:
            dens_arr = np.array(dens)
            dens_stats = {"mean": float(np.mean(dens_arr)), 
                          "min": float(np.min(dens_arr)), 
                          "max": float(np.max(dens_arr))}
        summaries[c] = {
            "count": data["count"],
            "centroid": (cx, cy),
            "top_formulas": top,
            "top_spacegroups": top_sg,
            "densities": dens_stats,
        }
    return summaries


def to_math_formula(formula: str) -> str:
    if not formula:
        return "N/A"
    clean = str(formula).replace(" ", r"\,")

    def _sub(m):
        return f"{m.group(1)}_{{{m.group(2)}}}"

    formatted = re.sub(r"([A-Za-z\)])([0-9\.]+)", _sub, clean)
    return r"$\mathrm{" + formatted + "}$"


def plot(records, summaries: Dict[str, Dict], out_path: Path, annotate: bool) -> None:
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(11, 6.5), dpi=150)
    gs = GridSpec(1, 2, width_ratios=[3, 1.3], wspace=0.2)
    ax = fig.add_subplot(gs[0])
    ax_text = fig.add_subplot(gs[1])
    labels = sorted(summaries.keys(), key=lambda x: float(x) \
                    if x.lstrip("-").isdigit() else x)
    cmap = plt.get_cmap("tab20")
    color_map = {lab: cmap(i % 20) for i, lab in enumerate(labels)}

    for lab in labels:
        xs = [float(r["embed_x"]) for r in records if str(r["cluster"]) == lab]
        ys = [float(r["embed_y"]) for r in records if str(r["cluster"]) == lab]
        ax.scatter(xs, ys, s=14, alpha=0.8, linewidths=0, c=[color_map[lab]], label=f"cluster {lab}")

    if annotate:
        for lab, info in summaries.items():
            cx, cy = info["centroid"]
            ax.text(cx, cy, lab, fontsize=9, weight="bold", 
                    ha="center", va="center")

    ax.set_xlabel(r"$\mathrm{embed}_{x}$")
    ax.set_ylabel(r"$\mathrm{embed}_{y}$")
    ax.set_title(f"Embedding with clusters from {out_path.name}")
    ax.legend(loc="lower left", bbox_to_anchor=(0.02, 0.02), framealpha=0.9)
    ax.grid(alpha=0.2, linestyle="--", linewidth=0.5)

    ax_text.axis("off")
    lines: List[str] = []
    for lab in sorted(summaries.keys(), key=lambda k: -summaries[k]["count"]):
        info = summaries[lab]
        cx, cy = info["centroid"]
        line = rf"cluster {lab}: n={info['count']}, μ=({cx:.2f},{cy:.2f})"
        lines.append(line)
        if info["top_formulas"]:
            forms = ", ".join([f"{to_math_formula(f)} (n={n})" \
                               for f, n in info["top_formulas"]])
            wrapped = textwrap.fill(forms, width=48, subsequent_indent="    ")
            lines.append("  top formulas: " + wrapped)
        if info["top_spacegroups"]:
            sgs = ", ".join([f"{sg} (n={n})" \
                             for sg, n in info["top_spacegroups"]])
            wrapped = textwrap.fill(sgs, width=48, subsequent_indent="    ")
            lines.append("  spacegroups: " + wrapped)
        if info["densities"]:
            d = info["densities"]
            lines.append(f"  density: mean={d['mean']:.3f}, range=[{d['min']:.3f}, {d['max']:.3f}]")
        lines.append("")

    ax_text.text(0.0, 1.0, "\n".join(lines), fontsize=9, va="top")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def to_records(df_or_rows) -> List[Dict[str, str]]:
    """Convert pd DataFrame or list of dict rows to list of dicts"""
    if hasattr(df_or_rows, "to_dict"):
        return df_or_rows.to_dict(orient="records")
    return df_or_rows

def main() -> None:
    args = parse_args()
    raw = load_table(args.csv, meta_path=args.meta_csv)
    records = to_records(raw)

    if args.sample and len(records) > args.sample:
        idx = np.random.choice(len(records), size=args.sample, replace=False)
        records = [records[i] for i in idx]

    summaries = summarize_clusters(records, top_forms=args.top_forms, 
                                   top_sgs=args.top_spacegroups)
    print(f"Found {len(summaries)} clusters in {len(records)} points.")
    for lab, info in summaries.items():
        cx, cy = info["centroid"]
        formulas = "; ".join([f"{f} (n={n})" \
                              for f, n in info["top_formulas"]]) or "N/A"
        sgs = "; ".join([f"{sg} (n={n})" \
                         for sg, n in info["top_spacegroups"]]) or "N/A"
        dens = info["densities"]
        dens_str = (
            f"mean={dens['mean']:.3f}, range=[{dens['min']:.3f}, {dens['max']:.3f}]" if dens else "N/A"
        )
        print(
            f"cluster {lab}: n={info['count']}, centroid=({cx:.3f}, {cy:.3f}), "
            f"top formulas: {formulas}; spacegroups: {sgs}; density: {dens_str}"
        )

    plot(records, summaries, args.out, annotate=args.annotate)
    print(f"Saved plot to {args.out}")


if __name__ == "__main__":
    main()
