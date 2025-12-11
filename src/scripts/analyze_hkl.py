#!/usr/bin/env python
"""Extract HKL reflection summaries from CIFs and save to CSV.

Uses `HKLAnalyzer` to compute XRD patterns and emit top-N reflections per entry.

Example:
    python src/scripts/analyze_hkl.py \
        --cif-root data/raw/cif \
        --out data/clean/hkl_peaks.csv \
        --top-n 10
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features.hkl import HKLAnalyzer, peaks_to_rows


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Extract HKL peak info from CIFs.")
    ap.add_argument("--cif-root", type=Path, required=True, help="Root directory containing CIF files.")
    ap.add_argument("--out", type=Path, default=Path("data/clean/hkl_peaks.csv"), help="Output CSV.")
    ap.add_argument("--top-n", type=int, default=10, help="Top-N reflections to keep per entry (by intensity).")
    ap.add_argument(
        "--two-theta-range",
        type=float,
        nargs=2,
        default=(0.0, 90.0),
        help="Two-theta range for XRD simulation (deg).",
    )
    ap.add_argument("--limit", type=int, default=None, help="Optional limit on number of CIFs processed.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    analyzer = HKLAnalyzer(two_theta_range=tuple(args.two_theta_range))

    cif_paths = sorted(args.cif_root.rglob("*.cif"))
    if args.limit:
        cif_paths = cif_paths[: args.limit]

    rows: List[dict] = []
    for p in cif_paths:
        try:
            peaks = analyzer.peaks_from_cif(p, entry_id=p.stem, top_n=args.top_n)
        except Exception:
            continue
        rows.extend(peaks_to_rows(peaks))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["entry_id", "h", "k", "l", "d_spacing", "two_theta", "intensity"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved HKL peaks for {len(rows)} reflections from {len(cif_paths)} CIFs to {args.out}")


if __name__ == "__main__":
    main()
