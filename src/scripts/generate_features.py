from __future__ import annotations
import argparse
import csv
from pathlib import Path
import sys
from typing import List, Sequence, Tuple
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features import fingerprints as fp


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate extra feature CSV (SOAP/XRD/motif) for PolyMine.")
    ap.add_argument("--cif-root", type=Path, required=True, help="Root directory of CIFs.")
    ap.add_argument("--out", type=Path, default=Path("data/clean/extra_features.csv"), help="Output CSV path.")
    ap.add_argument("--resolution", type=int, default=300, help="XRD intensity resolution.")
    ap.add_argument("--n-max", type=int, default=8, help="SOAP n_max.")
    ap.add_argument("--l-max", type=int, default=6, help="SOAP l_max.")
    ap.add_argument("--r-cut", type=float, default=5.0, help="SOAP cutoff radius.")
    ap.add_argument("--limit", type=int, default=None, help="Optional limit on files processed.")
    ap.add_argument(
        "--occupancy-tol",
        type=float,
        default=1.0,
        help="CIF occupancy_tolerance passed to CifParser (lower to be stricter).",
    )
    ap.add_argument(
        "--merge-tol",
        type=float,
        default=None,
        help="CIF merge_tol passed to CifParser (increase to merge nearby sites and suppress occupancy warnings).",
    )
    ap.add_argument(
        "--no-disorder-normalize",
        action="store_true",
        help="Disable OrderDisorderedStructureTransformation (skip disorder normalization).",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    Structure, _, CifParser, OrderDisorderedStructureTransformation = \
        fp._safe_import_pymatgen()
    if Structure is None:
        raise SystemExit("pymatgen required for feature generation.")

    cif_paths = sorted(args.cif_root.rglob("*.cif"))
    if args.limit:
        cif_paths = cif_paths[: args.limit]

    # Load structs and collect global species set so SOAP dimension is stable
    structures: List[Tuple[str, object]] = []
    species: set[str] = set()
    skipped = 0
    for p in cif_paths:
        try:
            structure = fp.load_structure(
                p,
                normalize_disorder=not args.no_disorder_normalize,
                occupancy_tolerance=args.occupancy_tol,
                merge_tol=args.merge_tol,
            )
        except Exception as e:
            print(f"[warn] Skipping {p}: {e}")
            skipped += 1
            continue
        structures.append((p.stem, structure))
        try:
            species.update(str(s) for s in structure.symbol_set)
        except Exception:
            pass

    if not structures:
        raise SystemExit("No CIFs parsed successfully.")

    species_list: Sequence[str] | None = sorted(species) if species else None

    # Second pass: compute features, tracking maximum dimension for padding.
    computed: List[Tuple[str, np.ndarray | None, np.ndarray | None, np.ndarray | None]] = []
    max_dims = {"soap": 0, "xrd": args.resolution, "motif": 0}
    for entry_id, structure in structures:
        soap = fp.compute_soap(structure, species=species_list, n_max=args.n_max, l_max=args.l_max, r_cut=args.r_cut)
        if soap is not None:
            max_dims["soap"] = max(max_dims["soap"], soap.size)

        xrd = fp.compute_xrd_intensities(structure, resolution=args.resolution)
        if xrd is not None:
            max_dims["xrd"] = max(max_dims["xrd"], xrd.size)

        motif = fp.compute_motif_features(structure)
        if motif is not None:
            max_dims["motif"] = max(max_dims["motif"], motif.size)

        computed.append((entry_id, soap, xrd, motif))

    def _pad_or_truncate(arr: np.ndarray | None, target: int) -> np.ndarray:
        if target == 0:
            return np.array([], dtype=float)
        if arr is None:
            return np.zeros(target, dtype=float)
        flat = np.asarray(arr, dtype=float).ravel()
        if flat.size == target:
            return flat
        if flat.size > target:
            return flat[:target]
        out = np.zeros(target, dtype=float)
        out[: flat.size] = flat
        return out

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        total_dim = max_dims["soap"] + max_dims["xrd"] + max_dims["motif"]
        header = ["entry_id"] + [f"f{i}" for i in range(total_dim)]
        writer.writerow(header)

        for entry_id, soap, xrd, motif in computed:
            soap_vec = _pad_or_truncate(soap, max_dims["soap"])
            xrd_vec = _pad_or_truncate(xrd, max_dims["xrd"])
            motif_vec = _pad_or_truncate(motif, max_dims["motif"])
            feat = np.concatenate([v for v in (soap_vec, xrd_vec, motif_vec) if v.size], axis=0)
            if feat.size != total_dim:
                # defensive: ensure consistent row length
                feat = _pad_or_truncate(feat, total_dim)
            writer.writerow([entry_id] + feat.tolist())
    print(f"Wrote {len(structures)} structures to {args.out} (skipped {skipped})")


if __name__ == "__main__":
    main()
