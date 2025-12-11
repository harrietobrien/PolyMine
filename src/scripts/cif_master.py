#!/usr/bin/env python
"""
Usage (from repo root):

    python -m analysis.build_master \
        --cif-root data/raw/cif \
        --out data/processed/master.csv \
        --limit 1000
"""

from __future__ import annotations

import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import warnings

warnings.filterwarnings("ignore", module="pymatgen")


class CifMasterBuilder:
    """
    Walk a CIF tree, extract structural + symmetry features, and
    write them to a CSV (master.csv)

    Main public methods:
      - build()       -> populate internal records list
      - to_dataframe()
      - to_csv(path)
    """

    METHOD_MAP: dict[str, str] = {
        "get_space_group_symbol": "sg_symbol",
        "get_space_group_number": "sg_number",
        "get_hall": "hall_symbol",
    }

    def __init__(self, cif_root: str | Path = "data/raw/cif", limit: int | None = None):
        self.cif_root = Path(cif_root)
        self.limit = limit
        self.records: list[dict] = []


    def build(self) -> None:
        count = 0
        for root, _, files in os.walk(self.cif_root):
            for fname in files:
                if not fname.endswith(".cif"):
                    continue

                codid = fname.split(".")[0]
                path = Path(root) / fname

                struct = self._safe_load_struct(path)
                if not isinstance(struct, Structure):
                    print(f"[WARN] Skipping {path}: not a valid Structure (got {type(struct)})")
                    continue

                rec = self._compute_features(codid, path, struct)
                self.records.append(rec)

                count += 1
                if self.limit is not None and count >= self.limit:
                    break
            if self.limit is not None and count >= self.limit:
                break

        print(f"[INFO] Processed {count} CIF files.")


    def to_dataframe(self) -> pd.DataFrame:
        """Return the collected records as a pandas DataFrame"""
        if not self.records:
            self.build()
        return pd.DataFrame(self.records)

    def to_csv(self, out_path: str | Path) -> None:
        """Write master table to CSV."""
        df = self.to_dataframe()
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"Wrote {len(df)} rows to {out_path}")


    def _safe_load_struct(self, path):
        """
        Try to load a Structure from CIF, expanding from asymmetric.
        If it fails, try again with '?' replaced by '0'.
        Return None if still broken.

        Guarantees: returns either a pymatgen Structure or None.
        """

        # First attempt:     def _safe_load_struct(self, path):
        """
        Try to load a Structure from CIF, expanding from asymmetric.
        If it fails, try again with '?' replaced by '0'.
        Return None if still broken.
        """
        # First attempt: normal load
        try:
            struct = Structure.from_file(path)
            return self.expand_from_asymmetric(struct, 1e-3, 5.0, 1e-5)
        except Exception as e:
            print(f"[WARN] Parse failed for {path}: {e}")

        # Second attempt: read text, replace '?', then parse from string
        try:
            with open(path, "r", errors="ignore") as f:
                text = f.read().replace("?", "0")
            struct = Structure.from_str(text, fmt="cif")
            return self.expand_from_asymmetric(struct, 1e-3, 5.0, 1e-5)
        except Exception as e2:
            print(f"[WARN] Second attempt failed for {path}: {e2}")
            struct = None
        try:
            struct = Structure.from_file(path)
            struct = self.expand_from_asymmetric(struct, 1e-3, 5.0, 1e-5)
        except Exception as e:
            print(f"[WARN] Parse failed for {path}: {e}")
            struct = None

        # Second attempt if needed
        if not isinstance(struct, Structure):
            try:
                with open(path, "r", errors="ignore") as f:
                    text = f.read().replace("?", "0")
                struct = Structure.from_str(text, fmt="cif")
                struct = self.expand_from_asymmetric(struct, 1e-3, 5.0, 1e-5)
            except Exception as e2:
                print(f"[WARN] Second attempt failed for {path}: {e2}")
                struct = None

        if not isinstance(struct, Structure):
            return None
        return struct


    def _compute_features(self, codid, path, struct):
        rec = {
            "cod_id": codid,
            "path": str(path),
            "formula": struct.composition.reduced_formula,
            "n_sites": len(struct),
            "n_species": len(struct.composition.elements),
            "volume": struct.volume,
            "density": struct.density,
            "a": struct.lattice.a,
            "b": struct.lattice.b,
            "c": struct.lattice.c,
            "alpha": struct.lattice.alpha,
            "beta": struct.lattice.beta,
            "gamma": struct.lattice.gamma,
        }

        try:
            analyzer = SpacegroupAnalyzer(struct)
            rec["sg_symbol"] = analyzer.get_space_group_symbol()
            rec["sg_number"] = analyzer.get_space_group_number()
            rec["hall_symbol"] = analyzer.get_hall()
        except Exception as e:
            print(f"[WARN] Symmetry undetermined for {codid}: {e}")
            rec["sg_symbol"] = np.nan
            rec["sg_number"] = np.nan
            rec["hall_symbol"] = np.nan

        return rec



    @staticmethod
    def expand_from_asymmetric(
        self,
        struct: Structure,
        symprec: float = 1e-3,
        angle_tolerance: float = 5.0,
        dedup_tol: float = 1e-5,
    ) -> Structure:
        """
        Build the full unit cell by applying spglib symmetry operations.
        If symmetry fails for any reason, just return the original struct.
        """
        try:
            sga = SpacegroupAnalyzer(struct, symprec=symprec, 
                                     angle_tolerance=angle_tolerance)
            data = sga.get_symmetry_dataset()
        except Exception as e:
            print(f"[WARN] expand_from_asymmetric: symmetry dataset failed, returning original struct: {e}")
            return struct

        rotations = data["rotations"]
        translations = data["translations"]

        species = []
        frac_coords = []

        def bucket_key(frac):
            wrapped = frac - np.floor(frac)
            return tuple(np.round(wrapped / dedup_tol).astype(int).tolist())

        seen = set()

        for site in struct.sites:
            f = np.asarray(site.frac_coords, float)
            for R, t in zip(rotations, translations):
                f_new = (R @ f + t).astype(float)
                f_new -= np.floor(f_new)
                key = (site.species_string, bucket_key(f_new))
                if key in seen:
                    continue
                seen.add(key)
                species.append(site.species)
                frac_coords.append(f_new)

        return Structure(struct.lattice, species, frac_coords,
                          coords_are_cartesian=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build master.csv table from COD CIF files."
    )
    parser.add_argument(
        "--cif-root",
        type=str,
        default="data/raw/cif",
        help="Root directory containing .cif files.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="data/processed/master.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of CIFs to process (testing).",
    )
    args = parser.parse_args()

    builder = CifMasterBuilder(cif_root=args.cif_root, limit=args.limit)
    builder.to_csv(args.out)


if __name__ == "__main__":
    main()
