##!/usr/bin/env python
"""
Usage (from repo root):

    python src/scripts/cif_minimal.py \
        --cif-root data/raw/cif \
        --out data/processed/master_minimal.csv \
        --limit 1000
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

import numpy as np  # (still unused, but kept in case you extend features)
import pandas as pd
from pymatgen.core import Structure
from tqdm.auto import tqdm
import warnings

warnings.filterwarnings("ignore", module="pymatgen")


class SimpleCIFTable:
    def __init__(self, cif_root: str | Path, limit: Optional[int] = None):
        self.cif_root = Path(cif_root)
        self.limit = limit
        self.records: list[dict] = []

    def build(self) -> None:
        ok = 0
        parse_fail = 0
        type_fail = 0

        # Collect (codid, path) pairs so tqdm knows the total
        cif_paths: list[tuple[str, Path]] = []
        for root, _, files in os.walk(self.cif_root):
            for fname in files:
                if not fname.endswith(".cif"):
                    continue
                codid = fname.split(".")[0]
                path = Path(root) / fname
                cif_paths.append((codid, path))

        if not cif_paths:
            print(f"[WARN] No .cif files found under {self.cif_root}")
            return

        iterator = tqdm(
            cif_paths,
            desc="Processing CIFs",
            unit="cif",
            leave=False,
        )

        for codid, path in iterator:
            # stop after collecting requested # of valid structures
            if self.limit is not None and ok >= self.limit:
                break

            struct = self._safe_load_struct(path)
            if struct is None:
                parse_fail += 1
                continue
            if not isinstance(struct, Structure):
                type_fail += 1
                continue

            # struct is a valid pymatgen Structure
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
            self.records.append(rec)
            ok += 1

            # update progress bar description
            iterator.set_postfix(ok=ok, parse_fail=parse_fail, type_fail=type_fail)

        iterator.close()
        print(f"[INFO] OK={ok}, parse_fail={parse_fail}, type_fail={type_fail}")

    def to_dataframe(self) -> pd.DataFrame:
        if not self.records:
            self.build()
        return pd.DataFrame(self.records)

    def to_csv(self, out_path: str | Path) -> None:
        df = self.to_dataframe()
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"[INFO] Wrote {len(df)} rows to {out_path}")

    def _safe_load_struct(self, path: Path) -> Structure | None:
        # first attempt
        try:
            struct = Structure.from_file(path)
        except Exception:
            struct = None

        # second attempt if needed
        if not isinstance(struct, Structure):
            try:
                with open(path, "r", errors="ignore") as f:
                    text = f.read().replace("?", "0")
                struct = Structure.from_str(text, fmt="cif")
            except Exception:
                struct = None

        if not isinstance(struct, Structure):
            return None

        return struct


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Minimal CIF → CSV builder.")
    p.add_argument(
        "--cif-root",
        type=str,
        default="data/raw/cif",
        help="Root directory with COD .cif files.",
    )
    p.add_argument(
        "--out",
        type=str,
        default="data/processed/master_minimal.csv",
        help="Output CSV path.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of valid CIFs to include (for testing).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tbl = SimpleCIFTable(cif_root=args.cif_root, limit=args.limit)
    tbl.to_csv(args.out)


if __name__ == "__main__":
    main()
