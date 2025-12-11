#!/usr/bin/env python
"""
Compute SOAP fingerprints for all structures listed in master_minimal.csv

Usage (from repo root):

    # Option 1: explicitly provide species
    python src/scripts/soap_features.py \
        --master data/processed/master_minimal.csv \
        --out data/processed/soap_features.npz \
        --species Si Al O N Y

    # Option 2: let the script infer species from the formula column
    python src/scripts/soap_features.py \
        --master data/processed/master_minimal.csv \
        --out data/processed/soap_features.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from dscribe.descriptors import SOAP as DscribeSOAP
from pymatgen.core import Structure, Composition
import warnings

warnings.filterwarnings("ignore", module="pymatgen")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute SOAP fingerprints for COD structures.")
    p.add_argument("--master", type=str, required=True,
                   help="CSV with at least 'cod_id' and 'path' columns (e.g. master_minimal.csv).")
    p.add_argument("--out", type=str, required=True,
                   help="Output .npz path for SOAP features.")
    p.add_argument(
        "--species",
        nargs="+",
        default=None,
        help=(
            "List of atomic species for SOAP, e.g. Si Al O N Y. "
            "If omitted, species will be inferred from the 'formula' column."
        ),
    )
    return p.parse_args()


def infer_species_from_formulas(formulas: pd.Series) -> list[str]:
    """Infer a sorted list of element symbols from a Series of formulas."""
    elems = set()
    for f in formulas.dropna().unique():
        try:
            comp = Composition(f)
            elems.update(str(el) for el in comp.elements)
        except Exception:
            continue
    return sorted(elems)


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.master)
    if "cod_id" not in df.columns or "path" not in df.columns:
        raise ValueError("master CSV must contain 'cod_id' and 'path' columns.")

    # ---- Decide species list ----
    if args.species is None:
        if "formula" not in df.columns:
            raise ValueError(
                "No --species provided and no 'formula' column in master CSV to infer from."
            )
        species_list = infer_species_from_formulas(df["formula"])
        print(f"[INFO] Inferred species from formulas: {species_list}")
    else:
        species_list = args.species
        print(f"[INFO] Using user-specified species: {species_list}")

    if not species_list:
        raise ValueError("Species list is empty; cannot construct SOAP descriptor.")

    # ---- Construct SOAP descriptor (DScribe) ----
    soap = DscribeSOAP(
        species_list,    # species (must be first in your DScribe version)
        5.0,             # rcut
        8,               # nmax
        6,               # lmax
        0.5,             # sigma
        periodic=True,
        average="outer",
        sparse=False,
    )

    features = []
    ids = []

    # ---- Loop over structures ----
    for _, row in df.iterrows():
        path = row["path"]
        cod_id = row["cod_id"]

        try:
            struct = Structure.from_file(path)
        except Exception as e:
            print(f"[WARN] Could not reload {path} for SOAP: {e}")
            continue

        try:
            atoms = struct.to_ase()
            v = soap.create(atoms)          # shape: (n_centers, n_features) or (1, n_features)
            v = np.atleast_2d(v)
            v_mean = v.mean(axis=0)         # ensure one vector per structure
        except Exception as e:
            print(f"[WARN] SOAP failed for {path}: {e}")
            continue

        ids.append(cod_id)
        features.append(v_mean)

    if not features:
        raise RuntimeError("No SOAP features were computed; all structures failed?")

    X = np.vstack(features)
    ids = np.array(ids)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, cod_id=ids, X=X)

    print(f"[INFO] Saved SOAP features for {len(ids)} structures to {out_path}")
    print(f"[INFO] Feature matrix shape: {X.shape}")


if __name__ == "__main__":
    main()
