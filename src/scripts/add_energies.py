#!/usr/bin/env python
"""
Augment master_minimal.csv with Materials Project / OQMD energies.

Input:  CSV with at least columns:
    cod_id, path, formula, n_sites, n_species, volume, density, a, b, c, alpha, beta, gamma

Output: Same rows + energy-related columns:
    mp_id, mp_e_form_per_atom, mp_e_above_hull,
    oqmd_best_id, oqmd_delta_e, oqmd_stability  (if OQMD available)

Usage (from repo root):

    export MP_API_KEY="YOUR_MP_KEY"   # required for MP

    python src/scripts/add_energies.py \
        --in data/processed/master_minimal.csv \
        --out data/processed/master_with_energies.csv
"""

from __future__ import annotations
import argparse
import os
import time
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore", module="pymatgen")

try:
    from mp_api.client import MPRester
except ImportError:
    MPRester = None

try:
    import qmpy_rester as qr
except ImportError:
    qr = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Add MP/OQMD energies to master_minimal.csv.")
    p.add_argument(
        "--in",
        dest="in_path",
        type=str,
        required=True,
        help="Input CSV (e.g. data/processed/master_minimal.csv).",
    )
    p.add_argument(
        "--out",
        dest="out_path",
        type=str,
        required=True,
        help="Output CSV (e.g. data/processed/master_with_energies.csv).",
    )
    p.add_argument(
        "--sleep-mp",
        type=float,
        default=0.1,
        help="Sleep (s) between MP queries to be polite.",
    )
    p.add_argument(
        "--sleep-oqmd",
        type=float,
        default=0.1,
        help="Sleep (s) between OQMD queries to be polite.",
    )
    p.add_argument(
        "--no-oqmd",
        action="store_true",
        help="Disable OQMD queries (only use Materials Project).",
    )
    p.add_argument(
        "--limit-formulas",
        type=int,
        default=None,
        help="Optional limit on the number of unique formulas to query (for testing).",
    )
    return p.parse_args()


def get_mp_energies_for_formula(
    formula: str,
    mpr: Optional[MPRester],
    sleep: float,
    cache: Dict[str, dict],
) -> dict:
    """
    Cached query to Materials Project for a given formula
    Returns a dict with keys:
        mp_id, mp_e_form_per_atom, mp_e_above_hull
    """
    if formula in cache:
        return cache[formula]

    empty = {
        "mp_id": np.nan,
        "mp_e_form_per_atom": np.nan,
        "mp_e_above_hull": np.nan,
    }

    if mpr is None:
        cache[formula] = empty
        return empty

    try:
        docs = mpr.materials.summary.search(
            formula=formula,
            fields=[
                "material_id",
                "formula_pretty",
                "formation_energy_per_atom",
                "energy_above_hull",
            ],
        )
    except Exception as e:
        print(f"[WARN] MP query failed for {formula}: {e}")
        time.sleep(sleep)
        cache[formula] = empty
        return empty

    time.sleep(sleep)

    if not docs:
        cache[formula] = empty
        return empty

    def e_hull(doc):
        val = getattr(doc, "energy_above_hull", None)
        try:
            return float(val)
        except (TypeError, ValueError):
            return 1e9

    best = min(docs, key=e_hull)

    rec = {
        "mp_id": str(best.material_id),
        "mp_e_form_per_atom": getattr(best, "formation_energy_per_atom", np.nan),
        "mp_e_above_hull": getattr(best, "energy_above_hull", np.nan),
    }
    cache[formula] = rec
    return rec

# not implemented
def get_oqmd_energies_for_formula(
    formula: str,
    sleep: float,
    cache: Dict[str, dict],
) -> dict:
    """
    Cached query to OQMD for a given formula.
    Returns a dict with keys:
        oqmd_best_id, oqmd_delta_e, oqmd_stability
    """
    if formula in cache:
        return cache[formula]

    empty = {
        "oqmd_best_id": np.nan,
        "oqmd_delta_e": np.nan,
        "oqmd_stability": np.nan,
    }

    if qr is None:
        cache[formula] = empty
        return empty

    try:
        with qr.QMPYRester() as q:
            data = q.get_oqmd_phases(
                composition=formula,
                fields="id,delta_e,stability",
                limit=200,
            )
    except Exception as e:
        print(f"[WARN] OQMD query failed for {formula}: {e}")
        time.sleep(sleep)
        cache[formula] = empty
        return empty

    time.sleep(sleep)

    if not data:
        cache[formula] = empty
        return empty

    def stability(rec):
        try:
            return float(rec.get("stability", 1e9))
        except (TypeError, ValueError):
            return 1e9

    best = min(data, key=stability)

    rec = {
        "oqmd_best_id": best.get("id", np.nan),
        "oqmd_delta_e": best.get("delta_e", np.nan),
        "oqmd_stability": best.get("stability", np.nan),
    }
    cache[formula] = rec
    return rec


def main() -> None:
    args = parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)

    if not in_path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {in_path}")

    df = pd.read_csv(in_path)
    print(f"[INFO] Loaded {df.shape[0]} rows from {in_path}")

    if "formula" not in df.columns:
        raise ValueError("Input CSV must contain a 'formula' column.")

    formulas = df["formula"].dropna().unique().tolist()
    formulas.sort()

    if args.limit_formulas is not None:
        formulas = formulas[: args.limit_formulas]

    print(f"[INFO] Unique formulas to query: {len(formulas)}")

    # Set up MP client
    mp_key = os.environ.get("MP_API_KEY", None)
    if mp_key is None:
        print("[WARN] MP_API_KEY not set; MP energies will be NaN.")
        mpr = None
    else:
        if MPRester is None:
            print("[WARN] mp_api not installed; MP energies will be NaN.")
            mpr = None
        else:
            mpr = MPRester(mp_key)

    mp_cache: Dict[str, dict] = {}
    oqmd_cache: Dict[str, dict] = {}

    mp_records = {}
    oqmd_records = {}

    # Query per formula
    for i, f in enumerate(formulas, start=1):
        if i % 50 == 0 or i == len(formulas):
            print(f"[INFO] Processing formula {i}/{len(formulas)}: {f}")

        mp_rec = get_mp_energies_for_formula(
            formula=f,
            mpr=mpr,
            sleep=args.sleep_mp,
            cache=mp_cache,
        )
        mp_records[f] = mp_rec

        if not args.no_oqmd:
            oqmd_rec = get_oqmd_energies_for_formula(
                formula=f,
                sleep=args.sleep_oqmd,
                cache=oqmd_cache,
            )
            oqmd_records[f] = oqmd_rec
        else:
            oqmd_records[f] = {
                "oqmd_best_id": np.nan,
                "oqmd_delta_e": np.nan,
                "oqmd_stability": np.nan,
            }

    # Map results back onto the dataframe by formula
    def mp_col(formula: str, key: str):
        rec = mp_records.get(formula, None)
        if rec is None:
            return np.nan
        return rec.get(key, np.nan)

    def oqmd_col(formula: str, key: str):
        rec = oqmd_records.get(formula, None)
        if rec is None:
            return np.nan
        return rec.get(key, np.nan)

    df["mp_id"] = df["formula"].apply(lambda f: mp_col(f, "mp_id"))
    df["mp_e_form_per_atom"] = df["formula"].apply(lambda f: mp_col(f, "mp_e_form_per_atom"))
    df["mp_e_above_hull"] = df["formula"].apply(lambda f: mp_col(f, "mp_e_above_hull"))

    df["oqmd_best_id"] = df["formula"].apply(lambda f: oqmd_col(f, "oqmd_best_id"))
    df["oqmd_delta_e"] = df["formula"].apply(lambda f: oqmd_col(f, "oqmd_delta_e"))
    df["oqmd_stability"] = df["formula"].apply(lambda f: oqmd_col(f, "oqmd_stability"))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[INFO] Wrote {df.shape[0]} rows with energies to {out_path}")


if __name__ == "__main__":
    main()
