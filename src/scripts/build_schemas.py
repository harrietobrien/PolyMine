import os
from pathlib import Path
import pandas as pd
import argparse
import numpy as np
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core.operations import SymmOp
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import warnings
warnings.filterwarnings("ignore", module="pymatgen")


class CifIndex:
    METHODS : dict = {'get_space_group_symbol': "Space Group (H-M)",
               'get_space_group_number': "Int\'l SG Number",
               'get_hall': "Hall Symbol"}
               # 'get_symmetry_dataset': "Symmetry Data"}

    def __init__(self, cif_dir="../../data/raw/cif"):
        self.cif_root = cif_dir
        self.cif_dict = self._cif_dict(self.cif_root)
        print(self._get_spacegroup())

    @staticmethod
    def _cif_dict(cif_root):
        cif = dict()
        for root, dirs, files in os.walk(cif_root):
            for file in files:
                if file.endswith(".cif"):
                    codid = file.split('.')[0]
                    cif[codid] = dict()
                    cif[codid]['path'] = \
                        os.path.join(root, file)
        return cif


    def _safe_load_struct(self, path):
        try:
            return self.expand_from_asymmetric(Structure.from_file(path), 1e-3, 5.0, 1e-5)
            # return Structure.from_file(path)
        except (ValueError, KeyError, IndexError, TypeError) as ugh:
            print(f"Parse failed for {path}: {ugh}")
            try:
                with open(path, 'r', errors="ignore") as file:
                    text = file.read().replace("?", "0")
                return self.expand_from_asymmetric(Structure.from_file(path), 1e-3, 5.0, 1e-5)
            except Exception:
                return None


    def _get_spacegroup(self):
        """
        :TODO: Incorrect stoich error bc some CIFs only include asymmetric unit
        """
        for codid in self.cif_dict:
            path = self.cif_dict[codid]['path']
            struct = self._safe_load_struct(path)
            if struct:
                analyzer = SpacegroupAnalyzer(struct)
                for method in CifIndex.METHODS:
                    func = getattr(analyzer, method)
                    key = CifIndex.METHODS[method]
                    self.cif_dict[codid][key] = func
        print(self.cif_dict)
        return self.cif_dict


    def expand_from_asymmetric(self,
                               struct: Structure,
                               symprec: float = 1e-3,
                               angle_tolerance: float = 5.0,
                               dedup_tol: float = 1e-5) -> Structure:
        """
        Build the full unit cell by applying spglib symmetry operations to the
        asymmetric-unit structure and deduplicating equivalent images.
        """
        sga = SpacegroupAnalyzer(struct, symprec=symprec, angle_tolerance=angle_tolerance)
        data = sga.get_symmetry_dataset()  # spglib dataset

        # Rotations (3x3 int), translations (3-vector floats) in fractional coords
        rotations = data["rotations"]
        translations = data["translations"]

        species = []
        frac_coords = []

        # Hash helper for deduplication on a fixed epsilon grid
        def bucket_key(frac):
            wrapped = frac - np.floor(frac)  # wrap into [0, 1)
            return tuple(np.round(wrapped / dedup_tol).astype(int).tolist())

        seen = set()

        for site in struct.sites:
            f = site.frac_coords
            # Apply symmetry operation in fractional space
            for R, t in zip(rotations, translations):
                # f' = R f + t
                f_new = (R @ f + t).astype(float)
                f_new -= np.floor(f_new)  # wrap into [0,1)
                key = (site.species_string, bucket_key(f_new))
                if key in seen:
                    continue
                seen.add(key)
                species.append(site.species)
                frac_coords.append(f_new)
        return Structure(lattice=struct.lattice, species=species, coords=frac_coords, coords_are_cartesian=False)

class CifMasterBuilder:
    """
    Walk a CIF tree, extract structural + symmetry features, and
    write them to a CSV (master.csv).

    Main public methods:
      - build()       -> populate internal records list
      - to_dataframe()
      - to_csv(path)
    """

    # Use CSV-friendly column names
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
        """Walk the CIF directory and collect feature records."""
        count = 0
        for root, _, files in os.walk(self.cif_root):
            for fname in files:
                if not fname.endswith(".cif"):
                    continue

                codid = fname.split(".")[0]
                path = Path(root) / fname

                struct = self._safe_load_struct(path)
                if struct is None:
                    continue

                rec = self._compute_features(codid, path, struct)
                self.records.append(rec)

                count += 1
                if self.limit is not None and count >= self.limit:
                    break
            if self.limit is not None and count >= self.limit:
                break

    def to_dataframe(self) -> pd.DataFrame:
        """Return the collected records as a pandas DataFrame."""
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


    def _safe_load_struct(self, path: Path) -> Structure | None:
        """
        Robust loading of a CIF file, then expansion from asymmetric unit.

        Returns None if the structure cannot be parsed.
        """
        try:
            struct = Structure.from_file(path)
            return self.expand_from_asymmetric(struct, 1e-3, 5.0, 1e-5)
        except Exception as e:
            print(f"[WARN] Parse failed for {path}: {e}")
            # Try again with '?' replaced by 0 and manual from_str
            try:
                with open(path, "r", errors="ignore") as f:
                    text = f.read().replace("?", "0")
                struct = Structure.from_str(text, fmt="cif")
                return self.expand_from_asymmetric(struct, 1e-3, 5.0, 1e-5)
            except Exception as e2:
                print(f"[WARN] Second attempt failed for {path}: {e2}")
                return None

    def _compute_features(self, codid: str, path: Path, struct: Structure) -> dict:
        """
        Compute per-structure features for master.csv.
        """
        analyzer = SpacegroupAnalyzer(struct)

        rec: dict = {
            "cod_id": codid,
            "path": str(path),

            # Composition / size
            "formula": struct.composition.reduced_formula,
            "n_sites": len(struct),
            "n_species": len(struct.composition.elements),

            # Cell geometry
            "volume": struct.volume,
            "density": struct.density,
            "a": struct.lattice.a,
            "b": struct.lattice.b,
            "c": struct.lattice.c,
            "alpha": struct.lattice.alpha,
            "beta": struct.lattice.beta,
            "gamma": struct.lattice.gamma,
        }

        # Symmetry info using METHOD_MAP
        for method_name, colname in self.METHOD_MAP.items():
            func = getattr(analyzer, method_name)
            try:
                rec[colname] = func()
            except Exception as e:
                print(f"[WARN] {method_name} failed for {codid}: {e}")
                rec[colname] = np.nan

        return rec

    @staticmethod
    def expand_from_asymmetric(
        struct: Structure,
        symprec: float = 1e-3,
        angle_tolerance: float = 5.0,
        dedup_tol: float = 1e-5,
    ) -> Structure:
        """
        Build the full unit cell by applying spglib symmetry operations to the
        asymmetric-unit structure and deduplicating equivalent images
        """
        sga = SpacegroupAnalyzer(struct, symprec=symprec, angle_tolerance=angle_tolerance)
        data = sga.get_symmetry_dataset()

        rotations = data["rotations"]      # (N_ops, 3, 3)
        translations = data["translations"]  # (N_ops, 3)

        species = []
        frac_coords = []

        def bucket_key(frac):
            wrapped = frac - np.floor(frac)  # wrap into [0, 1)
            return tuple(np.round(wrapped / dedup_tol).astype(int).tolist())

        seen = set()

        for site in struct.sites:
            f = np.asarray(site.frac_coords, float)
            for R, t in zip(rotations, translations):
                f_new = (R @ f + t).astype(float)
                f_new -= np.floor(f_new)  # wrap into [0,1)
                key = (site.species_string, bucket_key(f_new))
                if key in seen:
                    continue
                seen.add(key)
                species.append(site.species)
                frac_coords.append(f_new)

        return Structure(
            lattice=struct.lattice,
            species=species,
            coords=frac_coords,
            coords_are_cartesian=False,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a master.csv table from COD CIF files."
    )
    parser.add_argument(
        "--cif-root",
        type=str,
        default="data/raw/cif",
        help="Root directory containing .cif files (recursive).",
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
        help="Optional max number of CIFs to process (for testing).",
    )
    args = parser.parse_args()

    builder = CifMasterBuilder(cif_root=args.cif_root, limit=args.limit)
    builder.to_csv(args.out)


if __name__ == "__main__":
    main()
