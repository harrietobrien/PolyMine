import os
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


class CifIndex:
    METHODS : dict = {'get_space_group_symbol': "Space Group (H-M)",
               'get_space_group_number': "Int\'l SG Number",
               'get_hall': "Hall Symbol",
               'get_symmetry_dataset': "Symmetry Data"}

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


    @staticmethod
    def _safe_load_struct(path):
        try:
            return Structure.from_file(path)
        except (ValueError, KeyError, IndexError, TypeError) as ugh:
            print(f"Parse failed for {path}: {ugh}")
            try:
                with open(path, 'r', errors="ignore") as file:
                    text = file.read().replace("?", "0")
                return Structure.from_str(text, fmt='cif')
            except Exception:
                return None


    def _get_spacegroup(self):
        """
        :TODO: Incorrect stoich error bc some CIFs only include asymmetric unit
        """
        for codid in self.cif_dict:
            path = self.cif_dict[codid]['path']
            self._safe_load_struct(path)
            struct = self._safe_load_struct(path)
            if struct:
                analyzer = SpacegroupAnalyzer(struct)
                for method in CifIndex.METHODS:
                    func = getattr(analyzer, method)
                    key = CifIndex.METHODS[method]
                    self.cif_dict[codid][key] = func
        return self.cif_dict



if __name__ == "__main__":
    CifIndex()