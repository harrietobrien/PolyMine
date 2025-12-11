import os
from pathlib import Path

from iotbx import cif
from cctbx import xray
import numpy as np
import gemmi
from sanitize import CifSanitizer

class CifIndex:

    def __init__(self, cif_dir="../../data/raw/cif"):
        self.cif_root = cif_dir
        self.cif_dict = self._cif_dict(self.cif_root)
        print(self.ah())

    @staticmethod
    def _cif_dict(cif_root):
        cif = dict()
        for root, dirs, files in os.walk(cif_root):
            for file in files:
                if file.endswith(".cif"):
                    cod_id = file.split('.')[0]
                    cif[cod_id] = dict()
                    cif[cod_id]['path'] = \
                        os.path.join(root, file)
        return cif

    def ah(self):
        total, ok, failed = 0, 0, 0
        self.cif_dict = {cid: {**data, 'model': None,
                               'xray_struct': None}
                         for cid, data in self.cif_dict.items()}
        for cod_id in self.cif_dict:
            total += 1
            path = self.cif_dict[cod_id]['path']
            cif_model = cif.reader(file_path=path).model()
            self.cif_dict[cod_id]['model'] = cif_model
            try:
                struct = xray.structure.from_cif(file_path=path)
                self.cif_dict[cod_id]['xray_struct'] = struct
                ok += 1
                # print(struct)
                # return xray.structure.from_cif(file_path=path)
            except Exception as e1:
                failed += 1
                continue
                '''
                fixed = str(Path(path).with_suffix(".fixed.cif"))
                try:
                    print(fixed)
                    # sanitizer = CifSanitizer(path)
                    # sanitizer.sanitize(drop_incomplete=True)
                    # sanitizer.write(fixed)
                    # return xray.structure.from_cif(file_path=fixed)
                    try:
                        self.cif_dict[cod_id]['xray_struct']  = \
                            xray.structure.from_cif(file_path=fixed)
                    except Exception as e2:
                        raise RuntimeError(f"cctbx failed: {e1}\nAfter sanitize: {e2}")
                except Exception as e2:
                    raise RuntimeError(f"cctbx failed: {e1}\nAfter sanitize: {e2}")
                '''

        print("OK:\t{}".format(ok))
        print("Failed:\t{}".format(failed))
        print("Total:\t{}".format(total))
        return None


if __name__ == "__main__":
    CifIndex()
