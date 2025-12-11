from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional
import gemmi

class CifSanitizer:
    def __init__(self, path: str):
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(path)
        self.path = p
        self.doc = gemmi.cif.read_file(str(p))
        self.report: Dict[str, Dict[str, int]] = {}

    @staticmethod
    def _is_num_like(s: str) -> bool:
        # if s in ("", ".", "?"):
            # return False
        try:
            return True if float(s) else False
            return True
        except Exception:
            return False

    @staticmethod
    def _as_loop(obj):
        """Return a gemmi.cif.Loop from either a Loop or a Column (older gemmi)"""
        # If it already behaves like a Loop
        if hasattr(obj, "width") and hasattr(obj, "length") \
            and hasattr(obj, "tag") and hasattr(obj, "value"):
            return obj
        # Older gemmi may return a Column with a .loop backref
        if hasattr(obj, "loop"):
            lp = obj.loop
            if hasattr(lp, "width") and hasattr(lp, "length"):
                return lp
        return None

    @staticmethod
    def _find_atom_loop(block):
        """
        Portable atom-site loop finder using block.find_loop(tag),
        handling both Loop and Column returns.
        """
        for tag in (
                "_atom_site_type_symbol",
                "_atom_site_label",
                "_atom_site_fract_x",
                "_atom_site_Cartn_x",
        ):
            obj = block.find_loop(tag)
            lp = CifSanitizer._as_loop(obj)
            if lp is not None:
                return lp
        return None

    @staticmethod
    def _col_index(lp, tag: str) -> int:
        """Return column index for tag or -1 if absent"""
        for j in range(lp.width()):
            if lp.tag(j) == tag:
                return j
        return -1
    @staticmethod
    def _make_cell(block: gemmi.cif.Block) -> Optional[gemmi.UnitCell]:
        req = ["_cell_length_a","_cell_length_b","_cell_length_c",
               "_cell_angle_alpha","_cell_angle_beta","_cell_angle_gamma"]
        vals = [block.find_value(k) for k in req]
        if not all(v not in ("", ".", "?", None) for v in vals):
            return None
        a,b,c,al,be,ga = map(float, vals)
        return gemmi.UnitCell(a,b,c,al,be,ga)

    def sanitize(self, drop_incomplete: bool = True, digits: int = 8) -> \
        Dict[str, Dict[str, int]]:
        self.report.clear()
        for block in self.doc:
            bname = block.name
            repaired = dropped = kept = 0
            loop = self._find_atom_loop(block)
            if loop is None:
                self.report[bname] = {"repaired": 0, "dropped": 0, "kept": 0, "total_in": 0}
                continue

            def ensure_col(loop: gemmi.cif.Loop, tag: str) -> \
                tuple[gemmi.cif.Loop, int]:
                """Ensure that a column exists in the loop and return (loop, column_index)"""
                idx = self._col_index(loop, tag)
                if idx >= 0:
                    return loop, idx
                # rebuild loop with a new empty column
                col_names = [loop.tag(c) for c in range(loop.width())] + [tag]
                new_cols = [[loop.value(r, c) for r in range(loop.length())] \
                            for c in range(loop.width())]
                new_cols.append(["" for _ in range(loop.length())])
                block.remove_loop(loop)
                new_loop = block.add_new_loop(col_names)
                for r in range(loop.length()):
                    for c in range(len(col_names)):
                        new_loop.add_value(new_cols[c][r])
                return new_loop, len(col_names) - 1

            col_fx = self._col_index(loop, "_atom_site_fract_x")
            col_fy = self._col_index(loop, "_atom_site_fract_y")
            col_fz = self._col_index(loop, "_atom_site_fract_z")
            col_cx = self._col_index(loop, "_atom_site_Cartn_x")
            col_cy = self._col_index(loop, "_atom_site_Cartn_y")
            col_cz = self._col_index(loop, "_atom_site_Cartn_z")

            total_in = loop.length()
            cell = self._make_cell(block)
            rows_to_keep: List[int] = []

            for i in range(total_in):
                fx = loop.value(i, col_fx) if col_fx >= 0 else ""
                fy = loop.value(i, col_fy) if col_fy >= 0 else ""
                fz = loop.value(i, col_fz) if col_fz >= 0 else ""
                has_frac = all(self._is_num_like(v) for v in (fx, fy, fz))

                if not has_frac and cell and min(col_cx, col_cy, col_cz) >= 0:
                    cx, cy, cz = loop.value(i, col_cx), loop.value(i, col_cy), loop.value(i, col_cz)
                    if all(self._is_num_like(v) for v in (cx, cy, cz)):
                        cart = gemmi.Position(float(cx), float(cy), float(cz))
                        frac = cell.fractionalize(cart)
                        loop, col_fx = ensure_col(loop, "_atom_site_fract_x")
                        loop, col_fy = ensure_col(loop, "_atom_site_fract_y")
                        loop, col_fz = ensure_col(loop, "_atom_site_fract_z")
                        loop.set_value(i, col_fx, f"{frac.x:.{digits}f}")
                        loop.set_value(i, col_fy, f"{frac.y:.{digits}f}")
                        loop.set_value(i, col_fz, f"{frac.z:.{digits}f}")
                        repaired += 1
                        has_frac = True

                if has_frac or not drop_incomplete:
                    rows_to_keep.append(i)
                    kept += 1
                else:
                    dropped += 1

            # rebuild loop if rows were dropped
            if drop_incomplete and dropped > 0:
                col_names = [loop.tag(c) for c in range(loop.width())]
                new_cols = [[loop.value(r, c) for r in rows_to_keep] \
                            for c in range(loop.width())]
                block.remove_loop(loop)
                new_loop = block.add_new_loop(col_names)
                for r in range(len(rows_to_keep)):
                    for c in range(len(col_names)):
                        new_loop.add_value(new_cols[c][r])
                loop = new_loop

            self.report[bname] = {
                "repaired": repaired,
                "dropped": dropped if drop_incomplete else 0,
                "kept": kept,
                "total_in": total_in,
            }

        return self.report

    def write(self, out_path: str) -> None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        self.doc.write_file(str(out_path))

    def summary(self) -> str:
        if not self.report:
            return "No report available (run sanitize() first)."
        lines = ["CIF Sanitizer Summary:"]
        for blk, d in self.report.items():
            lines.append(
                f"  {blk:20s}  total={d['total_in']:4d}  kept={d['kept']:4d}  "
                f"repaired={d['repaired']:4d}  dropped={d['dropped']:4d}"
            )
        return "\n".join(lines)
