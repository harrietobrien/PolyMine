#!/usr/bin/env python3
from __future__ import annotations
import os, sys, math, struct, argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List
import numpy as np
import zstandard as zstd
from concurrent.futures import ProcessPoolExecutor

try:
    import gemmi
except Exception:
    print("ERROR: requires 'gemmi' (conda-forge): conda install -c conda-forge gemmi", file=sys.stderr)
    raise

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kw): return x


def _get_id_from_block(block: "gemmi.cif.Block", fallback: str) -> str:
    for tag in (
        "_database_code_COD",
        "_cod_database_code",
        "_database_code_depnum",
        "_database_code",
        "_entry.id",
        "_entry_id",
    ):
        v = block.find_value(tag)
        if v and v != "?":
            return str(v).strip()
    return fallback

def _spacegroup_number(block: "gemmi.cif.Block") -> int:
    for tag in ("_space_group.IT_number", "_space_group_IT_number",
                "_symmetry.Int_Tables_number", "_symmetry_Int_Tables_number"):
        v = block.find_value(tag)
        if v and v != "?":
            try:
                n = int(str(v))
                if 1 <= n <= 230:
                    return n
            except Exception:
                pass
    for tag in ("_space_group.name_H-M_ref", "_symmetry.space_group_name_H-M"):
        v = block.find_value(tag)
        if v and v != "?":
            try:
                sg = gemmi.find_spacegroup_by_name(str(v))
                if sg.number >= 1:
                    return int(sg.number)
            except Exception:
                pass
    v = block.find_value("_space_group.name_Hall")
    if v and v != "?":
        try:
            sg = gemmi.find_spacegroup_by_name(str(v))
            if sg.number >= 1:
                return int(sg.number)
        except Exception:
            pass
    return 0

def _read_cell(block: "gemmi.cif.Block") -> \
    Tuple[float,float,float,float,float,float]:
    def gv(tag):
        v = block.find_value(tag)
        return float(str(v)) if (v and v != "?") else float("nan")
    a = gv("_cell.length_a"); b = gv("_cell.length_b"); c = gv("_cell.length_c")
    al = gv("_cell.angle_alpha"); be = gv("_cell.angle_beta"); ga = gv("_cell.angle_gamma")
    return a,b,c,al,be,ga

def _density_from_block(block: "gemmi.cif.Block") -> float:
    for tag in ("_exptl_crystal.density_diffrn", "_exptl_crystal.density_meas", "_exptl_crystal.density_calc",
                "_exptl_crystal_density_diffrn", "_exptl_crystal_density_meas", "_exptl_crystal_density_calc"):
        v = block.find_value(tag)
        if v and v != "?":
            try:
                return float(str(v))
            except Exception:
                pass
    return float("nan")

def _element_bitset_from_block(block: "gemmi.cif.Block") -> \
    Tuple[np.uint64, np.uint64, bool]:
    lo = np.uint64(0); hi = np.uint64(0); ok = False
    try:
        loop = block.find_loop("_atom_site.type_symbol")
        if loop is not None and loop.width() > 0:
            for s in loop.values("_atom_site.type_symbol"):
                sym = str(s).strip()
                sym = "".join([ch for ch in sym if ch.isalpha()])
                if not sym:
                    continue
                try:
                    z = gemmi.Element(sym).atomic_number
                except Exception:
                    continue
                if z <= 0:
                    continue
                ok = True
                if z < 64:
                    lo |= (np.uint64(1) << np.uint64(z))
                else:
                    hi |= (np.uint64(1) << np.uint64(z-64))
    except Exception:
        pass
    return lo, hi, ok

def _process_one(i: int, path: str) -> tuple:
    """
    Returns: (i, id_str, sg, (a,b,c,al,be,ga), dens, lo, hi, flags, path_str)
    FLAGS bits:
      1: PARSE_OK, 2: HAS_SG, 4: HAS_CELL, 8: HAS_DENS, 16: HAS_ELEMS
    """
    flags = 0
    p = Path(path)
    try:
        doc = gemmi.cif.read_file(str(p))
        if len(doc) == 0:
            raise ValueError("Empty CIF")
        block = doc.sole_block() if hasattr(doc, "sole_block") else doc[0]
        flags |= 1
        id_str = _get_id_from_block(block, p.stem)
        sg = _spacegroup_number(block)
        if sg > 0: flags |= 2
        a,b,c,al,be,ga = _read_cell(block)
        if all((not math.isnan(x)) and x > 0.0 for x in (a,b,c,al,be,ga)):
            flags |= 4
        dens = _density_from_block(block)
        if not math.isnan(dens): flags |= 8
        lo, hi, ok = _element_bitset_from_block(block)
        if ok: flags |= 16
        return (i, id_str, int(sg), 
                (float(a),float(b),float(c),
                 float(al),float(be),float(ga)),
                 float(dens), int(lo), int(hi), 
                 int(flags), str(p))
    except Exception:
        return (i, p.stem, 0, (float("nan"),)*6, float("nan"), 0, 0, int(flags), str(p))

@dataclass
class IngestCIFs:
    input_dir: Path
    out_dir: Path
    workers: int = max(1, os.cpu_count() // 2)
    max_inflight: int = 8
    zstd_level: int = 6
    prune_dirs: tuple = ("manual-checks","logs","external-logs","doc",
                         "dictionaries","checks","bin",".git","__pycache__")
    extensions: tuple = (".cif", ".CIF")
    show_progress: bool = True

    def _discover(self) -> List[Path]:
        root = self.input_dir
        exts = set(self.extensions)
        skip = set(self.prune_dirs)
        out: List[Path] = []
        for p in root.rglob("*"):
            if p.is_dir():
                if p.name in skip:
                    continue
            elif p.is_file() and p.suffix in exts:
                if any(parent.name in skip for parent in p.parents):
                    continue
                out.append(p)
        out.sort(key=lambda x: x.as_posix())
        return out

    @staticmethod
    def _compute_meta_layout(N: int):
        off = 0
        def take(nbytes):
            nonlocal off
            s = off; off += nbytes; return s
        layout = {}
        layout["sgnum"]   = take(N * np.dtype(np.uint16).itemsize)
        layout["cell"]    = take(N * 6 * np.dtype(np.float32).itemsize)
        layout["density"] = take(N * np.dtype(np.float32).itemsize)
        layout["elem_lo"] = take(N * np.dtype(np.uint64).itemsize)
        layout["elem_hi"] = take(N * np.dtype(np.uint64).itemsize)
        layout["flags"]   = take(N * np.dtype(np.uint8).itemsize)
        layout["cif_off"] = take(N * np.dtype(np.uint64).itemsize)
        layout["cif_len"] = take(N * np.dtype(np.uint64).itemsize)
        return off, layout

    @staticmethod
    def _open_memmap(path: Path, dtype, shape, offset=0):
        return np.memmap(str(path), mode="r+", dtype=dtype, shape=shape, offset=offset, order="C")

    def _prepare_meta(self, N: int):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        meta_path = self.out_dir / "meta.bin"
        total_bytes, off = self._compute_meta_layout(N)
        with open(meta_path, "wb") as f: f.truncate(total_bytes)

        sgnum   = self._open_memmap(meta_path, np.uint16, (N,),    off["sgnum"])
        cell    = self._open_memmap(meta_path, np.float32, (N, 6), off["cell"])
        density = self._open_memmap(meta_path, np.float32, (N,),   off["density"])
        elem_lo = self._open_memmap(meta_path, np.uint64, (N,),    off["elem_lo"])
        elem_hi = self._open_memmap(meta_path, np.uint64, (N,),    off["elem_hi"])
        flags   = self._open_memmap(meta_path, np.uint8,  (N,),    off["flags"])
        cif_off = self._open_memmap(meta_path, np.uint64, (N,),    off["cif_off"])
        cif_len = self._open_memmap(meta_path, np.uint64, (N,),    off["cif_len"])

        sgnum[:] = 0; cell[:] = np.nan; density[:] = np.nan
        elem_lo[:] = 0; elem_hi[:] = 0; flags[:] = 0
        cif_off[:] = 0; cif_len[:] = 0

        return meta_path, (sgnum, cell, density, elem_lo, elem_hi, flags, cif_off, cif_len)

    def _prepare_ids_temp(self, N: int):
        ids_off_path = self.out_dir / "ids.off.tmp"
        with open(ids_off_path, "wb") as f:
            f.truncate(N * np.dtype(np.uint32).itemsize)

        offsets = np.memmap(str(ids_off_path), mode="r+", 
                            dtype=np.uint32, shape=(N,), 
                            offset=0, order="C")
        
        bytes_file = open(self.out_dir / "ids.bytes.tmp", 
                          "wb", buffering=1024*1024)
        
        return offsets, bytes_file, 0

    def _finalize_ids(self, offsets_memmap, bytes_len: int):
        ids_path = self.out_dir / "ids.bin"
        off_tmp = self.out_dir / "ids.off.tmp"
        bytes_tmp = self.out_dir / "ids.bytes.tmp"

        with open(ids_path, "wb") as out_f, open(bytes_tmp, "rb") as b:
            out_f.write(struct.pack("<II", offsets_memmap.shape[0], bytes_len))
            out_f.write(memoryview(offsets_memmap))
            for chunk in iter(lambda: b.read(1<<20), b""):
                out_f.write(chunk)

        try:
            offsets_memmap._mmap.close()
        except Exception:
            pass

        try:
            os.remove(off_tmp)
            os.remove(bytes_tmp)
        except Exception:
            pass
        return ids_path

    def _stream_compress_into(self, in_path: Path, 
                              out_stream, 
                              compressor: zstd.ZstdCompressor) -> Tuple[int, int]:
        """
        Stream a file into open zstd frame appended to out_stream; return (start, length)
        Uses closefd=False so out_stream remains open; flushes frame for clean boundary
        """
        start = out_stream.tell()
        with open(in_path, "rb", buffering=4 << 20) as fin:  # 4 MiB chunks
            with compressor.stream_writer(out_stream, closefd=False) as zw:
                for chunk in iter(lambda: fin.read(4 << 20), b""):  # 4 MiB reads
                    zw.write(chunk)
                zw.flush(zstd.FLUSH_FRAME)
        end = out_stream.tell()
        return start, (end - start)

    def run(self) -> None:
        files = self._discover()
        if not files:
            print("No .cif files found.", file=sys.stderr)
            sys.exit(1)
        N = len(files)

        meta_path, (sgnum, cell, density, 
                    elem_lo, elem_hi, flags, 
                    cif_off, cif_len) = self._prepare_meta(N)
        ids_offsets, ids_bytes_f, id_bytes_len = self._prepare_ids_temp(N)

        blob_path = self.out_dir / "cif.blob"
        blob_f = open(blob_path, "wb", buffering=4 << 20)

        zc = zstd.ZstdCompressor(level=self.zstd_level, threads=0)

        inflight = []
        next_idx = 0
        total = N
        desc = "Ingesting (low-mem)"

        with ProcessPoolExecutor(max_workers=self.workers) as ex:
            # prime
            while next_idx < total and len(inflight) < self.max_inflight:
                i = next_idx; next_idx += 1
                inflight.append(ex.submit(_process_one, i, str(files[i])))

            pbar = tqdm(total=total, desc=desc) if self.show_progress else None

            while inflight:
                fut = inflight.pop(0)
                idx, id_str, sg, (a,b,c,al,be,ga), dens, lo, hi, fl, path_str = fut.result()

                # backfill queue
                if next_idx < total:
                    inflight.append(ex.submit(_process_one, next_idx, str(files[next_idx])))
                    next_idx += 1

                # write meta
                sgnum[idx] = np.uint16(sg)
                cell[idx] = (a,b,c,al,be,ga)
                density[idx] = dens
                elem_lo[idx] = np.uint64(lo)
                elem_hi[idx] = np.uint64(hi)
                flags[idx]   = np.uint8(fl)

                # stream-compress raw CIF into blob (optimized)
                start, length = self._stream_compress_into(Path(path_str), blob_f, zc)
                cif_off[idx] = np.uint64(start)
                cif_len[idx] = np.uint64(length)

                # write id bytes
                ids_offsets[idx] = np.uint32(id_bytes_len)
                b = id_str.encode("utf-8") + b"\x00"
                ids_bytes_f.write(b)
                id_bytes_len += len(b)

                if pbar: pbar.update(1)
            if pbar: pbar.close()

        blob_f.flush(); blob_f.close()
        ids_bytes_f.flush(); ids_bytes_f.close()
        ids_path = self._finalize_ids(ids_offsets, id_bytes_len)

        for p in (meta_path, ids_path, blob_path):
            try:
                fd = os.open(p, os.O_RDONLY); os.fsync(fd); os.close(fd)
            except Exception:
                pass

        print(f"Done.\n  meta.bin : {meta_path}\n  ids.bin  : {ids_path}\n  cif.blob : {blob_path}")

def _parse_args():
    ap = argparse.ArgumentParser(description="Ingest CIFs → meta.bin, ids.bin, cif.blob (low memory, class-based)")
    ap.add_argument("input_dir", type=Path, help="Directory containing CIFs (scanned recursively).")
    ap.add_argument("--out", type=Path, default=Path("data/processed"), help="Output directory for binaries.")
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count()//2), help="Process workers for metadata parse.")
    ap.add_argument("--max-inflight", type=int, default=8, help="Bound the queued tasks to reduce RAM spikes.")
    ap.add_argument("--zstd-level", type=int, default=6, help="Compression level (lower is faster/less RAM).")
    ap.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bar.")
    return ap.parse_args()

def main():
    args = _parse_args()
    ing = IngestCIFs(
        input_dir=args.input_dir,
        out_dir=args.out,
        workers=args.workers,
        max_inflight=args.max_inflight,
        zstd_level=args.zstd_level,
        show_progress=not args.no_progress,
    )
    ing.run()

if __name__ == "__main__":
    main()

