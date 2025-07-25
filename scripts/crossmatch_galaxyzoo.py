#!/usr/bin/env python3
"""
Fast KD-tree cross-match:
CSV  ×  {Train, Val, Test} HDF5  →  single output HDF5

Example
-------
python crossmatch_hdf5.py \
      --csv  gz_decals_auto_posteriors.csv \
      --out  galaxies_plus_id.hdf5   \
      --radius 1.0                   \
      --id-name  ID                  \
      --unique

If you *omit* --hdf5 the script automatically expands

    ../split_files/Train/*.hdf5
    ../split_files/Val/*.hdf5
    ../split_files/Test/*.hdf5
"""
from __future__ import annotations
import argparse, glob, math, sys, re
from pathlib import Path
from typing import List, Dict

import h5py, numpy as np, pandas as pd
from scipy.spatial import cKDTree
from tqdm import tqdm

VLEN_STR = h5py.string_dtype(encoding="utf-8")  

# ------------------------------------------------------------------ helpers
def radec_to_xyz(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    ra, dec = np.deg2rad(ra_deg), np.deg2rad(dec_deg)
    cosd = np.cos(dec)
    return np.column_stack((cosd * np.cos(ra),
                            cosd * np.sin(ra),
                            np.sin(dec)))

def _normalize(arr: np.ndarray) -> np.ndarray:
    """
    Turn any object-dtype array into something h5py can store.
    • pure strings  → fixed-len UTF-8
    • nullable Int64/UInt64 → float64
    • mixed junk   → fixed-len UTF-8 of the string repr
    """
    if arr.dtype.kind != "O":
        return arr                                 # already safe

    # strings?
    if all(isinstance(x, str) or x is None for x in arr):
        maxlen = max(len(str(s)) for s in arr)
        return arr.astype(h5py.string_dtype("utf-8", length=maxlen or 1))

    # all items look numeric?  try float64
    try:
        return arr.astype("float64")
    except (TypeError, ValueError):
        pass

    # fallback: stringify everything
    maxlen = max(len(str(s)) for s in arr)
    return arr.astype(h5py.string_dtype("utf-8", length=maxlen or 1))

# ------------------------------------------------------------------ core
def crossmatch_stream(*,
    csv_path: Path,
    out_path: Path,
    hdf5_paths: List[Path],
    csv_ra: str,
    csv_dec: str,
    hdf5_ra: str,
    hdf5_dec: str,
    id_name: str,
    radius_arcsec: float,
    hdf5_chunk: int,
    ensure_unique: bool,
) -> None:

    # 1 ── read CSV & KD-tree ------------------------------------------------
    df = pd.read_csv(csv_path).reset_index(drop=True)
    bad = (~np.isfinite(df[csv_ra])) | (~np.isfinite(df[csv_dec]))
    if bad.any():
        print(f"Dropping {bad.sum():,} bad-coord rows"); df = df.loc[~bad]
    n_csv = len(df);  print(f"Reading CSV → {n_csv:,} rows")

    if id_name in df.columns:
        print(f"Column '{id_name}' already exists – overwriting")
    df[id_name] = pd.NA                         # defer dtype inference

    xyz_csv   = radec_to_xyz(df[csv_ra].values, df[csv_dec].values)
    csv_tree  = cKDTree(xyz_csv)
    unique_ok = np.zeros(n_csv, bool) if ensure_unique else None

    radius_rad   = math.radians(radius_arcsec / 3600)
    radius_chord = 2 * math.sin(radius_rad / 2)

    # 2 ── prepare output file (extendible datasets) -------------------------
    out_file   = h5py.File(out_path, "w")
    dsets: Dict[str, h5py.Dataset] = {}        # lazily created
    col_order  = list(df.columns)              # CSV first, keeps order
    append_len = 0                             # global number of matched rows

    def _ensure(col: str, sample: np.ndarray) -> None:
        """
        Create the target dataset on first use.
        • 1-D strings  → variable-len UTF-8
        • n-D numeric  → shape = (0, *sample.shape[1:])
        """
        if col in dsets:
            return

        if sample.dtype.kind in {"O", "U", "S"}:          # strings → VLEN UTF-8
            dt   = VLEN_STR
            sh   = (0,)                                   # rank-1 growable
            msha = (None,)
        else:                                             # numeric / images
            dt   = sample.dtype
            sh   = (0,) + sample.shape[1:]                # keep inner dims
            msha = (None,) + sample.shape[1:]

        dsets[col] = out_file.create_dataset(
            col, shape=sh, maxshape=msha,
            dtype=dt, compression="gzip", compression_opts=4, shuffle=True
        )
        
    def _append_block(col: str, block: np.ndarray) -> None:
        d = dsets[col]

        # --- dtype / string safety -------------------------------------------
        if h5py.check_string_dtype(d.dtype) is not None:      # dataset stores UTF-8
            if block.dtype.kind not in {"O", "U", "S"}:
                block = block.astype("U")                     # numeric → unicode
            block = block.astype(VLEN_STR, copy=False)
        else:                                                 # numeric dataset
            block = _normalize(block)

        # --- resize along the first axis only --------------------------------
        n = len(block)
        d.resize(len(d) + n, axis=0)
        d[-n:] = block

    # 3 ── stream each split file -------------------------------------------
    split_re = re.compile(r"/(Train|Val|Test)/", re.I)
    for h5_path in tqdm(hdf5_paths, desc="HDF5 files"):
        split_match = split_re.search(h5_path.as_posix())
        split_tag   = (split_match.group(1).capitalize()
                       if split_match else "Unknown")

        with h5py.File(h5_path, "r") as f:
            for need in (hdf5_ra, hdf5_dec, id_name):
                if need not in f:
                    raise KeyError(f"{h5_path}: dataset '{need}' not found")

            n_rows = len(f[hdf5_ra])
            inner  = tqdm(range(0, n_rows, hdf5_chunk),
                          desc=f"{h5_path.name} rows", leave=False)

            for start in inner:
                end = min(start + hdf5_chunk, n_rows)
                slc = slice(start, end)

                ra  = f[hdf5_ra][slc]; dec = f[hdf5_dec][slc]
                gid = f[id_name][slc]

                xyz  = radec_to_xyz(ra, dec)
                dist, idx = csv_tree.query(xyz, k=1,
                                           distance_upper_bound=radius_chord)

                good = np.isfinite(dist)
                if good.any():
                    good &= (2*np.arcsin(dist/2) <= radius_rad)

                if ensure_unique and good.any():
                    good &= ~unique_ok[idx]
                    unique_ok[idx[good]] = True

                if not good.any():
                    continue
                
                split_arr = np.asarray([split_tag] * good.sum(), dtype=VLEN_STR)
                _ensure("split", split_arr)
                _append_block("split", split_arr)

                # -------- gather blocks (CSV side)
                for col in col_order:
                    block = df.iloc[idx[good]][col].to_numpy()
                    block = _normalize(block)
                    _ensure(col, block)
                    _append_block(col, block)

                # -------- gather blocks (HDF5 side) – iterate keys lazily
                for col in f.keys():
                    if col in (hdf5_ra, hdf5_dec):  # already present via CSV
                        continue
                    block = f[col][slc][good]
                    block = _normalize(block)
                    _ensure(col, block)
                    _append_block(col, block)

                # -------- split tag column
                _ensure("split", np.asarray([split_tag], dtype="S"))
                _append_block("split",
                              np.asarray([split_tag.encode()] * good.sum(),
                                         dtype="S"))

                append_len += good.sum()

            tqdm.write(f"  {h5_path.name}: processed {n_rows:,} rows")

    out_file.close()
    print(f"✓ Finished: wrote {append_len:,} matched rows → {out_path}")


# ------------------------------------------------------------------ CLI
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--csv",  required=True, type=Path, help="Input CSV file")
    p.add_argument("--out",  required=True, type=Path, help="Output HDF5 path")
    p.add_argument("--hdf5", nargs="*", help="HDF5 paths/globs; "
                   "if omitted, defaults to Train/Val/Test globs")
    p.add_argument("--id-name",   default="ID",  help="ID dataset/column name")
    p.add_argument("--csv-ra",    default="ra",  help="CSV RA column (deg)")
    p.add_argument("--csv-dec",   default="dec", help="CSV Dec column (deg)")
    p.add_argument("--hdf5-ra",   default="ra",  help="HDF5 RA dataset (deg)")
    p.add_argument("--hdf5-dec",  default="dec", help="HDF5 Dec dataset (deg)")
    p.add_argument("--radius",    type=float, default=1.0, help="Match radius (arcsec)")
    p.add_argument("--hdf5-chunk",type=int, default=250_000, help="Rows per HDF5 slice")
    p.add_argument("--unique",    action="store_true", help="Enforce one-to-one CSV match")
    return p.parse_args()


def expand_hdf5_globs(user_globs: List[str] | None) -> List[Path]:
    patterns = (user_globs if user_globs else
                ["../split_files/Train/*.hdf5",
                 "../split_files/Val/*.hdf5",
                 "../split_files/Test/*.hdf5"])
    files: set[Path] = set()
    for pat in patterns:
        matches = glob.glob(pat)
        if not matches:
            print(f" pattern '{pat}' matched no files", file=sys.stderr)
        files.update(Path(m).resolve() for m in matches)
    if not files:
        sys.exit("No HDF5 files found – aborting.")
    return sorted(files)


def main() -> None:
    a = parse_args()
    crossmatch_stream(
        csv_path=a.csv,
        out_path=a.out,
        hdf5_paths=expand_hdf5_globs(a.hdf5),
        csv_ra=a.csv_ra,  csv_dec=a.csv_dec,
        hdf5_ra=a.hdf5_ra, hdf5_dec=a.hdf5_dec,
        id_name=a.id_name,
        radius_arcsec=a.radius,
        hdf5_chunk=a.hdf5_chunk,
        ensure_unique=a.unique,
    )


if __name__ == "__main__":
    main()

