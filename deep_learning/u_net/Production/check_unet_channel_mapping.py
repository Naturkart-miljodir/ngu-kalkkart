#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_unet_channel_mapping.py

Diagnostics for potential channel mismatches between:
1) training chips used by the U-Net, and
2) production rasters / VRTs used at inference time.

Main goals
----------
- verify that the production stack can reconstruct the same tile tensor as the training chip
- detect channel-order mismatches
- detect wrong multiband reads (especially AlphaEarth rows where the intended band from
  channel_map.csv is not being honored)
- flag duplicated source files, suspicious differences, and categorical range issues

Typical usage
-------------
python check_unet_channel_mapping.py

Optional arguments
------------------
--tile_id <tile_id>              test one specific tile id
--predictor_dir <folder>         override production predictor folder
--chips_root <folder>            override chip root folder
--channel_map <file>             override channel_map.csv
--tile_metadata <file>           override tile_metadata.csv
--out_dir <folder>               override output folder
--production_mode band1|intended compare production-style band1 reads, intended-band reads, or both
--max_rows N                     how many suspicious rows to print

Notes
-----
- Point predictor_dir to the same folder your production script uses. If that folder contains
  a .vrt for AlphaEarth, that is correct; the VRT should be read directly.
- The script does NOT modify any input files.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import warnings
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds

warnings.filterwarnings("ignore", category=RuntimeWarning)

# -----------------------------------------------------------------------------
# DEFAULT USER PATHS
# -----------------------------------------------------------------------------
DEFAULT_CHIPS_ROOT = r"E:\Test\National_test\DL_tiles_5pct_noSoftBgd\DL_AE_chips"
DEFAULT_CHANNEL_MAP = r"E:\Test\National_test\DL_tiles_5pct_noSoftBgd\DL_AE_chips\tile_metadata\channel_map.csv"
DEFAULT_TILE_METADATA = r"E:\Test\National_test\DL_tiles_5pct_noSoftBgd\DL_AE_chips\tile_metadata\tile_metadata.csv"
DEFAULT_PREDICTOR_DIR = r"C:\Users\acosta_pedro\OneDrive - Norges geologiske undersøkelse\Geochemistry NGU_2026\Kalk_project\Modelling\Covariates_to_model"
DEFAULT_OUT_DIR = os.path.join(DEFAULT_CHIPS_ROOT, "diagnostics_channel_check")

QUATERNARY_NAME = "quaternary_forenkletk_cog"
LANDUSE_NAME = "landuse_code_18_cog"
QUATERNARY_NUM_CLASSES = 22
LANDUSE_NUM_CLASSES = 33
ATOL = 1e-6
RTOL = 1e-5
MEAN_ABS_DIFF_WARN = 1e-3


@dataclass
class MapMeta:
    name_col: str
    band_col: Optional[str]
    channel_idx_col: Optional[str]
    source_col: Optional[str]
    normalized_col: Optional[str]


def normalize_name(s: str) -> str:
    s = str(s).strip().lower()
    s = s.replace(".tif", "").replace(".vrt", "")
    s = s.replace(" ", "_")
    return s


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def load_channel_map(path: str) -> Tuple[pd.DataFrame, MapMeta]:
    df = pd.read_csv(path)
    possible_cols = [c for c in df.columns if c.lower() in ("channel_name", "predictor_name", "name")]
    if not possible_cols:
        raise ValueError("channel_map.csv must contain one of: channel_name, predictor_name, name")
    name_col = possible_cols[0]
    df["__predictor_name__"] = df[name_col].astype(str)

    band_col = next((c for c in df.columns if c.lower() == "band"), None)
    channel_idx_col = next((c for c in df.columns if c.lower() in ("channel_idx", "channel", "index", "idx")), None)
    source_col = next((c for c in df.columns if c.lower() in ("source_file", "filename", "file", "path", "source")), None)
    normalized_col = next((c for c in df.columns if c.lower() in ("normalized", "is_normalized")), None)

    return df, MapMeta(name_col, band_col, channel_idx_col, source_col, normalized_col)


def load_tile_metadata(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"tile_id", "xmin", "ymin", "xmax", "ymax"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"tile_metadata.csv missing required columns: {sorted(missing)}")
    df["tile_id"] = df["tile_id"].astype(str)
    return df


def choose_tile(x_dir: str, tile_id: Optional[str]) -> Tuple[str, str]:
    x_paths = sorted(glob.glob(os.path.join(x_dir, "*.npy")))
    if not x_paths:
        raise RuntimeError(f"No .npy chips found in {x_dir}")
    if tile_id:
        p = os.path.join(x_dir, f"{tile_id}.npy")
        if not os.path.isfile(p):
            raise RuntimeError(f"Requested tile not found: {p}")
        return tile_id, p
    p = x_paths[0]
    return os.path.splitext(os.path.basename(p))[0], p


def read_chip(path: str) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D chip array, got {arr.shape} from {path}")
    # heuristic for channels_first
    if arr.shape[0] < 20 and arr.shape[-1] >= 20:
        arr = np.transpose(arr, (1, 2, 0))
    return arr.astype(np.float32)


def get_tile_bounds(meta_df: pd.DataFrame, tile_id: str) -> Tuple[float, float, float, float]:
    row = meta_df.loc[meta_df["tile_id"] == str(tile_id)]
    if row.empty:
        raise RuntimeError(f"tile_id not found in tile_metadata.csv: {tile_id}")
    row = row.iloc[0]
    return float(row["xmin"]), float(row["ymin"]), float(row["xmax"]), float(row["ymax"])


def build_predictor_lookup(predictor_dir: str) -> Tuple[Dict[str, str], List[str]]:
    tif_paths = glob.glob(os.path.join(predictor_dir, "*.tif"))
    vrt_paths = glob.glob(os.path.join(predictor_dir, "*.vrt"))
    all_paths = tif_paths + vrt_paths
    by_norm = {normalize_name(os.path.basename(p)): p for p in all_paths}
    return by_norm, all_paths


def match_predictor_files_like_production(predictor_dir: str, df_map: pd.DataFrame) -> Tuple[List[str], List[str]]:
    predictor_names = df_map["__predictor_name__"].tolist()
    by_norm, predictor_paths = build_predictor_lookup(predictor_dir)
    ordered_paths: List[str] = []
    ordered_names: List[str] = []

    for pred_name in predictor_names:
        key = normalize_name(pred_name)
        if key in by_norm:
            ordered_paths.append(by_norm[key])
            ordered_names.append(pred_name)
            continue

        candidates = []
        for p in predictor_paths:
            stem = normalize_name(os.path.basename(p))
            if key in stem or stem in key:
                candidates.append(p)

        if len(candidates) == 1:
            ordered_paths.append(candidates[0])
            ordered_names.append(pred_name)
        elif len(candidates) > 1:
            raise RuntimeError(f"Ambiguous match for predictor '{pred_name}': {candidates}")
        else:
            raise RuntimeError(f"Missing predictor '{pred_name}' in {predictor_dir}")

    return ordered_names, ordered_paths


def infer_alphaearth_row(row: pd.Series) -> bool:
    txt = " ".join([str(v) for v in row.values if pd.notna(v)]).lower()
    return ("alphaearth" in txt) or ("alpha_earth" in txt) or ("alpha earth" in txt) or ("ae_" in txt)


def read_tile_from_raster(path: str, xmin: float, ymin: float, xmax: float, ymax: float,
                          out_shape: Tuple[int, int], band: int) -> Tuple[np.ndarray, int, str]:
    with rasterio.open(path) as ds:
        win = from_bounds(xmin, ymin, xmax, ymax, ds.transform)
        arr = ds.read(band, window=win, boundless=True, fill_value=np.nan, out_shape=out_shape).astype(np.float32)
        nod = ds.nodata
        if nod is not None:
            arr[arr == nod] = np.nan
        return arr, ds.count, os.path.splitext(path)[1].lower()


def summarize_array(a: np.ndarray) -> Dict[str, float]:
    finite = np.isfinite(a)
    if not np.any(finite):
        return {"min": np.nan, "max": np.nan, "mean": np.nan, "std": np.nan,
                "nan_count": int(np.size(a)), "finite_count": 0}
    return {
        "min": float(np.nanmin(a)),
        "max": float(np.nanmax(a)),
        "mean": float(np.nanmean(a)),
        "std": float(np.nanstd(a)),
        "nan_count": int(np.isnan(a).sum()),
        "finite_count": int(np.isfinite(a).sum()),
    }


def safe_allclose(a: np.ndarray, b: np.ndarray, atol: float = ATOL, rtol: float = RTOL) -> bool:
    mask = np.isfinite(a) & np.isfinite(b)
    if not np.any(mask):
        return True
    return bool(np.allclose(a[mask], b[mask], atol=atol, rtol=rtol))


def mean_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if not np.any(mask):
        return np.nan
    return float(np.mean(np.abs(a[mask] - b[mask])))


def max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if not np.any(mask):
        return np.nan
    return float(np.max(np.abs(a[mask] - b[mask])))


def cast_categorical(arr: np.ndarray) -> np.ndarray:
    return np.rint(np.nan_to_num(arr, nan=0.0)).astype(np.int32)


def make_stack(df_map: pd.DataFrame, map_meta: MapMeta, ordered_names: List[str], ordered_paths: List[str],
               xmin: float, ymin: float, xmax: float, ymax: float, out_shape: Tuple[int, int],
               mode: str) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    mode = 'band1'    -> emulate suspicious production behavior by always reading band 1
    mode = 'intended' -> read the per-row band from channel_map when available, else band 1
    """
    rows = []
    bands = []

    for i, (pred_name, path) in enumerate(zip(ordered_names, ordered_paths)):
        row = df_map.iloc[i]
        intended_band = 1
        if map_meta.band_col and pd.notna(row[map_meta.band_col]):
            try:
                intended_band = int(row[map_meta.band_col])
            except Exception:
                intended_band = 1

        band_to_read = 1 if mode == "band1" else intended_band
        arr, dataset_band_count, file_ext = read_tile_from_raster(
            path=path, xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax, out_shape=out_shape, band=band_to_read
        )
        bands.append(arr)

        rows.append({
            "row_idx": i,
            "predictor_name": pred_name,
            "matched_file": os.path.basename(path),
            "matched_path": path,
            "file_ext": file_ext,
            "dataset_band_count": dataset_band_count,
            "intended_band_from_channel_map": intended_band,
            "band_used_for_read": band_to_read,
            "channel_idx_from_channel_map": row[map_meta.channel_idx_col] if map_meta.channel_idx_col else np.nan,
            "source_file_from_channel_map": row[map_meta.source_col] if map_meta.source_col else np.nan,
            "normalized_flag": row[map_meta.normalized_col] if map_meta.normalized_col else np.nan,
            "alphaearth_like": infer_alphaearth_row(row),
        })

    stack = np.stack(bands, axis=-1).astype(np.float32)
    return stack, pd.DataFrame(rows)


def compare_stack_to_chip(chip: np.ndarray, stack: np.ndarray, report_df: pd.DataFrame) -> pd.DataFrame:
    n_compare = min(chip.shape[-1], stack.shape[-1])
    out_rows = []
    for i in range(n_compare):
        a = chip[..., i]
        b = stack[..., i]
        s_a = summarize_array(a)
        s_b = summarize_array(b)
        row = report_df.iloc[i].to_dict()
        row.update({
            "chip_min": s_a["min"], "chip_max": s_a["max"], "chip_mean": s_a["mean"], "chip_std": s_a["std"],
            "ras_min": s_b["min"], "ras_max": s_b["max"], "ras_mean": s_b["mean"], "ras_std": s_b["std"],
            "mean_abs_diff": mean_abs_diff(a, b),
            "max_abs_diff": max_abs_diff(a, b),
            "allclose": safe_allclose(a, b),
        })

        pname = normalize_name(row["predictor_name"])
        if pname == normalize_name(QUATERNARY_NAME):
            a_cat = cast_categorical(a)
            b_cat = cast_categorical(b)
            row["chip_unique"] = ",".join(map(str, np.unique(a_cat)[:50]))
            row["ras_unique"] = ",".join(map(str, np.unique(b_cat)[:50]))
            row["chip_out_of_range"] = int(np.sum((a_cat < 0) | (a_cat > QUATERNARY_NUM_CLASSES)))
            row["ras_out_of_range"] = int(np.sum((b_cat < 0) | (b_cat > QUATERNARY_NUM_CLASSES)))
        if pname == normalize_name(LANDUSE_NAME):
            a_cat = cast_categorical(a)
            b_cat = cast_categorical(b)
            row["chip_unique"] = ",".join(map(str, np.unique(a_cat)[:50]))
            row["ras_unique"] = ",".join(map(str, np.unique(b_cat)[:50]))
            row["chip_out_of_range"] = int(np.sum((a_cat < 0) | (a_cat > LANDUSE_NUM_CLASSES)))
            row["ras_out_of_range"] = int(np.sum((b_cat < 0) | (b_cat > LANDUSE_NUM_CLASSES)))

        out_rows.append(row)
    return pd.DataFrame(out_rows)


def print_top(title: str, df: pd.DataFrame, cols: List[str], n: int) -> None:
    print(f"\n{title}")
    if df.empty:
        print("  none")
    else:
        print(df[cols].head(n).to_string(index=False))


def main() -> int:
    parser = argparse.ArgumentParser(description="Check U-Net chip vs production channel mapping")
    parser.add_argument("--chips_root", default=DEFAULT_CHIPS_ROOT)
    parser.add_argument("--channel_map", default=DEFAULT_CHANNEL_MAP)
    parser.add_argument("--tile_metadata", default=DEFAULT_TILE_METADATA)
    parser.add_argument("--predictor_dir", default=DEFAULT_PREDICTOR_DIR)
    parser.add_argument("--tile_id", default=None)
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--production_mode", default="both", choices=["band1", "intended", "both"])
    parser.add_argument("--max_rows", type=int, default=30)
    args = parser.parse_args()

    x_dir = os.path.join(args.chips_root, "X")
    if not os.path.isdir(x_dir):
        raise FileNotFoundError(f"Missing X folder: {x_dir}")
    if not os.path.isfile(args.channel_map):
        raise FileNotFoundError(f"Missing channel map: {args.channel_map}")
    if not os.path.isfile(args.tile_metadata):
        raise FileNotFoundError(f"Missing tile metadata: {args.tile_metadata}")
    if not os.path.isdir(args.predictor_dir):
        raise FileNotFoundError(f"Missing predictor dir: {args.predictor_dir}")

    out_dir = ensure_dir(args.out_dir)

    print("=" * 88)
    print("CHECKING POTENTIAL U-NET CHANNEL MISMATCH / ALPHAEARTH BAND MISMATCH")
    print("=" * 88)
    print(f"chips_root    : {args.chips_root}")
    print(f"channel_map   : {args.channel_map}")
    print(f"tile_metadata : {args.tile_metadata}")
    print(f"predictor_dir : {args.predictor_dir}")
    print(f"out_dir       : {out_dir}")

    df_map, map_meta = load_channel_map(args.channel_map)
    meta_df = load_tile_metadata(args.tile_metadata)
    tile_id, chip_path = choose_tile(x_dir, args.tile_id)
    chip = read_chip(chip_path)
    H, W, C = chip.shape
    xmin, ymin, xmax, ymax = get_tile_bounds(meta_df, tile_id)

    print(f"\nselected tile : {tile_id}")
    print(f"chip path     : {chip_path}")
    print(f"chip shape    : {chip.shape}")
    print(f"tile bounds   : xmin={xmin}, ymin={ymin}, xmax={xmax}, ymax={ymax}")
    print(f"channel rows  : {len(df_map)}")
    print(f"name col      : {map_meta.name_col}")
    print(f"band col      : {map_meta.band_col}")
    print(f"channel idx   : {map_meta.channel_idx_col}")
    print(f"source col    : {map_meta.source_col}")

    ordered_names, ordered_paths = match_predictor_files_like_production(args.predictor_dir, df_map)
    print(f"matched files : {len(ordered_paths)}")
    if len(ordered_paths) != C:
        print(f"[WARNING] chip channels ({C}) != channel_map rows ({len(ordered_paths)})")

    dup_files = Counter(os.path.basename(p) for p in ordered_paths)
    dup_df = pd.DataFrame(sorted([(k, v) for k, v in dup_files.items() if v > 1], key=lambda x: (-x[1], x[0])),
                          columns=["matched_file", "n_rows"])

    modes = [args.production_mode] if args.production_mode != "both" else ["band1", "intended"]
    summary_rows = []

    for mode in modes:
        print(f"\n{'-' * 88}\nMODE: {mode}\n{'-' * 88}")
        stack, report_df = make_stack(
            df_map=df_map, map_meta=map_meta, ordered_names=ordered_names, ordered_paths=ordered_paths,
            xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax, out_shape=(H, W), mode=mode
        )
        comp_df = compare_stack_to_chip(chip, stack, report_df)

        not_close = comp_df.loc[~comp_df["allclose"].fillna(False)].copy()
        suspicious = comp_df.loc[comp_df["mean_abs_diff"].fillna(0) > MEAN_ABS_DIFF_WARN].copy()
        alpha_multi = comp_df.loc[(comp_df["alphaearth_like"] == True) & (comp_df["dataset_band_count"] > 1)].copy()
        alpha_wrong_band = alpha_multi.loc[alpha_multi["intended_band_from_channel_map"] != alpha_multi["band_used_for_read"]].copy()
        any_wrong_band = comp_df.loc[(comp_df["dataset_band_count"] > 1) &
                                     (comp_df["intended_band_from_channel_map"] != comp_df["band_used_for_read"])].copy()

        for idx in [10, 11]:
            if idx < len(comp_df):
                r = comp_df.iloc[idx]
                print(
                    f"row {idx:>2} | predictor={r['predictor_name']} | file={r['matched_file']} | "
                    f"intended_band={r['intended_band_from_channel_map']} | used_band={r['band_used_for_read']} | "
                    f"mean_abs_diff={r['mean_abs_diff']:.6g} | allclose={r['allclose']}"
                )

        print(f"not allclose rows                    : {len(not_close)} / {len(comp_df)}")
        print(f"mean_abs_diff > {MEAN_ABS_DIFF_WARN} rows      : {len(suspicious)}")
        print(f"multiband rows with wrong used band  : {len(any_wrong_band)}")
        print(f"AlphaEarth multiband wrong-band rows : {len(alpha_wrong_band)}")

        print_top(
            "Top suspicious rows by mean_abs_diff",
            suspicious.sort_values("mean_abs_diff", ascending=False),
            ["row_idx", "predictor_name", "matched_file", "dataset_band_count",
             "intended_band_from_channel_map", "band_used_for_read", "mean_abs_diff", "max_abs_diff", "allclose"],
            args.max_rows,
        )

        print_top(
            "Top multiband wrong-band rows",
            any_wrong_band.sort_values(["matched_file", "row_idx"]),
            ["row_idx", "predictor_name", "matched_file", "dataset_band_count",
             "intended_band_from_channel_map", "band_used_for_read", "mean_abs_diff", "allclose"],
            args.max_rows,
        )

        comp_csv = os.path.join(out_dir, f"{tile_id}_comparison_{mode}.csv")
        comp_df.to_csv(comp_csv, index=False)
        summary_rows.append({
            "mode": mode,
            "tile_id": tile_id,
            "n_rows": len(comp_df),
            "n_not_allclose": int(len(not_close)),
            "n_suspicious": int(len(suspicious)),
            "n_multiband_wrong_band": int(len(any_wrong_band)),
            "n_alphaearth_wrong_band": int(len(alpha_wrong_band)),
            "comparison_csv": comp_csv,
        })
        print(f"saved comparison csv: {comp_csv}")

    mapping_df = pd.DataFrame({
        "row_idx": np.arange(len(ordered_names)),
        "predictor_name": ordered_names,
        "matched_file": [os.path.basename(p) for p in ordered_paths],
        "matched_path": ordered_paths,
    })
    mapping_csv = os.path.join(out_dir, f"{tile_id}_mapping.csv")
    mapping_df.to_csv(mapping_csv, index=False)
    print(f"saved mapping csv   : {mapping_csv}")

    if not dup_df.empty:
        dup_csv = os.path.join(out_dir, f"{tile_id}_duplicate_source_files.csv")
        dup_df.to_csv(dup_csv, index=False)
        print(f"saved duplicate csv : {dup_csv}")

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(out_dir, f"{tile_id}_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"saved summary csv   : {summary_csv}")

    txt_path = os.path.join(out_dir, f"{tile_id}_readme.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Interpretation guide\n")
        f.write("====================\n\n")
        f.write("1) Start with mode=band1. This emulates the suspicious case where a multiband dataset\n")
        f.write("   is repeatedly read as band 1 for several channel_map rows.\n\n")
        f.write("2) Compare against mode=intended. If intended-band mode is much closer to the chips\n")
        f.write("   than band1 mode, then the likely bug is wrong multiband handling in production.\n\n")
        f.write("3) Pay special attention to rows 10 and 11 in the console output and CSV files.\n")
        f.write("   If they are not close to the chips, your dominant predictors are also misaligned.\n\n")
        f.write("4) Repeated matched_file values are not automatically wrong. They are expected for multiband\n")
        f.write("   sources like a VRT, but they become suspicious if the intended band differs per row and\n")
        f.write("   production reads only band 1.\n")
    print(f"saved readme        : {txt_path}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
