#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Create an RGBA visualization raster for Klass_Ca classification where each class
uses its own color gradient driven by class probability.

Requested style:
- Class 1 (Kalkfattig): light pink gradient
- Class 2 (Intermediaer): light green gradient
- Class 3 (Kalkrik / Ca-rich): dark blue gradient

The class map determines hue family per pixel, and the corresponding class
probability controls where the pixel lands on that class gradient.
"""

from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import ColorInterp, Resampling
from rasterio import shutil as rio_shutil


# Input rasters from production run
BASE_DIR = Path(
    r"G:\National_maps\Ca_class\XGBoost"
)
CLASS_MAP_PATH = BASE_DIR / "KlassCa_class_map.tif"
PROB_1_PATH = BASE_DIR / "KlassCa_prob_1_Kalkfattig.tif"
PROB_2_PATH = BASE_DIR / "KlassCa_prob_2_Intermediaer.tif"
PROB_3_PATH = BASE_DIR / "KlassCa_prob_3_Kalkrik.tif"

# Output visualization
OUT_RGBA_PATH = BASE_DIR / "KlassCa_class_map_probstyled_rgba.tif"
TMP_RGBA_PATH = BASE_DIR / "KlassCa_class_map_probstyled_rgba_tmp.tif"

# Gamma controls gradient contrast: >1 emphasizes high-confidence colors.
GAMMA = 1.0


# Per-class gradients: (low_prob_rgb, high_prob_rgb)
CLASS_GRADIENTS = {
    1: (np.array([255, 244, 250], dtype=np.float32), np.array([255, 148, 196], dtype=np.float32)),
    2: (np.array([244, 255, 244], dtype=np.float32), np.array([120, 214, 140], dtype=np.float32)),
    3: (np.array([210, 231, 255], dtype=np.float32), np.array([10, 45, 145], dtype=np.float32)),
}


def _is_cog(path: Path) -> bool:
    if not path.exists():
        return False
    with rasterio.open(path) as src:
        layout = src.tags(ns="IMAGE_STRUCTURE").get("LAYOUT", "")
    return layout.upper() == "COG"


def _resolve_input_path(primary_path: Path) -> Path:
    if _is_cog(primary_path):
        return primary_path

    stem = primary_path.stem
    candidates = [
        BASE_DIR / f"{stem}_cog.tif",
        BASE_DIR / f"{stem}.cog.tif",
        BASE_DIR / f"{stem}_COG.tif",
    ]
    for cand in candidates:
        if _is_cog(cand):
            return cand

    return primary_path


def _check_inputs() -> None:
    missing = [
        str(p)
        for p in [CLASS_MAP_PATH, PROB_1_PATH, PROB_2_PATH, PROB_3_PATH]
        if not p.exists()
    ]
    if missing:
        raise FileNotFoundError("Missing input raster(s):\n- " + "\n- ".join(missing))


def _colorize_block(class_arr: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray):
    h, w = class_arr.shape
    rgb = np.zeros((3, h, w), dtype=np.uint8)
    alpha = np.zeros((h, w), dtype=np.uint8)

    valid = np.isfinite(class_arr)
    class_int = np.zeros((h, w), dtype=np.int16)
    class_int[valid] = np.rint(class_arr[valid]).astype(np.int16, copy=False)
    valid &= np.isin(class_int, [1, 2, 3])

    alpha[valid] = 255

    prob_by_class = {1: p1, 2: p2, 3: p3}

    for cls in (1, 2, 3):
        m = valid & (class_int == cls)
        if not np.any(m):
            continue

        p = np.clip(prob_by_class[cls][m], 0.0, 1.0).astype(np.float32)
        t = np.power(p, GAMMA, dtype=np.float32)[:, None]

        low_rgb, high_rgb = CLASS_GRADIENTS[cls]
        c = low_rgb[None, :] + t * (high_rgb - low_rgb)[None, :]
        c = np.clip(np.rint(c), 0, 255).astype(np.uint8)

        rgb[0][m] = c[:, 0]
        rgb[1][m] = c[:, 1]
        rgb[2][m] = c[:, 2]

    return rgb, alpha


def main() -> None:
    global CLASS_MAP_PATH, PROB_1_PATH, PROB_2_PATH, PROB_3_PATH

    _check_inputs()

    CLASS_MAP_PATH = _resolve_input_path(CLASS_MAP_PATH)
    PROB_1_PATH = _resolve_input_path(PROB_1_PATH)
    PROB_2_PATH = _resolve_input_path(PROB_2_PATH)
    PROB_3_PATH = _resolve_input_path(PROB_3_PATH)

    print("Using inputs:")
    print(f"  class map: {CLASS_MAP_PATH}")
    print(f"  prob1: {PROB_1_PATH}")
    print(f"  prob2: {PROB_2_PATH}")
    print(f"  prob3: {PROB_3_PATH}")

    with rasterio.open(CLASS_MAP_PATH) as src_cls, rasterio.open(PROB_1_PATH) as src_p1, rasterio.open(
        PROB_2_PATH
    ) as src_p2, rasterio.open(PROB_3_PATH) as src_p3:
        if not (
            src_cls.width == src_p1.width == src_p2.width == src_p3.width
            and src_cls.height == src_p1.height == src_p2.height == src_p3.height
        ):
            raise ValueError("Input rasters do not have identical dimensions.")

        profile = src_cls.profile.copy()
        profile.update(
            dtype="uint8",
            count=4,
            nodata=None,
            compress="deflate",
            tiled=True,
            blockxsize=512,
            blockysize=512,
            BIGTIFF="IF_SAFER",
        )

        with rasterio.open(TMP_RGBA_PATH, "w", **profile) as dst:
            for _, window in src_cls.block_windows(1):
                class_arr = src_cls.read(1, window=window)
                p1 = src_p1.read(1, window=window)
                p2 = src_p2.read(1, window=window)
                p3 = src_p3.read(1, window=window)

                rgb, alpha = _colorize_block(class_arr, p1, p2, p3)

                dst.write(rgb[0], 1, window=window)
                dst.write(rgb[1], 2, window=window)
                dst.write(rgb[2], 3, window=window)
                dst.write(alpha, 4, window=window)

            dst.colorinterp = (ColorInterp.red, ColorInterp.green, ColorInterp.blue, ColorInterp.alpha)
            overviews = [2, 4, 8, 16, 32, 64]
            dst.build_overviews(overviews, Resampling.average)
            dst.update_tags(ns="rio_overview", resampling="average")

    # Convert temporary GeoTIFF to Cloud Optimized GeoTIFF.
    cog_options = {
        "compress": "DEFLATE",
        "blocksize": 512,
        "overview_resampling": "average",
        "resampling": "nearest",
        "bigtiff": "IF_SAFER",
    }
    rio_shutil.copy(TMP_RGBA_PATH, OUT_RGBA_PATH, driver="COG", **cog_options)

    if not _is_cog(OUT_RGBA_PATH):
        print("[WARNING] Output is not flagged as COG after first write; retrying COG conversion.")
        rio_shutil.copy(TMP_RGBA_PATH, OUT_RGBA_PATH, driver="COG", **cog_options)

    if not _is_cog(OUT_RGBA_PATH):
        raise RuntimeError(f"Failed to create COG output: {OUT_RGBA_PATH}")

    if TMP_RGBA_PATH.exists():
        TMP_RGBA_PATH.unlink()

    print(f"Wrote styled RGBA COG: {OUT_RGBA_PATH}")
    print("Class gradients:")
    print("  1 Kalkfattig: light pink")
    print("  2 Intermediaer: light green")
    print("  3 Kalkrik: dark blue")


if __name__ == "__main__":
    main()
