# pyright: reportMissingImports=false, reportMissingModuleSource=false
# Run the code with the kalk_process environment
import argparse
import builtins
import json
import os
import time
from functools import partial
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import Window, transform as window_transform

print = partial(builtins.print, flush=True)

# Keep this modest to avoid oversubscribing CPU and disk during large warps.
REPROJECT_NUM_THREADS = max(2, min(4, (os.cpu_count() or 4) // 2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Process AlphaEarth folders zone-by-zone: align mask, export masked + de-quantized "
            "GeoTIFF images to one common folder, and build a national VRT from those outputs."
        )
    )

    parser.add_argument(
        "--alpha-root",
        type=Path,
        default=Path(r"C:\Users\acosta_pedro\Norges geologiske undersøkelse\Kay Sindre Skogseth - Pedro\Alpha_earth_data"),
        help="Root folder containing zone subfolders (31N, 32N, ...).",
    )

    parser.add_argument(
        "--mask-path",
        type=Path,
        default=Path(r"G:\Covariates_to_model\Topo_dtm_ch.tif"),
        help="Mask raster that will be checked/aligned to each AlphaEarth source grid.",
    )

    parser.add_argument(
        "--snap-raster-path",
        type=Path,
        default=Path(r"G:\Covariates_to_model\Topo_dtm_ch.tif"),
        help=(
            "Optional canonical target grid raster. When provided, every de-quantized image and the final "
            "VRT are forced to this raster's exact CRS, transform, width, and height. If omitted, --mask-path "
            "is used as the target grid."
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(r"E:\Alpha_earth"),
        help="Output root folder.",
    )

    parser.add_argument(
        "--zones",
        nargs="*",
        default=None,
        help="Optional zone names. If omitted, process all subfolders under --alpha-root.",
    )

    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=0.0,
        help="Mask values > threshold are kept.",
    )

    parser.add_argument(
        "--output-dtype",
        choices=["float32"],
        default="float32",
        help="GeoTIFF dtype for de-quantized outputs.",
    )

    parser.add_argument(
        "--force-north-up",
        action="store_true",
        help="Flip south-up outputs to north-up without resampling.",
    )

    parser.add_argument(
        "--target-crs",
        type=str,
        default="EPSG:25833",
        help="Final output CRS for de-quantized images.",
    )

    parser.add_argument(
        "--target-resolution",
        type=float,
        default=10.0,
        help="Final output pixel size in target CRS units.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing aligned masks / de-quantized images / VRT.",
    )

    parser.add_argument(
        "--skip-national-vrt",
        action="store_true",
        help="Skip building the national VRT.",
    )

    parser.add_argument(
        "--skip-qc-cog",
        action="store_true",
        help="Skip creating 1-band QC COG from the final VRT.",
    )

    parser.add_argument(
        "--qc-band",
        type=int,
        default=1,
        help="Band index to export as national QC COG.",
    )

    return parser.parse_args()


def rasters_are_aligned(ds1, ds2, tol=1e-9) -> bool:
    """
    Check whether two raster datasets are already perfectly aligned.
    This compares CRS, dimensions, and affine transform.
    """
    return (
        ds1.crs == ds2.crs
        and ds1.width == ds2.width
        and ds1.height == ds2.height
        and all(abs(a - b) <= tol for a, b in zip(ds1.transform, ds2.transform))
    )


def discover_zone_dirs(alpha_root: Path, selected_zones: list[str] | None) -> list[Path]:
    if selected_zones:
        zone_dirs = [alpha_root / z for z in selected_zones]
    else:
        zone_dirs = [p for p in sorted(alpha_root.iterdir()) if p.is_dir()]

    missing = [str(z) for z in zone_dirs if not z.exists()]
    if missing:
        raise FileNotFoundError(f"Zone folders not found: {missing}")

    return zone_dirs


def discover_zone_rasters(zone_dir: Path) -> list[Path]:
    preferred_ext = {".tif", ".tiff", ".vrt", ".img"}
    candidates = [p for p in sorted(zone_dir.iterdir()) if p.is_file()]

    preferred = [p for p in candidates if p.suffix.lower() in preferred_ext]
    others = [p for p in candidates if p.suffix.lower() not in preferred_ext]

    rasters = []

    for path in preferred + others:
        try:
            with rasterio.open(path):
                rasters.append(path)
        except Exception:
            continue

    if not rasters:
        raise FileNotFoundError(f"No readable rasters found in {zone_dir}")

    return rasters


def resolve_existing_raster_path(raster_path: Path) -> Path:
    if raster_path.exists():
        return raster_path

    # Allow passing mask path without extension, e.g. E:\mask\Mask_land_Kalk_cog
    if raster_path.suffix == "":
        for ext in [".tif", ".tiff", ".img", ".vrt"]:
            candidate = Path(f"{raster_path}{ext}")
            if candidate.exists():
                return candidate

    raise FileNotFoundError(f"Raster not found: {raster_path}")


def grid_window_for_source(
    grid_ds,
    source_raster_path: Path,
):
    with rasterio.open(source_raster_path) as src_ds:
        if src_ds.crs is None:
            raise ValueError(f"Source CRS missing: {source_raster_path}")
        if grid_ds.crs is None:
            raise ValueError("Grid CRS is missing")

        src_bounds_in_grid = transform_bounds(
            src_ds.crs,
            grid_ds.crs,
            *src_ds.bounds,
            densify_pts=21,
        )

    grid_bounds = grid_ds.bounds
    left = max(src_bounds_in_grid[0], grid_bounds.left)
    bottom = max(src_bounds_in_grid[1], grid_bounds.bottom)
    right = min(src_bounds_in_grid[2], grid_bounds.right)
    top = min(src_bounds_in_grid[3], grid_bounds.top)

    if left >= right or bottom >= top:
        return None

    row_tl, col_tl = grid_ds.index(left, top, op=np.floor)
    row_br, col_br = grid_ds.index(right, bottom, op=np.ceil)

    row_off = max(0, int(min(row_tl, row_br)))
    col_off = max(0, int(min(col_tl, col_br)))
    row_end = min(grid_ds.height, int(max(row_tl, row_br)))
    col_end = min(grid_ds.width, int(max(col_tl, col_br)))

    if col_end <= col_off or row_end <= row_off:
        return None

    return Window(
        col_off=col_off,
        row_off=row_off,
        width=col_end - col_off,
        height=row_end - row_off,
    )


def print_mask_match_status(
    mask_path: Path,
    reference_raster_path: Path,
) -> None:
    with rasterio.open(reference_raster_path) as ref_ds, rasterio.open(mask_path) as mask_ds:
        is_aligned = rasters_are_aligned(mask_ds, ref_ds)
        if is_aligned:
            print(f"[MASK CHECK] Match: YES - source and mask already aligned for {reference_raster_path.name}")
        else:
            print(
                f"[MASK CHECK] Match: NO (No match) - source={ref_ds.crs}, mask={mask_ds.crs}; "
                "reprojecting source to mask grid"
            )


def source_overlaps_target_grid(
    target_ds,
    source_raster_path: Path,
) -> bool:
    with rasterio.open(source_raster_path) as src_ds:
        if src_ds.crs is None:
            raise ValueError(f"Source CRS missing: {source_raster_path}")
        if target_ds.crs is None:
            raise ValueError("Target grid CRS is missing")

        src_bounds_in_target = transform_bounds(
            src_ds.crs,
            target_ds.crs,
            *src_ds.bounds,
            densify_pts=21,
        )

    target_bounds = target_ds.bounds
    left = max(src_bounds_in_target[0], target_bounds.left)
    bottom = max(src_bounds_in_target[1], target_bounds.bottom)
    right = min(src_bounds_in_target[2], target_bounds.right)
    top = min(src_bounds_in_target[3], target_bounds.top)
    return left < right and bottom < top


def north_up_transform_if_needed(transform: Affine, height: int) -> tuple[Affine, bool]:
    if transform.e < 0:
        return transform, False

    new_f = transform.f + transform.e * (height - 1)
    new_transform = Affine(
        transform.a,
        transform.b,
        transform.c,
        transform.d,
        -transform.e,
        new_f,
    )

    return new_transform, True


def dequantize_and_write_image(
    source_raster_path: Path,
    mask_path: Path,
    target_grid_path: Path,
    mask_threshold: float,
    out_image_path: Path,
    force_north_up: bool,
    overwrite: bool,
) -> dict:

    if out_image_path.exists() and not overwrite:
        return {
            "out_image": str(out_image_path),
            "status": "skipped_exists",
            "valid_fraction": None,
            "north_up": None,
        }

    out_image_path.parent.mkdir(parents=True, exist_ok=True)
    out_nodata = -9999.0

    with (
        rasterio.open(source_raster_path) as src_ds,
        rasterio.open(mask_path) as mask_ds,
        rasterio.open(target_grid_path) as target_ds,
    ):
        if src_ds.crs is None:
            raise ValueError(f"Source CRS missing: {source_raster_path}")
        if mask_ds.crs is None:
            raise ValueError(f"Mask CRS missing: {mask_path}")
        if target_ds.crs is None:
            raise ValueError(f"Target grid CRS missing: {target_grid_path}")

        target_window = grid_window_for_source(target_ds, source_raster_path)
        if target_window is None or not source_overlaps_target_grid(target_ds, source_raster_path):
            print(f"[MASK CHECK] No overlap with target grid extent: {source_raster_path.name}")
            return {
                "out_image": str(out_image_path),
                "status": "skipped_no_overlap",
                "valid_fraction": None,
                "north_up": None,
                "output_crs": str(target_ds.crs),
            }

        target_window_transform = window_transform(target_window, target_ds.transform)
        target_window_height = int(target_window.height)
        target_window_width = int(target_window.width)
        out_transform = target_window_transform
        out_height = int(target_window_height)
        out_width = int(target_window_width)
        out_crs = target_ds.crs

        if rasters_are_aligned(mask_ds, target_ds):
            mask_arr = mask_ds.read(1, window=target_window)
        else:
            mask_arr = np.full((target_window_height, target_window_width), 0.0, dtype=np.float32)
            reproject(
                source=rasterio.band(mask_ds, 1),
                destination=mask_arr,
                src_transform=mask_ds.transform,
                src_crs=mask_ds.crs,
                src_nodata=mask_ds.nodata,
                dst_transform=out_transform,
                dst_crs=out_crs,
                dst_nodata=0.0,
                resampling=Resampling.nearest,
                num_threads=REPROJECT_NUM_THREADS,
            )
        mask_keep = np.isfinite(mask_arr) & (mask_arr > mask_threshold)

        raw_band1 = np.full((target_window_height, target_window_width), -128, dtype=np.int16)
        reproject(
            source=rasterio.band(src_ds, 1),
            destination=raw_band1,
            src_transform=src_ds.transform,
            src_crs=src_ds.crs,
            src_nodata=-128,
            dst_transform=out_transform,
            dst_crs=out_crs,
            dst_nodata=-128,
            resampling=Resampling.nearest,
            num_threads=REPROJECT_NUM_THREADS,
        )

        ae_valid = raw_band1 != -128

        valid = mask_keep & ae_valid

        did_flip = False
        if force_north_up:
            out_transform, did_flip = north_up_transform_if_needed(out_transform, out_height)

        profile = src_ds.profile.copy()
        profile.update(
            dtype="float32",
            nodata=out_nodata,
            compress="deflate",
            predictor=3,
            tiled=True,
            blockxsize=256,
            blockysize=256,
            crs=out_crs,
            transform=out_transform,
            height=out_height,
            width=out_width,
            BIGTIFF="YES",
        )

        with rasterio.open(out_image_path, "w", **profile) as out_ds:
            for band_idx in range(src_ds.count):
                if band_idx == 0:
                    raw_band = raw_band1
                else:
                    raw_band = np.full((target_window_height, target_window_width), -128, dtype=np.int16)
                    reproject(
                        source=rasterio.band(src_ds, band_idx + 1),
                        destination=raw_band,
                        src_transform=src_ds.transform,
                        src_crs=src_ds.crs,
                        src_nodata=-128,
                        dst_transform=out_transform,
                        dst_crs=out_crs,
                        dst_nodata=-128,
                        resampling=Resampling.nearest,
                        num_threads=REPROJECT_NUM_THREADS,
                    )

                band_f = raw_band.astype(np.float32)
                band_deq = ((band_f / 127.5) ** 2) * np.sign(band_f)
                band_deq[~valid] = out_nodata

                if did_flip:
                    band_deq = band_deq[::-1, :]

                out_ds.write(band_deq.astype(np.float32, copy=False), band_idx + 1)

            if src_ds.descriptions:
                for band_idx, desc in enumerate(src_ds.descriptions, start=1):
                    if desc:
                        out_ds.set_band_description(band_idx, desc)

        return {
            "out_image": str(out_image_path),
            "status": "written",
            "valid_fraction": float(valid.mean()),
            "north_up": did_flip,
            "output_crs": str(out_crs),
            "grid": {
                "height": out_height,
                "width": out_width,
                "transform": [float(v) for v in out_transform[:6]],
            },
            "window": {
                "row_off": int(target_window.row_off),
                "col_off": int(target_window.col_off),
                "height": target_window_height,
                "width": target_window_width,
            },
        }


def sanitize_crs_tag(crs_text: str) -> str:
    return crs_text.lower().replace(":", "")


def format_duration(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours, rem = divmod(total_seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def print_progress(done: int, total: int, start_ts: float) -> None:
    if total <= 0:
        return

    elapsed = time.time() - start_ts
    rate = done / elapsed if elapsed > 0 else 0.0
    remaining = max(total - done, 0)
    eta = (remaining / rate) if rate > 0 else 0.0

    frac = min(max(done / total, 0.0), 1.0)
    bar_width = 28
    filled = int(round(bar_width * frac))
    bar = "#" * filled + "-" * (bar_width - filled)

    print(
        f"[PROGRESS] |{bar}| {done}/{total} ({frac * 100:5.1f}%) "
        f"elapsed={format_duration(elapsed)} eta={format_duration(eta)}"
    )


def expected_dequant_output_path(
    deq_image_dir: Path,
    zone_name: str,
    source_raster: Path,
    target_crs: CRS,
) -> Path:

    crs_tag = sanitize_crs_tag(target_crs.to_string())
    return deq_image_dir / f"{zone_name}_{source_raster.stem}_{crs_tag}_dequant.tif"


def build_vrt(
    raster_paths: list[Path],
    vrt_path: Path,
    overwrite: bool,
    reference_raster_path: Path | None = None,
) -> None:

    if not raster_paths:
        raise ValueError("No raster paths provided for VRT build")

    if vrt_path.exists() and not overwrite:
        return

    vrt_path.parent.mkdir(parents=True, exist_ok=True)

    from osgeo import gdal

    gdal.UseExceptions()

    vrt_kwargs: dict[str, object] = {"allowProjectionDifference": True}

    if reference_raster_path is not None:
        with rasterio.open(reference_raster_path) as ref_ds:
            ref_bounds = ref_ds.bounds
            vrt_kwargs.update(
                {
                    "outputBounds": (ref_bounds.left, ref_bounds.bottom, ref_bounds.right, ref_bounds.top),
                    "xRes": abs(ref_ds.transform.a),
                    "yRes": abs(ref_ds.transform.e),
                    "outputSRS": ref_ds.crs.to_string() if ref_ds.crs else None,
                    "resolution": "user",
                }
            )

    opts = gdal.BuildVRTOptions(**vrt_kwargs)

    vrt = gdal.BuildVRT(
        str(vrt_path),
        [str(p) for p in raster_paths],
        options=opts,
    )

    if vrt is None:
        raise RuntimeError("GDAL BuildVRT returned None")

    vrt.FlushCache()
    vrt = None


def build_qc_cog_from_vrt(
    vrt_path: Path,
    qc_dir: Path,
    target_crs: CRS,
    qc_band: int,
    overwrite: bool,
) -> Path:

    qc_dir.mkdir(parents=True, exist_ok=True)

    crs_tag = sanitize_crs_tag(target_crs.to_string())
    cog_path = qc_dir / f"alphaearth_mosaic_{crs_tag}_band{qc_band}_qc_cog.tif"

    if cog_path.exists() and not overwrite:
        return cog_path

    from osgeo import gdal

    gdal.UseExceptions()

    options = gdal.TranslateOptions(
        format="COG",
        bandList=[qc_band],
        creationOptions=[
            "COMPRESS=DEFLATE",
            "PREDICTOR=3",
            "OVERVIEWS=AUTO",
            "BLOCKSIZE=512",
            "BIGTIFF=IF_SAFER",
            "RESAMPLING=AVERAGE",
        ],
    )

    ds = gdal.Translate(
        str(cog_path),
        str(vrt_path),
        options=options,
    )

    if ds is None:
        raise RuntimeError("GDAL Translate to COG returned None")

    ds = None

    return cog_path


def main() -> None:

    args = parse_args()
    if not args.alpha_root.exists():
        raise FileNotFoundError(f"Alpha root not found: {args.alpha_root}")

    args.mask_path = resolve_existing_raster_path(args.mask_path)
    if args.snap_raster_path is None:
        args.snap_raster_path = args.mask_path
    else:
        args.snap_raster_path = resolve_existing_raster_path(args.snap_raster_path)

    print(f"[INFO] Using mask raster: {args.mask_path}")
    print(f"[INFO] Using target grid raster: {args.snap_raster_path}")

    with rasterio.open(args.mask_path) as mask_ds:
        if mask_ds.crs is None:
            raise ValueError(f"Mask CRS missing: {args.mask_path}")
        mask_crs = CRS.from_user_input(mask_ds.crs)

    with rasterio.open(args.snap_raster_path) as snap_ds:
        if snap_ds.crs is None:
            raise ValueError(f"Target grid CRS missing: {args.snap_raster_path}")
        target_grid_crs = CRS.from_user_input(snap_ds.crs)

    requested_target_crs = CRS.from_user_input(args.target_crs)
    if requested_target_crs != target_grid_crs:
        print(
            f"[INFO] --target-crs ({requested_target_crs}) differs from target grid CRS ({target_grid_crs}). "
            "Using target grid CRS to guarantee perfect alignment with modeling grid."
        )
    target_crs = target_grid_crs

    zone_dirs = discover_zone_dirs(args.alpha_root, args.zones)
    zone_rasters_map = {zone_dir: discover_zone_rasters(zone_dir) for zone_dir in zone_dirs}
    total_rasters = sum(len(rasters) for rasters in zone_rasters_map.values())
    processed_rasters = 0
    progress_start_ts = time.time()

    deq_image_dir = args.output_root / "dequant_images_all"
    vrt_dir = args.output_root / "vrt"
    report_dir = args.output_root / "reports"
    qc_dir = args.output_root / "qc"

    report_dir.mkdir(parents=True, exist_ok=True)

    all_deq_images = []
    reports = []

    for zone_dir in zone_dirs:

        zone_name = zone_dir.name
        print(f"\n=== Processing zone folder: {zone_name} ===")

        rasters = zone_rasters_map[zone_dir]

        if not args.overwrite:
            expected_outputs = [
                expected_dequant_output_path(
                    deq_image_dir,
                    zone_name,
                    r,
                    target_crs,
                )
                for r in rasters
            ]

            if expected_outputs and all(p.exists() for p in expected_outputs):
                print(
                    f"Zone {zone_name}: all de-quantized images already exist, skipping zone."
                )
                all_deq_images.extend(expected_outputs)
                processed_rasters += len(rasters)
                print_progress(processed_rasters, total_rasters, progress_start_ts)
                continue

        zone_written = 0
        zone_skipped = 0

        for source_raster in rasters:

            source_tag = source_raster.stem
            print_mask_match_status(
                mask_path=args.mask_path,
                reference_raster_path=source_raster,
            )

            out_image_path = expected_dequant_output_path(
                deq_image_dir,
                zone_name,
                source_raster,
                target_crs,
            )

            result = dequantize_and_write_image(
                source_raster_path=source_raster,
                mask_path=args.mask_path,
                target_grid_path=args.snap_raster_path,
                mask_threshold=args.mask_threshold,
                out_image_path=out_image_path,
                force_north_up=args.force_north_up,
                overwrite=args.overwrite,
            )

            if result["status"] == "written":
                zone_written += 1
            else:
                zone_skipped += 1

            all_deq_images.append(out_image_path)

            reports.append(
                {
                    "zone": zone_name,
                    "source": str(source_raster),
                    "mask_path": str(args.mask_path),
                    "source_tag": source_tag,
                    **result,
                }
            )

            processed_rasters += 1
            print_progress(processed_rasters, total_rasters, progress_start_ts)

        print(
            f"Zone {zone_name}: dequant images written={zone_written}, skipped(existing)={zone_skipped}"
        )

    unique_images = sorted({p for p in all_deq_images if p.exists()})

    national_vrt_path = (
        vrt_dir
        / f"alphaearth_dequant_national_{sanitize_crs_tag(target_crs.to_string())}.vrt"
    )

    if args.skip_national_vrt:
        print("\n[INFO] Skipping national VRT by user option.")
    else:
        try:
            build_vrt(
                unique_images,
                national_vrt_path,
                overwrite=args.overwrite,
                reference_raster_path=args.snap_raster_path,
            )

            print(f"\nNational VRT ready: {national_vrt_path}")

            if args.skip_qc_cog:
                print("[INFO] Skipping QC COG by user option.")
            else:
                qc_cog_path = build_qc_cog_from_vrt(
                    vrt_path=national_vrt_path,
                    qc_dir=qc_dir,
                    target_crs=target_crs,
                    qc_band=args.qc_band,
                    overwrite=args.overwrite,
                )

                print(f"QC COG ready band {args.qc_band}: {qc_cog_path}")

        except Exception as exc:
            print(f"\n[WARNING] National VRT build failed: {exc}")
            print("[INFO] De-quantized images were still exported successfully.")

    summary_path = report_dir / "alphaearth_dequant_image_report.json"

    summary_path.write_text(
        json.dumps(reports, indent=2),
        encoding="utf-8",
    )

    print(f"Report written: {summary_path}")


if __name__ == "__main__":
    main()