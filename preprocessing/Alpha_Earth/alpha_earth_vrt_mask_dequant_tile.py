# pyright: reportMissingImports=false, reportMissingModuleSource=false

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import array_bounds
from rasterio.warp import calculate_default_transform, reproject


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
        default=Path(r"E:\Test\National_test\Mask\National_mask_aligned.tif"),
        help="Mask raster that will be reprojected/aligned to each AlphaEarth source grid.",
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
        help="Flip south-up outputs to north-up without resampling (recommended for easier VRT/GIS use).",
    )
    parser.add_argument(
        "--target-crs",
        type=str,
        default="EPSG:25833",
        help="Final output CRS for de-quantized images (default: EPSG:25833).",
    )
    parser.add_argument(
        "--target-resolution",
        type=float,
        default=10.0,
        help="Final output pixel size in target CRS units (default: 10.0).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing aligned masks / de-quantized images / VRT.",
    )
    parser.add_argument(
        "--skip-national-vrt",
        action="store_true",
        help="Skip building the national VRT from de-quantized image outputs.",
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
        help="Band index (1-based) to export as national QC COG (default: 1).",
    )
    return parser.parse_args()


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

    rasters: list[Path] = []
    for path in preferred + others:
        try:
            with rasterio.open(path):
                rasters.append(path)
        except Exception:
            continue

    if not rasters:
        raise FileNotFoundError(f"No readable rasters found in {zone_dir}")

    return rasters


def align_mask_to_reference(
    mask_path: Path,
    reference_raster_path: Path,
    aligned_mask_path: Path,
    mask_threshold: float,
    overwrite: bool,
) -> None:
    if aligned_mask_path.exists() and not overwrite:
        return

    aligned_mask_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(reference_raster_path) as ref_ds, rasterio.open(mask_path) as mask_ds:
        source = rasterio.band(mask_ds, 1)
        destination = np.zeros((ref_ds.height, ref_ds.width), dtype=np.float32)

        reproject(
            source=source,
            destination=destination,
            src_transform=mask_ds.transform,
            src_crs=mask_ds.crs,
            src_nodata=mask_ds.nodata,
            dst_transform=ref_ds.transform,
            dst_crs=ref_ds.crs,
            dst_nodata=0,
            resampling=Resampling.nearest,
        )

        keep = np.isfinite(destination) & (destination > mask_threshold)
        out = keep.astype(np.uint8)

        profile = ref_ds.profile.copy()
        profile.update(
            count=1,
            dtype="uint8",
            nodata=0,
            compress="deflate",
            predictor=2,
            tiled=True,
            blockxsize=256,
            blockysize=256,
        )

        with rasterio.open(aligned_mask_path, "w", **profile) as out_ds:
            out_ds.write(out, 1)


def north_up_transform_if_needed(transform: Affine, height: int) -> tuple[Affine, bool]:
    if transform.e < 0:
        return transform, False

    new_f = transform.f + transform.e * (height - 1)
    new_transform = Affine(transform.a, transform.b, transform.c, transform.d, -transform.e, new_f)
    return new_transform, True


def dequantize_and_write_image(
    source_raster_path: Path,
    aligned_mask_path: Path,
    out_image_path: Path,
    force_north_up: bool,
    target_crs: CRS,
    target_resolution: float,
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

    with rasterio.open(source_raster_path) as src_ds, rasterio.open(aligned_mask_path) as mask_ds:
        if src_ds.width != mask_ds.width or src_ds.height != mask_ds.height:
            raise ValueError(
                f"Grid mismatch between source and aligned mask: {source_raster_path.name}"
            )

        raw = src_ds.read().astype(np.int16)
        mask_keep = mask_ds.read(1).astype(bool)

        ae_valid = np.all(raw != -128, axis=0)
        valid = mask_keep & ae_valid

        raw_f = raw.astype(np.float32)
        deq = ((raw_f / 127.5) ** 2) * np.sign(raw_f)
        deq[:, ~valid] = out_nodata

        out_transform = src_ds.transform
        did_flip = False
        if force_north_up:
            out_transform, did_flip = north_up_transform_if_needed(src_ds.transform, src_ds.height)
            if did_flip:
                deq = deq[:, ::-1, :]

        source_crs = src_ds.crs
        if source_crs is None:
            raise ValueError(f"Source CRS missing: {source_raster_path}")

        out_crs = source_crs
        out_height = src_ds.height
        out_width = src_ds.width

        if source_crs != target_crs:
            left, bottom, right, top = array_bounds(src_ds.height, src_ds.width, out_transform)
            dst_transform, dst_width, dst_height = calculate_default_transform(
                source_crs,
                target_crs,
                src_ds.width,
                src_ds.height,
                left,
                bottom,
                right,
                top,
                resolution=(target_resolution, target_resolution),
            )

            reprojected = np.full(
                (deq.shape[0], dst_height, dst_width), out_nodata, dtype=np.float32
            )
            for band_idx in range(deq.shape[0]):
                reproject(
                    source=deq[band_idx],
                    destination=reprojected[band_idx],
                    src_transform=out_transform,
                    src_crs=source_crs,
                    src_nodata=out_nodata,
                    dst_transform=dst_transform,
                    dst_crs=target_crs,
                    dst_nodata=out_nodata,
                    resampling=Resampling.nearest,
                )

            deq = reprojected
            out_transform = dst_transform
            out_crs = target_crs
            out_height = dst_height
            out_width = dst_width

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
            out_ds.write(deq.astype(np.float32, copy=False))
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
        }


def sanitize_crs_tag(crs_text: str) -> str:
    return crs_text.lower().replace(":", "")


def expected_dequant_output_path(
    deq_image_dir: Path,
    zone_name: str,
    source_raster: Path,
    target_crs: CRS,
) -> Path:
    crs_tag = sanitize_crs_tag(target_crs.to_string())
    return deq_image_dir / f"{zone_name}_{source_raster.stem}_{crs_tag}_dequant.tif"


def build_vrt(raster_paths: list[Path], vrt_path: Path, overwrite: bool) -> None:
    if not raster_paths:
        raise ValueError("No raster paths provided for VRT build")

    if vrt_path.exists() and not overwrite:
        return

    vrt_path.parent.mkdir(parents=True, exist_ok=True)

    from osgeo import gdal

    gdal.UseExceptions()
    opts = gdal.BuildVRTOptions(allowProjectionDifference=True)
    vrt = gdal.BuildVRT(str(vrt_path), [str(p) for p in raster_paths], options=opts)
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
    ds = gdal.Translate(str(cog_path), str(vrt_path), options=options)
    if ds is None:
        raise RuntimeError("GDAL Translate to COG returned None")
    ds = None

    return cog_path


def main() -> None:
    args = parse_args()
    target_crs = CRS.from_user_input(args.target_crs)

    if not args.alpha_root.exists():
        raise FileNotFoundError(f"Alpha root not found: {args.alpha_root}")
    if not args.mask_path.exists():
        raise FileNotFoundError(f"Mask raster not found: {args.mask_path}")

    zone_dirs = discover_zone_dirs(args.alpha_root, args.zones)

    aligned_mask_dir = args.output_root / "mask_aligned"
    deq_image_dir = args.output_root / "dequant_images_all"
    vrt_dir = args.output_root / "vrt"
    report_dir = args.output_root / "reports"
    qc_dir = args.output_root / "qc"
    report_dir.mkdir(parents=True, exist_ok=True)

    all_deq_images: list[Path] = []
    reports: list[dict] = []

    for zone_dir in zone_dirs:
        zone_name = zone_dir.name
        print(f"\n=== Processing zone folder: {zone_name} ===")

        rasters = discover_zone_rasters(zone_dir)

        if not args.overwrite:
            expected_outputs = [
                expected_dequant_output_path(deq_image_dir, zone_name, r, target_crs) for r in rasters
            ]
            if expected_outputs and all(p.exists() for p in expected_outputs):
                print(f"Zone {zone_name}: all de-quantized images already exist, skipping zone.")
                all_deq_images.extend(expected_outputs)
                continue

        zone_written = 0
        zone_skipped = 0

        for source_raster in rasters:
            source_tag = source_raster.stem
            aligned_mask_path = aligned_mask_dir / f"mask_to_{zone_name}_{source_tag}.tif"
            align_mask_to_reference(
                mask_path=args.mask_path,
                reference_raster_path=source_raster,
                aligned_mask_path=aligned_mask_path,
                mask_threshold=args.mask_threshold,
                overwrite=args.overwrite,
            )

            out_image_path = expected_dequant_output_path(
                deq_image_dir, zone_name, source_raster, target_crs
            )
            result = dequantize_and_write_image(
                source_raster_path=source_raster,
                aligned_mask_path=aligned_mask_path,
                out_image_path=out_image_path,
                force_north_up=args.force_north_up,
                target_crs=target_crs,
                target_resolution=args.target_resolution,
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
                    "aligned_mask": str(aligned_mask_path),
                    **result,
                }
            )

        print(f"Zone {zone_name}: dequant images written={zone_written}, skipped(existing)={zone_skipped}")

    unique_images = sorted({p for p in all_deq_images if p.exists()})
    national_vrt_path = vrt_dir / f"alphaearth_dequant_national_{sanitize_crs_tag(target_crs.to_string())}.vrt"

    if args.skip_national_vrt:
        print("\n[INFO] Skipping national VRT by user option.")
    else:
        try:
            build_vrt(unique_images, national_vrt_path, overwrite=args.overwrite)
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
                print(f"QC COG ready (band {args.qc_band}): {qc_cog_path}")
        except Exception as exc:
            print(f"\n[WARNING] National VRT build failed: {exc}")
            print("[INFO] De-quantized images were still exported successfully.")

    summary_path = report_dir / "alphaearth_dequant_image_report.json"
    summary_path.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(f"Report written: {summary_path}")


if __name__ == "__main__":
    main()
