import argparse
import shutil
from pathlib import Path

from osgeo import gdal
import rasterio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reproject and align topographic rasters to EPSG:25833 using a snap raster "
            "and export Cloud Optimized GeoTIFFs."
        )
    )
    parser.add_argument(
        "--input-dirs",
        nargs="+",
        type=Path,
        default=[
            Path(r"F:\Topodata_basins\Mosaic_Norge\Final_rasts_Jose"),
            Path(r"F:\Topodata_tiles\Final_rasters_Bertil"),
        ],
        help="One or more folders containing input GeoTIFF rasters.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            r"C:\Users\acosta_pedro\OneDrive - Norges geologiske undersøkelse\Geochemistry NGU_2025\kalk_prosjekt3.0\National\Topography_25833"
        ),
        help="Folder where aligned COGs will be written.",
    )
    parser.add_argument(
        "--snap-raster",
        type=Path,
        default=Path(r"E:\Alpha_earth\qc\alphaearth_mosaic_epsg25833_band1_qc_cog"),
        help="Snap raster path (with or without .tif extension).",
    )
    parser.add_argument(
        "--resampling",
        choices=["nearest", "bilinear", "cubic", "cubicspline", "lanczos", "average", "mode"],
        default="bilinear",
        help="Resampling method for reprojection/alignment.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs.",
    )
    return parser.parse_args()


def resolve_snap_path(path: Path) -> Path:
    if path.exists():
        return path

    tif_candidate = path.with_suffix(".tif")
    if tif_candidate.exists():
        return tif_candidate

    tiff_candidate = path.with_suffix(".tiff")
    if tiff_candidate.exists():
        return tiff_candidate

    raise FileNotFoundError(f"Snap raster not found: {path}")


def list_rasters(folder: Path) -> list[Path]:
    if not folder.exists():
        raise FileNotFoundError(f"Input folder not found: {folder}")

    rasters = []
    for ext in ("*.tif", "*.tiff", "*.TIF", "*.TIFF"):
        rasters.extend(folder.rglob(ext))

    seen = set()
    unique = []
    for raster in sorted(rasters):
        if raster not in seen:
            unique.append(raster)
            seen.add(raster)

    return unique


def output_path_for(input_root: Path, src: Path, output_dir: Path) -> Path:
    relative = src.relative_to(input_root)
    subdir = output_dir / input_root.name / relative.parent
    out_name = f"{relative.stem}_epsg25833_cog.tif"
    return subdir / out_name


def warp_to_snap_and_translate_cog(
    src: Path,
    dst: Path,
    snap_crs: str,
    snap_bounds: tuple[float, float, float, float],
    snap_width: int,
    snap_height: int,
    resampling: str,
) -> None:
    temp_path = dst.with_suffix(".tmp.tif")

    src_ds = gdal.Open(str(src), gdal.GA_ReadOnly)
    if src_ds is None:
        raise RuntimeError(f"Could not open source raster: {src}")

    nodata = None
    band1 = src_ds.GetRasterBand(1)
    if band1 is not None:
        nodata = band1.GetNoDataValue()

    warp_options = gdal.WarpOptions(
        format="GTiff",
        dstSRS=snap_crs,
        outputBounds=snap_bounds,
        width=snap_width,
        height=snap_height,
        resampleAlg=resampling,
        srcNodata=nodata,
        dstNodata=nodata,
        multithread=True,
        creationOptions=[
            "COMPRESS=DEFLATE",
            "PREDICTOR=2",
            "TILED=YES",
            "BLOCKXSIZE=512",
            "BLOCKYSIZE=512",
            "BIGTIFF=IF_SAFER",
        ],
    )

    warped = gdal.Warp(str(temp_path), src_ds, options=warp_options)
    if warped is None:
        src_ds = None
        raise RuntimeError(f"Warp failed for {src}")
    warped = None
    src_ds = None

    translate_options = gdal.TranslateOptions(
        format="COG",
        creationOptions=[
            "COMPRESS=DEFLATE",
            "PREDICTOR=2",
            "BLOCKSIZE=512",
            "OVERVIEWS=AUTO",
            "BIGTIFF=IF_SAFER",
            "RESAMPLING=AVERAGE",
        ],
    )

    cog = gdal.Translate(str(dst), str(temp_path), options=translate_options)
    if cog is None:
        raise RuntimeError(f"COG creation failed for {src}")
    cog = None

    if temp_path.exists():
        temp_path.unlink()


def main() -> None:
    args = parse_args()
    gdal.UseExceptions()

    snap_path = resolve_snap_path(args.snap_raster)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(snap_path) as snap_ds:
        snap_crs = snap_ds.crs.to_string() if snap_ds.crs else None
        if snap_crs is None:
            raise ValueError(f"Snap raster has no CRS: {snap_path}")
        snap_bounds = tuple(snap_ds.bounds)
        snap_width = snap_ds.width
        snap_height = snap_ds.height

    print(f"Snap raster: {snap_path}")
    print(f"Snap CRS: {snap_crs}")
    print(f"Snap size: {snap_width} x {snap_height}")

    all_rasters: list[tuple[Path, Path]] = []
    for input_dir in args.input_dirs:
        rasters = list_rasters(input_dir)
        print(f"Found {len(rasters)} rasters in {input_dir}")
        for raster in rasters:
            all_rasters.append((input_dir, raster))

    if not all_rasters:
        print("No input rasters found. Nothing to do.")
        return

    written = 0
    skipped = 0
    failed = 0

    for index, (input_root, src) in enumerate(all_rasters, start=1):
        dst = output_path_for(input_root, src, args.output_dir)
        dst.parent.mkdir(parents=True, exist_ok=True)

        if dst.exists() and not args.overwrite:
            skipped += 1
            print(f"[{index}/{len(all_rasters)}] SKIP {src.name} -> {dst}")
            continue

        if dst.exists() and args.overwrite:
            dst.unlink()

        temp_path = dst.with_suffix(".tmp.tif")
        if temp_path.exists():
            temp_path.unlink()

        try:
            warp_to_snap_and_translate_cog(
                src=src,
                dst=dst,
                snap_crs=snap_crs,
                snap_bounds=snap_bounds,
                snap_width=snap_width,
                snap_height=snap_height,
                resampling=args.resampling,
            )
            written += 1
            print(f"[{index}/{len(all_rasters)}] OK   {src.name} -> {dst}")
        except Exception as exc:
            failed += 1
            print(f"[{index}/{len(all_rasters)}] FAIL {src} :: {exc}")
            if temp_path.exists():
                temp_path.unlink()

    print("\n=== Summary ===")
    print(f"Written: {written}")
    print(f"Skipped: {skipped}")
    print(f"Failed: {failed}")
    print(f"Output dir: {args.output_dir}")


if __name__ == "__main__":
    main()
