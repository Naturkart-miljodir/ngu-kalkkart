"""
Mask National tiles: water mask + snow (NDSI)
Band mapping (image band -> Sentinel band):
1->B2, 2->B3, 3->B4, 4->B5, 5->B6, 6->B7, 7->B8, 8->B8A, 9->B11, 10->B12
NDSI uses B3 (green) and B11 (SWIR1): image bands 2 and 9.
Applies water mask + NDSI snow masking to nationwide Sentinel tiles.
"""
from osgeo import gdal
import numpy as np
import os
import glob
from datetime import datetime

# Configuration
tiles_dir = r"E:\Sentinel_tiles"
mask_src = r"F:\Topodata_basins\Mosaic_Norge\snap_raster\dtm_mask_srast_fixed.tif"
output_dir = r"E:\Test\National_test\masked_snow"

# Resume/logging settings
SKIP_EXISTING = True
REUSE_EXISTING_MASK = False
LOG_PATH = os.path.join(output_dir, "mask_national_tiles_snow_ndsi.log")

NDSI_THRESHOLD = 0.4
BAND_GREEN_IDX = 2  # image band 2 -> Sentinel B3
BAND_SWIR_IDX = 9   # image band 9 -> Sentinel B11

# Optional secondary snow filter (conservative; complements base NDSI filter)
# Keep OFF for baseline reproducibility, turn ON only when residual snow remains.
USE_SECONDARY_SNOW_FILTER = False
SECONDARY_NDSI_THRESHOLD = 0.42
SECONDARY_VIS_MEAN_MIN = 1800.0
SECONDARY_SWIR1_MAX = 2200.0
SECONDARY_NDVI_MAX = 0.15

# Additional bands used by secondary filter
BAND_BLUE_IDX = 1   # image band 1 -> Sentinel B2
BAND_RED_IDX = 3    # image band 3 -> Sentinel B4
BAND_NIR_IDX = 7    # image band 7 -> Sentinel B8

os.makedirs(output_dir, exist_ok=True)
os.chdir(tiles_dir)

gdal.UseExceptions()

def log_line(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")

# Step 1: Find tiles and bounds
pattern = "tile_*.tif"
tiles = sorted(glob.glob(pattern))
if not tiles:
    raise RuntimeError(f"No tiles found matching pattern: {pattern}")

log_line(f"Step 1: Found {len(tiles)} tiles")
for t in tiles:
    log_line(f"  {t}")

x_mins, y_mins, x_maxs, y_maxs = [], [], [], []
target_crs = None
pixel_res = None

for tile_path in tiles:
    ds = gdal.Open(tile_path)
    gt = ds.GetGeoTransform()
    xs, ys = ds.RasterXSize, ds.RasterYSize

    if target_crs is None:
        target_crs = ds.GetProjection()
        srs = gdal.osr.SpatialReference(wkt=target_crs)
        epsg_code = srs.GetAuthorityCode(None)
        log_line(f"  Target CRS: EPSG:{epsg_code}" if epsg_code else "  Target CRS detected")

    x_min = gt[0]
    y_max = gt[3]
    x_max = x_min + xs * gt[1]
    y_min = y_max + ys * gt[5]

    x_mins.append(x_min)
    y_mins.append(y_min)
    x_maxs.append(x_max)
    y_maxs.append(y_max)

    if pixel_res is None:
        pixel_res = abs(gt[1])
        log_line(f"  Pixel resolution: {pixel_res}m")

    ds = None

combined_bounds = [min(x_mins), min(y_mins), max(x_maxs), max(y_maxs)]
log_line(f"Combined extent: {combined_bounds}")

# Step 2: Align mask to combined bounds
mask_aligned = os.path.join(output_dir, "National_mask_aligned.tif")
if os.path.exists(mask_aligned) and REUSE_EXISTING_MASK:
    log_line(f"Step 2: Using existing aligned mask: {mask_aligned}")
else:
    if os.path.exists(mask_aligned):
        try:
            os.remove(mask_aligned)
        except PermissionError:
            log_line(f"  Note: Could not delete existing {os.path.basename(mask_aligned)}, will overwrite")

    log_line("Step 2: Aligning mask to National extent")

    mask_opts = gdal.WarpOptions(
        format="GTiff",
        dstSRS=target_crs,
        xRes=pixel_res,
        yRes=pixel_res,
        outputBounds=combined_bounds,
        resampleAlg="near",
        creationOptions=["COMPRESS=LZW", "TILED=YES", "BIGTIFF=YES"],
    )

    gdal.Warp(mask_aligned, mask_src, options=mask_opts)

mask_ds = gdal.Open(mask_aligned)
mask_nd = mask_ds.GetRasterBand(1).GetNoDataValue()
log_line(f"  Mask NoData value: {mask_nd}")
log_line(
    f"  Secondary snow filter: {'ON' if USE_SECONDARY_SNOW_FILTER else 'OFF'}"
)

# Step 3: Mask each tile
masked_tiles = []
if SKIP_EXISTING:
    existing_masked = sorted(glob.glob(os.path.join(output_dir, "National_*_masked_snow.tif")))
    masked_tiles.extend(existing_masked)
    log_line(f"Pre-existing masked tiles: {len(existing_masked)}")

for tile_path in tiles:
    tile_name = os.path.basename(tile_path)
    base_name = os.path.splitext(tile_name)[0]
    masked_path = os.path.join(output_dir, f"National_{base_name}_masked_snow.tif")

    if os.path.exists(masked_path):
        if SKIP_EXISTING:
            log_line(f"Step 3: Skipping {tile_name} (already masked)")
            continue
        try:
            os.remove(masked_path)
        except PermissionError:
            log_line(f"  Note: Could not delete existing {os.path.basename(masked_path)}, will overwrite")

    log_line(f"Step 3: Masking {tile_name}")

    try:
        src = gdal.Open(tile_path)
        if src is None:
            raise RuntimeError("Could not open tile")
        xsize = src.RasterXSize
        ysize = src.RasterYSize
        bands = src.RasterCount

        gt = src.GetGeoTransform()
        x_min = gt[0]
        y_max = gt[3]

        driver = gdal.GetDriverByName("GTiff")
        out = driver.Create(
            masked_path,
            xsize,
            ysize,
            bands,
            gdal.GDT_Float32,
            options=["COMPRESS=LZW", "TILED=YES", "BIGTIFF=YES", "BLOCKXSIZE=512", "BLOCKYSIZE=512"],
        )

        out.SetGeoTransform(src.GetGeoTransform())
        out.SetProjection(src.GetProjection())

        # Mask offset
        mask_gt = mask_ds.GetGeoTransform()
        x_off = int((x_min - mask_gt[0]) / mask_gt[1])
        y_off = int((y_max - mask_gt[3]) / mask_gt[5])

        mask_band = mask_ds.GetRasterBand(1)
        mask_data = mask_band.ReadAsArray(x_off, y_off, xsize, ysize)

        # Binary land mask
        binary = np.where(np.isclose(mask_data, mask_nd, rtol=1e-5), 0, 1).astype(np.uint8)

        # NDSI (compute only on valid land pixels to avoid water/artefact false positives)
        band_green = src.GetRasterBand(BAND_GREEN_IDX).ReadAsArray().astype(np.float32)
        band_swir = src.GetRasterBand(BAND_SWIR_IDX).ReadAsArray().astype(np.float32)

        denom = band_green + band_swir
        ndsi = np.full_like(band_green, np.nan)
        artefact = (band_green == 0) | (band_swir == 0) | (band_green >= 60000) | (band_swir >= 60000)
        valid = (binary == 1) & (denom != 0) & (~artefact)
        ndsi[valid] = (band_green[valid] - band_swir[valid]) / denom[valid]

        snow_mask_primary = valid & (ndsi > NDSI_THRESHOLD)

        if USE_SECONDARY_SNOW_FILTER:
            band_blue = src.GetRasterBand(BAND_BLUE_IDX).ReadAsArray().astype(np.float32)
            band_red = src.GetRasterBand(BAND_RED_IDX).ReadAsArray().astype(np.float32)
            band_nir = src.GetRasterBand(BAND_NIR_IDX).ReadAsArray().astype(np.float32)

            finite_secondary = (
                np.isfinite(band_blue)
                & np.isfinite(band_green)
                & np.isfinite(band_red)
                & np.isfinite(band_nir)
                & np.isfinite(band_swir)
            )

            # Ratio-free inequalities to avoid instability:
            # NDSI >= t  <=>  (G - S) >= t*(G + S)
            # NDVI <= t  <=>  (N - R) <= t*(N + R)
            ndsi_cond = (band_green - band_swir) >= (
                SECONDARY_NDSI_THRESHOLD * (band_green + band_swir)
            )
            ndvi_cond = (band_nir - band_red) <= (
                SECONDARY_NDVI_MAX * (band_nir + band_red)
            )
            vis_mean = (band_blue + band_green + band_red) / 3.0

            snow_mask_secondary = (
                (binary == 1)
                & finite_secondary
                & (~artefact)
                & ndsi_cond
                & (vis_mean >= SECONDARY_VIS_MEAN_MIN)
                & (band_swir <= SECONDARY_SWIR1_MAX)
                & ndvi_cond
            )
            snow_mask = snow_mask_primary | snow_mask_secondary
        else:
            snow_mask_secondary = np.zeros_like(snow_mask_primary, dtype=bool)
            snow_mask = snow_mask_primary

        snow_count = int(np.sum(snow_mask))
        snow_primary_count = int(np.sum(snow_mask_primary))
        snow_secondary_count = int(np.sum(snow_mask_secondary))
        snow_secondary_only = int(np.sum(snow_mask_secondary & (~snow_mask_primary)))
        land_count = int(np.sum(binary == 1))
        valid_count = int(np.sum(valid))
        snow_pct_land = (100 * snow_count / land_count) if land_count else 0.0
        snow_pct_valid = (100 * snow_count / valid_count) if valid_count else 0.0
        log_line(f"    Snow pixels (final): {snow_count:,} ({snow_pct_land:.2f}% of land, {snow_pct_valid:.2f}% of valid)")
        if USE_SECONDARY_SNOW_FILTER:
            log_line(
                f"    Primary: {snow_primary_count:,} | "
                f"Secondary: {snow_secondary_count:,} | "
                f"Secondary-only add: {snow_secondary_only:,}"
            )

        for b in range(1, bands + 1):
            data = src.GetRasterBand(b).ReadAsArray().astype(np.float32)

            # filter artifacts
            bin_local = binary
            bin_local = np.where(data >= 60000, 0, bin_local).astype(np.uint8)
            bin_local = np.where(data == 0, 0, bin_local).astype(np.uint8)
            bin_local = np.where(snow_mask, 0, bin_local).astype(np.uint8)

            masked = np.where(bin_local == 1, data, np.nan).astype(np.float32)

            out_band = out.GetRasterBand(b)
            out_band.WriteArray(masked)
            out_band.SetNoDataValue(np.nan)

        src = None
        out = None

        # Verify masked tile has valid data (not all NaN)
        check_ds = gdal.Open(masked_path)
        check_band = check_ds.GetRasterBand(1)
        check_data = check_band.ReadAsArray()
        valid_pixels = (~np.isnan(check_data)).sum()
        check_ds = None
        
        if valid_pixels == 0:
            log_line(f"  CRITICAL ERROR: Tile {os.path.basename(masked_path)} is 100% NaN - mask misalignment detected!")
            log_line(f"  Stopping process to prevent further corrupted output.")
            raise RuntimeError(f"All-NaN tile created: {masked_path} - Mask may be misaligned")

        masked_tiles.append(masked_path)
        log_line(f"  Created: {os.path.basename(masked_path)}")
    except Exception as exc:
        log_line(f"  ERROR masking {tile_name}: {exc}")
        try:
            src = None
            out = None
        except Exception:
            pass
        continue

mask_ds = None

# Step 4: VRT
vrt_path = os.path.join(output_dir, "National_mosaic_snow.vrt")
if os.path.exists(vrt_path):
    try:
        os.remove(vrt_path)
    except PermissionError:
        log_line(f"  Note: Could not delete existing {os.path.basename(vrt_path)}, will overwrite")

log_line("Step 4: Creating VRT mosaic")
vrt_opts = gdal.BuildVRTOptions(resolution="highest", resampleAlg="near")
gdal.BuildVRT(vrt_path, masked_tiles, options=vrt_opts)

log_line("Done!")
log_line(f"  Masked tiles: {len(masked_tiles)}")
log_line(f"  Output directory: {output_dir}")
