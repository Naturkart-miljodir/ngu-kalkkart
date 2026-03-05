"""
Mask individual Trondelag Sentinel tiles (water + snow) and create VRT mosaic
Water: Masked using external water mask (set to NaN)
Snow: Detected via NDSI > 0.4, set to NaN
"""
from osgeo import gdal
import numpy as np
import os
import glob

# Configuration
tiles_dir = r"E:\Test\Trondelag_test\Tiles"
mask_src = r"F:\Topodata_basins\Mosaic_Norge\snap_raster\dtm_mask_srast_fixed.tif"
output_dir = r"E:\Test\Trondelag_test\masked"
NDSI_THRESHOLD = 0.4
BAND_3_IDX = 3    # Green (for NDSI)
BAND_11_IDX = 11  # SWIR1 (for NDSI)

# Create output directory
os.makedirs(output_dir, exist_ok=True)
os.chdir(tiles_dir)

gdal.UseExceptions()

# Step 1: Find all tiles and get combined bounds
tile_pattern = "tile_*.tif"
tiles = sorted(glob.glob(tile_pattern))

if not tiles:
    print(f"No tiles found matching pattern: {tile_pattern}")
    exit(1)

print(f"Step 1: Found {len(tiles)} tiles")
for t in tiles:
    print(f"  {t}")

# Get combined bounds and detect CRS from first tile
x_mins, y_mins, x_maxs, y_maxs = [], [], [], []
target_crs = None

for tile_path in tiles:
    ds = gdal.Open(tile_path)
    gt = ds.GetGeoTransform()
    xs, ys = ds.RasterXSize, ds.RasterYSize
    
    # Get CRS from first tile
    if target_crs is None:
        target_crs = ds.GetProjection()
        srs = gdal.osr.SpatialReference(wkt=target_crs)
        epsg_code = srs.GetAuthorityCode(None)
        print(f"  Target CRS: EPSG:{epsg_code}" if epsg_code else f"  Target CRS detected")
    
    x_min = gt[0]
    y_max = gt[3]
    x_max = x_min + xs * gt[1]
    y_min = y_max + ys * gt[5]
    
    x_mins.append(x_min)
    y_mins.append(y_min)
    x_maxs.append(x_max)
    y_maxs.append(y_max)
    
    # Get pixel resolution from first tile
    if tile_path == tiles[0]:
        pixel_res = abs(gt[1])
        print(f"  Pixel resolution: {pixel_res}m")
    
    ds = None

combined_bounds = [min(x_mins), min(y_mins), max(x_maxs), max(y_maxs)]
print(f"\nCombined extent: {combined_bounds}")

# Step 2: Align mask to combined bounds
mask_aligned = os.path.join(output_dir, "Trondelag_mask_aligned.tif")
if os.path.exists(mask_aligned):
    os.remove(mask_aligned)

print(f"\nStep 2: Aligning mask to Trondelag extent")

mask_opts = gdal.WarpOptions(
    format="GTiff",
    dstSRS=target_crs,  # Use detected CRS from tiles
    xRes=pixel_res,
    yRes=pixel_res,
    outputBounds=combined_bounds,
    resampleAlg="near",
    creationOptions=["COMPRESS=LZW", "TILED=YES", "BIGTIFF=YES"],
)

gdal.Warp(mask_aligned, mask_src, options=mask_opts)

mask_ds = gdal.Open(mask_aligned)
mask_nd = mask_ds.GetRasterBand(1).GetNoDataValue()
print(f"  Mask NoData value: {mask_nd}")

# Step 3: Mask each tile individually
masked_tiles = []

for tile_path in tiles:
    tile_name = os.path.basename(tile_path)
    base_name = os.path.splitext(tile_name)[0]
    masked_path = os.path.join(output_dir, f"Trondelag_{base_name}_masked.tif")
    
    if os.path.exists(masked_path):
        os.remove(masked_path)
    
    print(f"\nStep 3: Masking {tile_name}")
    
    # Open source tile
    src = gdal.Open(tile_path)
    xsize = src.RasterXSize
    ysize = src.RasterYSize
    bands = src.RasterCount
    
    # Get tile bounds for mask extraction
    gt = src.GetGeoTransform()
    x_min = gt[0]
    y_max = gt[3]
    x_max = x_min + xsize * gt[1]
    y_min = y_max + ysize * gt[5]
    
    # Create output as Float32 with NaN for NoData
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
    
    # Calculate mask offset in aligned mask coordinates
    mask_gt = mask_ds.GetGeoTransform()
    x_off = int((x_min - mask_gt[0]) / mask_gt[1])
    y_off = int((y_max - mask_gt[3]) / mask_gt[5])
    
    # Read mask for this tile's extent
    mask_band = mask_ds.GetRasterBand(1)
    mask_data = mask_band.ReadAsArray(x_off, y_off, xsize, ysize)
    
    # Compute NDSI for snow detection (bands 3 & 11)
    band3_data = src.GetRasterBand(BAND_3_IDX).ReadAsArray().astype(np.float32)
    band11_data = src.GetRasterBand(BAND_11_IDX).ReadAsArray().astype(np.float32)
    
    denom = band3_data + band11_data
    ndsi = np.full_like(band3_data, np.nan)
    valid_denom = denom != 0
    ndsi[valid_denom] = (band3_data[valid_denom] - band11_data[valid_denom]) / denom[valid_denom]
    
    # Snow mask: NDSI > threshold
    snow_mask = (ndsi > NDSI_THRESHOLD) & ~np.isnan(band3_data)
    snow_count = np.sum(snow_mask)
    print(f"    Snow pixels: {snow_count:,} ({100*snow_count/(xsize*ysize):.2f}%)")
    
    # Process each band
    for b in range(1, bands + 1):
        src_band = src.GetRasterBand(b)
        data = src_band.ReadAsArray().astype(np.float32)
        
        # Create binary mask: 1=land, 0=water/nodata
        binary = np.where(np.isclose(mask_data, mask_nd, rtol=1e-5), 0, 1).astype(np.uint8)
        
        # Also mask fake 0s and GDAL fill artifacts (>=60000)
        binary = np.where(data >= 60000, 0, binary).astype(np.uint8)
        binary = np.where(data == 0, 0, binary).astype(np.uint8)
        
        # Mask snow pixels (NDSI-based)
        binary = np.where(snow_mask, 0, binary).astype(np.uint8)
        
        # Apply mask: set NoData pixels to NaN, keep land values
        masked = np.where(binary == 1, data, np.nan).astype(np.float32)
        
        out_band = out.GetRasterBand(b)
        out_band.WriteArray(masked)
        out_band.SetNoDataValue(np.nan)
    
    src = None
    out = None
    
    masked_tiles.append(masked_path)
    print(f"  Created: {os.path.basename(masked_path)}")

mask_ds = None

# Step 4: Create VRT mosaic
vrt_path = os.path.join(output_dir, "Trondelag_mosaic.vrt")
if os.path.exists(vrt_path):
    os.remove(vrt_path)

print(f"\nStep 4: Creating VRT mosaic")
vrt_opts = gdal.BuildVRTOptions(resolution="highest", resampleAlg="near")
gdal.BuildVRT(vrt_path, masked_tiles, options=vrt_opts)

print(f"\nDone!")
print(f"  Masked tiles: {len(masked_tiles)}")
print(f"  Output directory: {output_dir}")
print(f"  VRT mosaic: {os.path.basename(vrt_path)}")
