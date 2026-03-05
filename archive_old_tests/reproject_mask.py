"""
Reproject and clip mask to match Sentinel tiles exactly
"""
import os
os.add_dll_directory(r'C:\ProgramData\anaconda3\envs\Kalk_DL_tiling\Library\bin')

from osgeo import gdal

print("Reprojecting mask to match Sentinel CRS...")

# Input/output paths
mask_input = r'F:\Topodata_basins\Mosaic_Norge\snap_raster\dtm_mask_srast_fixed.tif'
mask_output = r'E:\Test\sentinel_reprojected\dtm_mask_etrs89.tif'
sentinel_vrt = r'E:\Test\sentinel_reprojected\temp_mosaic.vrt'

# Get Sentinel extent and CRS
sentinel = gdal.Open(sentinel_vrt)
s_gt = sentinel.GetGeoTransform()
s_proj = sentinel.GetProjection()
s_width = sentinel.RasterXSize
s_height = sentinel.RasterYSize

# Calculate extent
s_minx = s_gt[0]
s_maxx = s_gt[0] + s_width * s_gt[1]
s_maxy = s_gt[3]
s_miny = s_gt[3] + s_height * s_gt[5]

print(f"Target extent: ({s_minx}, {s_miny}, {s_maxx}, {s_maxy})")
print(f"Target size: {s_width} x {s_height}")
print(f"Target resolution: 10m")

# Warp mask to match Sentinel exactly
warp_options = gdal.WarpOptions(
    format='GTiff',
    dstSRS=s_proj,
    outputBounds=(s_minx, s_miny, s_maxx, s_maxy),
    xRes=10, yRes=10,
    targetAlignedPixels=True,
    resampleAlg='near',  # Nearest neighbor for binary mask
    creationOptions=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=IF_SAFER']
)

ds = gdal.Warp(mask_output, mask_input, options=warp_options)

if ds is None:
    print("ERROR: Failed to reproject mask!")
else:
    print(f"\n✓ Mask reprojected successfully:")
    print(f"  Output: {mask_output}")
    print(f"  Size: {ds.RasterXSize} x {ds.RasterYSize}")
    print(f"  Resolution: {ds.GetGeoTransform()[1]}, {ds.GetGeoTransform()[5]}")
    ds = None

sentinel = None
print("\nNow creating new masked VRT with reprojected mask...")
