"""
Check if the underlying Sentinel tiles and mask have actual data
"""
import os
os.add_dll_directory(r'C:\ProgramData\anaconda3\envs\Kalk_DL_tiling\Library\bin')

from osgeo import gdal
import numpy as np

print("Checking data sources...\n")

# Check temp_mosaic.vrt (Sentinel tiles)
print("1. Checking Sentinel mosaic (temp_mosaic.vrt)...")
sentinel_vrt = gdal.Open(r'E:\Test\sentinel_reprojected\temp_mosaic.vrt')
band1 = sentinel_vrt.GetRasterBand(1)
# Read same area as test export
data = band1.ReadAsArray(13000, 14000, 2000, 2000)
print(f"   Min: {data.min()}, Max: {data.max()}, Mean: {data.mean():.2f}")
print(f"   Non-zero pixels: {(data > 0).sum()} ({(data > 0).sum()/data.size*100:.1f}%)")
sentinel_vrt = None

# Check mask
print("\n2. Checking DTM mask...")
mask = gdal.Open(r'F:\Topodata_basins\Mosaic_Norge\snap_raster\dtm_mask_srast_fixed.tif')
print(f"   Mask size: {mask.RasterXSize} x {mask.RasterYSize}")
print(f"   Mask geotransform: {mask.GetGeoTransform()}")

# Check masked VRT
print("\n3. Checking Sentinel mosaic geotransform...")
sentinel_vrt = gdal.Open(r'E:\Test\sentinel_reprojected\temp_mosaic.vrt')
print(f"   Sentinel size: {sentinel_vrt.RasterXSize} x {sentinel_vrt.RasterYSize}")
print(f"   Sentinel geotransform: {sentinel_vrt.GetGeoTransform()}")
sentinel_vrt = None

print("\n4. Checking masked VRT...")
masked_vrt = gdal.Open(r'E:\Test\sentinel_reprojected\Sentinel_masked_final.vrt')
band1 = masked_vrt.GetRasterBand(1)
data = band1.ReadAsArray(13000, 14000, 2000, 2000)
print(f"   Min: {data.min()}, Max: {data.max()}, Mean: {data.mean():.2f}")
print(f"   Non-zero pixels: {(data > 0).sum()} ({(data > 0).sum()/data.size*100:.1f}%)")
masked_vrt = None

mask = None
