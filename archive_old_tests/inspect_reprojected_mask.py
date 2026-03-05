"""
Check dtm_mask_etrs89.tif values in detail
"""
import os
os.add_dll_directory(r'C:\ProgramData\anaconda3\envs\Kalk_DL_tiling\Library\bin')

from osgeo import gdal
import numpy as np

mask_file = r'E:\Test\sentinel_reprojected\dtm_mask_etrs89.tif'

print("Checking reprojected mask file...\n")

ds = gdal.Open(mask_file)
if ds is None:
    print(f"ERROR: Cannot open {mask_file}")
    exit(1)

band = ds.GetRasterBand(1)
print(f"Mask info:")
print(f"  Size: {ds.RasterXSize} x {ds.RasterYSize}")
print(f"  Data type: {band.DataType}")
print(f"  NoData value: {band.GetNoDataValue()}")

# Read full array
data = band.ReadAsArray()

print(f"\nData analysis:")
print(f"  Array shape: {data.shape}")
print(f"  Array dtype: {data.dtype}")
print(f"  Unique values: {np.unique(data)}")

# Show distribution
unique, counts = np.unique(data, return_counts=True)
print(f"\nValue distribution:")
for val, count in zip(unique, counts):
    pct = count / data.size * 100
    print(f"  {val}: {count} pixels ({pct:.2f}%)")

# Show sample values from different locations
print(f"\nSample values:")
print(f"  Center: {data[data.shape[0]//2, data.shape[1]//2]}")
print(f"  Top-left: {data[0, 0]}")
print(f"  Top-right: {data[0, -1]}")
print(f"  Bottom-left: {data[-1, 0]}")
print(f"  Bottom-right: {data[-1, -1]}")

ds = None
