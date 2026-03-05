"""
Check mask values to see if it's inverted
"""
import os
os.add_dll_directory(r'C:\ProgramData\anaconda3\envs\Kalk_DL_tiling\Library\bin')

from osgeo import gdal
import numpy as np

print("Checking mask values...\n")

mask = gdal.Open(r'E:\Test\sentinel_reprojected\dtm_mask_etrs89.tif')
band = mask.GetRasterBand(1)

# Read center area
data = band.ReadAsArray(10000, 10000, 1000, 1000)

print(f"Mask statistics in center area:")
print(f"  Unique values: {np.unique(data)}")
print(f"  Min: {data.min()}, Max: {data.max()}")
print(f"  Data type: {band.DataType}")

# Check what value corresponds to land vs water
print(f"\nValue counts:")
unique, counts = np.unique(data, return_counts=True)
for val, count in zip(unique, counts):
    print(f"  Value {val}: {count} pixels ({count/data.size*100:.1f}%)")

print("\nIf 1 is predominantly in the center area, mask is: 1=land, 0=water")
print("If 0 is predominantly in the center area, mask is inverted: 0=land, 1=water")

mask = None
