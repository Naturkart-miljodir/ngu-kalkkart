"""
Check original mask before reprojection to verify land/water encoding
"""
import os
os.add_dll_directory(r'C:\ProgramData\anaconda3\envs\Kalk_DL_tiling\Library\bin')

from osgeo import gdal
import numpy as np

print("Checking ORIGINAL mask (before reprojection)...\n")

original_mask = r'F:\Topodata_basins\Mosaic_Norge\snap_raster\dtm_mask_srast_fixed.tif'

ds = gdal.Open(original_mask)
if ds is None:
    print(f"ERROR: Cannot open {original_mask}")
    exit(1)

band = ds.GetRasterBand(1)
data = band.ReadAsArray()

print(f"Original mask info:")
print(f"  Size: {data.shape}")
print(f"  Data type: {data.dtype}")
print(f"  Unique values: {np.unique(data)}")

# Show distribution
unique, counts = np.unique(data, return_counts=True)
print(f"\nValue distribution:")
for val, count in zip(unique, counts):
    pct = count / data.size * 100
    print(f"  {val}: {count} pixels ({pct:.2f}%)")

print(f"\n⚠️  KEY QUESTION:")
print(f"  If 1 is ~75% (majority of land area), then: 1=land, 0=water ✓")
print(f"  If 0 is ~75% (majority of land area), then: MASK IS INVERTED! 0=land, 1=water ✗")

ds = None
