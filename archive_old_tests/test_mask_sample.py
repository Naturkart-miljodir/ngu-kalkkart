"""
Quick test to verify Sentinel_masked_final.vrt masking is working.
Reads a small window and shows if zeros exist (water masked).
"""
import os
os.add_dll_directory(r'C:\ProgramData\anaconda3\envs\Kalk_DL_tiling\Library\bin')

from osgeo import gdal
import numpy as np

# Open the masked VRT
vrt_path = r'E:\Test\sentinel_reprojected\Sentinel_masked_final.vrt'
ds = gdal.Open(vrt_path)

if ds is None:
    print("Failed to open VRT!")
    exit(1)

print(f"VRT opened successfully")
print(f"Size: {ds.RasterXSize} x {ds.RasterYSize}")
print(f"Bands: {ds.RasterCount}")

# Read a 1000x1000 window from center
x_off = ds.RasterXSize // 2 - 500
y_off = ds.RasterYSize // 2 - 500
window_size = 1000

print(f"\nReading {window_size}x{window_size} window at offset ({x_off}, {y_off})...")

band1 = ds.GetRasterBand(1)
data = band1.ReadAsArray(x_off, y_off, window_size, window_size)

print(f"\nBand 1 statistics in sample window:")
print(f"  Min: {data.min()}")
print(f"  Max: {data.max()}")
print(f"  Mean: {data.mean():.2f}")
print(f"  Std: {data.std():.2f}")
print(f"  Zeros (water/masked): {(data == 0).sum()} pixels ({(data == 0).sum()/data.size*100:.1f}%)")
print(f"  Non-zeros (land): {(data > 0).sum()} pixels ({(data > 0).sum()/data.size*100:.1f}%)")

if (data > 0).sum() > 0:
    print(f"\nLand pixel values (non-zero):")
    print(f"  Min: {data[data > 0].min()}")
    print(f"  Max: {data[data > 0].max()}")
    print(f"  Mean: {data[data > 0].mean():.2f}")

print("\n✓ Masking is working if you see both zeros (water) and non-zeros (land)")

ds = None
