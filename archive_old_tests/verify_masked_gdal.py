"""
Verify masked VRT using pure GDAL (no rasterio)
"""
print("Starting verification...")
print("Importing GDAL...")
import sys
sys.stdout.flush()

from osgeo import gdal
import numpy as np

print("GDAL imported OK")
sys.stdout.flush()

vrt_path = r"E:\Test\sentinel_reprojected\Sentinel_masked_final.vrt"

print(f"\nOpening: {vrt_path}")
sys.stdout.flush()

ds = gdal.Open(vrt_path)
if ds is None:
    print("ERROR: Could not open VRT")
    exit(1)

print(f"VRT opened successfully!")
print(f"Size: {ds.RasterXSize} x {ds.RasterYSize}")
print(f"Bands: {ds.RasterCount}")
sys.stdout.flush()

# Read a small window from band 1
band = ds.GetRasterBand(1)
print(f"\nBand 1 NoData: {band.GetNoDataValue()}")
sys.stdout.flush()

# Read 100x100 window at position with land
xoff, yoff = 13000, 14000
xsize, ysize = 100, 100

print(f"\nReading {xsize}x{ysize} window at ({xoff}, {yoff})...")
sys.stdout.flush()

data = band.ReadAsArray(xoff, yoff, xsize, ysize)

print(f"Data shape: {data.shape}")
print(f"Data type: {data.dtype}")
print(f"Min value: {np.min(data)}")
print(f"Max value: {np.max(data)}")
print(f"Mean value: {np.mean(data)}")
print(f"Unique values count: {len(np.unique(data))}")

# Show first few unique values
unique_vals = np.unique(data)[:20]
print(f"First 20 unique values: {unique_vals}")

ds = None
print("\n" + "="*70)
print("Verification complete!")
print("="*70)
