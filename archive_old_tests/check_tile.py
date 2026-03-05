"""
Check one actual snapped tile to verify Sentinel data quality
"""
import os
os.add_dll_directory(r'C:\ProgramData\anaconda3\envs\Kalk_DL_tiling\Library\bin')

from osgeo import gdal

print("Checking a Sentinel tile directly...\n")

tile = r'E:\Test\sentinel_reprojected\tile_02_01_snapped.tif'

ds = gdal.Open(tile)
print(f"Tile: {tile}")
print(f"Size: {ds.RasterXSize} x {ds.RasterYSize}")
print(f"Bands: {ds.RasterCount}")

for b in [1, 2, 3]:
    band = ds.GetRasterBand(b)
    # Read small sample
    data = band.ReadAsArray(0, 0, 100, 100)
    print(f"  Band {b}: dtype={data.dtype}, min={data.min()}, max={data.max()}, mean={data.mean():.0f}")

ds = None
