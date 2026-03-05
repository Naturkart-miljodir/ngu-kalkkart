"""
Check what values are in the Sentinel mosaic BEFORE masking
"""
import os
os.add_dll_directory(r'C:\ProgramData\anaconda3\envs\Kalk_DL_tiling\Library\bin')

from osgeo import gdal
import numpy as np

print("Checking Sentinel mosaic values...\n")

sentinel_vrt = r'E:\Test\sentinel_reprojected\temp_mosaic.vrt'
mask_file = r'E:\Test\sentinel_reprojected\dtm_mask_etrs89.tif'

# Check sentinel
print("[1] Sentinel mosaic (temp_mosaic.vrt)")
s_ds = gdal.Open(sentinel_vrt)
for band_num in [1, 5, 10]:
    band = s_ds.GetRasterBand(band_num)
    # Read sample from center area
    data = band.ReadAsArray(13000, 14000, 100, 100)
    print(f"    Band {band_num}: min={data.min()}, max={data.max()}, mean={data.mean():.0f}")
    print(f"              unique values: {np.unique(data)[:5]}...")  # First 5 unique values

# Check mask
print("\n[2] DTM mask (dtm_mask_etrs89.tif)")
m_ds = gdal.Open(mask_file)
band = m_ds.GetRasterBand(1)
data = band.ReadAsArray(13000, 14000, 100, 100)
print(f"    Unique values: {np.unique(data)}")
print(f"    Distribution: {np.bincount(data.ravel())}")
print(f"    0 (water): {(data == 0).sum()} pixels")
print(f"    1 (land): {(data == 1).sum()} pixels")

s_ds = None
m_ds = None
