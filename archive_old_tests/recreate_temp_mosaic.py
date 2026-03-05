"""
Recreate temp_mosaic.vrt WITHOUT -separate flag
Each Sentinel file has 10 bands, we just want to mosaic them, not separate
"""
import os
import subprocess
import glob

out_folder = r"E:\Test\sentinel_reprojected"

print("Recreating temp_mosaic.vrt (WITHOUT -separate flag)...\n")

# Find snapped tiles
snapped_tiles = sorted(glob.glob(os.path.join(out_folder, "tile_*_snapped.tif")))

print(f"Found {len(snapped_tiles)} tiles:")
for tile in snapped_tiles:
    print(f"  - {os.path.basename(tile)}")

# Create mosaic VRT WITHOUT -separate (this preserves the 10 bands)
temp_vrt = os.path.join(out_folder, "temp_mosaic.vrt")
cmd = ["gdalbuildvrt", temp_vrt] + snapped_tiles  # No -separate flag!

print(f"\nRunning: gdalbuildvrt (no -separate)")
result = subprocess.run(cmd, capture_output=True, text=True)

if result.returncode != 0:
    print(f"ERROR: {result.stderr}")
    exit(1)

print(f"✓ Created: {os.path.basename(temp_vrt)}")

# Verify
ds_import = __import__('osgeo').gdal.Open(temp_vrt)
if ds_import:
    print(f"  Bands: {ds_import.RasterCount} (should be 10, not 30)")
    print(f"  Size: {ds_import.RasterXSize} x {ds_import.RasterYSize}")
    ds_import = None
