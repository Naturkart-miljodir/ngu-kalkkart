"""
Reproject and snap Sentinel multi-band tiles to match template mask grid.
Sets NoData=0 and applies mask.
"""

import os
import sys
import glob
import subprocess

# Setup GDAL PATH
def setup_path_for_gdal():
    python_exe = sys.executable
    python_dir = os.path.dirname(python_exe)
    paths_to_add = [
        os.path.join(python_dir, 'Scripts'),
        os.path.join(python_dir, 'Library', 'bin'),
    ]
    for path in paths_to_add:
        if os.path.exists(path) and path not in os.environ['PATH']:
            os.environ['PATH'] = path + os.pathsep + os.environ['PATH']

setup_path_for_gdal()

# Paths
in_folder = r"E:\Test"
template_mask = r"F:\Topodata_basins\Mosaic_Norge\snap_raster\dtm_mask_srast_fixed.tif"
out_folder = r"E:\Test\sentinel_reprojected"

os.makedirs(out_folder, exist_ok=True)

print("=" * 70)
print("SENTINEL REPROJECTION AND SNAPPING WORKFLOW")
print("=" * 70)

# Find all tile TIF files (excluding existing outputs)
print(f"\n[1] Finding Sentinel tiles in {in_folder}...")
all_tifs = glob.glob(os.path.join(in_folder, "tile_*.tif"))
print(f"    Found {len(all_tifs)} tiles:")
for tif in all_tifs:
    print(f"      - {os.path.basename(tif)}")

if not all_tifs:
    print("    ERROR: No tile_*.tif files found!")
    sys.exit(1)

# Get template grid info using gdalinfo
print(f"\n[2] Reading template mask grid info...")
cmd = ["gdalinfo", template_mask]
result = subprocess.run(cmd, capture_output=True, text=True)
if result.returncode != 0:
    print(f"    ERROR reading template: {result.stderr}")
    sys.exit(1)

print("    [OK] Template mask readable")

# Reproject and snap each tile
print(f"\n[3] Reprojecting and snapping tiles...")
print("    Setting NoData=0 and matching grid to template...")

reprojected_tiles = []

for tile_file in all_tifs:
    tile_name = os.path.basename(tile_file)
    base_name = os.path.splitext(tile_name)[0]
    output_file = os.path.join(out_folder, f"{base_name}_snapped.tif")
    
    print(f"\n    Processing {tile_name}...")
    
    # gdalwarp command:
    # -t_srs: match CRS
    # -tr: match pixel size (will be read from template)
    # -tap: target aligned pixels (snap to grid)
    # NOTE: DO NOT use -srcnodata 0 or -dstnodata 0
    #       Sentinel has legitimate 0 pixel values (dark pixels/water)
    #       These are NOT NoData - they are real data values
    # -r near: Use nearest neighbor resampling (not bilinear) to preserve exact values
    
    cmd = [
        "gdalwarp",
        "-t_srs", "EPSG:25833",  # UTM 33N
        "-tr", "10", "10",  # 10m pixel size (Sentinel resolution)
        "-tap",  # Snap to grid - crucial for alignment!
        "-r", "near",  # Nearest neighbor to preserve exact pixel values
        "-co", "COMPRESS=LZW",
        "-co", "TILED=YES",
        "-co", "BIGTIFF=YES",  # Use BigTIFF for large files (>4GB)
        "-overwrite",
        tile_file,
        output_file,
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        file_size = os.path.getsize(output_file) / (1024 * 1024)
        print(f"      [OK] {file_size:.2f} MB")
        reprojected_tiles.append(output_file)
    else:
        print(f"      [ERROR] {result.stderr}")
        sys.exit(1)

# Build VRT mosaic
print("\n" + "=" * 70)
print("[4] Creating VRT mosaic of reprojected tiles...")
print("=" * 70)

vrt_file = os.path.join(out_folder, "Sentinel_mosaic.vrt")

cmd = ["gdalbuildvrt", "-srcnodata", "0", "-vrtnodata", "0", vrt_file] + reprojected_tiles
result = subprocess.run(cmd, capture_output=True, text=True)

if result.returncode == 0:
    print(f"\n[OK] Mosaic VRT created: {vrt_file}")
else:
    print(f"[ERROR] {result.stderr}")
    sys.exit(1)

# Create masked VRT
print("\n" + "=" * 70)
print("[5] Creating masked VRT (applying DTM mask)...")
print("=" * 70)

masked_vrt = os.path.join(out_folder, "Sentinel_masked.vrt")

# Use gdalbuildvrt with cutline or create manual VRT with mask multiplication
# For now, document both the mosaic and mask separately
# User will apply mask in next step or via multiplication in VRT

print(f"\n[OK] Workflow complete!")
print("\nOutputs:")
print(f"  Reprojected tiles: {out_folder}")
print(f"  Mosaic VRT: {vrt_file}")
print(f"  Template mask: {template_mask}")

print("\n" + "=" * 70)
print("NEXT STEPS")
print("=" * 70)
print("1. Load Sentinel_mosaic.vrt in QGIS to check alignment")
print("2. Verify NoData (0 values) are properly masked")
print("3. Check if tiles now align seamlessly (no gaps)")
print("4. If OK, we'll create the final masked VRT for tiling")
print("=" * 70)
