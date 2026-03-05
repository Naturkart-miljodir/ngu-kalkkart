"""
Verify masked VRT is working correctly
Read sample window and show statistics
"""

print("Starting verification...")
import sys
sys.stdout.flush()

try:
    print("Importing rasterio...")
    sys.stdout.flush()
    import rasterio
    print("  rasterio OK")
    sys.stdout.flush()
except Exception as e:
    print(f"  rasterio FAILED: {e}")
    sys.stdout.flush()
    exit(1)

try:
    print("Importing numpy...")
    sys.stdout.flush()
    import numpy as np
    print("  numpy OK")
    sys.stdout.flush()
except Exception as e:
    print(f"  numpy FAILED: {e}")
    sys.stdout.flush()
    exit(1)

try:
    print("Importing os...")
    sys.stdout.flush()
    import os
    print("  os OK")
    sys.stdout.flush()
except Exception as e:
    print(f"  os FAILED: {e}")
    sys.stdout.flush()
    exit(1)

masked_vrt = r"E:\Test\sentinel_reprojected\Sentinel_masked_final.vrt"
print(f"VRT path: {masked_vrt}")
sys.stdout.flush()
print(f"VRT exists: {os.path.exists(masked_vrt)}")
sys.stdout.flush()

print("=" * 70)
print("VERIFYING MASKED SENTINEL VRT")
print("=" * 70)
sys.stdout.flush()

try:
    print("Opening VRT with rasterio...")
    sys.stdout.flush()
    with rasterio.open(masked_vrt) as src:
        print(f"\nVRT Info:")
        print(f"  Bands: {src.count}")
        print(f"  Size: {src.width} x {src.height}")
        print(f"  Data type: {src.dtypes[0]}")
        print(f"  CRS: {src.crs}")
        sys.stdout.flush()
        
        # Read a small sample window (1000x1000 pixels) from band 1
        print(f"\n[1] Reading sample window (1000x1000 from top-left)...")
        sys.stdout.flush()
        window = rasterio.windows.Window(0, 0, 1000, 1000)
        data = src.read(1, window=window)
        
        print(f"  Shape: {data.shape}")
        print(f"  Data type: {data.dtype}")
        sys.stdout.flush()

except Exception as e:
    print(f"ERROR: {e}")
    sys.stdout.flush()
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
