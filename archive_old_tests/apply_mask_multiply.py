"""
Apply DTM mask to Sentinel mosaic: set water (mask=0) to NoData
Uses gdal_translate with multiple source bands and mask
"""

import os
import sys
import subprocess
import glob

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

out_folder = r"E:\Test\sentinel_reprojected"
template_mask = r"F:\Topodata_basins\Mosaic_Norge\snap_raster\dtm_mask_srast_fixed.tif"

print("=" * 70)
print("APPLYING DTM MASK TO SENTINEL MOSAIC")
print("=" * 70)

# Find all snapped tile files
snapped_tiles = sorted(glob.glob(os.path.join(out_folder, "tile_*_snapped.tif")))

if not snapped_tiles:
    print("ERROR: No snapped tiles found!")
    exit(1)

print(f"\nFound {len(snapped_tiles)} tiles:")
for tile in snapped_tiles:
    print(f"  - {os.path.basename(tile)}")

# Create mosaic VRT first
print("\n[1] Creating temporary mosaic VRT...")
temp_vrt = os.path.join(out_folder, "temp_mosaic.vrt")
cmd = ["gdalbuildvrt", "-separate", temp_vrt] + snapped_tiles
result = subprocess.run(cmd, capture_output=True, text=True)

if result.returncode != 0:
    print(f"ERROR: {result.stderr}")
    exit(1)

print(f"    [OK] {os.path.basename(temp_vrt)}")

# Now create final masked mosaic with mask applied
print("\n[2] Applying DTM mask (water pixels -> NoData)...")

# Build VRT XML with mask multiplication
masked_vrt_file = os.path.join(out_folder, "Sentinel_masked_final.vrt")

# Create VRT with each band multiplied by mask using PixelFunctionType
# Since mask is on different drive, use absolute paths
mask_path = template_mask.replace("\\", "/")
temp_vrt_path = temp_vrt.replace("\\", "/")

vrt_xml = """<VRTDataset rasterXSize="67456" rasterYSize="63296">
  <SRS>EPSG:25833</SRS>
  <GeoTransform>446730, 10, 0, 6762690, 0, -10</GeoTransform>
  <VRTRasterBand dataType="UInt16" band="1" subClass="VRTDerivedRasterBand">
    <PixelFunctionType>mul</PixelFunctionType>
    <ComplexSource>
      <SourceFilename>""" + temp_vrt_path + """</SourceFilename>
      <SourceBand>1</SourceBand>
    </ComplexSource>
    <ComplexSource>
      <SourceFilename>""" + mask_path + """</SourceFilename>
      <SourceBand>1</SourceBand>
    </ComplexSource>
  </VRTRasterBand>
  <VRTRasterBand dataType="UInt16" band="2" subClass="VRTDerivedRasterBand">
    <PixelFunctionType>mul</PixelFunctionType>
    <ComplexSource>
      <SourceFilename>""" + temp_vrt_path + """</SourceFilename>
      <SourceBand>2</SourceBand>
    </ComplexSource>
    <ComplexSource>
      <SourceFilename>""" + mask_path + """</SourceFilename>
      <SourceBand>1</SourceBand>
    </ComplexSource>
  </VRTRasterBand>
  <VRTRasterBand dataType="UInt16" band="3" subClass="VRTDerivedRasterBand">
    <PixelFunctionType>mul</PixelFunctionType>
    <ComplexSource>
      <SourceFilename>""" + temp_vrt_path + """</SourceFilename>
      <SourceBand>3</SourceBand>
    </ComplexSource>
    <ComplexSource>
      <SourceFilename>""" + mask_path + """</SourceFilename>
      <SourceBand>1</SourceBand>
    </ComplexSource>
  </VRTRasterBand>
  <VRTRasterBand dataType="UInt16" band="4" subClass="VRTDerivedRasterBand">
    <PixelFunctionType>mul</PixelFunctionType>
    <ComplexSource>
      <SourceFilename>""" + temp_vrt_path + """</SourceFilename>
      <SourceBand>4</SourceBand>
    </ComplexSource>
    <ComplexSource>
      <SourceFilename>""" + mask_path + """</SourceFilename>
      <SourceBand>1</SourceBand>
    </ComplexSource>
  </VRTRasterBand>
  <VRTRasterBand dataType="UInt16" band="5" subClass="VRTDerivedRasterBand">
    <PixelFunctionType>mul</PixelFunctionType>
    <ComplexSource>
      <SourceFilename>""" + temp_vrt_path + """</SourceFilename>
      <SourceBand>5</SourceBand>
    </ComplexSource>
    <ComplexSource>
      <SourceFilename>""" + mask_path + """</SourceFilename>
      <SourceBand>1</SourceBand>
    </ComplexSource>
  </VRTRasterBand>
  <VRTRasterBand dataType="UInt16" band="6" subClass="VRTDerivedRasterBand">
    <PixelFunctionType>mul</PixelFunctionType>
    <ComplexSource>
      <SourceFilename>""" + temp_vrt_path + """</SourceFilename>
      <SourceBand>6</SourceBand>
    </ComplexSource>
    <ComplexSource>
      <SourceFilename>""" + mask_path + """</SourceFilename>
      <SourceBand>1</SourceBand>
    </ComplexSource>
  </VRTRasterBand>
  <VRTRasterBand dataType="UInt16" band="7" subClass="VRTDerivedRasterBand">
    <PixelFunctionType>mul</PixelFunctionType>
    <ComplexSource>
      <SourceFilename>""" + temp_vrt_path + """</SourceFilename>
      <SourceBand>7</SourceBand>
    </ComplexSource>
    <ComplexSource>
      <SourceFilename>""" + mask_path + """</SourceFilename>
      <SourceBand>1</SourceBand>
    </ComplexSource>
  </VRTRasterBand>
  <VRTRasterBand dataType="UInt16" band="8" subClass="VRTDerivedRasterBand">
    <PixelFunctionType>mul</PixelFunctionType>
    <ComplexSource>
      <SourceFilename>""" + temp_vrt_path + """</SourceFilename>
      <SourceBand>8</SourceBand>
    </ComplexSource>
    <ComplexSource>
      <SourceFilename>""" + mask_path + """</SourceFilename>
      <SourceBand>1</SourceBand>
    </ComplexSource>
  </VRTRasterBand>
  <VRTRasterBand dataType="UInt16" band="9" subClass="VRTDerivedRasterBand">
    <PixelFunctionType>mul</PixelFunctionType>
    <ComplexSource>
      <SourceFilename>""" + temp_vrt_path + """</SourceFilename>
      <SourceBand>9</SourceBand>
    </ComplexSource>
    <ComplexSource>
      <SourceFilename>""" + mask_path + """</SourceFilename>
      <SourceBand>1</SourceBand>
    </ComplexSource>
  </VRTRasterBand>
  <VRTRasterBand dataType="UInt16" band="10" subClass="VRTDerivedRasterBand">
    <PixelFunctionType>mul</PixelFunctionType>
    <ComplexSource>
      <SourceFilename>""" + temp_vrt_path + """</SourceFilename>
      <SourceBand>10</SourceBand>
    </ComplexSource>
    <ComplexSource>
      <SourceFilename>""" + mask_path + """</SourceFilename>
      <SourceBand>1</SourceBand>
    </ComplexSource>
  </VRTRasterBand>
</VRTDataset>
"""

with open(masked_vrt_file, 'w') as f:
    f.write(vrt_xml)

print(f"    [OK] Masked VRT created: {os.path.basename(masked_vrt_file)}")

print("\n" + "=" * 70)
print("COMPLETE!")
print("=" * 70)
print(f"\nLoad this in QGIS: {masked_vrt_file}")
print("\nThis VRT:")
print("  - Multiplies each band by the mask (water=0 -> band*0=0)")
print("  - Water pixels become 0 (can be displayed as NoData)")
print("  - Respects the mask boundaries")
print("=" * 70)
