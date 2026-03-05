"""
Apply DTM mask to Sentinel mosaic properly
Read actual VRT metadata and create correct masked VRT
"""

import os
import sys
import subprocess
import glob
import xml.etree.ElementTree as ET

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
template_mask = r"E:\Test\sentinel_reprojected\dtm_mask_binary.tif"  # Binary mask (1=land, 0=water)
mosaic_vrt = os.path.join(out_folder, "Sentinel_mosaic.vrt")

print("=" * 70)
print("APPLYING DTM MASK TO SENTINEL MOSAIC")
print("=" * 70)

if not os.path.exists(mosaic_vrt):
  print(f"ERROR: Mosaic VRT not found: {mosaic_vrt}")
  exit(1)

print(f"\nUsing mosaic VRT: {mosaic_vrt}")

# Read the temp VRT to get actual dimensions and geotransform
print("\n[2] Reading VRT metadata...")
tree = ET.parse(mosaic_vrt)
root = tree.getroot()

# Extract dimensions
raster_x_size = root.get('rasterXSize')
raster_y_size = root.get('rasterYSize')

# Extract geotransform
geo_elem = root.find('.//GeoTransform')
if geo_elem is None or geo_elem.text is None:
    print("ERROR: Could not read GeoTransform from VRT")
    exit(1)

geo_text = geo_elem.text.strip()
print(f"    Dimensions: {raster_x_size} x {raster_y_size}")
print(f"    GeoTransform: {geo_text}")

# Now create masked VRT with correct metadata
print("\n[3] Creating masked VRT...")

masked_vrt_file = os.path.join(out_folder, "Sentinel_masked_final.vrt")

mask_path = template_mask.replace("\\", "/")
mosaic_vrt_path = mosaic_vrt.replace("\\", "/")

# Build VRT XML - just 3 RGB bands for now to test
vrt_xml = f"""<VRTDataset rasterXSize="{raster_x_size}" rasterYSize="{raster_y_size}">
  <SRS>EPSG:25833</SRS>
  <GeoTransform>{geo_text}</GeoTransform>
  <VRTRasterBand dataType="UInt16" band="1" subClass="VRTDerivedRasterBand">
    <PixelFunctionType>mul</PixelFunctionType>
    <ComplexSource>
      <SourceFilename>{mosaic_vrt_path}</SourceFilename>
      <SourceBand>1</SourceBand>
    </ComplexSource>
    <ComplexSource>
      <SourceFilename>{mask_path}</SourceFilename>
      <SourceBand>1</SourceBand>
    </ComplexSource>
  </VRTRasterBand>
  <VRTRasterBand dataType="UInt16" band="2" subClass="VRTDerivedRasterBand">
    <PixelFunctionType>mul</PixelFunctionType>
    <ComplexSource>
      <SourceFilename>{mosaic_vrt_path}</SourceFilename>
      <SourceBand>2</SourceBand>
    </ComplexSource>
    <ComplexSource>
      <SourceFilename>{mask_path}</SourceFilename>
      <SourceBand>1</SourceBand>
    </ComplexSource>
  </VRTRasterBand>
  <VRTRasterBand dataType="UInt16" band="3" subClass="VRTDerivedRasterBand">
    <PixelFunctionType>mul</PixelFunctionType>
    <ComplexSource>
      <SourceFilename>{mosaic_vrt_path}</SourceFilename>
      <SourceBand>3</SourceBand>
    </ComplexSource>
    <ComplexSource>
      <SourceFilename>{mask_path}</SourceFilename>
      <SourceBand>1</SourceBand>
    </ComplexSource>
  </VRTRasterBand>
  <VRTRasterBand dataType="UInt16" band="4" subClass="VRTDerivedRasterBand">
    <PixelFunctionType>mul</PixelFunctionType>
    <ComplexSource>
      <SourceFilename>{mosaic_vrt_path}</SourceFilename>
      <SourceBand>4</SourceBand>
    </ComplexSource>
    <ComplexSource>
      <SourceFilename>{mask_path}</SourceFilename>
      <SourceBand>1</SourceBand>
    </ComplexSource>
  </VRTRasterBand>
  <VRTRasterBand dataType="UInt16" band="5" subClass="VRTDerivedRasterBand">
    <PixelFunctionType>mul</PixelFunctionType>
    <ComplexSource>
      <SourceFilename>{mosaic_vrt_path}</SourceFilename>
      <SourceBand>5</SourceBand>
    </ComplexSource>
    <ComplexSource>
      <SourceFilename>{mask_path}</SourceFilename>
      <SourceBand>1</SourceBand>
    </ComplexSource>
  </VRTRasterBand>
  <VRTRasterBand dataType="UInt16" band="6" subClass="VRTDerivedRasterBand">
    <PixelFunctionType>mul</PixelFunctionType>
    <ComplexSource>
      <SourceFilename>{mosaic_vrt_path}</SourceFilename>
      <SourceBand>6</SourceBand>
    </ComplexSource>
    <ComplexSource>
      <SourceFilename>{mask_path}</SourceFilename>
      <SourceBand>1</SourceBand>
    </ComplexSource>
  </VRTRasterBand>
  <VRTRasterBand dataType="UInt16" band="7" subClass="VRTDerivedRasterBand">
    <PixelFunctionType>mul</PixelFunctionType>
    <ComplexSource>
      <SourceFilename>{mosaic_vrt_path}</SourceFilename>
      <SourceBand>7</SourceBand>
    </ComplexSource>
    <ComplexSource>
      <SourceFilename>{mask_path}</SourceFilename>
      <SourceBand>1</SourceBand>
    </ComplexSource>
  </VRTRasterBand>
  <VRTRasterBand dataType="UInt16" band="8" subClass="VRTDerivedRasterBand">
    <PixelFunctionType>mul</PixelFunctionType>
    <ComplexSource>
      <SourceFilename>{mosaic_vrt_path}</SourceFilename>
      <SourceBand>8</SourceBand>
    </ComplexSource>
    <ComplexSource>
      <SourceFilename>{mask_path}</SourceFilename>
      <SourceBand>1</SourceBand>
    </ComplexSource>
  </VRTRasterBand>
  <VRTRasterBand dataType="UInt16" band="9" subClass="VRTDerivedRasterBand">
    <PixelFunctionType>mul</PixelFunctionType>
    <ComplexSource>
      <SourceFilename>{mosaic_vrt_path}</SourceFilename>
      <SourceBand>9</SourceBand>
    </ComplexSource>
    <ComplexSource>
      <SourceFilename>{mask_path}</SourceFilename>
      <SourceBand>1</SourceBand>
    </ComplexSource>
  </VRTRasterBand>
  <VRTRasterBand dataType="UInt16" band="10" subClass="VRTDerivedRasterBand">
    <PixelFunctionType>mul</PixelFunctionType>
    <ComplexSource>
      <SourceFilename>{mosaic_vrt_path}</SourceFilename>
      <SourceBand>10</SourceBand>
    </ComplexSource>
    <ComplexSource>
      <SourceFilename>{mask_path}</SourceFilename>
      <SourceBand>1</SourceBand>
    </ComplexSource>
  </VRTRasterBand>
</VRTDataset>
"""

with open(masked_vrt_file, 'w') as f:
    f.write(vrt_xml)

print(f"    [OK] Masked VRT created")

print("\n" + "=" * 70)
print("COMPLETE!")
print("=" * 70)
print(f"\nLoad this in QGIS: {masked_vrt_file}")
print("\nThis VRT multiplies each band by the mask")
print("(water pixels where mask=0 will become 0)")
print("=" * 70)
