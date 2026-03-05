"""
Compare extents and resolutions of mask vs Sentinel tiles
"""
import os
os.add_dll_directory(r'C:\ProgramData\anaconda3\envs\Kalk_DL_tiling\Library\bin')

from osgeo import gdal

print("Comparing Sentinel tiles and mask...\n")

# Check Sentinel mosaic
sentinel = gdal.Open(r'E:\Test\sentinel_reprojected\temp_mosaic.vrt')
s_gt = sentinel.GetGeoTransform()
s_width = sentinel.RasterXSize
s_height = sentinel.RasterYSize
s_minx = s_gt[0]
s_maxx = s_gt[0] + s_width * s_gt[1]
s_maxy = s_gt[3]
s_miny = s_gt[3] + s_height * s_gt[5]

print("Sentinel mosaic (temp_mosaic.vrt):")
print(f"  Size: {s_width} x {s_height}")
print(f"  Resolution: {s_gt[1]}, {s_gt[5]}")
print(f"  Extent: ({s_minx:.1f}, {s_miny:.1f}) to ({s_maxx:.1f}, {s_maxy:.1f})")
print(f"  CRS: {sentinel.GetProjection()[:50]}...")

# Check mask
mask = gdal.Open(r'F:\Topodata_basins\Mosaic_Norge\snap_raster\dtm_mask_srast_fixed.tif')
m_gt = mask.GetGeoTransform()
m_width = mask.RasterXSize
m_height = mask.RasterYSize
m_minx = m_gt[0]
m_maxx = m_gt[0] + m_width * m_gt[1]
m_maxy = m_gt[3]
m_miny = m_gt[3] + m_height * m_gt[5]

print("\nDTM mask:")
print(f"  Size: {m_width} x {m_height}")
print(f"  Resolution: {m_gt[1]}, {m_gt[5]}")
print(f"  Extent: ({m_minx:.1f}, {m_miny:.1f}) to ({m_maxx:.1f}, {m_maxy:.1f})")
print(f"  CRS: {mask.GetProjection()[:50]}...")

# Check overlap
print("\nOverlap analysis:")
if s_minx >= m_maxx or s_maxx <= m_minx or s_miny >= m_maxy or s_maxy <= m_miny:
    print("  ❌ NO OVERLAP! Sentinel and mask don't intersect")
else:
    overlap_minx = max(s_minx, m_minx)
    overlap_maxx = min(s_maxx, m_maxx)
    overlap_miny = max(s_miny, m_miny)
    overlap_maxy = min(s_maxy, m_maxy)
    print(f"  ✓ Overlap extent: ({overlap_minx:.1f}, {overlap_miny:.1f}) to ({overlap_maxx:.1f}, {overlap_maxy:.1f})")
    
    # Check if Sentinel is fully within mask
    if s_minx >= m_minx and s_maxx <= m_maxx and s_miny >= m_miny and s_maxy <= m_maxy:
        print("  ✓ Sentinel tiles fully within mask extent")
    else:
        print("  ⚠ Sentinel extends beyond mask")

sentinel = None
mask = None
