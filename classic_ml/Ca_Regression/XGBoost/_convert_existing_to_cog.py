from pathlib import Path
import os
import sys
import rasterio
from rasterio import shutil as rio_shutil

out_dir = Path(r"G:/National_maps/Ca_class/XGBoost")
files = [
    out_dir / "KlassCa_class_map.tif",
    out_dir / "KlassCa_entropy.tif",
    out_dir / "KlassCa_prob_1_Kalkfattig.tif",
    out_dir / "KlassCa_prob_2_Intermediaer.tif",
    out_dir / "KlassCa_prob_3_Kalkrik.tif",
]
missing = [str(p) for p in files if not p.exists()]
if missing:
    print("FAIL: missing output maps")
    for m in missing:
        print(" -", m)
    sys.exit(2)

for src in files:
    tmp = src.with_suffix('.cog_tmp.tif')
    if tmp.exists():
        try:
            tmp.unlink()
        except OSError:
            pass
    print(f"Converting to COG: {src.name}")
    rio_shutil.copy(
        str(src),
        str(tmp),
        driver="COG",
        COMPRESS="DEFLATE",
        PREDICTOR="3",
        BLOCKSIZE="512",
        BIGTIFF="IF_SAFER",
        OVERVIEWS="AUTO",
        RESAMPLING="NEAREST",
    )
    os.replace(str(tmp), str(src))

print("COG conversion finished for all target outputs.")
