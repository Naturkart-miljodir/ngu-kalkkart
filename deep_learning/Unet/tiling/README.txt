Here we have two codes designed for national and local (in this case labeled as "trondelag") tests. The codes create deep-learning "chips" (used the term
"tiles", which can be a bit misleading) for training/testing of the U-Net model
National_dl_AlphaEarth_tiling_final.py is build for alpha earth data and normalizes only topographic data and gephysical data.
National_dl_tiling.py is build for Sentinel stack, and uses Sentinel indices (NVI, etc., )