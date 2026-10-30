"""
compare_baselines.py
--------------------
3-way comparison of Ca prediction methods:
  1. Old median-per-lithological-unit Ca raster  (baseline)
  2. EBK kriging raster from ArcGIS Pro          (spatial interpolation)
  3. RF regression (this work)                   (cv_predictions.csv)

All three are evaluated on the IDENTICAL set of sample points
(the same ~20 K lab points used for RF training/CV).

Usage:
  conda run --no-capture-output -n kalk_rf python -u compare_baselines.py

Edit the paths in the CONFIG block below before running.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ─────────────────────────────────────────────────────────────────
#  CONFIG  ← edit these paths
# ─────────────────────────────────────────────────────────────────

# Target variable and the RF config subfolder to use
TARGET_VARIABLE = "CaO"          # "CaO" | "CaMg_LOI_v" | "Comp_1" | "UMAP1"
RF_CONFIG       = "Top_30"       # "All" | "Top_20" | "Top_30" | ...

# Root output folder (same as OUT_DIR in RF_window_Ca_modelling.py)
MODEL_DIR = (
    r"C:\Users\acosta_pedro\OneDrive - Norges geologiske undersøkelse"
    r"\Geochemistry NGU_2026\Kalk_project\Modelling\RandForest"
    r"\Ca_conc_modelling\Model\Model_XRF_total_Lab"
)

# Path to the old median-per-lithological-unit Ca raster (.tif / .img)
BASELINE_RASTER = r"FILL_IN_PATH_TO_OLD_MEDIAN_RASTER.tif"

# Path to the EBK prediction raster from ArcGIS Pro (.tif / .img)
EBK_RASTER = r"FILL_IN_PATH_TO_EBK_RASTER.tif"

# Output folder for comparison results
OUT_DIR_COMP = os.path.join(MODEL_DIR, TARGET_VARIABLE, "Comparison")

# ─────────────────────────────────────────────────────────────────

def check_rasterio():
    try:
        import rasterio  # noqa: F401
        return True
    except ImportError:
        return False


def sample_raster_at_points(raster_path: str, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Sample a raster at (x, y) coordinates (EPSG:25833, metres).
    Returns 1-D array of sampled values; nodata → NaN.
    """
    import rasterio
    from rasterio.transform import rowcol

    values = np.full(len(x), np.nan, dtype=np.float64)
    with rasterio.open(raster_path) as src:
        nodata = src.nodata
        for i, (xi, yi) in enumerate(zip(x, y)):
            try:
                row, col = rowcol(src.transform, xi, yi)
                if 0 <= row < src.height and 0 <= col < src.width:
                    val = src.read(1, window=((row, row + 1), (col, col + 1)))
                    v = float(val[0, 0])
                    if nodata is not None and v == nodata:
                        values[i] = np.nan
                    else:
                        values[i] = v
            except Exception:
                pass  # out-of-bounds or other issue → stays NaN
    return values


def batch_sample_raster(raster_path: str, x: np.ndarray, y: np.ndarray,
                         batch_size: int = 5000) -> np.ndarray:
    """
    Faster batch sampling using rasterio's sample() generator.
    Returns 1-D array; nodata / out-of-bounds → NaN.
    """
    import rasterio

    values = np.full(len(x), np.nan, dtype=np.float64)
    coords = list(zip(x.tolist(), y.tolist()))

    with rasterio.open(raster_path) as src:
        nodata = src.nodata
        sampled = list(src.sample(coords, indexes=1))

    for i, v_arr in enumerate(sampled):
        v = float(v_arr[0])
        if nodata is not None and v == nodata:
            values[i] = np.nan
        elif np.isnan(v):
            values[i] = np.nan
        else:
            values[i] = v

    return values


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, label: str) -> dict:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    n = len(yt)
    rmse  = np.sqrt(mean_squared_error(yt, yp))
    mae   = mean_absolute_error(yt, yp)
    r2    = r2_score(yt, yp)
    bias  = float(np.mean(yp - yt))
    n_nan = int((~mask).sum())
    return {
        "Method": label,
        "N_used": n,
        "N_nan": n_nan,
        "R2":   round(r2,   4),
        "RMSE": round(rmse, 4),
        "MAE":  round(mae,  4),
        "Bias": round(bias, 4),
    }


def scatter_panel(ax, y_true, y_pred, label, metrics):
    ax.scatter(y_true, y_pred, s=3, alpha=0.25, color="steelblue", rasterized=True)
    mn = min(y_true.min(), y_pred.min())
    mx = max(y_true.max(), y_pred.max())
    ax.plot([mn, mx], [mn, mx], "r--", lw=1)
    ax.set_xlabel("Observed")
    ax.set_ylabel("Predicted")
    ax.set_title(
        f"{label}\nR²={metrics['R2']:.3f}  RMSE={metrics['RMSE']:.3f}  "
        f"MAE={metrics['MAE']:.3f}  Bias={metrics['Bias']:+.3f}"
    )
    ax.set_xlim(mn, mx)
    ax.set_ylim(mn, mx)
    ax.set_aspect("equal", adjustable="box")


def main():
    os.makedirs(OUT_DIR_COMP, exist_ok=True)

    # ── 1. Load RF CV predictions ─────────────────────────────────
    rf_csv = os.path.join(MODEL_DIR, TARGET_VARIABLE, RF_CONFIG, "cv_predictions.csv")
    if not os.path.exists(rf_csv):
        raise FileNotFoundError(f"RF predictions not found:\n  {rf_csv}")
    print(f"Loading RF predictions: {rf_csv}")
    rf_df = pd.read_csv(rf_csv)

    x_coords = rf_df["x"].values.astype(np.float64)
    y_coords = rf_df["y"].values.astype(np.float64)
    y_true   = rf_df["y_true"].values.astype(np.float64)
    y_rf     = rf_df["y_pred"].values.astype(np.float64)

    print(f"  Points: {len(y_true)}")

    # ── 2. Check rasterio ─────────────────────────────────────────
    if not check_rasterio():
        raise ImportError(
            "rasterio is not installed in this environment.\n"
            "Install with: conda install -c conda-forge rasterio"
        )

    # ── 3. Check raster paths are filled in ──────────────────────
    placeholder_prefix = "FILL_IN"
    if BASELINE_RASTER.startswith(placeholder_prefix):
        raise ValueError(
            "Please set BASELINE_RASTER to the actual path of the old "
            "median-per-unit Ca raster in the CONFIG block."
        )
    if EBK_RASTER.startswith(placeholder_prefix):
        raise ValueError(
            "Please set EBK_RASTER to the actual path of the EBK prediction "
            "raster in the CONFIG block."
        )

    # ── 4. Sample rasters at point locations ─────────────────────
    print(f"\nSampling baseline raster …\n  {BASELINE_RASTER}")
    y_baseline = batch_sample_raster(BASELINE_RASTER, x_coords, y_coords)
    n_nan_bl = int(np.isnan(y_baseline).sum())
    print(f"  Sampled; NaN={n_nan_bl}/{len(y_baseline)}")

    print(f"\nSampling EBK raster …\n  {EBK_RASTER}")
    y_ebk = batch_sample_raster(EBK_RASTER, x_coords, y_coords)
    n_nan_ebk = int(np.isnan(y_ebk).sum())
    print(f"  Sampled; NaN={n_nan_ebk}/{len(y_ebk)}")

    # ── 5. Compute metrics ────────────────────────────────────────
    metrics_rf   = compute_metrics(y_true, y_rf,       f"RF {RF_CONFIG}")
    metrics_bl   = compute_metrics(y_true, y_baseline, "Baseline (median/unit)")
    metrics_ebk  = compute_metrics(y_true, y_ebk,      "EBK (ArcGIS)")

    metrics_df = pd.DataFrame([metrics_bl, metrics_ebk, metrics_rf])
    print(f"\n{'='*60}")
    print(f"  Target: {TARGET_VARIABLE}")
    print(f"{'='*60}")
    print(metrics_df.to_string(index=False))
    print(f"{'='*60}\n")

    comp_csv = os.path.join(OUT_DIR_COMP, "comparison_metrics.csv")
    metrics_df.to_csv(comp_csv, index=False)
    print(f"Saved metrics: {comp_csv}")

    # ── 6. Save point-level comparison CSV ───────────────────────
    pts_df = pd.DataFrame({
        "x":           x_coords,
        "y":           y_coords,
        "y_true":      y_true,
        "y_pred_rf":   y_rf,
        "y_pred_bl":   y_baseline,
        "y_pred_ebk":  y_ebk,
    })
    if "point_id" in rf_df.columns:
        pts_df.insert(0, "point_id", rf_df["point_id"].values)
    pts_csv = os.path.join(OUT_DIR_COMP, "comparison_predictions.csv")
    pts_df.to_csv(pts_csv, index=False)
    print(f"Saved predictions: {pts_csv}")

    # ── 7. Scatter plot (3 panels) ────────────────────────────────
    # Use only points where ALL three methods have valid values
    mask_all = (
        np.isfinite(y_true) & np.isfinite(y_rf) &
        np.isfinite(y_baseline) & np.isfinite(y_ebk)
    )
    n_common = int(mask_all.sum())
    print(f"\nCommon valid points (all 3 methods): {n_common}")

    yt_c  = y_true[mask_all]
    yrf_c = y_rf[mask_all]
    ybl_c = y_baseline[mask_all]
    yebk_c = y_ebk[mask_all]

    m_rf_c  = compute_metrics(yt_c, yrf_c,  f"RF {RF_CONFIG}")
    m_bl_c  = compute_metrics(yt_c, ybl_c,  "Baseline (median/unit)")
    m_ebk_c = compute_metrics(yt_c, yebk_c, "EBK (ArcGIS)")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    scatter_panel(axes[0], yt_c, ybl_c,  "Baseline (median/unit)", m_bl_c)
    scatter_panel(axes[1], yt_c, yebk_c, "EBK (ArcGIS)",           m_ebk_c)
    scatter_panel(axes[2], yt_c, yrf_c,  f"RF {RF_CONFIG}",         m_rf_c)
    fig.suptitle(
        f"3-way comparison — {TARGET_VARIABLE}   (n={n_common} common points)",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    plot_path = os.path.join(OUT_DIR_COMP, "comparison_scatter.png")
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved scatter plot: {plot_path}")

    # ── 8. Bar chart of R² and RMSE ──────────────────────────────
    methods = [m["Method"] for m in [m_bl_c, m_ebk_c, m_rf_c]]
    r2s     = [m["R2"]   for m in [m_bl_c, m_ebk_c, m_rf_c]]
    rmses   = [m["RMSE"] for m in [m_bl_c, m_ebk_c, m_rf_c]]

    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    colors = ["#d9534f", "#f0ad4e", "#5cb85c"]

    bars1 = ax1.bar(methods, r2s, color=colors, edgecolor="k", linewidth=0.8)
    ax1.set_title(f"R² — {TARGET_VARIABLE}")
    ax1.set_ylim(0, max(r2s) * 1.25)
    ax1.set_ylabel("R²")
    for bar, val in zip(bars1, r2s):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    bars2 = ax2.bar(methods, rmses, color=colors, edgecolor="k", linewidth=0.8)
    ax2.set_title(f"RMSE — {TARGET_VARIABLE}")
    ax2.set_ylabel("RMSE")
    for bar, val in zip(bars2, rmses):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    bar_path = os.path.join(OUT_DIR_COMP, "comparison_bars.png")
    plt.savefig(bar_path, dpi=200, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved bar chart: {bar_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
