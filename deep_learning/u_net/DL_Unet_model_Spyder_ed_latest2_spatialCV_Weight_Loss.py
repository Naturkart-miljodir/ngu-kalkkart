
# -*- coding: utf-8 -*-
"""
U-Net training for Kalk project tiles
- Masked loss (background ignored)
- Class-weighted loss for classes 1–3
- Optional ADVANCED LOSSES:
      • Focal-Tversky
      • Dice + CE hybrid
- Spatial CV (N folds) with KFold
- Per-epoch macro-F1 (background ignored)
- MC DROPOUT UNCERTAINTY QUANTIFICATION
- QC plots + metric logs
"""

import os
import sys
import glob
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
try:
    from tqdm.auto import tqdm
except ImportError:
    # Fallback: no external dependency required
    def tqdm(x, **kwargs):
        return x

from sklearn.model_selection import train_test_split, KFold
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input, Concatenate, Dropout
from keras.models import Model
from keras.callbacks import Callback, ModelCheckpoint
import matplotlib.pyplot as plt

# Add zlib path for Windows (fixes zlibwapi.dll error)
if sys.platform == 'win32':
    conda_env = os.path.dirname(os.path.dirname(sys.executable))
    zlib_path = os.path.join(conda_env, 'Library', 'bin')
    if os.path.exists(zlib_path):
        # Add to PATH as fallback
        if zlib_path not in os.environ['PATH']:
            os.environ['PATH'] = zlib_path + os.pathsep + os.environ['PATH']
        # Python 3.8+: explicitly add DLL search path (more reliable)
        try:
            os.add_dll_directory(zlib_path)
        except AttributeError:
            pass  # Python < 3.8, PATH modification above is the fallback

# =====================================================================
# GPU CONFIGURATION
# =====================================================================
print("\n" + "=" * 70)
print("GPU CONFIGURATION")
print("=" * 70)

print(f"TensorFlow version: {tf.__version__}")
print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")

# List available GPUs
gpus = tf.config.list_physical_devices('GPU')
print(f"Number of GPUs available: {len(gpus)}")

if gpus:
    try:
        # Enable memory growth for all GPUs (prevents TensorFlow from allocating all GPU memory)
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✓ Enabled memory growth for: {gpu.name}")
        
        # Get logical devices
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"Physical GPUs: {len(gpus)}, Logical GPUs: {len(logical_gpus)}")
        
        # Test GPU availability
        if tf.test.gpu_device_name():
            print(f"✓ Default GPU device: {tf.test.gpu_device_name()}")
        else:
            print("⚠ GPU detected but not set as default device")
            
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(f"✗ GPU configuration error: {e}")
else:
    print("✗ WARNING: No GPU found!")
    print("  Training will run on CPU (this will be VERY slow for U-Net)")
    print("  Consider:")
    print("    1. Installing tensorflow-gpu")
    print("    2. Installing CUDA toolkit")
    print("    3. Installing cuDNN")

print("=" * 70)
print()

# =====================================================================
# 0. USER SWITCHES & DEFAULT ADVANCED LOSS PARAMETERS
# =====================================================================

DO_SPATIAL_CV = True
N_FOLDS       = 3
DO_MAIN_TRAIN = True

# UNCERTAINTY QUANTIFICATION
USE_MC_DROPOUT = True               # Enable MC Dropout for uncertainty
MC_DROPOUT_RATE = 0.3               # Dropout rate during training (keep same at inference)
MC_ITERATIONS = 20                  # Number of forward passes for uncertainty

# MAIN TOGGLE: Basic vs Advanced losses
USE_ADVANCED_LOSS = True             # ⬅️ Change this to True to activate advanced loss
ADVANCED_LOSS_TYPE = "dice_ce"   # options: "focal_tversky", "dice_ce"

# Default parameters for Focal-Tversky
FT_ALPHA = 0.7
FT_BETA  = 0.3
FT_GAMMA = 1.33

print("\n=== LOSS SETTINGS ===")
print("USE_ADVANCED_LOSS:", USE_ADVANCED_LOSS)
print("ADVANCED_LOSS_TYPE:", ADVANCED_LOSS_TYPE)
print(f"Focal-Tversky: α={FT_ALPHA}, β={FT_BETA}, γ={FT_GAMMA}")

print("\n=== UNCERTAINTY SETTINGS ===")
print("USE_MC_DROPOUT:", USE_MC_DROPOUT)
print("MC_DROPOUT_RATE:", MC_DROPOUT_RATE)
print("MC_ITERATIONS:", MC_ITERATIONS)
print()

# =====================================================================
# 1. Paths
# =====================================================================

TILE_DIR  = r"C:\Users\acosta_pedro\OneDrive - Norges geologiske undersøkelse\Geochemistry NGU_2026\Kalk_project\Modelling\Tiles"
MODEL_OUT = r"C:\Users\acosta_pedro\OneDrive - Norges geologiske undersøkelse\Geochemistry NGU_2026\Kalk_project\Modelling\Models"
QC_PLOTS  = r"C:\Users\acosta_pedro\OneDrive - Norges geologiske undersøkelse\Geochemistry NGU_2026\Kalk_project\Modelling\QC_plots"

X_dir = os.path.join(TILE_DIR, "X")
y_dir = os.path.join(TILE_DIR, "y")

os.makedirs(MODEL_OUT, exist_ok=True)
os.makedirs(QC_PLOTS, exist_ok=True)

# =====================================================================
# 2. Load tiles
# =====================================================================

X_paths = sorted(glob.glob(os.path.join(X_dir, "*.npy")))
y_paths = sorted(glob.glob(os.path.join(y_dir, "*.npy")))

print("Number of tiles:", len(X_paths))

X = np.array([np.load(p) for p in X_paths])

cleaned = []
for p in y_paths:
    arr = np.load(p).astype(np.int16)
    arr = np.where((arr < 0) | (arr > 3), 0, arr)
    cleaned.append(arr)

y = np.array(cleaned, dtype=np.int16)

print("Shapes -> X:", X.shape, "y:", y.shape)
print("Unique labels:", np.unique(y))

# =====================================================================
# 3. Convert X to channels-last (if needed)
# =====================================================================

# Check if data needs transposing from (N, C, H, W) to (N, H, W, C)
if X.ndim == 4 and X.shape[1] < X.shape[2]:
    print(f"[INFO] Transposing X from {X.shape} (channels_first) to channels_last format...")
    X = np.transpose(X, (0, 2, 3, 1))
    print(f"[INFO] X shape after transpose: {X.shape}")
elif X.ndim == 4 and X.shape[-1] < X.shape[1]:
    print(f"[INFO] Data already in channels_last format: {X.shape}")
else:
    print(f"[WARNING] Unexpected X shape: {X.shape}")

num_classes = 3
n_channels = X.shape[-1]
print("X shape:", X.shape)
print(f"Number of channels: {n_channels}")
print(f"[DEBUG] X after loading - has NaN: {np.isnan(X).any()}, range: [{np.nanmin(X):.4f}, {np.nanmax(X):.4f}]")

# Store NaN mask before cleaning (True where data is VALID)
nan_mask_original = np.isnan(X)
# Pixel is valid only if ALL channels are valid (not NaN)
valid_data_mask = ~np.any(nan_mask_original, axis=-1)  # Shape: (N, H, W)

# Clean NaN values and apply per-channel standardization
has_nan = np.isnan(X).any()
if has_nan:
    n_nan = np.isnan(X).sum()
    pct_nan = (n_nan / X.size) * 100
    print(f"[INFO] Found {n_nan} NaN pixels ({pct_nan:.2f}% of data).")

print(f"[INFO] Applying per-channel standardization (z-score normalization)...")
normalization_params = {}

for ch in range(n_channels):
    # Get valid (non-NaN) pixels for this channel
    valid_mask_ch = ~nan_mask_original[..., ch]
    valid_pixels = X[..., ch][valid_mask_ch]
    
    if len(valid_pixels) > 0:
        mean_ch = valid_pixels.mean()
        std_ch = valid_pixels.std()
        
        # Standardize: (x - mean) / std
        X[..., ch] = (X[..., ch] - mean_ch) / (std_ch + 1e-8)
        
        # Replace NaN with 0 (which is now the mean after standardization)
        X[..., ch] = np.nan_to_num(X[..., ch], nan=0.0)
        
        # Store parameters for inference
        normalization_params[ch] = {'mean': mean_ch, 'std': std_ch}
        
        print(f"  Channel {ch}: mean={mean_ch:.4f}, std={std_ch:.4f} (computed from {len(valid_pixels)} valid pixels)")
    else:
        print(f"  [WARNING] Channel {ch}: No valid pixels found!")
        normalization_params[ch] = {'mean': 0.0, 'std': 1.0}

print(f"[INFO] Standardization complete. NaN→0 (mean after normalization).")
print(f"[INFO] After preprocessing - has NaN: {np.isnan(X).any()}, range: [{np.min(X):.4f}, {np.max(X):.4f}]")
print(f"[INFO] Valid data mask will be applied to uncertainty outputs.")

# =====================================================================
# 4. CLASS WEIGHTS + LOSS FUNCTIONS
# =====================================================================

# ----- Class weights -----
flat_y = y.flatten()
counts = np.bincount(flat_y, minlength=4)
bg, c1, c2, c3 = counts

non_bg = c1 + c2 + c3

weights_np = np.zeros(4)
for c, cnt in zip([1, 2, 3], [c1, c2, c3]):
    weights_np[c] = non_bg / (3 * max(cnt, 1))

weights_np[1:4] /= weights_np[1:4].mean()
class_weights_tf = tf.constant(weights_np, dtype=tf.float32)

print("\nClass weights:", weights_np)

# ----- BASE LOSS (weighted CE) -----
@tf.autograph.experimental.do_not_convert
def masked_sparse_ce_weighted(y_true, y_pred):
    y_true = tf.cast(y_true, tf.int32)
    mask   = tf.cast(y_true > 0, tf.float32)

    y_shift = tf.where(mask > 0, y_true - 1, 0)
    ce = tf.keras.losses.sparse_categorical_crossentropy(y_shift, y_pred)

    w = tf.gather(class_weights_tf, tf.clip_by_value(y_true, 0, 3))
    ce = ce * mask * w
    return tf.reduce_sum(ce) / (tf.reduce_sum(mask * w) + 1e-7)

# ----- Focal-Tversky Loss -----
def tversky(y_true, y_pred, alpha=FT_ALPHA, beta=FT_BETA):
    y_true = tf.cast(y_true, tf.int32)
    mask   = tf.cast(y_true > 0, tf.float32)

    y_shift = tf.where(mask > 0, y_true - 1, 0)
    onehot  = tf.one_hot(y_shift, 3) * tf.expand_dims(mask, -1)

    p1 = onehot
    p0 = y_pred

    tp = tf.reduce_sum(p0 * p1, axis=[1,2])
    fp = tf.reduce_sum(p0 * (1 - p1), axis=[1,2])
    fn = tf.reduce_sum((1 - p0) * p1, axis=[1,2])

    return (tp + 1e-7) / (tp + alpha * fp + beta * fn + 1e-7)

@tf.autograph.experimental.do_not_convert
def focal_tversky_loss(y_true, y_pred):
    TI = tversky(y_true, y_pred)
    return tf.reduce_mean(tf.pow((1 - TI), FT_GAMMA))

# ----- Dice + CE hybrid -----
def dice_coef(y_true, y_pred):
    y_true = tf.cast(y_true, tf.int32)
    mask   = tf.cast(y_true > 0, tf.float32)

    y_shift = tf.where(mask > 0, y_true - 1, 0)
    onehot  = tf.one_hot(y_shift, 3) * tf.expand_dims(mask, -1)

    inter = tf.reduce_sum(onehot * y_pred, axis=[1,2])
    sums  = tf.reduce_sum(onehot + y_pred, axis=[1,2])
    return tf.reduce_mean((2*inter + 1e-7)/(sums + 1e-7))

@tf.autograph.experimental.do_not_convert
def dice_ce_loss(y_true, y_pred):
    ce   = masked_sparse_ce_weighted(y_true, y_pred)
    dice = dice_coef(y_true, y_pred)
    return 0.5 * ce + 0.5 * (1 - dice)

# ----- MASKED ACCURACY -----
@tf.autograph.experimental.do_not_convert
def masked_accuracy(y_true, y_pred):
    y_true = tf.cast(y_true, tf.int32)
    mask   = tf.cast(y_true > 0, tf.float32)

    y_shift = tf.where(mask > 0, y_true - 1, 0)

    # FIX: force argmax to int32
    y_pred_label = tf.cast(tf.argmax(y_pred, axis=-1), tf.int32)

    correct = tf.cast(tf.equal(y_shift, y_pred_label), tf.float32)
    correct = correct * mask

    return tf.reduce_sum(correct) / (tf.reduce_sum(mask) + 1e-7)

# ----- FINAL LOSS SELECTOR -----
if not USE_ADVANCED_LOSS:
    LOSS_FUNCTION = masked_sparse_ce_weighted
    print("\n➡ Using BASIC weighted CE loss")
else:
    if ADVANCED_LOSS_TYPE == "focal_tversky":
        LOSS_FUNCTION = focal_tversky_loss
        print("\n➡ Using ADVANCED LOSS: Focal-Tversky")
    else:
        LOSS_FUNCTION = dice_ce_loss
        print("\n➡ Using ADVANCED LOSS: Dice + CE hybrid")

# =====================================================================
# 5. Build U-Net
# =====================================================================

def conv_block(x, f):
    x = Conv2D(f, 3, padding="same", activation="relu")(x)
    if USE_MC_DROPOUT:
        x = Dropout(MC_DROPOUT_RATE)(x)  # Dropout layer - training mode controlled at call time
    x = Conv2D(f, 3, padding="same", activation="relu")(x)
    if USE_MC_DROPOUT:
        x = Dropout(MC_DROPOUT_RATE)(x)  # Dropout layer - training mode controlled at call time
    return x

def encoder_block(x, f):
    c = conv_block(x, f)
    return c, MaxPooling2D(2)(c)

def decoder_block(x, skip, f):
    x = UpSampling2D(2)(x)
    return conv_block(Concatenate()([x, skip]), f)

def build_unet(input_shape, num_classes):
    inp = Input(shape=input_shape)
    c1, p1 = encoder_block(inp, 32)
    c2, p2 = encoder_block(p1, 64)
    c3, p3 = encoder_block(p2, 128)
    c4, p4 = encoder_block(p3, 256)

    bn = conv_block(p4, 512)

    d1 = decoder_block(bn, c4, 256)
    d2 = decoder_block(d1, c3, 128)
    d3 = decoder_block(d2, c2, 64)
    d4 = decoder_block(d3, c1, 32)

    out = Conv2D(num_classes, 1, activation="softmax")(d4)
    return Model(inp, out)

# =====================================================================
# 6. F1 callback
# =====================================================================

class F1MetricsCallback(Callback):
    def __init__(self, Xv, yv, out_txt, prefix=""):
        super().__init__()
        self.Xv = Xv
        self.yv = yv
        self.out = out_txt
        self.prefix = prefix
        self.history = []

    def on_epoch_end(self, epoch, logs=None):
        y_prob = self.model.predict(self.Xv, batch_size=2, verbose=0)
        y_pred = np.argmax(y_prob, axis=-1).astype(np.int32) + 1


        mask = self.yv > 0
        y_true = self.yv[mask]
        y_pred = y_pred[mask]

        eps = 1e-7
        f1s = []

        ep_dict = {}
        for cls in [1,2,3]:
            tp = np.sum((y_true==cls)&(y_pred==cls))
            fp = np.sum((y_true!=cls)&(y_pred==cls))
            fn = np.sum((y_true==cls)&(y_pred!=cls))

            prec = tp/(tp+fp+eps)
            rec  = tp/(tp+fn+eps)
            f1   = 2*prec*rec/(prec+rec+eps)
            f1s.append(f1)

            ep_dict[f"f1_{cls}"] = f1

        ep_dict["macro_f1"] = np.mean(f1s)
        logs["val_macro_f1"] = ep_dict["macro_f1"]

        print(
    f"\n[{self.prefix} Epoch {epoch+1}] "
    f"macro_F1={ep_dict['macro_f1']:.4f}  "
    f"F1_1={ep_dict['f1_1']:.3f}  "
    f"F1_2={ep_dict['f1_2']:.3f}  "
    f"F1_3={ep_dict['f1_3']:.3f}",
    flush=True
)


        self.history.append(ep_dict)

    def on_train_end(self, logs=None):
        with open(self.out, "w") as f:
            for i, h in enumerate(self.history):
                f.write(f"Epoch {i+1}: {h}\n")
        print(f"Saved F1 log to {self.out}")

# =====================================================================
# 6. MC DROPOUT UNCERTAINTY FUNCTION
# =====================================================================

def compute_mc_uncertainty(model, X, valid_mask=None, mc_iterations=MC_ITERATIONS):
    """
    Compute pixel-wise uncertainty using MC Dropout.
    Explicitly enables all Dropout layers for MC sampling.
    
    Args:
        model: Trained model with MC Dropout
        X: Input data (N, H, W, C)
        valid_mask: Boolean mask (N, H, W) - True where data is valid, False for NaN regions
        mc_iterations: Number of forward passes
        
    Returns:
        mean_pred: Mean prediction (N, H, W, num_classes)
        std_pred: Standard deviation (N, H, W, num_classes)
        entropy: Predictive entropy (N, H, W) - NaN in masked regions
    """
    print("  [DEBUG] Enabling MC Dropout for inference...")
    print(f"  [DEBUG] Input X shape: {X.shape}, dtype: {X.dtype}")
    print(f"  [DEBUG] Input X range: [{np.min(X):.4f}, {np.max(X):.4f}]")
    print(f"  [DEBUG] Input has NaN: {np.isnan(X).any()}")
    
    predictions = []
    
    # Convert X to tensor once to speed up iterations
    X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
    
    for i in tqdm(range(mc_iterations), desc="MC Dropout passes", unit="pass"):
        # Use training=True to keep dropout active
        pred = model(X_tensor, training=True).numpy()
        
        # Check first prediction
        if i == 0:
            print(f"  [DEBUG] First prediction shape: {pred.shape}, dtype: {pred.dtype}")
            print(f"  [DEBUG] First prediction range: [{np.min(pred):.4f}, {np.max(pred):.4f}]")
            print(f"  [DEBUG] First prediction has NaN: {np.isnan(pred).any()}")
        
        predictions.append(pred)
    
    predictions = np.array(predictions)  # (mc_iterations, N, H, W, num_classes)
    
    # Diagnostic: check if dropout is having an effect
    pred_variance = np.var(predictions, axis=0)
    mean_variance = np.mean(pred_variance)
    print(f"  MC prediction variance (mean): {mean_variance:.6f}")
    
    # Check first few predictions to see if they differ
    if mc_iterations >= 2:
        pred_diff = np.abs(predictions[0] - predictions[1])
        max_diff = np.max(pred_diff)
        print(f"  Max diff between first two MC passes: {max_diff:.6f}")
    
    if mean_variance < 1e-8:
        print("  [WARNING] Very low variance across MC passes — dropout may not be active!")
    
    mean_pred = np.mean(predictions, axis=0)
    std_pred = np.std(predictions, axis=0)
    
    # Entropy: -sum(p * log(p)) with numerical stability
    # Clamp probabilities to avoid log(0) issues
    mean_pred_clamped = np.clip(mean_pred, 1e-10, 1.0 - 1e-10)
    
    # Normalize to ensure sum=1 across classes (in case of numerical drift)
    mean_pred_norm = mean_pred_clamped / np.sum(mean_pred_clamped, axis=-1, keepdims=True)
    
    # Compute entropy with stable log
    entropy = -np.sum(mean_pred_norm * np.log(mean_pred_norm), axis=-1)
    
    # Apply valid data mask to entropy (set NaN regions to NaN)
    if valid_mask is not None:
        entropy_masked = np.where(valid_mask, entropy, np.nan)
        n_masked = (~valid_mask).sum()
        n_total = valid_mask.size
        print(f"  Applied mask: {n_masked}/{n_total} pixels ({n_masked/n_total*100:.1f}%) marked as no-data (NaN)")
    else:
        entropy_masked = entropy
        print("  [INFO] No mask applied - computing uncertainty for all pixels")
    
    # Diagnostic: check entropy range (only valid pixels)
    valid_entropy = entropy_masked[~np.isnan(entropy_masked)]
    if len(valid_entropy) > 0:
        print(f"  Entropy range (valid pixels): [{np.min(valid_entropy):.4f}, {np.max(valid_entropy):.4f}]")
    else:
        print("  [WARNING] All entropy values are NaN!")
    
    return mean_pred, std_pred, entropy_masked
    
    return mean_pred, std_pred, entropy

def save_uncertainty_maps(mean_pred, std_pred, entropy, fold_idx, data_type="cv"):
    """Save uncertainty maps as numpy files."""
    out_dir = os.path.join(QC_PLOTS, f"uncertainty_{data_type}_fold{fold_idx}")
    os.makedirs(out_dir, exist_ok=True)
    
    np.save(os.path.join(out_dir, "mean_pred.npy"), mean_pred)
    np.save(os.path.join(out_dir, "std_pred.npy"), std_pred)
    np.save(os.path.join(out_dir, "entropy.npy"), entropy)
    
    # Calculate statistics for valid (non-NaN) pixels
    total_pixels = entropy.size
    nan_pixels = np.isnan(entropy).sum()
    valid_pixels = total_pixels - nan_pixels
    pct_valid = (valid_pixels / total_pixels) * 100
    pct_masked = (nan_pixels / total_pixels) * 100
    
    # Plot mean entropy per sample
    mean_entropy_per_sample = np.mean(entropy, axis=(1, 2))
    
    # Remove NaN values for histogram
    valid_entropy = mean_entropy_per_sample[~np.isnan(mean_entropy_per_sample)]
    
    if len(valid_entropy) > 0:
        plt.figure(figsize=(10, 5))
        plt.hist(valid_entropy, bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel("Mean Entropy per Sample")
        plt.ylabel("Frequency")
        plt.title(f"Predictive Uncertainty Distribution ({data_type.upper()} Fold {fold_idx})")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "entropy_histogram.png"), dpi=150)
        plt.close()
        
        print(f"Saved uncertainty maps to {out_dir}")
        print(f"  Valid pixels: {valid_pixels}/{total_pixels} ({pct_valid:.1f}%)")
        print(f"  Masked (no-data) pixels: {nan_pixels} ({pct_masked:.1f}%)")
        print(f"  Mean entropy (valid pixels): {np.mean(valid_entropy):.4f}")
        print(f"  Max entropy: {np.max(valid_entropy):.4f}")
        print(f"  Min entropy: {np.min(valid_entropy):.4f}")
    else:
        print(f"[WARNING] No valid entropy values found for {data_type} fold {fold_idx}")
        print(f"  All {total_pixels} pixels are masked (no-data)")
        print(f"Saved uncertainty maps to {out_dir} (all NaN - check data mask)")

# =====================================================================
# 7. SPATIAL CV  (with KFold shuffle)
# =====================================================================

if DO_SPATIAL_CV:
    print("\n=== Spatial CV ===")

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X), start=1):
        print(f"\n--- Fold {fold}/{N_FOLDS} ---")

        Xtr, Xva = X[tr_idx], X[va_idx]
        ytr, yva = y[tr_idx], y[va_idx]
        
        # Make copies to prevent any in-place modification during training
        Xva_clean = Xva.copy()
        
        # Debug: check for NaN before training
        print(f"  [DEBUG CV FOLD {fold}] Xva after split - has NaN: {np.isnan(Xva).any()}, range: [{np.nanmin(Xva):.4f}, {np.nanmax(Xva):.4f}]")

        model = build_unet(X.shape[1:], num_classes)
        model.compile(optimizer="adam", loss=LOSS_FUNCTION, metrics=[masked_accuracy])

        ckpt = ModelCheckpoint(os.path.join(MODEL_OUT, f"unet_cv_fold{fold}.keras"),
                               monitor="val_loss", save_best_only=True)

        f1_log = os.path.join(QC_PLOTS, f"cv_fold{fold}_f1.txt")
        f1cb = F1MetricsCallback(Xva, yva, f1_log, prefix=f"CV Fold {fold}")

        model.fit(
            Xtr, ytr,
            validation_data=(Xva, yva),
            epochs=10,
            batch_size=2,
            callbacks=[ckpt, f1cb],
            verbose=1
        )
        
        # Debug: check for NaN after training
        print(f"  [DEBUG CV FOLD {fold}] After fit - Xva has NaN: {np.isnan(Xva).any()}, Xva_clean has NaN: {np.isnan(Xva_clean).any()}")
        
        # Compute uncertainty on validation set
        if USE_MC_DROPOUT:
            print(f"\nComputing MC Dropout uncertainty (Fold {fold})...")
            # Get mask for validation samples
            va_mask = valid_data_mask[va_idx]
            mean_pred, std_pred, entropy = compute_mc_uncertainty(model, Xva_clean, valid_mask=va_mask, mc_iterations=MC_ITERATIONS)
            save_uncertainty_maps(mean_pred, std_pred, entropy, fold_idx=fold, data_type="cv")

# =====================================================================
# 8. MAIN TRAINING  (Option A → uses selected loss)
# =====================================================================

if DO_MAIN_TRAIN:
    print("\n=== Main Training ===")

    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Make copies to prevent any in-place modification during training
    Xva_clean = Xva.copy()
    print(f"  [DEBUG MAIN] Xva after split - has NaN: {np.isnan(Xva).any()}, range: [{np.nanmin(Xva):.4f}, {np.nanmax(Xva):.4f}]")

    model = build_unet(X.shape[1:], num_classes)
    model.compile(optimizer="adam", loss=LOSS_FUNCTION, metrics=[masked_accuracy])

    ckpt = ModelCheckpoint(os.path.join(MODEL_OUT, "unet_best_model.keras"),
                           monitor="val_loss", save_best_only=True)

    f1_main_log = os.path.join(QC_PLOTS, "main_train_f1.txt")
    f1cb = F1MetricsCallback(Xva, yva, f1_main_log, prefix="MAIN")

    history = model.fit(
        Xtr, ytr,
        validation_data=(Xva, yva),
        epochs=20,
        batch_size=2,
        callbacks=[ckpt, f1cb],
        verbose=1
    )

    # Plot QC training curves
    hist = history.history
    fig, ax = plt.subplots(1,2, figsize=(12,4))

    ax[0].plot(hist["loss"], label="loss")
    ax[0].plot(hist["val_loss"], label="val_loss")
    ax[0].legend()
    ax[0].set_title("Loss")

    ax[1].plot(hist["masked_accuracy"], label="masked_acc")
    ax[1].plot(hist["val_masked_accuracy"], label="val_masked_acc")
    ax[1].legend()
    ax[1].set_title("Masked Accuracy")

    out_png = os.path.join(QC_PLOTS, "training_curves.png")
    plt.savefig(out_png, dpi=150)
    plt.close()

    print("\n✔ Saved:", out_png)
    print("✔ F1 log:", f1_main_log)
    
    # Compute uncertainty on validation set (main training)
    if USE_MC_DROPOUT:
        print("\nComputing MC Dropout uncertainty (Main Training)...")
        print(f"  [DEBUG MAIN] Before MC - Xva has NaN: {np.isnan(Xva).any()}, Xva_clean has NaN: {np.isnan(Xva_clean).any()}")
        # Get indices for validation samples
        _, va_idx = train_test_split(np.arange(len(X)), test_size=0.2, random_state=42)
        va_mask_main = valid_data_mask[va_idx]
        mean_pred, std_pred, entropy = compute_mc_uncertainty(model, Xva_clean, valid_mask=va_mask_main, mc_iterations=MC_ITERATIONS)
        save_uncertainty_maps(mean_pred, std_pred, entropy, fold_idx=0, data_type="main")
    
    print("   Training completed successfully!")
    print("   Check the saved plots and F1 logs for results.")
