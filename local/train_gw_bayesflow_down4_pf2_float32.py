#!/usr/bin/env python
"""
train_gw_bayesflow_down4_pf2_float32.py
---------------------------------------
• Expects 1024 Hz dataset (x4 downsample): gw_bbh_down4_train.npz / gw_bbh_down4_val.npz
• Loads from /content/tmp_data, saves model/logs/scaler to Google Drive
• Channels-last (N, T, 1) → mean-pool ×2 (→ 4096 steps, ~512 Hz effective)
• Standardises using TRAIN mean/std (saved alongside model)
• TimeSeriesNetwork + CouplingFlow(depth=4)
• AdamW + warmup→cosine LR, EarlyStopping, CSV logs
• Mixed precision is DISABLED (float32) to avoid dtype mismatch in CouplingFlow
"""

# ───── 0) Environment ─────────────────────────────────────────────────
import os, json
from pathlib import Path
import numpy as np
import tensorflow as tf
import keras
import bayesflow as bf
from bayesflow.adapters import Adapter
from bayesflow.networks import TimeSeriesNetwork, CouplingFlow

os.environ["TF_CPP_MIN_LOG_LEVEL"]      = "2"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# XLA on, but force global float32 (no mixed precision)
try:
    tf.config.optimizer.set_jit(True)
    from keras import mixed_precision
    mixed_precision.set_global_policy("float32")   # ← IMPORTANT
    print("→ Mixed precision OFF (float32), XLA JIT ON.")
except Exception as e:
    print("! Policy setup warning:", e)

for gpu in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        pass

SEED = 42
tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)

# ───── 1) Paths ───────────────────────────────────────────────────────
PREFIX     = "gw_bbh_down4"                          # 1024 Hz dataset
DRIVE_DIR  = Path("/content/drive/MyDrive/gw_dataset")
DATA_DIR   = Path("/content/tmp_data")               # you copied the .npz here

TRAIN_NPZ  = DATA_DIR / f"{PREFIX}_train.npz"
VAL_NPZ    = DATA_DIR / f"{PREFIX}_val.npz"

SAVE_PATH  = DRIVE_DIR / "gw_bayesflow_approx.keras"
HIST_CSV   = DRIVE_DIR / f"{PREFIX}_train_history.csv"
HIST_NPY   = DRIVE_DIR / f"{PREFIX}_train_history.npy"
SCALER_NPZ = DRIVE_DIR / f"{PREFIX}_wave_scaler_runtime.npz"
META_JSON  = DRIVE_DIR / f"{PREFIX}_preproc_meta.json"

# Sanity check inputs exist
for p in (TRAIN_NPZ, VAL_NPZ):
    if not p.exists():
        raise FileNotFoundError(f"Missing dataset file: {p}")

# ───── 2) Load dataset ────────────────────────────────────────────────
train = np.load(TRAIN_NPZ)
val   = np.load(VAL_NPZ)

theta_train = train["parameters"].astype(np.float32)
x_train     = train["waveforms"].astype(np.float32)   # (N, 1, 8192) or (N, 8192, 1)
theta_val   = val["parameters"].astype(np.float32)
x_val       = val["waveforms"].astype(np.float32)

print(f"θ_train: {theta_train.shape}   x_train: {x_train.shape}")
print(f"θ_val:   {theta_val.shape}   x_val:   {x_val.shape}")

# ───── 3) Channels-last (N, T, 1) ─────────────────────────────────────
if x_train.ndim == 3 and x_train.shape[1] == 1 and x_train.shape[-1] != 1:
    x_train = np.transpose(x_train, (0, 2, 1))
    x_val   = np.transpose(x_val,   (0, 2, 1))
    print("→ Transposed to channels-last (N, T, 1)")

# ───── 4) Temporal mean-pooling (×2 for down4 data) ───────────────────
POOL_FACTOR = 2  # 1024 Hz → pool×2 → 4096 steps (~512 Hz eff.)

def mean_pool_1d(arr, k):
    # arr: (N, T, C) → (N, T//k, C)
    N, T, C = arr.shape
    cut = (T // k) * k
    if cut != T:
        arr = arr[:, :cut, :]
    return arr.reshape(N, cut // k, k, C).mean(axis=2, dtype=np.float32)

x_train = mean_pool_1d(x_train, POOL_FACTOR)
x_val   = mean_pool_1d(x_val,   POOL_FACTOR)
print(f"→ Mean-pooled ×{POOL_FACTOR}:  x_train {x_train.shape}, x_val {x_val.shape}")

# ───── 5) Standardise (use TRAIN stats) ───────────────────────────────
def per_channel_standardise(x, mean=None, std=None):
    # x: (N, T, C)
    if mean is None or std is None:
        mean = x.mean(axis=(0,1), keepdims=True, dtype=np.float64)
        std  = x.std(axis=(0,1),  keepdims=True, dtype=np.float64) + 1e-8
    x = (x - mean) / std
    return x.astype(np.float32), mean.squeeze(), std.squeeze()

xm, xs = float(x_train.mean()), float(x_train.std())
ym, ys = float(x_val.mean()),   float(x_val.std())
print(f"Waveform stats before std: train mean={xm:+.3f} std={xs:.3f} | val mean={ym:+.3f} std={ys:.3f}")

x_train, wmean, wstd = per_channel_standardise(x_train)
x_val,   _,    _     = per_channel_standardise(x_val, wmean[None,None,...], wstd[None,None,...])
np.savez(SCALER_NPZ, mean=wmean.astype(np.float32), std=wstd.astype(np.float32))
META_JSON.write_text(json.dumps({"pool_factor": POOL_FACTOR, "dataset_prefix": PREFIX}, indent=2))
print(f"→ Standardised. Saved scaler → {SCALER_NPZ}")

# ───── 6) Adapter ─────────────────────────────────────────────────────
adapter = Adapter().rename("waveforms", "summary_variables").to_array()

# ───── 7) Networks ────────────────────────────────────────────────────
summary_net = TimeSeriesNetwork(
    summary_dim   = 64,
    filters       = (48, 64, 96, 128),   # lighter convs (T=4096)
    kernel_sizes  = (5, 5, 3, 3),
    recurrent_dim = 128,                 # safe for 4096 steps
    bidirectional = False,               # flip True if VRAM allows
    dropout       = 0.35,
)

invertible_net = CouplingFlow(
    depth       = 4,
    transform   = "affine",
    permutation = "random",
    use_actnorm = True,
)

# ───── 8) Optimizer with warm-up → cosine LR ──────────────────────────
EPOCHS      = 80
BATCH_SIZE  = 64  # adjust per VRAM (32–96)

steps_per_epoch = max(1, len(x_train) // BATCH_SIZE)
total_steps     = EPOCHS * steps_per_epoch
warmup_steps    = int(0.05 * total_steps)  # 5%

class WarmUpCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr, warmup_steps, total_steps, final_lr_ratio=5e-3):
        super().__init__()
        self._base_lr        = float(base_lr)
        self._warmup_steps_i = int(warmup_steps)
        self._total_steps_i  = int(total_steps)
        self._final_ratio    = float(final_lr_ratio)
        self.base_lr      = tf.convert_to_tensor(self._base_lr, dtype=tf.float32)
        self.warmup_steps = tf.cast(self._warmup_steps_i, tf.float32)
        self.total_steps  = tf.cast(self._total_steps_i, tf.float32)
        self.cosine_decay = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate = self.base_lr,
            decay_steps           = self._total_steps_i - self._warmup_steps_i,
            alpha                 = self._final_ratio,
        )
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warm = self.base_lr * (step / tf.maximum(1.0, self.warmup_steps))
        cos  = self.cosine_decay(tf.cast(step - self.warmup_steps, tf.int64))
        return tf.where(step < self.warmup_steps, warm, cos)
    def get_config(self):
        return {"base_lr": self._base_lr,
                "warmup_steps": self._warmup_steps_i,
                "total_steps": self._total_steps_i,
                "final_lr_ratio": self._final_ratio}

lr_schedule = WarmUpCosine(
    base_lr=3e-4, warmup_steps=warmup_steps, total_steps=total_steps, final_lr_ratio=5e-3
)

optimizer = keras.optimizers.AdamW(
    learning_rate = lr_schedule,
    weight_decay  = 1e-4,
    clipnorm      = 1.0,
)

# ───── 9) Workflow ────────────────────────────────────────────────────
workflow = bf.workflows.BasicWorkflow(
    adapter             = adapter,
    summary_network     = summary_net,
    invertible_network  = invertible_net,
    optimizer           = optimizer,
    inference_variables = ["inference_variables"],
    summary_variables   = ["summary_variables"],
)

# ───── 10) Callbacks ─────────────────────────────────────────────────
early = keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
ckpt  = keras.callbacks.ModelCheckpoint(filepath=SAVE_PATH, monitor="val_loss",
                                        save_best_only=True, verbose=1)
csv   = keras.callbacks.CSVLogger(HIST_CSV, append=False)
nan   = keras.callbacks.TerminateOnNaN()

# ───── 11) Train ─────────────────────────────────────────────────────
history = workflow.fit_offline(
    data            = {"waveforms": x_train, "inference_variables": theta_train},
    validation_data = {"waveforms": x_val,   "inference_variables": theta_val},
    epochs          = EPOCHS,
    batch_size      = BATCH_SIZE,
    callbacks       = [early, ckpt, csv, nan],
    verbose         = 1,
)

np.save(HIST_NPY, {k: np.array(v) for k, v in history.history.items()}, allow_pickle=True)

print("\nBest model saved to →", SAVE_PATH.resolve())
print("History CSV →", HIST_CSV.resolve())
print("History NPY →", HIST_NPY.resolve())
print("Scaler NPZ  →", SCALER_NPZ.resolve())
print("Meta JSON   →", META_JSON.resolve())
