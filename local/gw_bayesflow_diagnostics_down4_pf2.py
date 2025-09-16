#!/usr/bin/env python
"""
gw_bayesflow_diagnostics_down4_pf2.py
-------------------------------------------
• Data: 1024 Hz (down×4) + POOL_FACTOR=2
• Preproc: channels-last (N,T,1) → mean-pool ×2 → standardize (train scaler)
• Draw N_POST_DRAWS posterior samples on a subset and run diagnostics:
  - BayesFlow default suite (loss, recovery, ECDF, z-score contraction)
  - Calibration histogram (rank statistic)  <-- ADDED
  - z-score normality (QQ + histogram)      <-- ADDED
"""

# 0) Environment ---------------------------------------------------------------
import os, json
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

from pathlib import Path
import numpy as np
from numpy.random import default_rng
import tensorflow as tf
import keras
import bayesflow as bf
from bayesflow.adapters import Adapter
from bayesflow.workflows import BasicWorkflow
from bayesflow.diagnostics import plots
from matplotlib import pyplot as plt

# Keep float32 to match the trained model
try:
    from keras import mixed_precision
    mixed_precision.set_global_policy("float32")
except Exception:
    pass

for gpu in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        pass

# 1) Paths & constants ---------------------------------------------------------
PREFIX     = "gw_bbh_down4"   # 1024 Hz dataset
DRIVE_DIR  = Path("/content/drive/MyDrive/gw_dataset")
DATA_DIR   = Path("/content/tmp_data")

VAL_PATH   = DATA_DIR / f"{PREFIX}_val.npz"
MODEL_PATH = DRIVE_DIR / "gw_bayesflow_approx.keras"

SCALER_CANDIDATES = [
    DRIVE_DIR / f"{PREFIX}_wave_scaler_runtime.npz",
    DRIVE_DIR / f"{PREFIX}_wave_scaler.npz",
]
META_JSON  = DRIVE_DIR / f"{PREFIX}_preproc_meta.json"

USE_SUBSET     = True
TEST_N         = 300
N_POST_DRAWS   = 1_000
BATCH_SIZE_SMP = 128

# 2) Load val set ---------------------------------------------------------------
val        = np.load(VAL_PATH)
theta_val  = val["parameters"].astype(np.float32)
x_val      = val["waveforms"].astype(np.float32)

print(f"Loaded: {VAL_PATH}")
print(f"θ_val: {theta_val.shape}   x_val: {x_val.shape}")

# 3) Preprocessing (match training) --------------------------------------------
# channels last
if x_val.ndim == 3 and x_val.shape[1] == 1 and x_val.shape[-1] != 1:
    x_val = np.transpose(x_val, (0, 2, 1))
    print("→ Transposed to channels-last (N, T, 1).  New shape:", x_val.shape)
elif x_val.ndim == 2:
    x_val = x_val[:, :, None]
    print("→ Expanded to channels-last (N, T, 1).  New shape:", x_val.shape)

# mean-pool ×2
pool_factor = 2
try:
    if META_JSON.exists():
        meta = json.loads(META_JSON.read_text())
        pool_factor = int(meta.get("pool_factor", 2))
except Exception:
    pass

def mean_pool_1d(arr, k):
    N, T, C = arr.shape
    cut = (T // k) * k
    if cut != T:
        arr = arr[:, :cut, :]
    return arr.reshape(N, cut // k, k, C).mean(axis=2, dtype=np.float32)

x_val = mean_pool_1d(x_val, pool_factor)
print(f"→ Temporal mean-pooled ×{pool_factor}.  New shape:", x_val.shape)

# standardize with training scaler
wave_mean = None; wave_std = None
for cand in SCALER_CANDIDATES:
    if cand.exists():
        s = np.load(cand)
        wave_mean = np.array(s["mean"], dtype=np.float32)
        wave_std  = np.array(s["std"],  dtype=np.float32)
        print("→ Loaded waveform scaler from:", cand)
        break
if wave_mean is None or wave_std is None:
    wave_mean = x_val.mean(axis=(0,1), keepdims=True).astype(np.float32)
    wave_std  = x_val.std(axis=(0,1),  keepdims=True).astype(np.float32) + 1e-8
    print("⚠️  Scaler not found; using val-set mean/std (fallback).")

if wave_mean.ndim == 0:
    wave_mean = wave_mean[None, None, None]
    wave_std  = wave_std[None,  None,  None]
elif wave_mean.ndim == 1:
    wave_mean = wave_mean[None, None, :]
    wave_std  = wave_std[None,  None, :]

x_val = ((x_val - wave_mean) / wave_std).astype(np.float32)

# 4) Workflow + model -----------------------------------------------------------
adapter  = Adapter().rename("waveforms", "summary_variables").to_array()
workflow = BasicWorkflow(adapter=adapter, optimizer=None)
workflow.approximator = keras.saving.load_model(MODEL_PATH, compile=False)
print("Loaded model:", MODEL_PATH)

# 5) Posterior sampling ---------------------------------------------------------
def batched_sample(wf, waves, draws=1_000, batch=128):
    outs = []
    for i in range(0, len(waves), batch):
        slab = waves[i : i + batch]
        s    = wf.sample(num_samples=draws, conditions={"waveforms": slab})
        outs.append(s["inference_variables"])
    return np.concatenate(outs, axis=0)

if USE_SUBSET:
    rng         = default_rng(42)
    idx         = rng.choice(len(x_val), TEST_N, replace=False)
    x_test      = x_val[idx]
    theta_test  = theta_val[idx]
    samples     = workflow.sample(
        num_samples=N_POST_DRAWS, conditions={"waveforms": x_test}
    )["inference_variables"]
else:
    x_test      = x_val
    theta_test  = theta_val
    samples     = batched_sample(workflow, x_test, draws=N_POST_DRAWS, batch=BATCH_SIZE_SMP)

# 6) Diagnostics ---------------------------------------------------------------
# 6a) Calibration histogram (rank statistic)  <-- THIS produces the chart you want
fig_cal_hist = plots.calibration_histogram(samples, theta_test)

# 6b) BayesFlow default suite (no duplicates)
metrics = workflow.compute_default_diagnostics(
    test_data={"waveforms": x_test, "inference_variables": theta_test},
)
figures = workflow.plot_default_diagnostics(
    test_data                 = {"waveforms": x_test, "inference_variables": theta_test},
    loss_kwargs               = {"figsize": (15, 3), "label_fontsize": 12},
    recovery_kwargs           = {"figsize": (15, 3), "label_fontsize": 12},
    calibration_ecdf_kwargs   = {"figsize": (15, 3), "legend_fontsize": 8,
                                 "difference": True, "label_fontsize": 12},
    z_score_contraction_kwargs= {"figsize": (15, 3), "label_fontsize": 12},
)

print(f"\n=== Summary diagnostics (subset={USE_SUBSET}) ===")
for key, val in metrics.items():
    try:
        print(f"{key:30s}: {float(val):.6f}")
    except (TypeError, ValueError):
        print(f"{key:30s}: {val}")

# 7) Added: z-score normality (QQ + histogram) ---------------------------------
def compute_z_scores(samps: np.ndarray, theta_true: np.ndarray) -> np.ndarray:
    mu  = samps.mean(axis=1)                      # (N, P)
    std = samps.std(axis=1, ddof=1) + 1e-8        # (N, P)
    return (theta_true - mu) / std                # (N, P)

# Use SciPy if available; otherwise a high-accuracy inverse normal fallback
try:
    from scipy.stats import norm
    def normal_quantiles(n: int) -> np.ndarray:
        p = (np.arange(1, n + 1) - 0.5) / n
        p = np.clip(p, 1e-12, 1 - 1e-12)
        return norm.ppf(p).astype(np.float64)
except Exception:
    def _inv_norm_cdf(p):
        p = np.asarray(p, dtype=np.float64)
        a = [-3.969683028665376e+01,  2.209460984245205e+02,
             -2.759285104469687e+02,  1.383577518672690e+02,
             -3.066479806614716e+01,  2.506628277459239e+00]
        b = [-5.447609879822406e+01,  1.615858368580409e+02,
             -1.556989798598866e+02,  6.680131188771972e+01,
             -1.328068155288572e+01]
        c = [-7.784894002430293e-03, -3.223964580411365e-01,
             -2.400758277161838e+00, -2.549732539343734e+00,
              4.374664141464968e+00,  2.938163982698783e+00]
        d = [ 7.784695709041462e-03,  3.224671290700398e-01,
              2.445134137142996e+00,  3.754408661907416e+00]
        plow, phigh = 0.02425, 1 - 0.02425
        q = np.zeros_like(p)
        m = p < plow
        if m.any():
            ql = np.sqrt(-2*np.log(p[m]))
            q[m] = (((((c[0]*ql + c[1])*ql + c[2])*ql + c[3])*ql + c[4])*ql + c[5]) / \
                    ((((d[0]*ql + d[1])*ql + d[2])*ql + d[3])*ql + 1)
        m = (p >= plow) & (p <= phigh)
        if m.any():
            r = p[m] - 0.5; t = r*r
            q[m] = (((((a[0]*t + a[1])*t + a[2])*t + a[3])*t + a[4])*t + a[5])*r / \
                    (((((b[0]*t + b[1])*t + b[2])*t + b[3])*t + b[4])*t + 1)
        m = p > phigh
        if m.any():
            qu = np.sqrt(-2*np.log(1 - p[m]))
            q[m] = -(((((c[0]*qu + c[1])*qu + c[2])*qu + c[3])*qu + c[4])*qu + c[5]) / \
                     ((((d[0]*qu + d[1])*qu + d[2])*qu + d[3])*qu + 1)
        return q
    def normal_quantiles(n: int) -> np.ndarray:
        p = (np.arange(1, n + 1) - 0.5) / n
        p = np.clip(p, 1e-12, 1 - 1e-12)
        return _inv_norm_cdf(p)

def plot_z_normality(z: np.ndarray, titles=None, max_abs=6.0):
    N, P = z.shape
    if titles is None:
        titles = [f"v_{j}" for j in range(P)]

    # QQ plots
    fig_qq, axs = plt.subplots(1, P, figsize=(3.2*P, 3.0), constrained_layout=True)
    axs = np.atleast_1d(axs)
    for j in range(P):
        zj = z[:, j]
        zj = zj[np.isfinite(zj)]
        zj = zj[np.abs(zj) <= max_abs]
        zj.sort()
        q = normal_quantiles(len(zj))
        ax = axs[j]
        ax.plot(q, zj, ".", ms=2)
        lo, hi = q[0], q[-1]
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_title(titles[j], fontsize=11)
        ax.set_xlabel("Theoretical N(0,1) quantile", fontsize=9)
        if j == 0: ax.set_ylabel("Empirical z quantile", fontsize=9)
        ax.grid(alpha=0.2)

    # Histograms
    from math import pi
    xs = np.linspace(-max_abs, max_abs, 512)
    pdf = np.exp(-0.5*xs*xs) / np.sqrt(2.0*pi)

    fig_hist, axs2 = plt.subplots(1, P, figsize=(3.2*P, 3.0), constrained_layout=True)
    axs2 = np.atleast_1d(axs2)
    for j in range(P):
        zj = z[:, j]
        zj = zj[np.isfinite(zj)]
        zj = zj[np.abs(zj) <= max_abs]
        ax = axs2[j]
        ax.hist(zj, bins=30, density=True, alpha=0.85)
        ax.plot(xs, pdf, "k--", lw=1)
        ax.set_title(titles[j], fontsize=11)
        ax.set_xlabel("z-score", fontsize=9)
        if j == 0: ax.set_ylabel("Density", fontsize=9)
        ax.grid(alpha=0.2)

    return fig_qq, fig_hist

def print_z_summary(z: np.ndarray, names=None):
    P = z.shape[1]
    if names is None: names = [f"v_{j}" for j in range(P)]
    print("\nPosterior z-score summary (mean ± std | skew | kurtosis_excess):")
    for j in range(P):
        zj = z[:, j]
        zj = zj[np.isfinite(zj)]
        if len(zj) == 0:
            print(f"  {names[j]:<18s}: (no finite values)")
            continue
        m  = float(zj.mean())
        s  = float(zj.std(ddof=1))
        if len(zj) > 3:
            c3 = float(((zj - m)**3).mean())
            c4 = float(((zj - m)**4).mean())
            skew = c3 / (s**3 + 1e-12)
            kurt = c4 / (s**4 + 1e-12) - 3.0
        else:
            skew = np.nan; kurt = np.nan
        print(f"  {names[j]:<18s}: {m:+.3f} ± {s:.3f} | skew {skew:+.3f} | kurt {kurt:+.3f}")

z_scores = compute_z_scores(samples, theta_test)
param_names = [f"v_{j}" for j in range(z_scores.shape[1])]
fig_qq, fig_hist = plot_z_normality(z_scores, titles=param_names)
print_z_summary(z_scores, names=param_names)

# Optional: save
# OUT_DIR = DRIVE_DIR / "diagnostics_down4_pf2"
# OUT_DIR.mkdir(parents=True, exist_ok=True)
# fig_cal_hist.savefig(OUT_DIR / "calibration_histogram.png", dpi=150, bbox_inches="tight")
# for name, fig in figures.items():
#     if hasattr(fig, "savefig"):
#         fig.savefig(OUT_DIR / f"{name}.png", dpi=150, bbox_inches="tight")
# fig_qq.savefig(OUT_DIR / "z_normality_qq.png", dpi=150, bbox_inches="tight")
# fig_hist.savefig(OUT_DIR / "z_normality_hist.png", dpi=150, bbox_inches="tight")
# print("Saved figures to:", OUT_DIR)
