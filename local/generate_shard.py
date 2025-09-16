#!/usr/bin/env python
# =============================================================================
#  generate_shard.py  —  4096 Hz → 1024 Hz (×4) via two-stage ×2×2 decimation
#  ----------------------------------------------------------------------------
#  • Uses simulator.simulate_event(...) to generate 8 s whitened strain @ 4096 Hz
#  • Anti-alias decimation in two gentle ×2 stages with Kaiser windows + padding
#  • Light Tukey end-taper reduces edge ripple before each stage
#  • Writes shards: datasets/raw/gw_bbh_down4_shard00.npz, 01, …
#  • Outputs:
#       parameters: (N, 6)  float32
#       waveforms : (N, 1, 8192) float32   # 8 s @ 1024 Hz
#
#  NOTE for training/diagnostics:
#    - Set POOL_FACTOR = 2 (since dataset is 1024 Hz, 8 s → 8192 samples).
# =============================================================================

from __future__ import annotations
import os, time
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_backend
from scipy.signal import resample_poly, windows

import priors
import simulator

# ─────────────── Config ──────────────────────────────────────────────────────
SHARD_SIZE  = 4_000
RAW_DIR     = Path("datasets/raw")
PREFIX      = "gw_bbh_down4"          # new prefix for 1024 Hz shards

FS_IN       = 4096                     # simulator output rate
DURATION    = 8.0                      # seconds
DOWN_TOTAL  = 4                        # 4096 → 1024 Hz
FS_OUT      = FS_IN // DOWN_TOTAL      # 1024 Hz
T           = int(FS_OUT * DURATION)   # 8192 samples

# Decimator tuning
KAISER_BETA = 12.0                     # stronger low-pass than default (8.0)
PAD_SAMPLES = 1024                     # pad each end before ×2 to reduce ripple
TAPER_ALPHA = 0.02                     # 2% Tukey taper at ends before each stage

N_PROC      = -1                       # use all CPU cores
OUT_DTYPE   = np.float32               # final dtype on disk

# Prevent thread over-subscription (faster + more stable)
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(v, "1")

# ─────────────── Helpers ─────────────────────────────────────────────────────
def _taper_ends(x: np.ndarray, alpha: float = TAPER_ALPHA) -> np.ndarray:
    """Apply a light Tukey taper to both ends (reduces Gibbs ringing)."""
    if alpha <= 0.0:
        return x
    w = windows.tukey(len(x), alpha=alpha, sym=True)
    return (x * w).astype(np.float64, copy=False)

def _pad_edges(x: np.ndarray, pad: int = PAD_SAMPLES) -> np.ndarray:
    """Pad with zeros on both ends before decimation (removed after)."""
    if pad <= 0:
        return x
    return np.pad(x, (pad, pad), mode="constant")

def _trim_edges(y: np.ndarray, down: int, pad: int = PAD_SAMPLES) -> np.ndarray:
    """Remove the decimated padding after downsampling by 'down'."""
    if pad <= 0:
        return y
    s = pad // down
    return y[s: -s] if s > 0 else y

def _decimate_x2(x: np.ndarray,
                 beta: float = KAISER_BETA,
                 pad: int = PAD_SAMPLES,
                 taper_alpha: float = TAPER_ALPHA) -> np.ndarray:
    """4096→2048 (or 2048→1024) in a single ×2 stage with anti-alias filtering."""
    x64 = _taper_ends(x.astype(np.float64, copy=False), alpha=taper_alpha)
    x64 = _pad_edges(x64, pad=pad)
    y   = resample_poly(x64, up=1, down=2, window=("kaiser", float(beta)))
    y   = _trim_edges(y, down=2, pad=pad)
    return y.astype(np.float64, copy=False)

def decimate_4096_to_1024(x: np.ndarray) -> np.ndarray:
    """Cascade ×2 then ×2. Keeps exact 8 s length after trims."""
    y = _decimate_x2(x, beta=KAISER_BETA, pad=PAD_SAMPLES, taper_alpha=TAPER_ALPHA)   # 2048 Hz
    y = _decimate_x2(y, beta=KAISER_BETA, pad=PAD_SAMPLES, taper_alpha=TAPER_ALPHA)   # 1024 Hz
    # Enforce exact length (minor rounding guard)
    if len(y) != T:
        y = y[:T] if len(y) > T else np.pad(y, (0, T - len(y)))
    return y.astype(OUT_DTYPE, copy=False)

# ─────────────── Simulation wrapper ──────────────────────────────────────────
def _sim_one(row_tup, idx: int, seed0: int) -> np.ndarray:
    """
    Run one simulation at 4096 Hz and decimate to 1024 Hz.
    Returns shape (1, T_out) for easy stacking into (N, 1, T_out).
    """
    # Ensure we pass a pandas.Series of PHYSICAL parameters (not z-scores)
    row = pd.Series(row_tup._asdict())
    ts  = simulator.simulate_event(
            row, seed=seed0 + idx,
            min_snr=8.0,        # safe floor via NOISE scaling inside simulator
            target_snr=None,    # DO NOT scale signal; preserve amplitude–distance info
            snr_jitter=None,
          )
    y4096 = ts.numpy()                        # float32, length = 8 s * 4096
    y1024 = decimate_4096_to_1024(y4096)      # float32, length = 8192
    assert y1024.shape == (T,), y1024.shape
    return y1024[None, :]                     # (1, T)

def _next_shard_idx() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    existing = sorted(RAW_DIR.glob(f"{PREFIX}_shard*.npz"))
    return len(existing)

# ─────────────── Main ────────────────────────────────────────────────────────
def main():
    shard_idx = _next_shard_idx()
    seed_base = shard_idx * SHARD_SIZE
    print(f"[Shard {shard_idx:02d}] Simulating {SHARD_SIZE:,} events at 4096 Hz → 1024 Hz …")

    df = priors.sample_prior(SHARD_SIZE, rng_seed=seed_base).astype(np.float32)

    t0 = time.time()
    with parallel_backend("loky", inner_max_num_threads=1):
        waves = Parallel(n_jobs=N_PROC, verbose=5)(
            delayed(_sim_one)(r, i, seed_base)
            for i, r in enumerate(df.itertuples(index=False))
        )
    waves = np.stack(waves, axis=0)          # (N, 1, T)
    assert waves.shape == (SHARD_SIZE, 1, T), waves.shape
    print(f"[Shard {shard_idx:02d}] Done in {time.time()-t0:.1f}s  →  waves {waves.shape}, dtype {waves.dtype}")

    out_path = RAW_DIR / f"{PREFIX}_shard{shard_idx:02d}.npz"
    np.savez_compressed(
        out_path,
        parameters=df.to_numpy().astype(np.float32, copy=False),
        waveforms=waves.astype(OUT_DTYPE, copy=False),
    )
    print(f"Saved → {out_path.resolve()}")

if __name__ == "__main__":
    main()
