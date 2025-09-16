#!/usr/bin/env python
# simulator.py  –  float32-strict, PSD-safe, preserves amplitude–distance info

from __future__ import annotations
import numpy as np, pandas as pd
from pycbc.waveform import get_td_waveform
from pycbc.noise    import noise_from_psd
from pycbc.psd      import aLIGOZeroDetHighPower
from pycbc.types    import TimeSeries, FrequencySeries

# ───────── constants ─────────
DELTA_T  = np.float32(1 / 4096)
DURATION = np.float32(8.0)
F_LOWER  = 40.0
APPROX   = "IMRPhenomD"
PSD_FLOW = 20.0

N        = int(DURATION / DELTA_T)
DELTA_F  = 1.0 / DURATION

# ───────── helpers ─────────
def _make_psd(n_freq: int) -> FrequencySeries:
    return aLIGOZeroDetHighPower(n_freq, DELTA_F, PSD_FLOW)

def _whiten(ts: TimeSeries, cal: float) -> TimeSeries:
    fd        = ts.to_frequencyseries()
    psd64     = _make_psd(len(fd)).numpy()
    psd64[psd64 < 1e-40] = 1e-40
    white_fd  = fd.numpy() / np.sqrt(psd64.astype(np.float32))
    white_fd *= np.sqrt(N * DELTA_F) / 2.0
    ts_white  = FrequencySeries(
        white_fd.astype(np.complex64), delta_f=fd.delta_f, copy=False
    ).to_timeseries()
    return TimeSeries(ts_white.numpy().astype(np.float32),
                      delta_t=ts_white.delta_t) * cal

def _optimal_snr(h: TimeSeries) -> float:
    h_fd  = h.to_frequencyseries()
    psd   = _make_psd(len(h_fd)).numpy()
    psd[psd < 1e-40] = 1e-40
    snr_sq = 4.0 * ((np.abs(h_fd.numpy()) ** 2) / psd).sum() * DELTA_F
    return float(np.sqrt(snr_sq))

# one-time σ-calibration for whitening
_noise = noise_from_psd(N, DELTA_T, _make_psd(N // 2 + 1), seed=12345)
_CAL   = 1.0 / _whiten(_noise, 1.0).numpy().std(dtype=np.float32)

# ───────── main ─────────
def simulate_event(
    theta: pd.Series,
    seed: int | None = None,
    *,
    min_snr:   float | None = 8.0,
    target_snr: float | None = None,   # ← default None (no signal scaling!)
    snr_jitter: tuple[float, float] | None = None,  # e.g. (8, 25) for optional spread
) -> TimeSeries:
    """
    Returns a whitened single-channel strain. We never scale the signal
    to a fixed SNR; if we need to enforce a minimum, we scale the NOISE.
    """
    rng = np.random.default_rng(seed)

    # --- clean signal at the given parameters (preserve distance!) ---
    hp, _ = get_td_waveform(
        approximant = APPROX,
        mass1=float(theta.m1),  mass2=float(theta.m2),
        spin1z=float(theta.chi1), spin2z=float(theta.chi2),
        distance=float(theta.D),  inclination=float(theta.inc),
        delta_t=float(DELTA_T),   f_lower=F_LOWER,
    )
    h_det = hp.astype(np.float32)

    # pad/crop to fixed length
    if len(h_det) < N:
        pad   = np.zeros(N - len(h_det), dtype=np.float32)
        h_det = TimeSeries(np.concatenate([h_det.numpy(), pad]), delta_t=DELTA_T)
    else:
        h_det = h_det[:N]
    h_det.start_time = 0.0

    # --- draw noise and (optionally) scale noise to meet SNR constraints ---
    noise = noise_from_psd(N, DELTA_T, _make_psd(N // 2 + 1), seed=seed).astype(np.float32)

    clean_snr = _optimal_snr(h_det)
    noise_scale = 1.0

    # enforce a minimum SNR by scaling NOISE (not the signal)
    if (min_snr is not None) and (clean_snr < min_snr) and (clean_snr > 0):
        noise_scale = clean_snr / float(min_snr)

    # optional SNR jitter band via further noise scaling
    if snr_jitter is not None and clean_snr > 0:
        lo, hi = snr_jitter
        target = float(rng.uniform(lo, hi))
        noise_scale = min(noise_scale, clean_snr / target)

    noise *= np.float32(noise_scale)

    # combine, random time shift, and whiten
    strain = h_det + noise
    shift  = int(rng.integers(0, N))
    strain = TimeSeries(np.roll(strain.numpy(), shift).astype(np.float32),
                        delta_t=DELTA_T)

    return _whiten(strain, _CAL)
