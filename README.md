# Parameter Estimation of Black-Hole Binary Waveforms with BayesFlow

Neural **simulation-based inference (SBI)** for recovering binary black-hole (BBH) parameters from simulated gravitational-wave strain. We generate BBH waveforms with **PyCBC**, add realistic colored noise, whiten and downsample the signals, then train a **BayesFlow** normalizing-flow model to approximate the posterior over physical parameters \((m_1, m_2, \chi_1, \chi_2, D, \iota)\).

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sabers13/Parameter-Estimation-of-Black-Hole-Binary-Waveforms-using-BayesFlow/blob/main/SBI_gw.ipynb)

- 📓 **Notebook:** `SBI_gw.ipynb` (Colab-ready)  
- 🏫 Course project (SBI), **TU Dortmund University** — Aug 2, 2025  

---

## Why SBI (and BayesFlow)?
Classical likelihoods for GW waveforms are intractable and repeated waveform synthesis is expensive for MCMC. **Amortized SBI** learns a fast posterior estimator from simulations: at inference time, one forward pass produces samples from \(p(\theta\mid x)\). We use **BayesFlow** (normalizing flows) as the posterior approximator.

---

## Skills & Tools (Resume‑ready)

- **Machine Learning & Bayesian Inference:** Simulation‑Based Inference (SBI), amortized posteriors, normalizing flows (CouplingFlow), uncertainty quantification, posterior calibration (rank histograms, ECDF).  
- **Deep Learning:** TensorFlow/Keras; Conv1D & GRU encoders; dropout, weight decay, gradient clipping; cosine LR schedules; early stopping & checkpointing; GPU runtimes (Colab).  
- **Signal Processing for GW:** PSD‑based frequency‑domain whitening, anti‑alias decimation (**4096→1024 Hz**), mean‑pooling, standardization/normalization of time‑series.  
- **Scientific Python:** Python 3.10, NumPy, SciPy, Matplotlib, PyCBC waveform generation, data serialization (NPZ); reproducible data pipelines.  
- **Software Engineering:** Git/GitHub, `.gitignore` for large artifacts, Colab ↔ GitHub workflows, environment setup, structured experiments.  
- **Statistics:** NRMSE, posterior contraction (PCON), calibration error (CAL), credible intervals & coverage analysis.  
- **Domain Knowledge:** gravitational‑wave BBH parameters (masses, spins, luminosity distance, inclination) and modeling assumptions (aligned spins, single detector).

 **Keywords:** BayesFlow · PyCBC · Simulation‑Based Inference · Normalizing Flows · TensorFlow · Time‑Series · Signal Processing · Bayesian Inference · Gravitational Waves

---

## Data & Simulator

- **Waveforms:** time-domain BBH using `pycbc.waveform.get_td_waveform`, duration **8 s** at **4096 Hz** (later downsampled).  
- **Parameters:** \(\theta=(m_1,m_2,\chi_1,\chi_2,D,\iota)\) with \(m_1\ge m_2\).  
- **Priors:**  
  - \(m_{1,2}\sim \mathcal U(5,80)~M_\odot\) (then sort),  
  - \(\chi_{1,2}\sim \mathcal U(0,0.99)\),  
  - \(D\) uniform in volume on **[100, 2000] Mpc**,  
  - isotropic orientation with \(\cos\iota\sim \mathcal U[-1,1]\).  
- **Noise & Whitening:** colored Gaussian noise from an analytic aLIGO-like PSD; whiten in frequency domain to ~unit-variance; decimate **4096→1024 Hz** in two anti-aliasing stages; during training mean-pool by 2 (effective ~512 Hz).  
- **Dataset:** **24,000** simulations → **20k train / 4k val**; standardize waveforms; z-score parameters using train stats.

---

## Model

- **Summary network (encoder):** 1-channel sequence (length ~4096 after pooling) → Conv blocks with filters **(48, 64, 96, 128)** and kernels **(5,5,3,3)** → **GRU(128)** → **dropout 0.35** → **summary dim 64**.  
- **Posterior network:** depth-4 **affine CouplingFlow** (with actnorm & random permutations), conditioned on the summary \(s(x)\).  
- **Target:** amortized posterior \(q_\phi(\theta\mid x)\) over \((m_1,m_2,\chi_1,\chi_2,D,\iota)\).

---

## Training

- **Objective:** negative log-likelihood under the flow posterior.  
- **Optimizer:** **AdamW**, weight decay \(1\times10^{-4}\), gradient clip-norm **1.0**.  
- **Schedule:** **5% warm-up** then **cosine decay**; base LR \(3\times10^{-4}\).  
- **Run:** up to **80 epochs**, batch **64**, **early stopping** (patience 8), best-checkpoint saved.

---

## Results & Diagnostics

Evaluated on **300** held-out events with 1,000 posterior samples per event.

- **Calibration:** rank histograms and ECDF curves are broadly acceptable; some parameters show mild bias/dispersion deviations.  
- **Recovery:** posterior means vs. truth show positive correlations; strongest for masses.

| Parameter | NRMSE | PCON | CAL |
|---|---:|---:|---:|
| \(m_1\) | 0.701 | 0.266 | 0.0239 |
| \(m_2\) | 0.430 | 0.536 | 0.0173 |
| \(\chi_1\) | 0.887 | 0.021 | 0.0245 |
| \(\chi_2\) | 0.843 | 0.033 | 0.0415 |
| \(D\) | 0.838 | −0.019 | 0.0183 |
| \(\iota\) | 0.996 | 0.005 | 0.0239 |

> **Example posterior (95% CI)** for one event:  
> \(m_1\) 35.4 \([33.1, 37.7]\), \(m_2\) 22.8 \([20.0, 25.5]\), \(\chi_1\) 0.50 \([0.36, 0.63]\), \(\chi_2\) 0.30 \([0.15, 0.46]\), \(D\) 503.5 Mpc \([460.0, 550.3]\), \(\iota\) 1.18 rad \([0.95, 1.38]\).

---

## Quickstart

### Option A — Colab (recommended)
Open the notebook in Google Colab and run all cells (GPU runtime recommended).

### Option B — Local
1. Create and activate a Python 3.10+ environment.  
2. Install deps (typical stack):
   ```bash
   pip install bayesflow pycbc numpy scipy matplotlib tensorflow
   ```
3. Run the notebook or convert to a script with:
   ```bash
   jupyter nbconvert --to script SBI_gw.ipynb
   python SBI_gw.py
   ```

> **Notes**  
> • Simulation can be compute-intensive; reduce dataset size for quick tests.  
> • Keep large data/artifacts out of git; add them to `.gitignore`.

---

## Repository Contents

- `SBI_gw.ipynb` — full pipeline: simulation → preprocessing → training → diagnostics → inference.  
- *(Optional)* `artifacts/` — trained model, scalers, and meta manifest (if you choose to save them).

---

## Limitations

- **Physics:** aligned spins only; no precession/higher harmonics; single-detector input → known degeneracies (e.g., \(D\)–\(\iota\)).  
- **Noise realism:** analytic PSD + Gaussian noise; real data are non-stationary and glitchy.  
- **Capacity:** shallow flow and summary network chosen for stability/speed; deeper/spline flows or attention may help.

---

## Roadmap

- Extend to precession & higher modes; multi-detector inputs.  
- Learnable multi-scale downsampling; reparameterize to \((M, q, \chi_{\mathrm{eff}}, \log D)\).  
- Larger/stratified held-out sets; non-stationary noise and glitch handling; post-hoc calibration.

---

## Acknowledgments

- **BayesFlow**: amortized Bayesian inference with normalizing flows.  
- **PyCBC**: waveform generation & GW utilities.  
- Course materials and guidance from the SBI course at TU Dortmund.

---

## Citation

If you use or build on this work, please cite the project report associated with this repository (course project, TU Dortmund, Aug 2025) and the BayesFlow & PyCBC papers/tooling.

