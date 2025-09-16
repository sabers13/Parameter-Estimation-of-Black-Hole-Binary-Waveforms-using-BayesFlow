
#!/usr/bin/env python
"""
priors.py
---------------
Sample BBH parameters exactly as specified:

* m1  :  Uniform or power-law on [5, 80] M⊙
* m2  :  Uniform on [m_min, m1]  ⇒  always  m2 ≤ m1
* chi :  Uniform on [0, 0.99]   (dimensionless spin magnitude)
* D   :  Uniform in comoving volume  (∝ D²) on [100, 2000] Mpc
* inc :  Isotropic  ⇒ cos ι ~ U(−1, 1)

Return value is a float32 `pandas.DataFrame` with the canonical column order
(m1, m2, chi1, chi2, D, inc).
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def sample_prior(
    n_samples: int,
    *,
    alpha: float = 0.0,          # 0 → uniform; >0 → p(m1) ∝ m1^{−α}
    m_min: float = 5.0,
    m_max: float = 80.0,
    D_min: float = 100.0,
    D_max: float = 2_000.0,
    rng_seed: int | None = None,
) -> pd.DataFrame:
    rng = np.random.default_rng(rng_seed)

    # --- primary mass  m1 ---------------------------------------
    if np.isclose(alpha, 0.0):
        m1 = rng.uniform(m_min, m_max, n_samples)
    else:
        expo = 1.0 - alpha
        u    = rng.random(n_samples)
        m1   = (u * (m_max**expo - m_min**expo) + m_min**expo) ** (1.0 / expo)

    # --- secondary mass  m2  (uniform in [m_min, m1]) -----------
    m2 = rng.uniform(m_min, m1)

    # --- dimensionless spins  -----------------------------------
    chi1 = rng.uniform(0.0, 0.99, n_samples)
    chi2 = rng.uniform(0.0, 0.99, n_samples)

    # --- luminosity distance  D  (uniform in volume) ------------
    uV  = rng.random(n_samples)
    D   = (D_min**3 + uV * (D_max**3 - D_min**3)) ** (1.0 / 3.0)

    # --- inclination  inc  (isotropic) --------------------------
    inc = np.arccos(rng.uniform(-1.0, 1.0, n_samples))

    return pd.DataFrame({
        "m1":   m1.astype(np.float32),
        "m2":   m2.astype(np.float32),
        "chi1": chi1.astype(np.float32),
        "chi2": chi2.astype(np.float32),
        "D":    D.astype(np.float32),
        "inc":  inc.astype(np.float32),
    })


# quick demo when run as a script
if __name__ == "__main__":
    df = sample_prior(5, alpha=1.6, rng_seed=0).round(3)
    print(df)
