# =============================================================================
#  merge_shards.py   –   build **20 000 train / 4 000 val** (24 000 total)
#  -----------------------------------------------------------------------------
#  Reads every gw_bbh_down2_shard*.npz, concatenates, splits, scales, saves:
#     datasets/gw_bbh_down2_train.npz
#     datasets/gw_bbh_down2_val.npz
#     datasets/gw_bbh_down2_param_scaler.pkl
# =============================================================================
from pathlib import Path
import sys, numpy as np, pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RAW_DIR  = Path("datasets/raw")
OUT_DIR  = Path("datasets")
PREFIX   = "gw_bbh_down4"

TRAIN_N  = 20_000
VAL_N    = 4_000
SEED     = 42

def load_shards():
    pars, waves = [], []
    for f in sorted(RAW_DIR.glob(f"{PREFIX}_shard*.npz")):
        d = np.load(f)
        pars.append(d["parameters"]); waves.append(d["waveforms"])
        print(f" loaded {f.name:32s}  {d['parameters'].shape[0]:,} samples")
    return np.concatenate(pars), np.concatenate(waves)

def main():
    if not RAW_DIR.exists():
        sys.exit("ERROR: datasets/raw/ missing. Generate shards first.")

    θ_all, x_all = load_shards()
    total = len(θ_all)
    required = TRAIN_N + VAL_N
    if total < required:
        sys.exit(f"ERROR: need ≥ {required:,} samples, found {total:,}.")

    θ_tr, θ_val, x_tr, x_val = train_test_split(
        θ_all, x_all,
        train_size   = TRAIN_N,
        test_size    = VAL_N,
        random_state = SEED,
        shuffle      = True,
    )

    scaler = StandardScaler()
    θ_tr  = scaler.fit_transform(θ_tr).astype(np.float32)
    θ_val = scaler.transform(θ_val).astype(np.float32)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_DIR / f"{PREFIX}_train.npz",
                        parameters=θ_tr, waveforms=x_tr)
    np.savez_compressed(OUT_DIR / f"{PREFIX}_val.npz",
                        parameters=θ_val, waveforms=x_val)
    with open(OUT_DIR / f"{PREFIX}_param_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print("\n✅ 20 000-train / 4 000-val dataset saved to", OUT_DIR.resolve())

if __name__ == "__main__":
    main()
