# plot_loss_trajectory.py
# Reads Keras CSVLogger history or the saved NPY history dict and renders a
# "Loss Trajectory" chart with raw points + moving averages.

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- 1) Point to your saved history (CSV preferred; NPY works too)
# If you used the paths from your training script:
csv_path = Path("/content/drive/MyDrive/gw_dataset/gw_bbh_down4_train_history.csv")
npy_path = Path("/content/drive/MyDrive/gw_dataset/gw_bbh_down4_train_history.npy")  # fallback
out_png  = csv_path.with_name("gw_bbh_down4_loss_trajectory.png")

# ---- 2) Robust moving average (NaN-aware, centered)
def moving_average(x, window):
    x = np.asarray(x, dtype=float)
    if window < 2:
        return x
    w = int(window) | 1  # force odd window
    mask = ~np.isnan(x)
    num = np.convolve(np.where(mask, x, 0.0), np.ones(w), mode="same")
    den = np.convolve(mask.astype(float), np.ones(w), mode="same")
    return np.divide(num, np.maximum(den, 1e-12))

# ---- 3) Load history
if csv_path.exists():
    df = pd.read_csv(csv_path)
    # Keras CSVLogger usually has 'loss' and 'val_loss'
    loss     = df["loss"].to_numpy()
    val_loss = df["val_loss"].to_numpy() if "val_loss" in df else None
elif npy_path.exists():
    hist = np.load(npy_path, allow_pickle=True).item()
    loss     = np.array(hist.get("loss", []))
    val_loss = np.array(hist.get("val_loss", [])) if "val_loss" in hist else None
else:
    raise FileNotFoundError("Neither history CSV nor NPY found.")

epochs = np.arange(1, len(loss) + 1)

# ---- 4) Choose smoothing window (~5% of epochs, >=5)
win = max(5, int(round(0.05 * len(loss))) | 1)  # make it odd
loss_ma = moving_average(loss, win)
val_ma  = moving_average(val_loss, win) if val_loss is not None else None

# ---- 5) Plot (matplotlib only)
plt.figure(figsize=(16, 4))

# raw points
plt.plot(epochs, loss, marker="o", ms=3, lw=1, alpha=0.35, label="Training")
if val_loss is not None:
    plt.plot(epochs, val_loss, marker="o", ms=3, lw=1, alpha=0.25, linestyle="--", label="Validation")

# moving averages
plt.plot(epochs, loss_ma, lw=2.5, label="Training (Moving Average)")
if val_ma is not None:
    plt.plot(epochs, val_ma, lw=2.5, linestyle="--", label="Validation (Moving Average)")

plt.title("Loss Trajectory")
plt.xlabel("Training epoch #")
plt.ylabel("Value")
plt.grid(axis="y", alpha=0.25)
plt.legend(loc="upper right", frameon=True)
plt.tight_layout()

# ---- 6) Save next to your history CSV
plt.savefig(out_png, dpi=180)
print(f"Saved plot → {out_png}")
plt.show()
