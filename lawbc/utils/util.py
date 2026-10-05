"""General-purpose utility helpers for training, export, and analysis."""

import csv
import os
import random
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.signal import bessel, sosfilt


# ===== Plotting =====

def save_loss_svg(
    train_losses: List[float],
    test_losses: List[float],
    out_path: str,
    title: str,
) -> None:
    """Save a train/test loss curve as SVG."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    plt.figure()
    plt.plot(train_losses, label="train")
    plt.plot(test_losses, label="test")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(title)
    plt.legend()
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.savefig(out_path, format="svg")
    plt.close()


def save_pred_csv(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: str,
) -> None:
    """Save y_true vs y_pred to CSV."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["y_true", "y_pred"])
        for yt, yp in zip(y_true.reshape(-1), y_pred.reshape(-1)):
            writer.writerow([f"{yt:.6g}", f"{yp:.6g}"])
    print(f"[CSV] saved: {out_path}  ({len(y_true)} rows)")


def save_pred_plots(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_dir: str,
    model_name: str,
) -> None:
    """Save scatter plot and per-class mean/std bar chart for predictions."""
    os.makedirs(out_dir, exist_ok=True)
    yt = np.round(y_true.reshape(-1).astype(np.float64), 2)
    yp = y_pred.reshape(-1).astype(np.float64)

    # Scatter: y_true vs y_pred
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(yt, yp, s=4, alpha=0.15, rasterized=True)
    mn, mx = min(yt.min(), yp.min()), max(yt.max(), yp.max())
    ax.plot([mn, mx], [mn, mx], "k--", linewidth=1, label="y=x")
    ax.set_xlabel("y_true (mu_terrain)")
    ax.set_ylabel("y_pred")
    ax.set_title(f"{model_name} — y_true vs y_pred scatter")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    scatter_path = os.path.join(out_dir, f"{model_name}_scatter.png")
    fig.savefig(scatter_path, dpi=150)
    plt.close(fig)
    print(f"[Plot] saved: {scatter_path}")

    # Mean +/- std per y_true class
    unique_vals = np.sort(np.unique(np.round(yt, 1)))
    means, stds, counts = [], [], []
    for v in unique_vals:
        mask = np.abs(yt - v) < 0.05
        preds = yp[mask]
        means.append(np.mean(preds))
        stds.append(np.std(preds, ddof=1) if len(preds) > 1 else 0.0)
        counts.append(len(preds))

    means = np.array(means)
    stds = np.array(stds)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(unique_vals, means, width=0.06, alpha=0.7, label="mean(y_pred)")
    ax.errorbar(
        unique_vals, means, yerr=stds, fmt="none",
        ecolor="black", capsize=3, linewidth=1, label="±1 std",
    )
    ax.plot(
        [unique_vals[0] - 0.05, unique_vals[-1] + 0.05],
        [unique_vals[0] - 0.05, unique_vals[-1] + 0.05],
        "k--", linewidth=1, alpha=0.5, label="y=x",
    )

    for v, m, s, n in zip(unique_vals, means, stds, counts):
        ax.annotate(
            f"μ={m:.3f}\nσ={s:.3f}\nn={n}",
            (v, m), textcoords="offset points", xytext=(0, 12),
            fontsize=7, ha="center",
        )

    ax.set_xlabel("y_true (mu_terrain)")
    ax.set_ylabel("y_pred")
    ax.set_title(f"{model_name} — prediction mean ± std per class")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    stats_path = os.path.join(out_dir, f"{model_name}_pred_stats.png")
    fig.savefig(stats_path, dpi=150)
    plt.close(fig)
    print(f"[Plot] saved: {stats_path}")


def plot_alpha_comparison(
    all_loss_logs: Dict[str, Dict],
    optimal_alpha: float,
    out_dir: str,
):
    """Plot baseline vs alpha-scaled train/test MSE per model."""
    model_names = [
        k for k, v in all_loss_logs.items()
        if "alpha_scaled" in v and "baseline" in v
    ]
    if not model_names:
        return

    n = len(model_names)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 5 * nrows), squeeze=False)

    for i, name in enumerate(model_names):
        ax = axes[i // ncols][i % ncols]
        bl = all_loss_logs[name]["baseline"]
        al = all_loss_logs[name]["alpha_scaled"]
        epochs = range(1, len(bl["train_mse"]) + 1)

        ax.plot(epochs, bl["train_mse"], "b-",  label="train (baseline)", alpha=0.8)
        ax.plot(epochs, bl["test_mse"],  "r-",  label="test (baseline)",  alpha=0.8)
        ax.plot(epochs, al["train_mse"], "b--", label=f"train (a={optimal_alpha:.3f})", alpha=0.8)
        ax.plot(epochs, al["test_mse"],  "r--", label=f"test (a={optimal_alpha:.3f})",  alpha=0.8)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE")
        ax.set_title(name)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    fig.suptitle(f"Baseline vs Alpha-Scaled Training (a={optimal_alpha:.4f})", fontsize=14)
    fig.tight_layout()
    out_path = os.path.join(out_dir, "alpha_comparison.svg")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[Alpha] Comparison plot saved: {out_path}")


# ===== Model export =====

def export_torchscript(
    model: nn.Module,
    example_base: torch.Tensor,
    example_joint: torch.Tensor,
    out_path: str,
) -> None:
    """Trace and save a TorchScript model."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if hasattr(model, "_orig_mod"):
        model = model._orig_mod

    model.eval()
    with torch.no_grad():
        traced = torch.jit.trace(model, (example_base, example_joint))
        traced.save(out_path)
    print(f"[TorchScript] saved: {out_path}")


# ===== Reproducibility =====

def set_seed(seed: int) -> None:
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def get_device(gpu_id: int) -> torch.device:
    """Return a CUDA device if available, otherwise CPU."""
    if torch.cuda.is_available():
        if gpu_id < 0 or gpu_id >= torch.cuda.device_count():
            raise RuntimeError(
                f"Invalid gpu_id={gpu_id}, device_count={torch.cuda.device_count()}"
            )
        torch.cuda.set_device(gpu_id)
        return torch.device(f"cuda:{gpu_id}")
    return torch.device("cpu")


# ===== String / array helpers =====

def _decode_str_array(arr) -> List[str]:
    """Decode a bytes/string array into a list of Python str."""
    out = []
    for v in arr:
        if isinstance(v, bytes):
            out.append(v.decode("utf-8", errors="ignore"))
        else:
            out.append(str(v))
    return out


# ===== Signal processing =====

def bessel_lowpass_nd(
    x: np.ndarray,
    fs: float,
    cutoff_hz: float,
    order: int = 2,
    axis: int = 0,
) -> np.ndarray:
    """Causal digital Bessel low-pass filter (SOS form, forward-only).

    Uses ``sosfilt`` (causal, forward-only) to match real-time deployment.
    The SOS coefficients can be reused in an online loop via ``sosfilt``
    with explicit ``zi`` state.
    """
    if cutoff_hz <= 0:
        return x

    orig_dtype = x.dtype
    x64 = np.asarray(x, dtype=np.float64)

    nyq = 0.5 * fs
    wn = cutoff_hz / nyq
    sos = bessel(order, wn, btype="low", analog=False, output="sos", norm="phase")

    y = sosfilt(sos, x64, axis=axis)
    return y.astype(orig_dtype)
