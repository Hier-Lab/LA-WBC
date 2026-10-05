"""
Optimization Helper for input scaling factor alpha
"""

from __future__ import annotations

import random
from typing import Callable, List

import numpy as np
from scipy.optimize import Bounds, NonlinearConstraint, minimize
from torch.utils.data import ConcatDataset, DataLoader
from tqdm.auto import tqdm


# Calculate condition number
def kappa(mat: np.ndarray, effective_rank: int | None = None) -> float:
    """Condition number of a matrix."""
    svals = np.linalg.svd(mat, compute_uv=False)
    if effective_rank is None:
        return float(svals[0] / svals[-1])
    idx = len(svals) - effective_rank
    idx = max(0, min(idx, len(svals) - 1))
    return float(svals[0] / svals[idx])


# Calculate frobenius norm divided by matrix 2-norm
def frobenius_over_two(mat: np.ndarray) -> float:
    return np.linalg.norm(mat, ord="fro") / np.linalg.norm(mat, ord=2)


# Scaled Covariance Matrix
def build_sigma_tilde(
    alpha: float,
    sigma_z: np.ndarray,
    col_list: List[int],
    *args,
    **kwargs,
) -> np.ndarray:
    """Return scaled matrix."""
    n_row, n_col = sigma_z.shape

    if n_row != n_col:
        raise ValueError("Given matrix is not a squared matrix.")

    sigma_scaled = sigma_z.copy()

    # Masking
    row_in = np.isin(np.arange(sigma_scaled.shape[0]), col_list)
    col_in = np.isin(np.arange(sigma_scaled.shape[1]), col_list)

    both_mask = row_in[:, None] & col_in
    xor_mask = row_in[:, None] ^ col_in

    # Scaled matrix
    scale = np.full_like(sigma_scaled, 1, dtype=float)
    scale[xor_mask] = alpha
    scale[both_mask] = alpha**2

    sigma_scaled = sigma_scaled * scale

    return sigma_scaled


# Objective function
def objective(
    alpha: np.ndarray,
    sigma_z: np.ndarray,
    sigma_builder: Callable[[float, np.ndarray], np.ndarray],
    effective_rank: int | None = None,
    *sb_args,
    **sb_kwargs,
) -> float:
    """Optimize for scaling factor."""
    alpha_val = float(alpha[0])
    sigma_tilde = sigma_builder(alpha_val, sigma_z, *sb_args, **sb_kwargs)
    ratio = kappa(sigma_tilde, effective_rank) / kappa(sigma_z, effective_rank)
    return ratio


# Constraints
def constraint_cond_number(
    alpha: np.ndarray,
    sigma_z: np.ndarray,
    sigma_builder: Callable[[float, np.ndarray], np.ndarray],
    effective_rank: int | None = None,
    *sb_args,
    **sb_kwargs,
) -> float:
    """After scaling, condition number should not increase."""
    alpha_val = float(alpha[0])
    sigma_tilde = sigma_builder(alpha_val, sigma_z, *sb_args, **sb_kwargs)
    return kappa(sigma_z, effective_rank) - kappa(sigma_tilde, effective_rank)


def constraint_norm_ratio(
    alpha: np.ndarray,
    sigma_z: np.ndarray,
    sigma_builder: Callable[[float, np.ndarray], np.ndarray],
    *sb_args,
    **sb_kwargs,
) -> float:
    """After scaling, delta should not decrease."""
    alpha_val = float(alpha[0])
    sigma_tilde = sigma_builder(alpha_val, sigma_z, *sb_args, **sb_kwargs)
    return frobenius_over_two(sigma_tilde) - frobenius_over_two(sigma_z)


# Solver
def solve_alpha(
    sigma_z: np.ndarray,
    sigma_builder: Callable[[float, np.ndarray], np.ndarray],
    initial_alpha: float | None = None,
    alpha_range=(0.001, 2),
    n_grid: int = 1001,
    effective_rank: int | None = None,
    *sb_args,
    **sb_kwargs,
) -> float:
    """Return optimal alpha.

    effective_rank: if set, condition number uses the k-th smallest singular
                    value instead of the absolute smallest.
    """
    bounds = Bounds([alpha_range[0]], [alpha_range[1]])

    # Global grid search
    grid = np.logspace(
        np.log10(alpha_range[0]), np.log10(alpha_range[1]), n_grid,
    )
    best_alpha = None
    best_val = np.inf
    for a in grid:
        # when constraints are satisfied
        if (constraint_cond_number(
                np.array([a]), sigma_z, sigma_builder,
                effective_rank, *sb_args, **sb_kwargs) >= 0
            and constraint_norm_ratio(
                np.array([a]), sigma_z, sigma_builder,
                *sb_args, **sb_kwargs) >= 0):
            val = objective(
                np.array([a]), sigma_z, sigma_builder,
                effective_rank, *sb_args, **sb_kwargs,
            )
            if val < best_val:
                best_val = val
                best_alpha = a

    if best_alpha is None:
        raise RuntimeError("No feasible alpha found in global search.")

    x0_alpha = best_alpha if initial_alpha is None else initial_alpha

    # Nonlinear constraints
    nlc1 = NonlinearConstraint(
        fun=lambda a: constraint_cond_number(
            a, sigma_z, sigma_builder, effective_rank, *sb_args, **sb_kwargs,
        ),
        lb=-1e-6, ub=np.inf,
    )
    nlc2 = NonlinearConstraint(
        fun=lambda a: constraint_norm_ratio(
            a, sigma_z, sigma_builder, *sb_args, **sb_kwargs,
        ),
        lb=-1e-6, ub=np.inf,
    )

    # Optimize (local refinement)
    res = minimize(
        fun=lambda a: objective(
            a, sigma_z, sigma_builder, effective_rank, *sb_args, **sb_kwargs,
        ),
        x0=np.array([x0_alpha]),
        method="SLSQP",
        bounds=bounds,
        constraints=[nlc1, nlc2],
        options=dict(disp=False, maxiter=100, ftol=1e-9),
    )

    if not res.success:
        print(f"Optimization Failed: {res.message}")
        return 1.0

    return float(res.x[0])


# ===== Sample collection & alpha computation =====

def collect_alpha_samples(
    tr_loader: DataLoader,
    norm_stats,
    n_samples: int = 200,
):
    """Collect normalized covariance matrices for alpha optimization."""
    q_mean = norm_stats.q_mean.reshape(1, 12)
    tau_mean = norm_stats.tau_mean.reshape(1, 12)
    q_std = float(norm_stats.q_std)
    qd_std = float(norm_stats.qd_std)
    tau_std = float(norm_stats.tau_std)

    ds = tr_loader.dataset
    if isinstance(ds, ConcatDataset):
        sub_datasets = list(ds.datasets)
    else:
        sub_datasets = [ds]

    # Flat list of (dataset, local_index), shuffled
    all_indices = []
    for sub_ds in sub_datasets:
        for local_i in range(len(sub_ds)):
            all_indices.append((sub_ds, local_i))
    rng = random.Random(42)
    rng.shuffle(all_indices)

    cov_list = []
    n_collected = 0

    pbar = tqdm(total=n_samples, desc="[Collect] alpha samples", leave=True)
    for sub_ds, local_i in all_indices:
        if n_collected >= n_samples:
            break

        base_seq, joint_seq, _ = sub_ds[local_i]

        j = joint_seq
        j_q = (j[..., 0] - q_mean) / q_std
        j_qd = j[..., 1] / qd_std
        j_tau = (j[..., 2] - tau_mean) / tau_std
        z = np.concatenate([j_q, j_qd, j_tau], axis=-1)  # (H, 36)

        cov = np.cov(z.T)
        if cov.shape != (36, 36) or not np.all(np.isfinite(cov)):
            n_collected += 1
            pbar.update(1)
            continue

        cov_list.append(cov.astype(np.float64))
        n_collected += 1
        pbar.update(1)

    pbar.close()

    if len(cov_list) == 0:
        raise RuntimeError("No valid samples collected for alpha optimization")

    sample_bundle = {
        "cov_list": cov_list,
        "n_collected": len(cov_list),
    }
    print(f"[Collect] valid samples = {sample_bundle['n_collected']}")
    return sample_bundle


def compute_optimal_alpha(sample_bundle) -> float:
    """Compute optimal alpha from pre-collected covariance samples."""
    tau_col_list = list(range(24, 36))
    cov_list = sample_bundle["cov_list"]

    alphas = []

    pbar = tqdm(total=len(cov_list), desc="[Alpha] solving", leave=True)
    for cov in cov_list:
        try:
            a = solve_alpha(
                cov,
                build_sigma_tilde,
                col_list=tau_col_list,
                effective_rank=None,
            )
            alphas.append(a)
        except RuntimeError:
            pass

        pbar.update(1)
        pbar.set_postfix(
            feasible=len(alphas),
            mean=f"{np.mean(alphas):.4f}" if alphas else "N/A",
        )
    pbar.close()

    if not alphas:
        print("[Alpha] No feasible alpha found in any sample, using 1.0")
        return 1.0

    optimal = float(np.mean(alphas))
    print(
        f"\n[Alpha] Computed from {len(alphas)}/{len(cov_list)} feasible samples: "
        f"mean={optimal:.4f}  std={np.std(alphas):.4f}  median={np.median(alphas):.4f}"
    )
    return optimal
