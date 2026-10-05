"""Compute the locomotion cost map from raw episode data (episodes/ + index.csv).

Usage::

    python scripts/compute_cost_map.py \
        --data dataset/w_cont_id/2026-02-04_18-30-04_w_cont_id \
        --output results/cost_maps/cost_map_w_cont_id.npz

Output NPZ keys:
    mu_terrain_grid, mu_ctrl_grid, vx_cmd_grid  — parameter axes
    cost_map                                    — (N, M, K) weighted total cost
    metric_<name> / metric_norm_<name>          — raw / normalised per-metric maps
    weights, weight_keys                        — metric weight vector
    metric_mins, metric_maxs                    — pre-normalisation bounds
    episode_count, flipped_count                — per-cell counts
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import numpy as np
from tqdm import tqdm

from lawbc.utils.frame_transforms import is_flipped_from_Rgb
from lawbc.utils.metrics import (
    prepare_episode_data,
    compute_episode_metrics,
    get_metric_names,
)

# Cost-map settings used in the paper.
METRIC_WEIGHTS = {  # normalized to sum to 1
    "vel_track_error": 1.0,
    "cost_of_transport": 1.0,
    "stability_periodic_wobble": 0.2,
}
BETA = 0.5          # stability_periodic_wobble: J = mean(exp(x(t) + beta*(1 - r(t))^2))
TRIM_STEPS = 2000   # initial (stand-up) steps skipped in each episode


# ===== Episode I/O =====

def load_episode_raw(episode_dir: Path) -> Optional[Dict[str, np.ndarray]]:
    """Load raw episode data from ``episodes/<name>/data.h5``."""
    data_h5 = episode_dir / "data.h5"
    if not data_h5.exists():
        return None

    try:
        with h5py.File(data_h5, "r") as f:
            if "data" not in f or "value" not in f["data"]:
                return None

            raw_data = f["data"]["value"][()]
            field_names = raw_data.dtype.names

            known_fields = [
                "q", "qd", "qdd_sim", "u", "u_des", "q_des", "qd_des",
                "f", "feet_pos", "feet_vel",
            ]
            return {k: raw_data[k] for k in known_fields if k in field_names}

    except Exception as e:
        print(f"Error loading {data_h5}: {e}")
        return None


# ===== Index loading =====

def load_index_csv(data_root: Path) -> List[Dict]:
    """Load index.csv and return list of episode info dicts."""
    index_csv = data_root / "index.csv"
    if not index_csv.exists():
        raise FileNotFoundError(f"index.csv not found: {index_csv}")

    episodes = []
    with open(index_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes.append({
                "episode": row["episode"],
                "status": row["status"],
                "mu_terrain": float(row["mu_terrain"]),
                "mu_ctrl": float(row["mu_ctrl"]),
                "vx_cmd": float(row["vx_cmd"]),
                "dt": float(row["dt"]),
                "n_steps": int(row["n_steps"]),
                "path": row["path"],
            })
    return episodes


# ===== Cost map construction =====

def build_cost_map(data_root: str, output_path: str) -> None:
    """Build 3-D cost map from raw episode data and save as NPZ."""
    trim, beta = TRIM_STEPS, BETA
    data_root = Path(data_root).expanduser().resolve()

    print(f"Loading episodes from: {data_root}")

    episodes = load_index_csv(data_root)
    print(f"Total episodes in index.csv: {len(episodes)}")

    episodes = [ep for ep in episodes if ep["status"] == "OK"]
    print(f"Episodes with status='OK': {len(episodes)}")

    metric_names = get_metric_names()
    if len(metric_names) == 0:
        raise ValueError("No metrics registered in METRIC_REGISTRY")
    print(f"Registered metrics ({len(metric_names)}): {metric_names}")
    print(f"stability_periodic_wobble beta = {beta}")

    # Weights
    weight_sum = sum(METRIC_WEIGHTS[m] for m in metric_names)
    metric_weights = {m: METRIC_WEIGHTS[m] / weight_sum for m in metric_names}
    print(f"Metric weights: {metric_weights}")

    # Parameter grid
    mu_terrain_vals = sorted(set(ep["mu_terrain"] for ep in episodes))
    mu_ctrl_vals = sorted(set(ep["mu_ctrl"] for ep in episodes))
    vx_cmd_vals = sorted(set(ep["vx_cmd"] for ep in episodes))

    print("\nParameter grid:")
    print(f"  mu_terrain: {len(mu_terrain_vals)} values "
          f"({mu_terrain_vals[0]:.1f} ~ {mu_terrain_vals[-1]:.1f})")
    print(f"  mu_ctrl: {len(mu_ctrl_vals)} values "
          f"({mu_ctrl_vals[0]:.1f} ~ {mu_ctrl_vals[-1]:.1f})")
    print(f"  vx_cmd: {len(vx_cmd_vals)} values "
          f"({vx_cmd_vals[0]:.1f} ~ {vx_cmd_vals[-1]:.1f})")

    # Group episodes by (mu_terrain, mu_ctrl, vx_cmd)
    param_groups = defaultdict(list)
    for i, ep in enumerate(episodes):
        param_groups[(ep["mu_terrain"], ep["mu_ctrl"], ep["vx_cmd"])].append(i)
    print(f"  Total parameter combinations with data: {len(param_groups)}")

    mu_terrain_to_idx = {v: i for i, v in enumerate(mu_terrain_vals)}
    mu_ctrl_to_idx = {v: i for i, v in enumerate(mu_ctrl_vals)}
    vx_cmd_to_idx = {v: i for i, v in enumerate(vx_cmd_vals)}

    shape = (len(mu_terrain_vals), len(mu_ctrl_vals), len(vx_cmd_vals))
    episode_count = np.zeros(shape, dtype=np.int32)
    flipped_count = np.zeros(shape, dtype=np.int32)
    metric_maps = {m: np.full(shape, np.nan, dtype=np.float32) for m in metric_names}

    # --- Pass 1: compute raw metrics (skip flipped episodes) ---
    print("\nComputing metrics for each parameter combination...")

    total_flipped = 0
    for (mu_t, mu_c, vx), ep_indices in tqdm(param_groups.items(), desc="Processing"):
        i_mu_t = mu_terrain_to_idx[mu_t]
        i_mu_c = mu_ctrl_to_idx[mu_c]
        i_vx = vx_cmd_to_idx[vx]

        all_metrics: List[Dict[str, float]] = []
        n_flipped_here = 0

        for ep_idx in ep_indices:
            ep_info = episodes[ep_idx]
            episode_dir = data_root / ep_info["path"]

            raw_data = load_episode_raw(episode_dir)
            if raw_data is None:
                continue

            episode_data = prepare_episode_data(
                raw_data, trim=trim,
                episode_name=ep_info["episode"],
                episode_path=str(episode_dir),
            )

            if is_flipped_from_Rgb(episode_data["R_gb"]):
                n_flipped_here += 1
                total_flipped += 1
                continue

            metrics = compute_episode_metrics(
                episode_data,
                episode_name=ep_info["episode"],
                vx_cmd=vx,
                dt=ep_info.get("dt", 0.001),
                beta=beta,
            )

            all_metrics.append(metrics)

        flipped_count[i_mu_t, i_mu_c, i_vx] = n_flipped_here

        if len(all_metrics) == 0:
            continue

        for metric_name in metric_names:
            vals = [m[metric_name] for m in all_metrics]
            metric_maps[metric_name][i_mu_t, i_mu_c, i_vx] = float(np.nanmean(vals))

        episode_count[i_mu_t, i_mu_c, i_vx] = len(all_metrics)

    print(f"  Flipped episodes excluded: {total_flipped}")

    # --- Pass 2: min-max normalise each metric across the 3-D grid ---
    print("\nNormalizing metrics (min-max -> [0,1])...")

    # cells where every episode flipped get worst-case cost
    flipped_only_mask = (episode_count == 0) & (flipped_count > 0)
    n_flipped_cells = int(np.sum(flipped_only_mask))
    if n_flipped_cells > 0:
        print(f"  Grid points with only flipped episodes: {n_flipped_cells}")

    metric_norm_maps = {}
    metric_mins = np.full(len(metric_names), np.nan, dtype=np.float32)
    metric_maxs = np.full(len(metric_names), np.nan, dtype=np.float32)

    for k, metric_name in enumerate(metric_names):
        raw = metric_maps[metric_name]
        finite_mask = np.isfinite(raw)

        if not np.any(finite_mask):
            metric_norm_maps[metric_name] = np.full(shape, np.nan, dtype=np.float32)
            print(f"  {metric_name}: no finite values — skipped")
            continue

        vmin = float(np.min(raw[finite_mask]))
        vmax = float(np.max(raw[finite_mask]))
        metric_mins[k] = vmin
        metric_maxs[k] = vmax

        normed = np.full(shape, np.nan, dtype=np.float32)
        if (vmax - vmin) < 1e-12:
            normed[finite_mask] = 0.0
            print(f"  {metric_name}: constant ({vmin:.6g}) -> all 0.0")
        else:
            normed[finite_mask] = (raw[finite_mask] - vmin) / (vmax - vmin)
            print(f"  {metric_name}: min={vmin:.6g}  max={vmax:.6g}")

        if n_flipped_cells > 0:
            worst = float(np.nanmax(normed[finite_mask])) if np.any(finite_mask) else 1.0
            normed[flipped_only_mask] = worst

        metric_norm_maps[metric_name] = normed

    # --- Pass 3: weighted total cost ---
    cost_map = np.full(shape, np.nan, dtype=np.float32)

    for idx in np.ndindex(shape):
        vals = []
        any_nan = False
        for metric_name in metric_names:
            v = metric_norm_maps[metric_name][idx]
            if not np.isfinite(v):
                any_nan = True
                break
            vals.append(metric_weights[metric_name] * v)

        cost_map[idx] = np.nan if any_nan else sum(vals)

    n_valid = int(np.sum(np.isfinite(cost_map)))
    n_total = cost_map.size
    print(f"\nCost map: {n_valid}/{n_total} valid entries ({100 * n_valid / n_total:.1f}%)")

    # --- Save NPZ ---
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving cost map to: {output_path}")

    save_dict = {
        "mu_terrain_grid": np.array(mu_terrain_vals, dtype=np.float32),
        "mu_ctrl_grid": np.array(mu_ctrl_vals, dtype=np.float32),
        "vx_cmd_grid": np.array(vx_cmd_vals, dtype=np.float32),
        "cost_map": cost_map,
        "episode_count": episode_count,
        "flipped_count": flipped_count,
        "weights": np.array([metric_weights[k] for k in metric_names], dtype=np.float32),
        "weight_keys": np.array(metric_names, dtype=object),
        "beta": np.float32(beta),
        "metric_mins": metric_mins,
        "metric_maxs": metric_maxs,
    }
    for metric_name in metric_names:
        save_dict[f"metric_{metric_name}"] = metric_maps[metric_name]
        save_dict[f"metric_norm_{metric_name}"] = metric_norm_maps[metric_name]

    np.savez_compressed(output_path, **save_dict)

    print("Done!")


# ===== CLI =====

def main():
    parser = argparse.ArgumentParser(description="Compute locomotion cost-map from raw episode data")
    parser.add_argument("--data", type=str, required=True,
                        help="Run folder (contains index.csv and episodes/)")
    parser.add_argument("--output", type=str, default="results/cost_maps/cost_map.npz",
                        help="Output NPZ file path")
    args = parser.parse_args()

    build_cost_map(data_root=args.data, output_path=args.output)


if __name__ == "__main__":
    main()
