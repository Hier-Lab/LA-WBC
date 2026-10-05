"""Locomotion performance metrics used to build the cost map.

All ``compute_metric_*`` functions take a structured ``episode_data`` array
(as produced by :func:`prepare_episode_data`) plus ``**kwargs`` and return a
scalar. Field access is by ``episode_data["<name>"]`` where the generalized
coordinates are laid out as::

    q  = [x, y, z, roll, pitch, yaw, joint_0..joint_11]    # (T, 18)
    qd = [vx, vy, vz, wx, wy, wz, jointvel_0..jointvel_11] # (T, 18)

Registered metrics (see ``METRIC_REGISTRY``)
--------------------------------------------
- ``vel_track_error``            commanded velocity tracking error
- ``cost_of_transport``          cost of transport (CoT)
- ``stability_periodic_wobble``  body-rotation wobble weighted by gait periodicity
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from scipy.spatial.transform import Rotation as R

from .frame_transforms import compute_vel_body


# Go2 total mass (kg), used as the default for cost-of-transport.
DEFAULT_MASS = 15.206408
GRAVITY = 9.81


# ===== Episode data preparation =====

def prepare_episode_data(
    raw_data: Dict[str, np.ndarray],
    trim: int = 2000,
    episode_name: str = "unknown",
    episode_path: str = "unknown",
) -> np.ndarray:
    """Trim, compute body-frame fields, and pack into a structured array.

    Args:
        raw_data: dict of raw telemetry arrays (``q``, ``qd`` required; torque
            ``u_des``/``u`` required; ``f``/``feet_pos``/``feet_vel`` optional).
        trim: number of initial timesteps to drop.
        episode_name / episode_path: only used for error messages.

    Returns:
        Structured ``np.ndarray`` of shape ``(T,)`` with body-frame ``vel_body``
        and ``R_gb`` derived and optional fields packed when present.
    """
    required_fields = ["q", "qd"]
    has_torque = ("u_des" in raw_data) or ("u" in raw_data)
    if not has_torque:
        raise ValueError(
            f"Episode '{episode_name}' ({episode_path}): "
            f"missing torque field. Available: {list(raw_data.keys())}"
        )
    for field in required_fields:
        if field not in raw_data:
            raise ValueError(
                f"Episode '{episode_name}' ({episode_path}): "
                f"missing '{field}'. Available: {list(raw_data.keys())}"
            )

    trimmed = {}
    for key, val in raw_data.items():
        trimmed[key] = val[trim:] if len(val) > trim else val

    vel_body, R_gb = compute_vel_body(trimmed["q"], trimmed["qd"])
    trimmed["vel_body"] = vel_body
    trimmed["R_gb"] = R_gb

    T = len(vel_body)

    dtype_list = [
        ("q", np.float64, (18,)),
        ("qd", np.float64, (18,)),
        ("vel_body", np.float32, (6,)),
        ("R_gb", np.float32, (3, 3)),
    ]
    optional_fields = [
        ("qdd_sim", np.float64, (18,)),
        ("u", np.float64, (12,)),
        ("u_des", np.float64, (12,)),
        ("q_des", np.float64, (12,)),
        ("qd_des", np.float64, (12,)),
        ("f", np.float64, (4,)),
        ("feet_pos", np.float64, (4, 3)),
        ("feet_vel", np.float64, (4, 3)),
    ]
    for field_name, field_dtype, field_shape in optional_fields:
        if field_name in trimmed:
            dtype_list.append((field_name, field_dtype, field_shape))

    episode_data = np.zeros(T, dtype=dtype_list)

    for entry in dtype_list:
        field_name = entry[0]
        target_shape = entry[2] if len(entry) > 2 else ()
        if field_name in trimmed:
            src = trimmed[field_name]
            if isinstance(target_shape, tuple) and len(target_shape) > 1:
                expected = (T,) + target_shape
                if src.shape != expected:
                    src = src.reshape(expected)
            episode_data[field_name] = src

    return episode_data


# ===== Metric functions =====

def _vel_tracking_error(vel_body_slice: np.ndarray, vx_cmd: float) -> float:
    """error_rate = ||mean(measured) - cmd|| / ||cmd||"""
    command = np.array([vx_cmd, 0.0, 0.0], dtype=np.float32)
    mean_measured = np.mean(vel_body_slice[:, :3], axis=0)
    err_norm = np.linalg.norm(mean_measured - command)
    cmd_norm = np.linalg.norm(command)
    if cmd_norm < 1e-6:
        return float(np.linalg.norm(mean_measured))
    return float(err_norm / cmd_norm)


def compute_metric_vel_track_error(episode_data: np.ndarray, **kwargs) -> float:
    """Velocity tracking error over the entire episode."""
    vx_cmd = kwargs.get("vx_cmd", 0.0)
    if "vel_body" not in episode_data.dtype.names:
        raise ValueError("Missing 'vel_body' for vel_track_error metric")
    return _vel_tracking_error(episode_data["vel_body"], vx_cmd)


def compute_metric_cost_of_transport(episode_data: np.ndarray, **kwargs) -> float:
    """CoT = E / (m * g * d)"""
    dt = kwargs.get("dt", 0.001)
    mass = kwargs.get("mass", DEFAULT_MASS)
    g = GRAVITY
    use_3d_vel = kwargs.get("use_3d_vel", True)

    if "vel_body" not in episode_data.dtype.names:
        raise ValueError("Missing 'vel_body' for cost_of_transport metric")
    if "qd" not in episode_data.dtype.names:
        raise ValueError("Missing 'qd' for cost_of_transport metric")

    if "u_des" in episode_data.dtype.names:
        torque = episode_data["u_des"]
    elif "u" in episode_data.dtype.names:
        torque = episode_data["u"]
    else:
        raise ValueError("Missing 'u_des' or 'u' for cost_of_transport metric")

    vel_body = episode_data["vel_body"]
    qd_joints = episode_data["qd"][:, 6:18]

    power = np.clip(np.sum(torque * qd_joints, axis=1), 0.0, None)

    if use_3d_vel:
        speed = np.linalg.norm(vel_body[:, :3], axis=1)
    else:
        speed = np.linalg.norm(vel_body[:, :2], axis=1)

    E = float(np.trapezoid(power, dx=dt))
    d = float(np.trapezoid(speed, dx=dt))

    if d < 1e-6:
        return np.nan

    return float(E / (mass * g * d))


def compute_metric_stability_periodic_wobble_cost(episode_data: np.ndarray, **kwargs) -> float:
    """J = mean( exp( x(t) + beta * (1 - r(t))^2 ))

    x(t) = |delta_theta(t)|,  r(t) = Pearson corr between current and previous gait cycle.
    """
    if "R_gb" not in episode_data.dtype.names:
        raise ValueError("Missing 'R_gb' for stability_periodic_wobble_cost metric")

    dt = kwargs.get("dt", 0.001)
    cycle_sec = kwargs.get("cycle_sec", 0.6)
    beta = kwargs.get("beta", 0.5)
    W = int(round(cycle_sec / dt))

    R_gb = episode_data["R_gb"]
    T = R_gb.shape[0]

    L = T - 1
    t_start = 2 * W - 1
    if L < t_start + 1:
        raise ValueError(
            f"Not enough data for stability_periodic_wobble_cost "
            f"(need >= {t_start + 2} timesteps, got {T})"
        )

    # x(t) = |delta_theta(t)|
    rot = R.from_matrix(R_gb)
    quats = rot.as_quat()

    norms = np.maximum(np.linalg.norm(quats, axis=1, keepdims=True), 1e-12)
    quats = quats / norms

    for k in range(T - 1):
        if np.dot(quats[k], quats[k + 1]) < 0:
            quats[k + 1] = -quats[k + 1]

    q_prev, q_next = quats[:-1], quats[1:]
    q_inv = q_prev.copy()
    q_inv[:, :3] = -q_inv[:, :3]

    px, py, pz, pw = q_inv[:, 0], q_inv[:, 1], q_inv[:, 2], q_inv[:, 3]
    qx, qy, qz, qw = q_next[:, 0], q_next[:, 1], q_next[:, 2], q_next[:, 3]

    dw = pw * qw - px * qx - py * qy - pz * qz
    dx = pw * qx + px * qw + py * qz - pz * qy
    dy = pw * qy - px * qz + py * qw + pz * qx
    dz = pw * qz + px * qy - py * qx + pz * qw

    s = np.sqrt(dx**2 + dy**2 + dz**2)
    x = np.abs(2.0 * np.arctan2(s, np.abs(dw)))  # (L,)

    # r(t): windowed Pearson correlation between current and previous gait cycle
    t_end = L - 1
    n_pts = t_end - t_start + 1

    from numpy.lib.stride_tricks import as_strided

    itemsize = x.strides[0]
    cur_start = t_start - W + 1
    prev_start = t_start - 2 * W + 1
    cur_windows = as_strided(
        x[cur_start:], shape=(n_pts, W), strides=(itemsize, itemsize),
    )
    prev_windows = as_strided(
        x[prev_start:], shape=(n_pts, W), strides=(itemsize, itemsize),
    )

    cur_c = cur_windows - cur_windows.mean(axis=1, keepdims=True)
    prev_c = prev_windows - prev_windows.mean(axis=1, keepdims=True)

    cov = np.sum(cur_c * prev_c, axis=1)
    denom = np.sqrt(np.sum(cur_c**2, axis=1)) * np.sqrt(np.sum(prev_c**2, axis=1))
    r = np.where(denom > 1e-12, cov / denom, 0.0)

    x_slice = x[t_start:t_end + 1]
    exponents = x_slice + beta * (1 - r) ** 2
    return float(np.mean(np.exp(exponents)))


# ===== Metric registry =====

METRIC_REGISTRY = [
    ("vel_track_error", compute_metric_vel_track_error),
    ("cost_of_transport", compute_metric_cost_of_transport),
    ("stability_periodic_wobble", compute_metric_stability_periodic_wobble_cost),
]


def get_metric_names() -> List[str]:
    return [name for name, _ in METRIC_REGISTRY]


def compute_episode_metrics(
    episode_data: np.ndarray,
    episode_name: str = "unknown",
    **kwargs,
) -> Dict[str, float]:
    """Compute all registered metrics for a single episode."""
    if episode_data is None or len(episode_data) == 0:
        raise ValueError(f"Episode '{episode_name}': empty episode_data")

    metrics = {}
    for metric_name, metric_func in METRIC_REGISTRY:
        metrics[metric_name] = metric_func(episode_data, **kwargs)
    return metrics
