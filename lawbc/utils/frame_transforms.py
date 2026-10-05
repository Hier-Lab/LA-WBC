"""Frame transformation and contact detection utilities."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R


def _as_np(x) -> np.ndarray:
    """Ensure *x* is a numpy array."""
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)


# ===== Kinematics =====

def euler_rates_to_body_omega(
    roll: np.ndarray,
    pitch: np.ndarray,
    yaw: np.ndarray,
    roll_rate: np.ndarray,
    pitch_rate: np.ndarray,
    yaw_rate: np.ndarray,
) -> np.ndarray:
    """Convert Euler angle rates to angular velocity in the body frame.

    Assumes XYZ intrinsic convention: ``R_gb = Rx(roll) * Ry(pitch) * Rz(yaw)``.

    Implements ``omega_body = E_b(pitch, yaw) @ [roll_rate, pitch_rate, yaw_rate]^T``
    where::

        E_b = [[ cos(pitch)*cos(yaw),   sin(yaw),  0 ],
               [-cos(pitch)*sin(yaw),   cos(yaw),  0 ],
               [ sin(pitch),            0,         1 ]]
    """
    roll = np.asarray(roll, dtype=np.float32)
    pitch = np.asarray(pitch, dtype=np.float32)
    yaw = np.asarray(yaw, dtype=np.float32)
    roll_rate = np.asarray(roll_rate, dtype=np.float32)
    pitch_rate = np.asarray(pitch_rate, dtype=np.float32)
    yaw_rate = np.asarray(yaw_rate, dtype=np.float32)

    sinp, cosp = np.sin(pitch), np.cos(pitch)
    siny, cosy = np.sin(yaw), np.cos(yaw)

    wx = cosp * cosy * roll_rate + siny * pitch_rate
    wy = -cosp * siny * roll_rate + cosy * pitch_rate
    wz = sinp * roll_rate + yaw_rate

    return np.stack([wx, wy, wz], axis=-1)


# ===== Body-frame velocity =====

def in_body_frame(
    q: np.ndarray,
    qd: np.ndarray,
    foot_pos: Optional[np.ndarray] = None,
    foot_vel: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Transform world-frame quantities into the base/body frame.

    Args:
        q:  (T, 18+) generalized positions  [x,y,z, roll,pitch,yaw, joint...]
        qd: (T, 18+) generalized velocities [vx,vy,vz, roll_rate,pitch_rate,yaw_rate, ...]
        foot_pos: (T, 12) foot positions in world frame (FL,FR,RL,RR x 3)
        foot_vel: (T, 12) foot velocities in world frame

    Returns:
        Dict with keys:
          R_gb          (T,3,3)  body-to-world rotation matrices
          vel_body      (T,6)    [linear_vel(3), omega_body(3)] in body frame
          foot_pos_body (T,12)   if *foot_pos* provided
          foot_vel_body (T,12)   if *foot_vel* provided (requires *foot_pos* too)
    """
    q = _as_np(q)
    qd = _as_np(qd)
    if q.ndim != 2 or q.shape[1] < 6 or qd.ndim != 2 or qd.shape[1] < 6:
        return {}

    T = q.shape[0]

    R_gb = R.from_euler("xyz", q[:, 3:6]).as_matrix().astype(np.float32)  # (T,3,3)
    Rt = np.transpose(R_gb, (0, 2, 1))  # world-to-body

    v_base_world = qd[:, 0:3].astype(np.float32)
    omega_body = euler_rates_to_body_omega(
        q[:, 3], q[:, 4], q[:, 5],
        qd[:, 3], qd[:, 4], qd[:, 5],
    )

    vel_body = np.zeros((T, 6), dtype=np.float32)
    vel_body[:, 0:3] = np.einsum("tij,tj->ti", Rt, v_base_world)
    vel_body[:, 3:6] = omega_body

    out: Dict[str, np.ndarray] = {
        "R_gb": R_gb,
        "vel_body": vel_body,
    }

    # Foot positions in body frame
    foot_pos_body = None
    if foot_pos is not None:
        fp = _as_np(foot_pos).astype(np.float32)
        if fp.ndim == 2 and fp.shape[0] == T and fp.shape[1] >= 12:
            base_xyz = q[:, 0:3].astype(np.float32)
            foot_pos_body = np.zeros((T, 12), dtype=np.float32)
            for k in range(4):
                seg = slice(3 * k, 3 * k + 3)
                r_world = fp[:, seg] - base_xyz
                foot_pos_body[:, seg] = np.einsum("tij,tj->ti", Rt, r_world)
            out["foot_pos_body"] = foot_pos_body

    # Foot velocities in body frame
    if foot_vel is not None and foot_pos_body is not None:
        fv = _as_np(foot_vel).astype(np.float32)
        if fv.ndim == 2 and fv.shape[0] == T and fv.shape[1] >= 12:
            foot_vel_body = np.zeros((T, 12), dtype=np.float32)
            for k in range(4):
                seg = slice(3 * k, 3 * k + 3)
                v_rel_world = fv[:, seg] - v_base_world
                v_rel_body_rot = np.einsum("tij,tj->ti", Rt, v_rel_world)
                r_body = foot_pos_body[:, seg]
                foot_vel_body[:, seg] = v_rel_body_rot - np.cross(omega_body, r_body)
            out["foot_vel_body"] = foot_vel_body

    return out


def compute_vel_body(
    q: np.ndarray,
    qd: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute body-frame velocity and rotation matrix.

    Returns:
        (vel_body (T,6), R_gb (T,3,3))
    """
    bf = in_body_frame(q, qd)
    if not bf:
        raise ValueError("q and qd must be (T, 6+) arrays")
    return bf["vel_body"], bf["R_gb"]


# ===== Contact / impact detection =====

def detect_contact(
    force_data: np.ndarray,
    zero_tolerance: int = 50,
    force_threshold: float = 5.0,
) -> Dict[str, List[int]]:
    """Detect foot touchdown events from ground reaction forces.

    Args:
        force_data: (T, 4) GRF magnitudes for legs [FL, FR, RL, RR]
        zero_tolerance: consecutive sub-threshold samples before declaring swing
        force_threshold: minimum GRF to consider as contact

    Returns:
        Dict mapping leg name to list of impact-start indices.
    """
    force_data = _as_np(force_data)
    if force_data.ndim != 2 or force_data.shape[1] < 4:
        return {"FL": [], "FR": [], "RL": [], "RR": []}

    leg_names = ["FL", "FR", "RL", "RR"]
    contact_moments: Dict[str, List[int]] = {n: [] for n in leg_names}

    for leg in range(4):
        col = force_data[:, leg]
        in_contact = False
        zero_count = 0
        start_idx: Optional[int] = None
        impacts: List[int] = []

        for i, v in enumerate(col):
            if v > force_threshold:
                if not in_contact:
                    in_contact = True
                    start_idx = int(i)
                zero_count = 0
            else:
                if in_contact:
                    zero_count += 1
                    if zero_count >= zero_tolerance:
                        if start_idx is not None:
                            impacts.append(int(start_idx))
                        in_contact = False
                        zero_count = 0
                        start_idx = None

        if in_contact and start_idx is not None:
            impacts.append(int(start_idx))

        contact_moments[leg_names[leg]] = impacts

    return contact_moments


# ===== Flip detection =====

def is_flipped_from_Rgb(
    R_gb: np.ndarray,
    up_z_thresh: float = 0.0,
    min_consecutive: int = 200,
) -> bool:
    """Check whether the robot has flipped over.

    Args:
        R_gb: (T,3,3) body-to-world rotation matrices
        up_z_thresh: threshold on world-z component of body up-vector
        min_consecutive: minimum consecutive flipped steps to declare flipped
    """
    if R_gb is None:
        return False
    R_gb = np.asarray(R_gb)
    if R_gb.ndim != 3 or R_gb.shape[1:] != (3, 3):
        return False

    up_z = R_gb[:, 2, 2]
    bad = up_z < float(up_z_thresh)

    run, best = 0, 0
    for b in bad:
        if b:
            run += 1
            if run > best:
                best = run
        else:
            run = 0
    return best >= int(min_consecutive)


def get_flipped_timesteps(
    R_gb: np.ndarray,
    up_z_thresh: float = 0.0,
    min_consecutive: int = 100,
) -> np.ndarray:
    """Return per-timestep boolean mask indicating flipped states.

    Only marks a contiguous run as flipped if it lasts >= *min_consecutive* steps.
    """
    if R_gb is None:
        return np.array([], dtype=bool)

    R_gb = np.asarray(R_gb)
    if R_gb.ndim != 3 or R_gb.shape[1:] != (3, 3):
        return np.zeros((R_gb.shape[0],), dtype=bool)

    T = R_gb.shape[0]
    up_z = R_gb[:, 2, 2]
    bad = up_z < float(up_z_thresh)
    flip_mask = np.zeros((T,), dtype=bool)

    i = 0
    while i < T:
        if bad[i]:
            j = i
            while j < T and bad[j]:
                j += 1
            if (j - i) >= min_consecutive:
                flip_mask[i:j] = True
            i = j
        else:
            i += 1

    return flip_mask
