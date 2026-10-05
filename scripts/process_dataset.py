"""Convert structured raw telemetry dataset into a single processed_dataset.h5.

Example::

    python scripts/process_dataset.py --src dataset/w_cont_id/2026-02-04_18-30-04_w_cont_id
"""

from __future__ import annotations

import argparse
import datetime
import json
from dataclasses import dataclass
from functools import reduce
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np

from lawbc.utils.frame_transforms import (
    in_body_frame,
    detect_contact,
    get_flipped_timesteps,
)

# Initial steps (stand-up transient) dropped from each episode.
TRIM_STEPS = 2000

# Fields to keep from the raw telemetry structured array.
_KEEP_VALUE_FIELDS = [
    "start_time_nano",
    "q", "qd", "qdd_sim",
    "u", "u_des",
    "q_des", "qd_des",
    "f",
    "feet_pos", "feet_vel",
    "ctrl_curr", "ctrl_des",
]


# ===== I/O helpers =====

def _load_json(path: Optional[Path]) -> Dict[str, Any]:
    """Read a JSON file, returning ``{}`` on any failure."""
    if path is None or not path.is_file():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _find_episodes(src: Path) -> List[Tuple[Path, Optional[Path]]]:
    """Return ``[(data_h5_path, meta_json_path | None), ...]``."""
    data_files = sorted(src.rglob("data.h5"))
    out: List[Tuple[Path, Optional[Path]]] = []
    for dh5 in data_files:
        mjson = dh5.parent / "meta.json"
        out.append((dh5, mjson if mjson.is_file() else None))
    return out


def _read_episode_timeseries(h5_path: Path) -> Dict[str, np.ndarray]:
    """Read the array datasets stored at ``/data/<key>`` of an episode H5."""
    out: Dict[str, np.ndarray] = {}
    with h5py.File(h5_path, "r") as f:
        if "data" not in f:
            return out
        for k, dset in f["data"].items():
            mode = dset.attrs.get("mode", "array")
            if isinstance(mode, bytes):
                mode = mode.decode("utf-8", errors="ignore")
            if mode == "array":
                out[k] = dset[()]
    return out


def _write_string_dataset(g: h5py.Group, name: str, values: List[str]) -> None:
    """Write a list of Python strings as a UTF-8 H5 dataset."""
    dt = h5py.string_dtype(encoding="utf-8")
    g.create_dataset(name, data=np.asarray(values, dtype=object), dtype=dt)


def _get_param_float(params: Dict[str, Any], key: str) -> float:
    """Extract a scalar float from *params*, returning NaN on failure."""
    v = params.get(key, np.nan)
    try:
        return float(v)
    except Exception:
        return float("nan")


# ===== dtype helpers =====

def _build_packed_dtype(
    src: np.dtype,
    fields: List[str],
    extra_descr: List[tuple],
) -> np.dtype:
    """Build a packed structured dtype from *src* keeping only *fields*, plus extras."""
    packed = []
    for name in fields:
        fdt = src.fields[name][0]
        packed.append((name, fdt))
    packed.extend(extra_descr)
    return np.dtype(packed, align=False)


# ===== Episode record =====

@dataclass
class _EpisodeRecord:
    episode_uid: str
    run_id: str
    ep_name: str
    data_h5: str
    meta_json: str
    status: str
    n_steps_raw: int
    n_steps_proc: int
    dt: float
    params: Dict[str, Any]


# ===== Processing =====

def process(
    src: Path,
    trim_steps: int = TRIM_STEPS,
    compression: str = "gzip",
    compression_opts: int = 4,
) -> None:
    """Walk *src* for episode H5 files and write a single processed dataset."""
    src = src.expanduser().resolve()
    out_path = Path(str(src) + "/processed_dataset.h5")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    episodes = _find_episodes(src)
    if len(episodes) == 0:
        raise FileNotFoundError(f"No episodes found under: {src}")

    records: List[_EpisodeRecord] = []

    with h5py.File(out_path, "w") as fout:
        fout.attrs["format"] = "processed_dataset_h5_v1"
        fout.attrs["created_at"] = datetime.datetime.now().isoformat(timespec="seconds")
        fout.attrs["source_root"] = str(src)
        fout.attrs["trim_steps"] = int(trim_steps)

        g_eps = fout.require_group("episodes")
        g_idx = fout.require_group("index")

        for data_h5, meta_json in episodes:
            ep_name = data_h5.parent.name
            run_id = "unknown_run"
            try:
                run_id = data_h5.parent.parent.parent.name
            except Exception:
                pass

            episode_uid = f"{run_id}__{ep_name}"
            meta = _load_json(meta_json)
            status = str(meta.get("status", "OK"))
            params = meta.get("params", {})
            if not isinstance(params, dict):
                params = {}

            ts = _read_episode_timeseries(data_h5)
            if not ts:
                print(f"[SKIP] {episode_uid}: cannot read timeseries: {data_h5}")
                continue

            # Determine raw timestep count from any array
            T_raw = None
            for v in ts.values():
                if isinstance(v, np.ndarray) and v.ndim >= 1:
                    T_raw = int(v.shape[0])
                    break
            if T_raw is None:
                print(f"[SKIP] {episode_uid}: no array-like fields")
                continue

            k = int(trim_steps)
            if k >= T_raw:
                print(f"[SKIP] {episode_uid}: trim_steps({k}) >= T_raw({T_raw})")
                continue

            # Trim time-aligned arrays (shape[0] == T_raw)
            ts_trim: Dict[str, np.ndarray] = {}
            for name, arr in ts.items():
                if not isinstance(arr, np.ndarray) or arr.ndim == 0:
                    continue
                if arr.shape[0] != T_raw:
                    continue
                ts_trim[name] = arr[k:, ...]

            if len(ts_trim) == 0:
                print(f"[SKIP] {episode_uid}: no time-aligned fields after trimming")
                continue

            T_proc = int(next(iter(ts_trim.values())).shape[0])
            dt = float(meta.get("dt", np.nan))

            if "value" not in ts_trim:
                print(
                    f"[SKIP] {episode_uid}: 'value' not found "
                    f"(keys={list(ts_trim.keys())})"
                )
                continue

            val = ts_trim["value"]
            names = val.dtype.names or ()

            keep = [f for f in _KEEP_VALUE_FIELDS if f in names]
            if len(keep) == 0:
                print(f"[SKIP] {episode_uid}: none of _KEEP_VALUE_FIELDS in value")
                continue

            # ── Derive additional signals ─────────────────────────────────

            # Impact flag
            impact_flag = None
            if "f" in names:
                try:
                    contact = detect_contact(val["f"])
                    impact_sets = [
                        np.asarray(contact[leg], dtype=int)
                        for leg in ["FL", "FR", "RL", "RR"]
                    ]
                    impact_sets = [a for a in impact_sets if a.size > 0]
                    if len(impact_sets) == 0:
                        impact = np.asarray([], dtype=int)
                    else:
                        impact = reduce(np.union1d, impact_sets)
                    impact_flag = np.zeros((T_proc,), dtype=np.uint8)
                    if impact.size > 0:
                        impact_flag[impact] = 1
                except Exception as e:
                    print(f"[WARN] {episode_uid}: impact derivation failed: {e}")

            # Body-frame quantities
            bf: Dict[str, np.ndarray] = {}
            flip_mask = None
            if ("q" in names) and ("qd" in names):
                try:
                    bf = in_body_frame(
                        val["q"],
                        val["qd"],
                        foot_pos=(val["feet_pos"] if "feet_pos" in names else None),
                        foot_vel=(val["feet_vel"] if "feet_vel" in names else None),
                    )
                    if "R_gb" in bf:
                        flip_mask = get_flipped_timesteps(
                            bf["R_gb"], up_z_thresh=0.1, min_consecutive=100,
                        )
                        if np.any(flip_mask):
                            n_flipped = int(np.sum(flip_mask))
                            print(f"Robot flipped at {n_flipped}/{T_proc} timesteps!")
                            status = f"{status}|FLIPPED"
                except Exception as e:
                    print(f"[WARN] {episode_uid}: body-frame derivation failed: {e}")

            # Per-timestep ground-truth mu_terrain
            mu_terrain_gt = None
            original_mu = params.get("mu_terrain", np.nan)
            try:
                original_mu = float(original_mu)
            except (ValueError, TypeError):
                original_mu = np.nan

            if not np.isnan(original_mu):
                mu_terrain_gt = np.full(
                    (T_proc,), round(original_mu, 2), dtype=np.float64,
                )
                if flip_mask is not None and np.any(flip_mask):
                    mu_terrain_gt[flip_mask] = 0.0

            # ── Pack derived fields into structured array ─────────────────

            extra_descr: List[tuple] = []
            if impact_flag is not None:
                extra_descr.append(("impact_flag", np.uint8))
            if "vel_body" in bf:
                extra_descr.append(("vel_body", np.float32, (6,)))
            if "foot_pos_body" in bf:
                extra_descr.append(("foot_pos_body", np.float32, (12,)))
            if "foot_vel_body" in bf:
                extra_descr.append(("foot_vel_body", np.float32, (12,)))
            if "R_gb" in bf:
                extra_descr.append(("R_gb", np.float32, (3, 3)))
            if mu_terrain_gt is not None:
                extra_descr.append(("mu_terrain_gt", np.float64))

            if extra_descr:
                new_dtype = _build_packed_dtype(val.dtype, keep, extra_descr)
                val_out = np.empty(val.shape[0], dtype=new_dtype)

                for kf in keep:
                    val_out[kf] = val[kf]
                if impact_flag is not None:
                    val_out["impact_flag"] = impact_flag
                if "vel_body" in bf:
                    val_out["vel_body"] = bf["vel_body"].astype(np.float32)
                if "foot_pos_body" in bf:
                    val_out["foot_pos_body"] = bf["foot_pos_body"].astype(np.float32)
                if "foot_vel_body" in bf:
                    val_out["foot_vel_body"] = bf["foot_vel_body"].astype(np.float32)
                if "R_gb" in bf:
                    val_out["R_gb"] = bf["R_gb"].astype(np.float32)
                if mu_terrain_gt is not None:
                    val_out["mu_terrain_gt"] = mu_terrain_gt
            else:
                val_out = val[keep]

            # ── Write episode to H5 ──────────────────────────────────────

            gep = g_eps.require_group(episode_uid)
            gep.attrs["run_id"] = run_id
            gep.attrs["ep_name"] = ep_name
            gep.attrs["source_data_h5"] = str(data_h5)
            gep.attrs["source_meta_json"] = (
                str(meta_json) if meta_json is not None else ""
            )
            gep.attrs["status"] = status
            gep.attrs["dt"] = dt
            gep.attrs["params_json"] = json.dumps(params, ensure_ascii=False)

            if "value" in gep:
                del gep["value"]
            gep.create_dataset(
                "value",
                data=np.ascontiguousarray(val_out),
                compression=compression,
                compression_opts=compression_opts,
                shuffle=True,
            )

            records.append(_EpisodeRecord(
                episode_uid=episode_uid,
                run_id=run_id,
                ep_name=ep_name,
                data_h5=str(data_h5),
                meta_json=str(meta_json) if meta_json is not None else "",
                status=status,
                n_steps_raw=T_raw,
                n_steps_proc=T_proc,
                dt=dt,
                params=params,
            ))

            print(
                f"[OK] {episode_uid}: "
                f"T_raw={T_raw} -> T_proc={T_proc} (trim={trim_steps})"
            )

        # ── Write index datasets ─────────────────────────────────────────

        _write_string_dataset(g_idx, "episode_uid", [r.episode_uid for r in records])
        _write_string_dataset(g_idx, "run_id", [r.run_id for r in records])
        _write_string_dataset(g_idx, "ep_name", [r.ep_name for r in records])
        _write_string_dataset(g_idx, "data_h5", [r.data_h5 for r in records])
        _write_string_dataset(g_idx, "meta_json", [r.meta_json for r in records])
        _write_string_dataset(g_idx, "status", [r.status for r in records])

        g_idx.create_dataset(
            "n_steps_raw",
            data=np.asarray([r.n_steps_raw for r in records], dtype=np.int32),
        )
        g_idx.create_dataset(
            "n_steps_proc",
            data=np.asarray([r.n_steps_proc for r in records], dtype=np.int32),
        )
        g_idx.create_dataset(
            "dt",
            data=np.asarray([r.dt for r in records], dtype=np.float32),
        )

        # Common params for quick filtering
        param_keys = ["mu_terrain", "mu_ctrl", "vx_cmd", "t_step", "h_step", "t_dwell"]
        for key in param_keys:
            vals = np.round(
                np.asarray(
                    [_get_param_float(r.params, key) for r in records],
                    dtype=np.float64,
                ),
                2,
            )
            g_idx.create_dataset(f"param_{key}", data=vals)


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", type=str, required=True,
                   help="Run folder containing episodes/ (output is written next to it).")
    args = p.parse_args()

    process(src=Path(args.src))


if __name__ == "__main__":
    main()
