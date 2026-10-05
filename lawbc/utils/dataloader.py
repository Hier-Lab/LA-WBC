"""Windowed dataset over processed_dataset.h5 files, normalization, and loaders."""

import glob
import json
import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from tqdm.auto import tqdm

from .util import _decode_str_array, bessel_lowpass_nd


# Read processed .h5 data
def load_index(h5_path: str) -> Dict[str, np.ndarray]:
    """
    Load index data from single processed .h5 file
    """
    with h5py.File(h5_path, "r") as f:
        g = f["index"]
        return {k: g[k][()] for k in g.keys()}


def load_multi_h5_index(h5_files: List[str], vx_cmd_key: str = "param_vx_cmd"):
    """
    Load and merge index data from multiple processed .h5 files.
    Training target is always param_mu_terrain.
    Returns (all_uids, mu_terrain_by_ep, vx_by_ep, mu_ctrl_by_ep, h5_by_ep)
    """
    all_uids = []
    mu_terrain_by_ep = {}
    vx_by_ep = {}
    mu_ctrl_by_ep = {}
    h5_by_ep = {}

    for h5_path in h5_files:
        dirname = os.path.basename(os.path.dirname(h5_path))
        prefix = f"{dirname}::"

        idx = load_index(h5_path)
        uids_raw = _decode_str_array(idx["episode_uid"])

        mu_terrain_all = np.round(np.asarray(idx["param_mu_terrain"], dtype=np.float64), 2)
        mu_ctrl_all = np.round(np.asarray(idx["param_mu_ctrl"], dtype=np.float64), 2)
        vx_all = None
        if vx_cmd_key in idx:
            vx_all = np.asarray(idx[vx_cmd_key], dtype=np.float32)

        for i, uid_raw in enumerate(uids_raw):
            uid = prefix + uid_raw
            all_uids.append(uid)
            mu_terrain_by_ep[uid] = float(mu_terrain_all[i])
            mu_ctrl_by_ep[uid] = float(mu_ctrl_all[i])
            h5_by_ep[uid] = h5_path
            if vx_all is not None:
                vx_by_ep[uid] = float(vx_all[i])

        print(f"  Loaded: {dirname} -> {len(uids_raw)} episodes")
    return all_uids, mu_terrain_by_ep, vx_by_ep, mu_ctrl_by_ep, h5_by_ep


def discover_h5_files(data_dir: str,
                      filename: str = "processed_dataset.h5") -> List[str]:
    """
    Recursively find all {filename}.h5 files under {data_dir}
    Returns the list of absolute paths to found files
    """
    pattern = os.path.join(data_dir, "**", filename)
    h5_files = glob.glob(pattern, recursive=True)
    h5_files = sorted([os.path.abspath(f) for f in h5_files])
    return h5_files


def print_h5_file_summary(h5_files: List[str]):
    """
    Print summary of H5 files that will be used for training
    """
    print("\n" + "=" * 70)
    print("DATA FILES TO BE USED FOR TRAINING")
    print("=" * 70)
    print(f"Total: {len(h5_files)} file(s)\n")
    for i, path in enumerate(h5_files, 1):
        dirname = os.path.basename(os.path.dirname(path))
        filename = os.path.basename(path)
        print(f"  [{i:3d}] {dirname}/{filename}")
        print(f"        -> {path}")
    print("=" * 70 + "\n")


# Dataloader
@dataclass
class EpisodeCache:
    vel_body: np.ndarray                 # (T,6)  float32 [vx, vy, vz, roll_rate, pitch_rate, yaw_rate]
    q: np.ndarray                        # (T,18) float32
    qd: np.ndarray                       # (T,18) float32
    tau: np.ndarray                      # (T,12) float32
    mu_terrain_gt: Optional[np.ndarray]  # (T,)   float32
    T: int                               # total timesteps

class H5WindowDataset(Dataset):
    """
    Generate window samples from processed_dataset.h5

    return:
      base_seq  (H,6)    [vx_body, vy_body, vrz_body, vx_cmd, vy_cmd, vrz_cmd]
      joint_seq (H,12,3) per joint [q, qd, tau_cmd]
      y         (,)      mu_terrain scalar
    """
    def __init__(
        self,
        h5_path: str,
        episode_uids: List[str],
        vx_cmd_by_episode: Optional[Dict[str, float]] = None,
        horizon_size: int = 120,
        stride: int = 5,
        lpf_cutoff: float = 30.0,
    ):

        # index info
        self.h5_path            = str(h5_path)
        self.episode_uids       = list(episode_uids)
        self.vx_cmd_by_episode  = dict(vx_cmd_by_episode or {})

        # data sampling params
        self.H           = int(horizon_size)
        self.stride      = int(stride)
        self.raw_span = self.H * self.stride
        self.lpf_cutoff  = lpf_cutoff
        assert self.H > 0 and self.stride > 0

        self.ep_cache: Dict[str, EpisodeCache] = {}
        self._preload_all_episodes()
        self.samples: List[Tuple[str, int]] = []  # (ep_uid, start)
        self._build_samples()

        if len(self.samples) == 0:
            raise RuntimeError("No samples built.")

    def _preload_all_episodes(self):
        """Preload all episodes into RAM to reduce training time."""
        print("pre-loading episodes ...")
        t0 = time.perf_counter()
        with h5py.File(self.h5_path, "r") as f:
            for ep_uid in tqdm(self.episode_uids, desc="[Preload] episodes", leave=True):
                dset = f[f"episodes/{ep_uid}/value"]
                names = dset.dtype.names
                T = int(dset.shape[0])

                vel = np.asarray(dset["vel_body"][()], dtype=np.float32)
                q   = np.asarray(dset["q"][()], dtype=np.float32)
                qd  = np.asarray(dset["qd"][()], dtype=np.float32)

                tau_key = "u_des" if ("u_des" in names) else "u"
                tau = np.asarray(dset[tau_key][()], dtype=np.float32)

                if float(self.lpf_cutoff) > 0.0:
                    tau = bessel_lowpass_nd(tau, fs=1000, cutoff_hz=self.lpf_cutoff,
                                           order=2, axis=0).astype(np.float32)

                # Load timestep-level mu_terrain ground truth
                mu_terrain_gt = None
                if "mu_terrain_gt" in names:
                    mu_terrain_gt = np.round(np.asarray(dset["mu_terrain_gt"][()], dtype=np.float64), 2).astype(np.float32)

                self.ep_cache[ep_uid] = EpisodeCache(
                    vel_body=vel, q=q, qd=qd, tau=tau,
                    mu_terrain_gt=mu_terrain_gt, T=T
                )
        dt = time.perf_counter() - t0

        # memory usage log
        total_bytes = 0
        for v in self.ep_cache.values():
            total_bytes += v.vel_body.nbytes + v.q.nbytes + v.qd.nbytes + v.tau.nbytes
        print(f"[Preload] done in {dt:.2f}s | cached_eps={len(self.ep_cache)} | approx_mem={total_bytes/1e9:.2f} GB")

    def _build_samples(self):
        """Build sliding-window start indices for each episode."""
        raw_span = self.raw_span

        for ep_uid in tqdm(self.episode_uids, desc="[Build samples] episodes", leave=False):

            ep = self.ep_cache[ep_uid]
            T = ep.T

            if T <= raw_span + 1:
                continue

            step = max(1, self.stride)
            for s in range(0, T - raw_span + 1, step):
                self.samples.append((ep_uid, int(s)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i: int):
        ep_uid, start = self.samples[i]
        t0 = int(start)
        t1 = t0 + self.raw_span

        vx_cmd  = float(self.vx_cmd_by_episode.get(ep_uid, 0.0))

        ep = self.ep_cache[ep_uid]
        if t1 > ep.T:
            raise IndexError(f"{ep_uid}: window out of range t1={t1} > T={ep.T}")

        vb  = ep.vel_body[t0:t1:self.stride]
        q   = ep.q[t0:t1:self.stride]
        qd  = ep.qd[t0:t1:self.stride]
        tau = ep.tau[t0:t1:self.stride]
        y   = float(ep.mu_terrain_gt[min(t1 - self.stride, ep.T - 1)])

        # base_seq: (H, 6)
        vx_body, vy_body, vrz_body = vb[:, 0], vb[:, 1], vb[:, 5]
        H = vb.shape[0]
        base_seq = np.stack(
            [vx_body, vy_body, vrz_body,
            np.full((H,), vx_cmd, dtype=np.float32),
            np.zeros((H,), dtype=np.float32),
            np.zeros((H,), dtype=np.float32)], axis=-1
        ).astype(np.float32)

        qj  = q[:, 6:18]
        qdj = qd[:, 6:18]
        joint_seq = np.stack([qj, qdj, tau], axis=-1).astype(np.float32)

        return (base_seq,
                joint_seq,
                np.float32(y),
                )

def collate_batch(batch):
    """Helper function to generate batch sample."""
    base_np  = np.stack([b[0] for b in batch], axis=0).astype(np.float32, copy=False)   # (B,H,6)
    joint_np = np.stack([b[1] for b in batch], axis=0).astype(np.float32, copy=False)   # (B,H,12,3)
    y_np     = np.asarray([b[2] for b in batch], dtype=np.float32)                      # (B,)

    base  = torch.from_numpy(base_np)
    joint = torch.from_numpy(joint_np)
    y     = torch.from_numpy(y_np)
    return base, joint, y


# Normalization
@dataclass
class NormStats:
    base_mean:    np.ndarray   # (6,)
    base_std:     np.ndarray   # (6,)

    q_mean:       np.ndarray   # (12,)  standing_q
    q_std:        float        # mean std across 12 joints

    qd_mean:      np.ndarray   # (12,)  zeros
    qd_std:       float        # mean std across 12 joints

    tau_mean:     np.ndarray   # (12,)  standing_tau_cmd
    tau_std:      float        # mean std across 12 joints

    def to_jsonable(self):
        return {
            "base_mean": self.base_mean.tolist(),
            "base_std":  self.base_std.tolist(),
            "q_mean":    self.q_mean.tolist(),
            "q_std":     float(self.q_std),
            "qd_mean":   self.qd_mean.tolist(),
            "qd_std":    float(self.qd_std),
            "tau_mean":  self.tau_mean.tolist(),
            "tau_std":   float(self.tau_std),
        }


@torch.no_grad()
def compute_norm_stats_streaming(
    ds,  # H5WindowDataset or ConcatDataset
    seed: int,
) -> NormStats:

    # ConcatDataset - collect stats from all sub-datasets
    if isinstance(ds, ConcatDataset):
        print(f"[Norm] ConcatDataset detected with {len(ds.datasets)} sub-datasets")
        largest_ds = max(ds.datasets, key=len)
        print(f"[Norm] Using largest sub-dataset ({len(largest_ds)} samples) for norm stats")
        return compute_norm_stats_streaming(largest_ds, seed)

    # Standing configs for normalization
    # Nominal Go2 standing pose / torque, measured in MuJoCo under the MPAC stand controller
    standing_q = np.array(
        [
            0.0309353, 0.967779, -1.7404,
            -0.0182785, 0.889244, -1.75452,
            -0.0250176, 0.940715, -1.78302,
            0.0387742, 0.916887, -1.78998,
        ],
        dtype=np.float32,
    )  # (12,)

    standing_tau_cmd = np.array(
        [
            -2.6261, -0.140439, 5.7334,
            2.58988, -0.142835, 5.73781,
            -2.89276, 0.371219, 5.67261,
            2.90343, 0.374715, 5.68071,
        ],
        dtype=np.float32,
    )  # (12,)

    # group starts by episode
    ep_to_starts = defaultdict(list)
    for i in range(len(ds)):
        ep_uid, start = ds.samples[i]
        ep_to_starts[ep_uid].append(int(start))

    # accumulators
    base_sum = np.zeros((6,), dtype=np.float64)
    base_sumsq = np.zeros((6,), dtype=np.float64)
    base_cnt = 0

    q_sumsq = np.zeros((12,), dtype=np.float64)
    q_cnt = 0

    qd_sumsq = np.zeros((12,), dtype=np.float64)
    qd_cnt = 0

    tau_sumsq = np.zeros((12,), dtype=np.float64)
    tau_cnt = 0

    raw_span = ds.raw_span
    stride = ds.stride
    H = ds.H

    # cache-only requirement
    if not ds.ep_cache:
        raise RuntimeError("Check if preload was successful. 'ep_cache' is empty.")

    # streaming over windows
    for ep_uid in tqdm(list(ep_to_starts.keys()), desc="[Norm] episodes", leave=True):
        ep = ds.ep_cache[ep_uid]
        T = ep.T

        vx_cmd = float(ds.vx_cmd_by_episode.get(ep_uid, 0.0))

        vel_body_full = ep.vel_body  # (T,6)
        q_full = ep.q               # (T,18)
        qd_full = ep.qd             # (T,18)
        tau_full = ep.tau           # (T,12)

        for t0 in ep_to_starts[ep_uid]:
            t1 = t0 + raw_span
            if t1 > T:
                continue

            vb = vel_body_full[t0:t1:stride]  # (H,6)
            q = q_full[t0:t1:stride]          # (H,18)
            qd = qd_full[t0:t1:stride]        # (H,18)
            tau = tau_full[t0:t1:stride]      # (H,12)

            if vb.shape[0] != H:
                continue

            # base_seq exactly like ds.__getitem__
            vx_body, vy_body, vrz_body = vb[:, 0], vb[:, 1], vb[:, 5]
            base_seq = np.stack(
                [
                    vx_body, vy_body, vrz_body,
                    np.full((H,), vx_cmd, dtype=np.float32),
                    np.zeros((H,), dtype=np.float32),
                    np.zeros((H,), dtype=np.float32),
                ],
                axis=-1,
            ).astype(np.float32)

            b2 = base_seq.reshape(-1, 6)  # (H,6)
            base_sum += b2.sum(axis=0)
            base_sumsq += (b2 ** 2).sum(axis=0)
            base_cnt += b2.shape[0]

            # use ONLY joints 6:18 for q/qd (exclude base-related first 6 dims)
            qj = q[:, 6:18].astype(np.float32)     # (H,12)
            qdj = qd[:, 6:18].astype(np.float32)   # (H,12)
            tauj = tau.astype(np.float32)          # (H,12)

            q_sumsq += (qj ** 2).sum(axis=0)
            q_cnt += qj.shape[0]

            qd_sumsq += (qdj ** 2).sum(axis=0)
            qd_cnt += qdj.shape[0]

            tau_sumsq += (tauj ** 2).sum(axis=0)
            tau_cnt += tauj.shape[0]

    # base: mean=0, std=max(per-channel std)
    base_var = base_sumsq / max(1, base_cnt) - (base_sum / max(1, base_cnt)) ** 2
    base_std_per_ch = np.sqrt(np.maximum(base_var, 1e-12)).astype(np.float32)
    base_std_scalar = float(np.max(base_std_per_ch))
    base_mean = np.zeros((6,), dtype=np.float32)
    base_std = np.ones((6,), dtype=np.float32) * base_std_scalar

    # q: mean=standing_q, std=mean(per-joint std)
    q_var = q_sumsq / max(1, q_cnt)
    q_std_joint = np.sqrt(np.maximum(q_var, 1e-12)).astype(np.float32)
    q_std_scalar = float(np.mean(q_std_joint))
    if q_std_scalar < 1e-6:
        q_std_scalar = 1.0
    q_mean = standing_q.copy()

    # qd: mean=0, std=mean(per-joint std)
    qd_var = qd_sumsq / max(1, qd_cnt)
    qd_std_joint = np.sqrt(np.maximum(qd_var, 1e-12)).astype(np.float32)
    qd_std_scalar = float(np.mean(qd_std_joint))
    if qd_std_scalar < 1e-6:
        qd_std_scalar = 1.0
    qd_mean = np.zeros((12,), dtype=np.float32)

    # tau: mean=standing_tau_cmd, std=mean(per-joint std)
    tau_var = tau_sumsq / max(1, tau_cnt)
    tau_std_joint = np.sqrt(np.maximum(tau_var, 1e-12)).astype(np.float32)
    tau_std_scalar = float(np.mean(tau_std_joint))
    if tau_std_scalar < 1e-6:
        tau_std_scalar = 1.0
    tau_mean = standing_tau_cmd.copy()

    return NormStats(
        base_mean=base_mean,
        base_std=base_std,
        q_mean=q_mean,
        q_std=q_std_scalar,
        qd_mean=qd_mean,
        qd_std=qd_std_scalar,
        tau_mean=tau_mean,
        tau_std=tau_std_scalar,
    )


# ===== Episode splitting =====

def split_episodes(
    uids: List[str], train_ratio: float, seed: int,
) -> Tuple[List[str], List[str]]:
    """Randomly split episode UIDs into train/test sets."""
    idxs = list(range(len(uids)))
    rng = random.Random(seed)
    rng.shuffle(idxs)
    ntr = max(1, int(len(idxs) * train_ratio))
    train = [uids[i] for i in idxs[:ntr]]
    test = [uids[i] for i in idxs[ntr:]]
    return train, test


# ===== Data preparation =====

def prepare_data(args, device) -> Tuple[DataLoader, DataLoader, NormStats]:
    """Load H5 data, split episodes, build datasets, and create DataLoaders."""

    def worker_init_fn(worker_id):
        worker_seed = args.seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    # Load episode index from H5 files
    h5_files = discover_h5_files(args.data_dir)
    if len(h5_files) == 0:
        raise FileNotFoundError(f"No processed_dataset.h5 found under {args.data_dir}")
    print_h5_file_summary(h5_files)

    print("Loading index data from all H5 files...")
    uids, mu_terrain_by_ep, vx_by_ep, mu_ctrl_by_ep, h5_by_ep = \
        load_multi_h5_index(h5_files)

    unique_mu_terrain = sorted(set(mu_terrain_by_ep[uid] for uid in uids))
    unique_mu_ctrl    = sorted(set(mu_ctrl_by_ep[uid]    for uid in uids))
    mu_values = [mu_terrain_by_ep[uid] for uid in uids]
    print(f"  unique mu_terrain: {unique_mu_terrain}")
    print(f"  unique mu_ctrl: {unique_mu_ctrl}")
    print(f"[ALL] episodes={len(uids)} label=mu_terrain "
          f"range=[{min(mu_values):.3f},{max(mu_values):.3f}]")

    # Episode-level train/test split
    train_uids, test_uids = split_episodes(uids, args.train_ratio, args.seed)
    print(f"[SPLIT:EPISODES] train={len(train_uids)} test={len(test_uids)}")

    # Build H5WindowDataset (multi-H5 aware)
    def _build_datasets(episode_uids: List[str]):
        uids_by_h5: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        for uid in episode_uids:
            h5_path = h5_by_ep[uid]
            original_uid = uid.split("::", 1)[1] if "::" in uid else uid
            uids_by_h5[h5_path].append((uid, original_uid))

        datasets = []
        for h5_path, uid_pairs in uids_by_h5.items():
            original_uids = [p[1] for p in uid_pairs]
            vx_local      = {orig: vx_by_ep.get(pref, 0.0)      for pref, orig in uid_pairs}

            dirname = os.path.basename(os.path.dirname(h5_path))
            print(f"  [{dirname}] {len(original_uids)} episodes")

            ds = H5WindowDataset(
                h5_path=h5_path,
                episode_uids=original_uids,
                vx_cmd_by_episode=vx_local,
                horizon_size=args.horizon,
                stride=args.stride,
            )
            datasets.append(ds)
        return datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)

    print("\nBuilding TRAIN dataset...")
    tr = _build_datasets(train_uids)
    print(f"  -> {len(tr)} training samples total")

    print("\nBuilding TEST dataset...")
    te = _build_datasets(test_uids)
    print(f"  -> {len(te)} test samples total")

    # Compute normalization stats (train set only)
    stats = compute_norm_stats_streaming(tr, args.seed)
    with open(os.path.join(args.out_dir, "norm_stats.json"), "w") as f:
        json.dump(stats.to_jsonable(), f, indent=2)
    print("[Norm] saved norm_stats.json\n")

    # DataLoaders
    use_mp = (args.num_workers and args.num_workers > 0)
    pin = (device.type == "cuda")

    g = torch.Generator()
    g.manual_seed(args.seed)

    tr_loader = DataLoader(
        tr, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=(4 if args.num_workers > 0 else None),
        pin_memory=pin, collate_fn=collate_batch,
        generator=g, drop_last=True, worker_init_fn=worker_init_fn,
    )
    te_loader = DataLoader(
        te, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=use_mp,
        prefetch_factor=(4 if use_mp else None),
        pin_memory=pin, collate_fn=collate_batch,
        drop_last=False, worker_init_fn=worker_init_fn,
    )

    return tr_loader, te_loader, stats
