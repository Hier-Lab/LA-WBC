"""Training loop for terrain friction estimation models."""

from __future__ import annotations

import os
import time
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .dataloader import NormStats
from .util import save_loss_svg


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    name: str,
    epochs: int,
    lr: float,
    min_lr: float,
    weight_decay: float,
    out_dir: Optional[str] = None,
    norm_stats: Optional[NormStats] = None,
    tau_alpha: float = 1.0,
):
    """Train a model with cosine LR schedule and return best-MAE checkpoint."""
    model = model.to(device)

    if norm_stats is not None:
        bm = torch.from_numpy(norm_stats.base_mean).to(device)
        bs = torch.from_numpy(norm_stats.base_std).to(device)
        qm = torch.from_numpy(norm_stats.q_mean).to(device).view(1, 1, 12)
        tm = torch.from_numpy(norm_stats.tau_mean).to(device).view(1, 1, 12)
        q_std = float(norm_stats.q_std)
        qd_std = float(norm_stats.qd_std)
        tau_std = float(norm_stats.tau_std)
    else:
        bm = bs = qm = tm = None
        q_std = qd_std = tau_std = None

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=min_lr)

    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    train_mse_hist: List[float] = []
    test_mse_hist: List[float] = []
    test_mae_hist: List[float] = []

    def _normalize(base, joint):
        if bm is not None:
            base = (base - bm) / bs
            joint_q = (joint[..., 0] - qm) / q_std
            joint_qd = joint[..., 1] / qd_std
            joint_tau = (joint[..., 2] - tm) / tau_std * tau_alpha
            joint = torch.stack([joint_q, joint_qd, joint_tau], dim=-1)
        return base, joint

    @torch.no_grad()
    def _evaluate():
        """Evaluate model on test_loader."""
        model.eval()
        mses, maes = [], []
        pbar = tqdm(test_loader, desc=f"[{name}] eval", leave=False)
        for bi, (base, joint, y) in enumerate(pbar):
            base = base.to(device, non_blocking=True)
            joint = joint.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            base, joint = _normalize(base, joint)

            pred = model(base, joint)
            mse = F.mse_loss(pred, y).item()
            mae = (pred - y).abs().mean().item()
            mses.append(mse)
            maes.append(mae)
            pbar.set_postfix(mse=f"{mse:.6f}", mae=f"{mae:.4f}")

        if not mses:
            return float("nan"), float("nan")
        return float(np.mean(mses)), float(np.mean(maes))

    best_mae = float("inf")
    best_state = None

    for ep in range(1, epochs + 1):
        t_ep0 = time.perf_counter()
        model.train()

        mse_sum, nb = 0.0, 0

        steps = len(train_loader)
        if steps <= 0:
            ns = getattr(getattr(train_loader, "sampler", None), "num_samples", None)
            if ns is None:
                raise RuntimeError("train_loader length is 0 and sampler.num_samples unknown")
            steps = max(1, ns // train_loader.batch_size)

        it = iter(train_loader)

        pbar = tqdm(
            range(steps),
            total=steps,
            desc=f"[{name}] train ep {ep:03d}/{epochs}",
            leave=False,
            mininterval=0.5,
        )

        for step_i in pbar:
            try:
                base, joint, y = next(it)
            except StopIteration:
                break

            base = base.to(device, non_blocking=True)
            joint = joint.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            base, joint = _normalize(base, joint)

            opt.zero_grad(set_to_none=True)

            if use_amp:
                with torch.amp.autocast(device_type="cuda"):
                    pred = model(base, joint)
                    loss = F.mse_loss(pred, y)
                scaler.scale(loss).backward()
            else:
                pred = model(base, joint)
                loss = F.mse_loss(pred, y)
                loss.backward()

            if use_amp:
                scaler.unscale_(opt)
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()

            loss_val = float(loss.detach().item())
            mse_sum += loss_val
            nb += 1

            pbar.set_postfix(
                mse=f"{loss_val:.6f}",
                lr=f"{opt.param_groups[0]['lr']:.2e}",
            )

        sch.step()

        train_mse = mse_sum / max(1, nb)
        test_mse, test_mae = _evaluate()

        if np.isfinite(test_mae) and test_mae < best_mae:
            best_mae = test_mae
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        train_mse_hist.append(float(train_mse))
        test_mse_hist.append(float(test_mse))
        test_mae_hist.append(float(test_mae))

        t_ep1 = time.perf_counter()
        d_time = t_ep1 - t_ep0

        print(
            f"[{name}] epoch {ep:03d}/{epochs} | epoch_time {(d_time // 60):.0f}m {(d_time % 60):.1f}s | "
            f"train_MSE {train_mse:.6f} | test_MSE {test_mse:.6f} | "
            f"lr {opt.param_groups[0]['lr']:.2e}",
            flush=True,
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        save_loss_svg(
            train_mse_hist,
            test_mse_hist,
            os.path.join(out_dir, f"{name}_mse.svg"),
            title=f"{name} train/test MSE",
        )
        save_loss_svg(
            test_mae_hist,
            test_mae_hist,
            os.path.join(out_dir, f"{name}_mae.svg"),
            title=f"{name} test MAE",
        )

    loss_log = {
        "train_mse": train_mse_hist,
        "test_mse": test_mse_hist,
        "test_mae": test_mae_hist,
    }
    return model, loss_log
