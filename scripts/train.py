"""Train and export neural network models for terrain friction estimation.

Example::

    python scripts/train.py --data_dir dataset/w_cont_id --out_dir w_cont_id --gpu_id 0

Outputs are written to ``results/<out_dir>/``.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch
import yaml

from lawbc.models.model import FCNNBaseline, GRUBaseline, RecGNN, TransformerEstimator
from lawbc.utils.dataloader import prepare_data
from lawbc.utils.scaling_factor_optimizer import collect_alpha_samples, compute_optimal_alpha
from lawbc.utils.trainer import train_model
from lawbc.utils.util import (
    export_torchscript, get_device, plot_alpha_comparison,
    save_pred_csv, save_pred_plots, set_seed,
)


# Training settings used in the paper (fixed for reproducibility).
PAPER_CONFIG = dict(
    seed=42,
    horizon=120,              # window length (inference ticks)
    stride=5,                 # 1 kHz -> 200 Hz
    train_ratio=0.8,          # episode-level train/test split
    epochs=30,
    batch_size=256,
    lr=1e-4,
    min_lr=1e-6,              # cosine schedule floor
    weight_decay=1e-4,
    alpha_sample_ratio=0.01,  # fraction of training windows used to solve for alpha
    alpha_max_samples=30000,
)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", type=str, required=True,
                   help="Directory searched recursively for processed_dataset.h5")
    p.add_argument("--out_dir", type=str, required=True,
                   help="Run name; outputs go to results/<out_dir>/")
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    args = argparse.Namespace(**PAPER_CONFIG, **vars(p.parse_args()))

    set_seed(args.seed)
    device = get_device(args.gpu_id)
    print("\nDevice:", device)
    if device.type == "cuda":
        print("CUDA:", torch.cuda.get_device_name(args.gpu_id))

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    args.out_dir = os.path.join(repo_root, "results", args.out_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "train_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # Load model configs
    cfg_dir = os.path.join(repo_root, "lawbc", "models")
    with open(os.path.join(cfg_dir, "fcnn_cfg.yaml")) as f:
        fcnn_cfg = yaml.safe_load(f)
    with open(os.path.join(cfg_dir, "gru_cfg.yaml")) as f:
        gru_cfg = yaml.safe_load(f)
    with open(os.path.join(cfg_dir, "transformer_cfg.yaml")) as f:
        transformer_cfg = yaml.safe_load(f)
    with open(os.path.join(cfg_dir, "recgnn_cfg.yaml")) as f:
        recgnn_cfg = yaml.safe_load(f)

    # ===== Timing / measurement bookkeeping =====
    t_pipeline0 = time.perf_counter()
    timings: Dict[str, float] = {}
    model_times: Dict[str, float] = {}

    # Load data
    t_data0 = time.perf_counter()
    tr_loader, te_loader, stats = prepare_data(args, device)
    timings["data_prep_sec"] = time.perf_counter() - t_data0

    n_train_samples = len(tr_loader.dataset)
    n_test_samples = len(te_loader.dataset)
    print("\n" + "=" * 60)
    print("DATASET SIZE")
    print("=" * 60)
    print(f"  train windows (samples used for training): {n_train_samples}")
    print(f"  test  windows                            : {n_test_samples}")
    print(f"  data preparation time                    : {timings['data_prep_sec']:.1f}s")
    print("=" * 60)

    # Normalized example tensors for TorchScript export
    ex_base, ex_joint, _ = next(iter(te_loader))
    ex_base = ex_base[:1].to(device)
    ex_joint = ex_joint[:1].to(device)
    bm_ts = torch.from_numpy(stats.base_mean).to(device)
    bs_ts = torch.from_numpy(stats.base_std).to(device)
    ex_base = (ex_base - bm_ts) / bs_ts
    qm_ts = torch.from_numpy(stats.q_mean).to(device).view(1, 1, 12)
    tm_ts = torch.from_numpy(stats.tau_mean).to(device).view(1, 1, 12)
    ex_joint_q = (ex_joint[..., 0] - qm_ts) / float(stats.q_std)
    ex_joint_qd = ex_joint[..., 1] / float(stats.qd_std)
    ex_joint_tau = (ex_joint[..., 2] - tm_ts) / float(stats.tau_std)
    ex_joint = torch.stack([ex_joint_q, ex_joint_qd, ex_joint_tau], dim=-1)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    # Norm tensors for test-set prediction collection
    bm_t = torch.from_numpy(stats.base_mean).to(device)
    bs_t = torch.from_numpy(stats.base_std).to(device)
    qm_t = torch.from_numpy(stats.q_mean).to(device).view(1, 1, 12)
    tm_t = torch.from_numpy(stats.tau_mean).to(device).view(1, 1, 12)

    @torch.no_grad()
    def _collect_test_predictions(model, tau_alpha: float = 1.0):
        """Collect y_true / y_pred on the test set."""
        model.eval()
        ys, ps = [], []
        for base, joint, y in te_loader:
            base = base.to(device, non_blocking=True)
            joint = joint.to(device, non_blocking=True)
            base = (base - bm_t) / bs_t
            joint = torch.stack([
                (joint[..., 0] - qm_t) / float(stats.q_std),
                joint[..., 1] / float(stats.qd_std),
                (joint[..., 2] - tm_t) / float(stats.tau_std) * tau_alpha,
            ], dim=-1)
            pred = model(base, joint)
            ys.append(y.numpy())
            ps.append(pred.cpu().numpy())
        return np.concatenate(ys), np.concatenate(ps)

    def _run_model(model, name: str, tag: str, tau_alpha: float = 1.0):
        """Train, save checkpoint, export TorchScript, and collect predictions."""
        t_model0 = time.perf_counter()
        trained, loss_log = train_model(
            model, tr_loader, te_loader, device, name,
            args.epochs, args.lr, args.min_lr, args.weight_decay,
            out_dir=args.out_dir, norm_stats=stats,
            tau_alpha=tau_alpha,
        )
        model_times[tag] = time.perf_counter() - t_model0
        print(f"[TIME] {name} total train time: {model_times[tag]:.1f}s "
              f"({model_times[tag] / max(1, args.epochs):.1f}s/epoch)", flush=True)
        state_path = os.path.join(args.out_dir, f"{tag}_state.pt")
        ts_path = os.path.join(args.out_dir, f"{tag}.ts.pt")
        torch.save(trained.state_dict(), state_path)
        export_torchscript(trained.to(device), ex_base, ex_joint, ts_path)

        y_true, y_pred = _collect_test_predictions(trained, tau_alpha=tau_alpha)
        save_pred_csv(y_true, y_pred, os.path.join(args.out_dir, f"{tag}_pred.csv"))
        save_pred_plots(y_true, y_pred, args.out_dir, tag)

        return trained, loss_log

    # ===== Train all models =====
    all_loss_logs: Dict[str, Dict] = {}

    # FCNN
    fcnn = FCNNBaseline(
        args.horizon,
        hidden=fcnn_cfg["hidden"],
        dropout=fcnn_cfg["dropout"],
    )
    _, log = _run_model(fcnn, "FCNN", "fcnn")
    all_loss_logs["FCNN"] = {"baseline": log}

    # GRU
    gru = GRUBaseline(
        input_dim=gru_cfg["input_dim"],
        hidden_dim=gru_cfg["hidden_dim"],
        num_layers=gru_cfg["num_layers"],
        head_hidden=gru_cfg["head_hidden"],
        dropout=gru_cfg["dropout"],
    )
    _, log = _run_model(gru, "GRU", "gru")
    all_loss_logs["GRU"] = {"baseline": log}

    # Transformer
    transformer = TransformerEstimator(
        horizon_size=args.horizon,
        input_dim=transformer_cfg["input_dim"],
        d_model=transformer_cfg["d_model"],
        nhead=transformer_cfg["nhead"],
        num_layers=transformer_cfg["num_layers"],
        dim_ff=transformer_cfg["dim_ff"],
        dropout=transformer_cfg["dropout"],
    )
    _, log = _run_model(transformer, "Transformer", "transformer")
    all_loss_logs["Transformer"] = {"baseline": log}

    # RecGNN
    recgnn = RecGNN(
        args.horizon,
        emb_dim=recgnn_cfg["emb_dim"],
        msg_dim=recgnn_cfg["msg_dim"],
        hidden_mlp=recgnn_cfg["hidden_mlp"],
        mp_steps_per_t=recgnn_cfg["mp_steps_per_t"],
        dropout=recgnn_cfg["dropout"],
        gru_hidden=recgnn_cfg["gru_hidden"],
        gru_layers=recgnn_cfg["gru_layers"],
    )
    _, log = _run_model(recgnn, "RecGNN", "recgnn")
    all_loss_logs["RecGNN"] = {"baseline": log}

    # ===== Torque-channel scale (alpha) =====
    print("\n" + "=" * 60)
    print("Torque-channel scale (alpha)")
    print("=" * 60)

    n_analysis = min(
        args.alpha_max_samples,
        max(1, int(len(tr_loader.dataset) * args.alpha_sample_ratio)),
    )
    print(f"  sample count: {n_analysis}")

    t_collect0 = time.perf_counter()
    analysis_samples = collect_alpha_samples(
        tr_loader, stats, n_samples=n_analysis,
    )
    timings["alpha_collect_sec"] = time.perf_counter() - t_collect0

    t_solve0 = time.perf_counter()
    optimal_alpha = compute_optimal_alpha(analysis_samples)
    timings["alpha_solve_sec"] = time.perf_counter() - t_solve0

    timings["alpha_total_sec"] = timings["alpha_collect_sec"] + timings["alpha_solve_sec"]
    n_alpha_requested = int(n_analysis)
    n_alpha_valid = int(analysis_samples.get("n_collected", n_analysis))
    print(f"[TIME] alpha sample collection: {timings['alpha_collect_sec']:.1f}s "
          f"| alpha solve: {timings['alpha_solve_sec']:.1f}s "
          f"| total: {timings['alpha_total_sec']:.1f}s", flush=True)

    model_constructors = {
        "RecGNN": lambda: RecGNN(
            args.horizon, emb_dim=recgnn_cfg["emb_dim"],
            msg_dim=recgnn_cfg["msg_dim"], hidden_mlp=recgnn_cfg["hidden_mlp"],
            mp_steps_per_t=recgnn_cfg["mp_steps_per_t"],
            dropout=recgnn_cfg["dropout"],
            gru_hidden=recgnn_cfg["gru_hidden"],
            gru_layers=recgnn_cfg["gru_layers"],
        ),
    }

    for name, ctor in model_constructors.items():
        print(f"\n--- {name} with alpha={optimal_alpha:.4f} ---")
        set_seed(args.seed)
        model_alpha = ctor()
        tag_alpha = f"{name.lower()}_alpha"
        _, alpha_log = _run_model(
            model_alpha, f"{name}_alpha", tag_alpha,
            tau_alpha=optimal_alpha,
        )
        if name not in all_loss_logs:
            all_loss_logs[name] = {}
        all_loss_logs[name]["alpha_scaled"] = alpha_log

    plot_alpha_comparison(all_loss_logs, optimal_alpha, args.out_dir)

    timings["total_pipeline_sec"] = time.perf_counter() - t_pipeline0

    # ===== Quantitative measurement summary =====
    def _fmt(sec: float) -> str:
        return f"{sec / 60:.1f} min ({sec:.1f}s)"

    measurement = {
        "n_train_samples": n_train_samples,
        "n_test_samples": n_test_samples,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "horizon": args.horizon,
        "stride": args.stride,
        "alpha_samples_requested": n_alpha_requested,
        "alpha_samples_valid": n_alpha_valid,
        "optimal_alpha": optimal_alpha,
        "timings_sec": timings,
        "per_model_train_sec": model_times,
    }

    print("\n" + "=" * 60)
    print("QUANTITATIVE MEASUREMENT SUMMARY")
    print("=" * 60)
    print(f"  train samples used            : {n_train_samples}")
    print(f"  test samples                  : {n_test_samples}")
    print(f"  epochs per model              : {args.epochs}")
    print(f"  data preparation time         : {_fmt(timings.get('data_prep_sec', 0.0))}")
    for tag, sec in model_times.items():
        print(f"  train time [{tag:<16}] : {_fmt(sec)}")
    print(f"  alpha samples (requested)     : {n_alpha_requested}")
    print(f"  alpha samples (valid)         : {n_alpha_valid}")
    print(f"  alpha collect time            : {_fmt(timings.get('alpha_collect_sec', 0.0))}")
    print(f"  alpha solve time              : {_fmt(timings.get('alpha_solve_sec', 0.0))}")
    print(f"  alpha total time              : {_fmt(timings.get('alpha_total_sec', 0.0))}")
    print(f"  optimal alpha                 : {optimal_alpha:.4f}")
    print(f"  TOTAL PIPELINE TIME           : {_fmt(timings['total_pipeline_sec'])}")
    print("=" * 60)

    # ===== Save loss log =====
    log_data: Dict = {
        "epochs": args.epochs,
        "models": all_loss_logs,
        "measurement": measurement,
        "optimal_alpha": optimal_alpha,
    }
    loss_log_path = os.path.join(args.out_dir, "loss_log.json")
    with open(loss_log_path, "w") as f:
        json.dump(log_data, f, indent=2)
    print(f"[Log] Loss trajectories saved: {loss_log_path}")

    # Standalone measurement report
    measurement_path = os.path.join(args.out_dir, "measurement_report.json")
    with open(measurement_path, "w") as f:
        json.dump(measurement, f, indent=2)
    print(f"[Log] Measurement report saved: {measurement_path}")

    print("Done. out_dir =", os.path.abspath(args.out_dir))


if __name__ == "__main__":
    main()
