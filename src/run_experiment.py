#!/usr/bin/env python
"""
Run reproducible experiments with result saving.

Usage:
    python src/run_experiment.py --config graph_const --output-dir results/
    python src/run_experiment.py --run-ablations --output-dir results/ablations/
"""
from __future__ import annotations

import argparse
import json
import os

import torch as th

from src.data.ngsim_loader import load_ngsim_detector_dataset
from src.eval.metrics import (
    wave_arrival_time, wave_arrival_error,
    trajectory_collision_count, detector_reconstruction_error, acceleration_stats
)
from src.experiment import (
    ExperimentConfig, ExperimentResult, ABLATION_CONFIGS,
    set_seed, create_output_dir, save_trajectories, save_predictions
)
from src.graph_prior import laplacian_penalty, compute_platooning_weights
from src.loss import detector_loss
from src.obs.detector_operator import detector_crosslane_at_times
from src.sim.rollout import rollout_idm_multilane
from src.sim.reparam import sigmoid_reparam, inverse_sigmoid_reparam


def run_experiment(cfg: ExperimentConfig, data: dict, device: str, dtype: th.dtype) -> ExperimentResult:
    """Run a single experiment with the given configuration."""
    set_seed(cfg.seed)

    N = data["s0"].shape[0]
    s0 = data["s0"]
    v0_obs = data["v0"]
    lane_id = data["lane_id"]
    leader_idx = data["leader_idx"]
    xq = data["xq"]
    x_dets = data["x_dets"]
    time_idx = data["time_idx_5s"]
    obs = data["obs_5s"]

    steps = int(round(cfg.duration_s / cfg.dt))

    # Initialize parameters
    u_T = th.nn.Parameter(inverse_sigmoid_reparam(
        th.full((N,), cfg.T_init, device=device, dtype=dtype),
        cfg.T_min, cfg.T_max
    ))

    params = [u_T]
    v0_param = None
    lane_indices = None

    if cfg.learn_v0 == "vector":
        v0_param = th.nn.Parameter(v0_obs.clone())
        params.append(v0_param)
    elif cfg.learn_v0 == "scalar":
        v0_param = th.nn.Parameter(th.tensor(float(v0_obs.mean().item()), device=device, dtype=dtype))
        params.append(v0_param)
    elif cfg.learn_v0 == "lane":
        unique_lanes = th.unique(lane_id)
        lane_to_idx = {int(lane.item()): i for i, lane in enumerate(unique_lanes)}
        lane_indices = th.tensor([lane_to_idx[int(l.item())] for l in lane_id], device=device)
        v0_lane_init = th.zeros(len(unique_lanes), device=device, dtype=dtype)
        for i, lane in enumerate(unique_lanes):
            mask = lane_id == lane
            v0_lane_init[i] = v0_obs[mask].mean()
        v0_param = th.nn.Parameter(v0_lane_init)
        params.append(v0_param)

    opt = th.optim.Adam(params, lr=cfg.lr)

    a_max = th.full((N,), cfg.a_max, device=device, dtype=dtype)
    b = th.full((N,), cfg.b_comfort, device=device, dtype=dtype)
    v_target = th.full((N,), cfg.v_target, device=device, dtype=dtype)

    history = {"loss": [], "L_det": [], "L_graph": [], "L_prior_T": [], "grad_T": []}

    # Training loop
    for _ in range(cfg.iters):
        opt.zero_grad()

        T = sigmoid_reparam(u_T, cfg.T_min, cfg.T_max)
        if cfg.learn_v0 == "vector":
            v0 = th.clamp(v0_param, 0.0, cfg.v0_max)
        elif cfg.learn_v0 == "scalar":
            v0 = th.ones_like(v0_obs) * th.clamp(v0_param, 0.0, cfg.v0_max)
        elif cfg.learn_v0 == "lane":
            v0_clamped = th.clamp(v0_param, 0.0, cfg.v0_max)
            v0 = v0_clamped[lane_indices]
        else:
            v0 = v0_obs

        S, V, _ = rollout_idm_multilane(
            s0_init=s0, v0_init=v0, leader_idx=leader_idx,
            num_steps=steps, dt=cfg.dt, a_max=a_max, b=b, v_target=v_target,
            s0=cfg.s0_idm, T_headway=T, ghost_v=cfg.ghost_v, ghost_gap0=cfg.ghost_gap0,
        )

        pred = detector_crosslane_at_times(
            S=S, V=V, lane_id=lane_id, xq=xq, x_dets=x_dets,
            time_indices=time_idx, sigma=cfg.sigma, half_window=10.0,
        )

        L_det = detector_loss(
            pred=pred, obs=obs,
            weights={"flow": 1.0, "speed": cfg.speed_weight},
            loss_type="huber", huber_delta=1.0,
        )

        if cfg.state_weighted_graph:
            weights = compute_platooning_weights(
                S, V, leader_idx, lane_id,
                s0=cfg.s0_platooning, vs=cfg.vs_platooning
            )
        else:
            weights = None

        L_graph = laplacian_penalty(T, leader_idx, lane_id, weights=weights)
        L_prior_T = ((T - cfg.T_bar) ** 2).mean()

        loss = L_det + cfg.lambda_T * L_graph + cfg.beta_T * L_prior_T

        if not th.isfinite(loss):
            break

        loss.backward()
        opt.step()

        history["loss"].append(float(loss.item()))
        history["L_det"].append(float(L_det.item()))
        history["L_graph"].append(float(L_graph.item()))
        history["L_prior_T"].append(float(L_prior_T.item()))
        history["grad_T"].append(float(u_T.grad.norm().item()) if u_T.grad is not None else 0.0)

    # Final evaluation
    with th.no_grad():
        T_final = sigmoid_reparam(u_T, cfg.T_min, cfg.T_max)

        # Run final simulation
        S, V, _ = rollout_idm_multilane(
            s0_init=s0, v0_init=v0_obs, leader_idx=leader_idx,
            num_steps=steps, dt=cfg.dt, a_max=a_max, b=b, v_target=v_target,
            s0=cfg.s0_idm, T_headway=T_final, ghost_v=cfg.ghost_v, ghost_gap0=cfg.ghost_gap0,
        )

        pred = detector_crosslane_at_times(
            S=S, V=V, lane_id=lane_id, xq=xq, x_dets=x_dets,
            time_indices=time_idx, sigma=cfg.sigma, half_window=10.0,
        )

        # Compute metrics
        recon_err = detector_reconstruction_error(pred, obs)
        collision_stats = trajectory_collision_count(S, leader_idx, s_min=cfg.s0_idm)
        accel_stats_result = acceleration_stats(V, cfg.dt)

        # Smoothness
        valid = leader_idx >= 0
        valid = valid & (lane_id == lane_id[th.clamp(leader_idx, min=0)])
        if valid.sum() > 0:
            i = th.where(valid)[0]
            j = leader_idx[i]
            smoothness = float((T_final[i] - T_final[j]).abs().mean().item())
        else:
            smoothness = 0.0

        # Wave metrics
        wave_mae, wave_rmse = None, None
        if cfg.duration_s >= cfg.wave_sustained_s + 10:
            time_s = time_idx.float() * cfg.dt
            t_pred = wave_arrival_time(pred["speed"], time_s, cfg.wave_v_threshold, cfg.wave_sustained_s, dt=5.0)
            t_obs = wave_arrival_time(obs["speed"], time_s, cfg.wave_v_threshold, cfg.wave_sustained_s, dt=5.0)
            wave_err = wave_arrival_error(t_pred, t_obs)
            if th.isfinite(wave_err["mae"]):
                wave_mae = float(wave_err["mae"].item())
                wave_rmse = float(wave_err["rmse"].item())

    return ExperimentResult(
        config=cfg.to_dict(),
        T=T_final.tolist(),
        history=history,
        L_det_final=history["L_det"][-1] if history["L_det"] else None,
        L_graph_final=history["L_graph"][-1] if history["L_graph"] else None,
        smoothness=smoothness,
        flow_mse=float(recon_err.get("flow_mse", th.tensor(float("nan"))).item()),
        speed_mse=float(recon_err.get("speed_mse", th.tensor(float("nan"))).item()),
        collision_count=int(collision_stats["collision_count"].item()),
        collision_fraction=float(collision_stats["collision_fraction"].item()),
        min_spacing=float(collision_stats["min_spacing"].item()),
        accel_max=float(accel_stats_result["accel_max"].item()),
        jerk_max=float(accel_stats_result["jerk_max"].item()),
        wave_mae=wave_mae,
        wave_rmse=wave_rmse,
    )


def main():
    parser = argparse.ArgumentParser(description="Run reproducible experiments")
    parser.add_argument("--config", type=str, choices=list(ABLATION_CONFIGS.keys()),
                        help="Predefined config name")
    parser.add_argument("--config-file", type=str, help="Path to custom config JSON")
    parser.add_argument("--run-ablations", action="store_true",
                        help="Run all ablation experiments")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Output directory for results")
    parser.add_argument("--csv", type=str, help="Override CSV path")
    parser.add_argument("--duration-s", type=float, help="Override duration")
    args = parser.parse_args()

    device = "cuda" if th.cuda.is_available() else "cpu"
    dtype = th.float32

    # Determine which configs to run
    if args.run_ablations:
        configs = list(ABLATION_CONFIGS.values())
    elif args.config:
        configs = [ABLATION_CONFIGS[args.config]]
    elif args.config_file:
        configs = [ExperimentConfig.load(args.config_file)]
    else:
        configs = [ExperimentConfig()]

    # Apply overrides
    for cfg in configs:
        if args.csv:
            cfg.csv = args.csv
        if args.duration_s:
            cfg.duration_s = args.duration_s

    # Create output directory
    output_dir = create_output_dir(args.output_dir)
    print(f"Output directory: {output_dir}")

    # Load data (same for all configs)
    cfg = configs[0]
    if not os.path.exists(cfg.csv):
        print(f"CSV not found: {cfg.csv}")
        return

    # Infer t0_base
    import csv as csv_module
    with open(cfg.csv, "r", newline="") as f:
        reader = csv_module.reader(f)
        header = next(reader)
        i_time = header.index("Global_Time")
        min_t = None
        for k, row in enumerate(reader):
            if k >= 1000:
                break
            try:
                t = int(float(row[i_time]))
            except ValueError:
                continue
            min_t = t if min_t is None else min(min_t, t)
    t0_ms = cfg.t0_ms if cfg.t0_ms is not None else min_t + int(round(cfg.offset_s * 1000.0))

    data = load_ngsim_detector_dataset(
        csv_path=cfg.csv,
        t0_ms=t0_ms,
        duration_s=cfg.duration_s,
        x_min=cfg.x_center - cfg.x_half,
        x_max=cfg.x_center + cfg.x_half,
        dt=cfg.dt,
        sigma=cfg.sigma,
        device=device,
        dtype=dtype,
    )

    if data is None:
        print("No usable vehicles found")
        return

    print(f"Loaded {data['s0'].shape[0]} vehicles")

    # Run experiments
    results = {}
    for cfg in configs:
        print(f"\nRunning: {cfg.name}")
        print(f"  lambda_T={cfg.lambda_T}, beta_T={cfg.beta_T}, soft_weights={cfg.state_weighted_graph}")

        result = run_experiment(cfg, data, device, dtype)
        results[cfg.name] = result

        # Save individual result
        result.save(os.path.join(output_dir, f"{cfg.name}_result.json"))
        cfg.save(os.path.join(output_dir, f"{cfg.name}_config.json"))

        print(f"  L_det={result.L_det_final:.6f}, smoothness={result.smoothness:.6f}")
        print(f"  flow_mse={result.flow_mse:.6f}, speed_mse={result.speed_mse:.6f}")
        if result.wave_mae is not None:
            print(f"  wave_mae={result.wave_mae:.2f}s")

    # Save summary
    summary = {
        name: {
            "L_det": r.L_det_final,
            "L_graph": r.L_graph_final,
            "smoothness": r.smoothness,
            "flow_mse": r.flow_mse,
            "speed_mse": r.speed_mse,
            "wave_mae": r.wave_mae,
        }
        for name, r in results.items()
    }

    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {output_dir}")

    # Print comparison table
    print("\n" + "=" * 80)
    print("Ablation Comparison")
    print("=" * 80)
    print(f"{'Config':<20} {'L_det':>12} {'L_graph':>12} {'smooth':>10} {'flow_mse':>12} {'wave_mae':>10}")
    print("-" * 80)
    for name, r in results.items():
        wave_str = f"{r.wave_mae:.2f}" if r.wave_mae is not None else "N/A"
        print(f"{name:<20} {r.L_det_final:>12.6f} {r.L_graph_final:>12.6f} {r.smoothness:>10.6f} {r.flow_mse:>12.6f} {wave_str:>10}")


if __name__ == "__main__":
    main()
