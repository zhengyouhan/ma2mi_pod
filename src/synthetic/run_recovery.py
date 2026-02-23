#!/usr/bin/env python
"""
Run T recovery experiment on synthetic wave scenario.

Visualizes:
1. Space-time diagram of ground truth vs prediction
2. Detector observations: predicted vs observed
3. T parameter recovery: estimated vs true
"""
from __future__ import annotations

import argparse
import os

import torch as th
import matplotlib.pyplot as plt
import numpy as np

from src.synthetic.wave_scenario import WaveScenarioConfig, create_wave_scenario
from src.sim.rollout import rollout_idm_multilane
from src.sim.reparam import sigmoid_reparam, inverse_sigmoid_reparam
from src.obs.detector_operator import detector_crosslane_at_times
from src.graph_prior import laplacian_penalty, compute_platooning_weights
from src.loss import detector_loss
from src.wavefront_loss import (
    wavefront_loss,
    compute_all_wavefront_features,
    compute_vehicle_wavefront_distance,
    compute_wavefront_gating,
    wavefront_gated_laplacian_penalty,
)


def fit_T(data: dict, cfg: WaveScenarioConfig, lambda_T: float = 0.01, iters: int = 100, lr: float = 0.05):
    """Fit T parameters from detector observations."""
    device = data["s0"].device
    dtype = data["s0"].dtype
    N = data["s0"].shape[0]

    s0 = data["s0"]
    v0 = data["v0"]
    lane_id = data["lane_id"]
    leader_idx = data["leader_idx"]
    xq = data["xq"]
    x_dets = data["x_dets"]
    time_idx = data["time_idx_5s"]
    obs = data["obs_5s"]

    steps = int(cfg.duration_s / cfg.dt)

    # Initialize T with prior mean
    T_min, T_max = cfg.T_min, cfg.T_max
    u_T = th.nn.Parameter(inverse_sigmoid_reparam(
        th.full((N,), cfg.T_mean, device=device, dtype=dtype),
        T_min, T_max
    ))

    opt = th.optim.Adam([u_T], lr=lr)

    a_max = th.full((N,), cfg.a_max, device=device, dtype=dtype)
    b = th.full((N,), cfg.b_comfort, device=device, dtype=dtype)
    v_target = th.full((N,), cfg.v_free, device=device, dtype=dtype)

    history = {"loss": [], "L_det": [], "L_graph": []}

    print(f"Fitting T with lambda_T={lambda_T}, iters={iters}, lr={lr}")

    for i in range(iters):
        opt.zero_grad()

        T = sigmoid_reparam(u_T, T_min, T_max)

        S, V, _ = rollout_idm_multilane(
            s0_init=s0,
            v0_init=v0,
            leader_idx=leader_idx,
            num_steps=steps,
            dt=cfg.dt,
            a_max=a_max,
            b=b,
            v_target=v_target,
            s0=cfg.s0,
            T_headway=T,
            ghost_v=cfg.ghost_v_base,
            ghost_gap0=cfg.ghost_gap0,
        )

        pred = detector_crosslane_at_times(
            S=S, V=V, lane_id=lane_id, xq=xq, x_dets=x_dets,
            time_indices=time_idx, sigma=cfg.sigma, half_window=10.0,
        )

        L_det = detector_loss(
            pred=pred, obs=obs,
            weights={"flow": 1.0, "speed": 0.1},
            loss_type="huber", huber_delta=1.0,
        )

        L_graph = laplacian_penalty(T, leader_idx, lane_id)
        loss = L_det + lambda_T * L_graph

        if not th.isfinite(loss):
            print(f"  NaN at iter {i}, stopping")
            break

        loss.backward()
        opt.step()

        history["loss"].append(float(loss.item()))
        history["L_det"].append(float(L_det.item()))
        history["L_graph"].append(float(L_graph.item()))

        if i % 20 == 0:
            print(f"  iter {i:3d}: loss={loss.item():.6f}, L_det={L_det.item():.6f}, L_graph={L_graph.item():.6f}")

    # Final T
    with th.no_grad():
        T_est = sigmoid_reparam(u_T, T_min, T_max)

    return T_est, history


def fit_T_with_wavefront(
    data: dict,
    cfg: WaveScenarioConfig,
    lambda_T: float = 0.01,
    iters: int = 100,
    lr: float = 0.05,
    alpha_pos: float = 0.01,
    alpha_speed: float = 0.001,
    alpha_amp: float = 0.01,
    tau: float = 2.0,
    use_gated_laplacian: bool = True,
    r_gate: float = 100.0,
    delta_r: float = 50.0,
    warmup_frac: float = 0.3,  # Fraction of iters for Stage A (L_det only)
):
    """Fit T parameters with wavefront-aware loss."""
    device = data["s0"].device
    dtype = data["s0"].dtype
    N = data["s0"].shape[0]

    s0 = data["s0"]
    v0 = data["v0"]
    lane_id = data["lane_id"]
    leader_idx = data["leader_idx"]
    xq = data["xq"]
    x_dets = data["x_dets"]
    time_idx = data["time_idx_5s"]
    obs = data["obs_5s"]

    steps = int(cfg.duration_s / cfg.dt)

    # Initialize T with prior mean
    T_min, T_max = cfg.T_min, cfg.T_max
    u_T = th.nn.Parameter(inverse_sigmoid_reparam(
        th.full((N,), cfg.T_mean, device=device, dtype=dtype),
        T_min, T_max
    ))

    opt = th.optim.Adam([u_T], lr=lr)

    a_max = th.full((N,), cfg.a_max, device=device, dtype=dtype)
    b = th.full((N,), cfg.b_comfort, device=device, dtype=dtype)
    v_target = th.full((N,), cfg.v_free, device=device, dtype=dtype)

    # Precompute observed wavefront features
    x_det_tensor = th.tensor(x_dets, device=device, dtype=dtype)
    V_obs = obs["speed"]  # [K, J]
    feat_obs = compute_all_wavefront_features(V_obs, x_det_tensor, cfg.obs_interval, tau)

    history = {
        "loss": [], "L_det": [], "L_graph": [],
        "L_wf_pos": [], "L_wf_speed": [], "L_wf_amp": [],
        "T_std": [],
    }

    warmup_iters = int(iters * warmup_frac)
    print(f"Fitting T with wavefront loss: lambda_T={lambda_T}, alpha_pos={alpha_pos}, alpha_speed={alpha_speed}, alpha_amp={alpha_amp}")
    print(f"  Stage A (L_det only): iters 0-{warmup_iters-1}")
    print(f"  Stage B (+ wavefront): iters {warmup_iters}-{iters-1}")

    for i in range(iters):
        opt.zero_grad()

        # Two-stage optimization: warmup with L_det only, then add wavefront loss
        use_wavefront = (i >= warmup_iters)

        T = sigmoid_reparam(u_T, T_min, T_max)

        S, V, _ = rollout_idm_multilane(
            s0_init=s0,
            v0_init=v0,
            leader_idx=leader_idx,
            num_steps=steps,
            dt=cfg.dt,
            a_max=a_max,
            b=b,
            v_target=v_target,
            s0=cfg.s0,
            T_headway=T,
            ghost_v=cfg.ghost_v_base,
            ghost_gap0=cfg.ghost_gap0,
        )

        pred = detector_crosslane_at_times(
            S=S, V=V, lane_id=lane_id, xq=xq, x_dets=x_dets,
            time_indices=time_idx, sigma=cfg.sigma, half_window=10.0,
        )

        # Standard detector loss
        L_det = detector_loss(
            pred=pred, obs=obs,
            weights={"flow": 1.0, "speed": 0.1},
            loss_type="huber", huber_delta=1.0,
        )

        # Wavefront loss (only after warmup)
        if use_wavefront:
            V_sim = pred["speed"]
            wf_result = wavefront_loss(
                V_sim, V_obs, x_det_tensor, cfg.obs_interval,
                tau=tau, alpha_pos=1.0, alpha_speed=1.0, alpha_amp=1.0,
            )
            L_wf_pos = wf_result["L_pos"]
            L_wf_speed = wf_result["L_speed"]
            L_wf_amp = wf_result["L_amp"]

            # Laplacian regularization (optionally gated)
            if use_gated_laplacian:
                # Compute vehicle distance to wavefront
                x_front_sim = wf_result["features_sim"]["x_front"]
                d = compute_vehicle_wavefront_distance(
                    S, x_front_sim, time_idx, cfg.dt, cfg.obs_interval
                )
                eta = compute_wavefront_gating(d, r=r_gate, delta_r=delta_r)
                L_graph = wavefront_gated_laplacian_penalty(T, leader_idx, lane_id, eta)
            else:
                L_graph = laplacian_penalty(T, leader_idx, lane_id)
        else:
            # Warmup: no wavefront loss, standard Laplacian
            L_wf_pos = th.tensor(0.0, device=device, dtype=dtype)
            L_wf_speed = th.tensor(0.0, device=device, dtype=dtype)
            L_wf_amp = th.tensor(0.0, device=device, dtype=dtype)
            L_graph = laplacian_penalty(T, leader_idx, lane_id)

        # Combined loss
        loss = (
            L_det
            + alpha_pos * L_wf_pos
            + alpha_speed * L_wf_speed
            + alpha_amp * L_wf_amp
            + lambda_T * L_graph
        )

        if not th.isfinite(loss):
            print(f"  NaN at iter {i}, stopping")
            break

        loss.backward()
        opt.step()

        # Record history
        history["loss"].append(float(loss.item()))
        history["L_det"].append(float(L_det.item()))
        history["L_graph"].append(float(L_graph.item()))
        history["L_wf_pos"].append(float(L_wf_pos.item()))
        history["L_wf_speed"].append(float(L_wf_speed.item()))
        history["L_wf_amp"].append(float(L_wf_amp.item()))
        history["T_std"].append(float(T.std().item()))

        if i % 20 == 0:
            stage = "B" if use_wavefront else "A"
            print(f"  [{stage}] iter {i:3d}: loss={loss.item():.4f}, L_det={L_det.item():.4f}, "
                  f"L_wf={L_wf_pos.item():.2f}/{L_wf_speed.item():.2f}/{L_wf_amp.item():.2f}, "
                  f"T_std={T.std().item():.3f}")

    # Final T
    with th.no_grad():
        T_est = sigmoid_reparam(u_T, T_min, T_max)

    return T_est, history


def visualize_results(
    data: dict,
    T_est: th.Tensor,
    cfg: WaveScenarioConfig,
    output_dir: str = "outputs",
):
    """Create visualization plots."""
    os.makedirs(output_dir, exist_ok=True)

    device = data["s0"].device
    dtype = data["s0"].dtype
    N = data["s0"].shape[0]

    # Re-run simulation with estimated T
    a_max = th.full((N,), cfg.a_max, device=device, dtype=dtype)
    b = th.full((N,), cfg.b_comfort, device=device, dtype=dtype)
    v_target = th.full((N,), cfg.v_free, device=device, dtype=dtype)
    steps = int(cfg.duration_s / cfg.dt)

    with th.no_grad():
        S_pred, V_pred, _ = rollout_idm_multilane(
            s0_init=data["s0"],
            v0_init=data["v0"],
            leader_idx=data["leader_idx"],
            num_steps=steps,
            dt=cfg.dt,
            a_max=a_max,
            b=b,
            v_target=v_target,
            s0=cfg.s0,
            T_headway=T_est,
            ghost_v=cfg.ghost_v_base,
            ghost_gap0=cfg.ghost_gap0,
        )

    S_true = data["S"]
    V_true = data["V"]
    T_true = data["T_true"]
    lane_id = data["lane_id"]

    # Convert to numpy
    S_true_np = S_true.cpu().numpy()
    V_true_np = V_true.cpu().numpy()
    S_pred_np = S_pred.cpu().numpy()
    V_pred_np = V_pred.cpu().numpy()
    T_true_np = T_true.cpu().numpy()
    T_est_np = T_est.cpu().numpy()
    lane_id_np = lane_id.cpu().numpy()

    time = np.arange(S_true_np.shape[0]) * cfg.dt

    # Figure 1: Space-time diagram comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Ground truth trajectories (colored by velocity)
    ax = axes[0, 0]
    for i in range(N):
        colors = plt.cm.RdYlGn((V_true_np[:, i] - 0) / 30.0)
        for j in range(len(time) - 1):
            ax.plot(time[j:j+2], S_true_np[j:j+2, i], c=colors[j], linewidth=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position (m)")
    ax.set_title("Ground Truth Trajectories (color = velocity)")
    ax.set_xlim(0, cfg.duration_s)

    # Predicted trajectories
    ax = axes[0, 1]
    for i in range(N):
        colors = plt.cm.RdYlGn((V_pred_np[:, i] - 0) / 30.0)
        for j in range(len(time) - 1):
            ax.plot(time[j:j+2], S_pred_np[j:j+2, i], c=colors[j], linewidth=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position (m)")
    ax.set_title("Predicted Trajectories (color = velocity)")
    ax.set_xlim(0, cfg.duration_s)

    # T recovery scatter
    ax = axes[1, 0]
    colors = ["C0", "C1", "C2"]
    for lane in range(cfg.num_lanes):
        mask = lane_id_np == lane
        ax.scatter(T_true_np[mask], T_est_np[mask], c=colors[lane], alpha=0.7, label=f"Lane {lane}")
    ax.plot([cfg.T_min, cfg.T_max], [cfg.T_min, cfg.T_max], "k--", label="Perfect")
    ax.set_xlabel("True T (s)")
    ax.set_ylabel("Estimated T (s)")
    ax.set_title("T Parameter Recovery")
    ax.legend()
    ax.set_xlim(cfg.T_min - 0.1, cfg.T_max + 0.1)
    ax.set_ylim(cfg.T_min - 0.1, cfg.T_max + 0.1)
    ax.set_aspect("equal")

    # T error distribution
    ax = axes[1, 1]
    T_error = T_est_np - T_true_np
    ax.hist(T_error, bins=20, edgecolor="black", alpha=0.7)
    ax.axvline(0, color="k", linestyle="--")
    ax.set_xlabel("T Error (estimated - true)")
    ax.set_ylabel("Count")
    mae = np.abs(T_error).mean()
    rmse = np.sqrt((T_error ** 2).mean())
    ax.set_title(f"T Error Distribution (MAE={mae:.3f}, RMSE={rmse:.3f})")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "trajectory_comparison.png"), dpi=150)
    print(f"Saved: {output_dir}/trajectory_comparison.png")
    plt.close()

    # Figure 2: Detector observations
    obs_true = data["obs_5s"]
    time_idx = data["time_idx_5s"]
    x_dets = data["x_dets"]

    with th.no_grad():
        obs_pred = detector_crosslane_at_times(
            S_pred, V_pred, lane_id, data["xq"], x_dets,
            time_indices=time_idx, sigma=cfg.sigma, half_window=10.0,
        )

    fig, axes = plt.subplots(2, len(x_dets), figsize=(4 * len(x_dets), 8))

    obs_times = time_idx.cpu().numpy() * cfg.dt

    for j, x_det in enumerate(x_dets):
        # Flow
        ax = axes[0, j]
        ax.plot(obs_times, obs_true["flow"][:, j].cpu().numpy(), "b-", label="True", linewidth=2)
        ax.plot(obs_times, obs_pred["flow"][:, j].cpu().numpy(), "r--", label="Pred", linewidth=2)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Flow")
        ax.set_title(f"Detector @ x={x_det}m")
        ax.legend()

        # Speed
        ax = axes[1, j]
        ax.plot(obs_times, obs_true["speed"][:, j].cpu().numpy(), "b-", label="True", linewidth=2)
        ax.plot(obs_times, obs_pred["speed"][:, j].cpu().numpy(), "r--", label="Pred", linewidth=2)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Speed (m/s)")
        ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "detector_comparison.png"), dpi=150)
    print(f"Saved: {output_dir}/detector_comparison.png")
    plt.close()

    # Figure 3: Space-time speed heatmap
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Create space-time grids
    x_bins = np.linspace(cfg.x_min, cfg.x_max, 50)
    t_bins = np.linspace(0, cfg.duration_s, 60)

    def create_heatmap(S, V, x_bins, t_bins):
        heatmap = np.zeros((len(t_bins) - 1, len(x_bins) - 1))
        counts = np.zeros_like(heatmap)

        for t_idx in range(S.shape[0]):
            t = t_idx * cfg.dt
            t_bin = np.searchsorted(t_bins, t) - 1
            if 0 <= t_bin < len(t_bins) - 1:
                for i in range(S.shape[1]):
                    x = S[t_idx, i]
                    x_bin = np.searchsorted(x_bins, x) - 1
                    if 0 <= x_bin < len(x_bins) - 1:
                        heatmap[t_bin, x_bin] += V[t_idx, i]
                        counts[t_bin, x_bin] += 1

        counts[counts == 0] = 1
        return heatmap / counts

    hm_true = create_heatmap(S_true_np, V_true_np, x_bins, t_bins)
    hm_pred = create_heatmap(S_pred_np, V_pred_np, x_bins, t_bins)

    vmin, vmax = 0, 30

    ax = axes[0]
    im = ax.imshow(hm_true, aspect="auto", origin="lower",
                   extent=[x_bins[0], x_bins[-1], t_bins[0], t_bins[-1]],
                   cmap="RdYlGn", vmin=vmin, vmax=vmax)
    ax.set_xlabel("Position (m)")
    ax.set_ylabel("Time (s)")
    ax.set_title("Ground Truth Speed (m/s)")
    plt.colorbar(im, ax=ax)

    ax = axes[1]
    im = ax.imshow(hm_pred, aspect="auto", origin="lower",
                   extent=[x_bins[0], x_bins[-1], t_bins[0], t_bins[-1]],
                   cmap="RdYlGn", vmin=vmin, vmax=vmax)
    ax.set_xlabel("Position (m)")
    ax.set_ylabel("Time (s)")
    ax.set_title("Predicted Speed (m/s)")
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "spacetime_heatmap.png"), dpi=150)
    print(f"Saved: {output_dir}/spacetime_heatmap.png")
    plt.close()

    # Print summary
    print("\n" + "=" * 50)
    print("T Recovery Summary")
    print("=" * 50)
    print(f"True T:  mean={T_true_np.mean():.3f}, std={T_true_np.std():.3f}")
    print(f"Est T:   mean={T_est_np.mean():.3f}, std={T_est_np.std():.3f}")
    print(f"MAE:     {mae:.4f}")
    print(f"RMSE:    {rmse:.4f}")
    print(f"Corr:    {np.corrcoef(T_true_np, T_est_np)[0, 1]:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Run T recovery on synthetic wave scenario")
    parser.add_argument("--lambda-T", type=float, default=0.01, help="Graph regularization weight")
    parser.add_argument("--iters", type=int, default=100, help="Optimization iterations")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate")
    parser.add_argument("--output-dir", type=str, default="outputs/wave_recovery", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--detector-spacing", type=float, default=200.0, help="Spacing between detectors (m)")
    parser.add_argument("--obs-interval", type=float, default=5.0, help="Observation interval (s)")
    # Wavefront loss arguments
    parser.add_argument("--wavefront", action="store_true", help="Use wavefront-aware loss")
    parser.add_argument("--alpha-pos", type=float, default=0.01, help="Wavefront position loss weight")
    parser.add_argument("--alpha-speed", type=float, default=0.001, help="Wavefront speed loss weight")
    parser.add_argument("--alpha-amp", type=float, default=0.01, help="Wavefront amplitude loss weight")
    parser.add_argument("--tau", type=float, default=2.0, help="Softmax temperature for front localization")
    parser.add_argument("--gated-laplacian", action="store_true", help="Use gated Laplacian (allow heterogeneity near front)")
    parser.add_argument("--warmup-frac", type=float, default=0.3, help="Fraction of iters for L_det-only warmup")
    args = parser.parse_args()

    device = "cuda" if th.cuda.is_available() else "cpu"
    dtype = th.float32

    print("=" * 60)
    print("Synthetic Wave Scenario: T Parameter Recovery")
    print("=" * 60)

    # Create scenario with custom detector settings
    print("\n1. Creating wave scenario...")
    cfg = WaveScenarioConfig(
        detector_spacing=args.detector_spacing,
        obs_interval=args.obs_interval,
    )
    data = create_wave_scenario(cfg, device=device, dtype=dtype, seed=args.seed)

    print(f"   Vehicles: {data['s0'].shape[0]}")
    print(f"   Duration: {cfg.duration_s}s")
    print(f"   Ghost driver: base_v={cfg.ghost_v_base} m/s")
    print(f"   Wave 1: t={cfg.wave1_time}s, drop={cfg.wave1_v_drop} m/s, duration={cfg.wave1_duration}s")
    print(f"   Wave 2: t={cfg.wave2_time}s, drop={cfg.wave2_v_drop} m/s, duration={cfg.wave2_duration}s")
    print(f"   True T: [{data['T_true'].min():.2f}, {data['T_true'].max():.2f}], mean={data['T_true'].mean():.2f}")
    print(f"   Detectors: {len(cfg.detector_positions)} positions ({cfg.detector_spacing}m spacing)")
    print(f"   Observations: every {cfg.obs_interval}s ({len(data['time_idx_5s'])} time points)")

    # Fit T
    print("\n2. Fitting T parameters...")
    if args.wavefront:
        print("   Using wavefront-aware loss")
        T_est, history = fit_T_with_wavefront(
            data, cfg,
            lambda_T=args.lambda_T,
            iters=args.iters,
            lr=args.lr,
            alpha_pos=args.alpha_pos,
            alpha_speed=args.alpha_speed,
            alpha_amp=args.alpha_amp,
            tau=args.tau,
            use_gated_laplacian=args.gated_laplacian,
            warmup_frac=args.warmup_frac,
        )
    else:
        T_est, history = fit_T(data, cfg, lambda_T=args.lambda_T, iters=args.iters, lr=args.lr)

    # Visualize
    print("\n3. Creating visualizations...")
    visualize_results(data, T_est, cfg, output_dir=args.output_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
