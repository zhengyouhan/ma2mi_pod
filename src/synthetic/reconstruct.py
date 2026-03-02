#!/usr/bin/env python
"""
Trajectory Reconstruction via Differentiable IDM.

Implements the revised plan (Steps 0–4):
  Step 0: Visualize current best reconstruction
  Step 1: Ablation on learnable global parameters (v0, a_max)
  Step 3: Reassess L_arr with trajectory metrics
  Step 4: Resolution scaling curve

Usage:
    python -m src.synthetic.reconstruct --step 0 --device cuda
    python -m src.synthetic.reconstruct --step 1 --device cuda
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch as th
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.synthetic.wave_scenario import (
    WaveScenarioConfig,
    create_wave_scenario,
    generate_detector_observations,
)
from src.sim.rollout import rollout_idm_multilane
from src.sim.reparam import sigmoid_reparam, inverse_sigmoid_reparam
from src.obs.detector_operator import (
    detector_crosslane_at_times,
    detector_perlane_at_times,
    micro_to_macro_gaussian,
)
from src.loss import detector_loss
from src.synthetic.run_2s_recovery import compute_arrival_spread


# ---------------------------------------------------------------------------
# Core: Reconstruction loop with optional learnable globals
# ---------------------------------------------------------------------------

def run_reconstruction(
    data: dict,
    cfg: WaveScenarioConfig,
    obs: dict,
    time_indices: th.Tensor,
    obs_interval: float,
    *,
    learn_v0: bool = False,
    learn_amax: bool = False,
    iters: int = 100,
    lr: float = 0.05,
    label: str = "",
) -> dict:
    """
    Gradient-based trajectory reconstruction.

    Learnable params:
      - Per-vehicle T (always)
      - Global v0 (if learn_v0=True)
      - Global a_max (if learn_amax=True)

    Returns dict with T_est, v0_est, amax_est, S_pred, V_pred, history.
    """
    device = data["s0"].device
    dtype = data["s0"].dtype
    N = data["s0"].shape[0]

    s0 = data["s0"]
    v0 = data["v0"]
    lane_id = data["lane_id"]
    leader_idx = data["leader_idx"]
    xq = data["xq"]
    x_dets = data["x_dets"]
    T_true = data["T_true"]

    steps = int(cfg.duration_s / cfg.dt)
    T_min, T_max = cfg.T_min, cfg.T_max

    # --- Learnable parameters ---
    params = []

    # Per-vehicle T (always learnable)
    u_T = th.nn.Parameter(inverse_sigmoid_reparam(
        th.full((N,), cfg.T_mean, device=device, dtype=dtype),
        T_min, T_max,
    ))
    params.append(u_T)

    # Global v0 (optional)
    v0_lo, v0_hi = 15.0, 40.0
    if learn_v0:
        u_v0 = th.nn.Parameter(inverse_sigmoid_reparam(
            th.tensor(cfg.v_free, device=device, dtype=dtype),
            v0_lo, v0_hi,
        ))
        params.append(u_v0)

    # Global a_max (optional)
    amax_lo, amax_hi = 0.5, 3.0
    if learn_amax:
        u_amax = th.nn.Parameter(inverse_sigmoid_reparam(
            th.tensor(cfg.a_max, device=device, dtype=dtype),
            amax_lo, amax_hi,
        ))
        params.append(u_amax)

    opt = th.optim.Adam(params, lr=lr)

    b_vec = th.full((N,), cfg.b_comfort, device=device, dtype=dtype)

    history = {
        "loss": [], "L_det": [],
        "T_Corr": [], "T_MAE": [], "T_std": [],
        "v0_est": [], "amax_est": [],
    }

    tag = f"[{label}]" if label else ""
    learn_str = "T"
    if learn_v0:
        learn_str += " + v0"
    if learn_amax:
        learn_str += " + a_max"
    print(f"{tag} Reconstruction: learnable={learn_str}, obs_interval={obs_interval}s, "
          f"iters={iters}, lr={lr}")

    T_true_np = T_true.cpu().numpy()

    for i in range(iters):
        opt.zero_grad()

        T = sigmoid_reparam(u_T, T_min, T_max)
        v_target_val = sigmoid_reparam(u_v0, v0_lo, v0_hi) if learn_v0 else th.tensor(cfg.v_free, device=device, dtype=dtype)
        amax_val = sigmoid_reparam(u_amax, amax_lo, amax_hi) if learn_amax else th.tensor(cfg.a_max, device=device, dtype=dtype)

        v_target_vec = v_target_val.expand(N)
        amax_vec = amax_val.expand(N)

        S, V, _ = rollout_idm_multilane(
            s0_init=s0, v0_init=v0, leader_idx=leader_idx,
            num_steps=steps, dt=cfg.dt,
            a_max=amax_vec, b=b_vec, v_target=v_target_vec,
            s0=cfg.s0, T_headway=T,
            ghost_v=cfg.ghost_v_base, ghost_gap0=cfg.ghost_gap0,
        )

        pred = detector_crosslane_at_times(
            S=S, V=V, lane_id=lane_id, xq=xq, x_dets=x_dets,
            time_indices=time_indices, sigma=cfg.sigma, half_window=10.0,
        )

        loss = detector_loss(
            pred=pred, obs=obs,
            weights={"flow": 1.0, "speed": 0.1},
            loss_type="huber", huber_delta=1.0,
        )

        if not th.isfinite(loss):
            print(f"{tag}   NaN at iter {i}, stopping")
            break

        loss.backward()
        opt.step()

        with th.no_grad():
            T_np = T.cpu().numpy()
            corr = float(np.corrcoef(T_true_np, T_np)[0, 1]) if T_np.std() > 1e-8 else 0.0
            mae = float(np.abs(T_true_np - T_np).mean())
            v0_val = float(v_target_val.item()) if learn_v0 else cfg.v_free
            amax_v = float(amax_val.item()) if learn_amax else cfg.a_max

        history["loss"].append(float(loss.item()))
        history["L_det"].append(float(loss.item()))
        history["T_Corr"].append(corr)
        history["T_MAE"].append(mae)
        history["T_std"].append(float(T.std().item()))
        history["v0_est"].append(v0_val)
        history["amax_est"].append(amax_v)

        if i % 20 == 0 or i == iters - 1:
            extras = ""
            if learn_v0:
                extras += f", v0={v0_val:.2f}"
            if learn_amax:
                extras += f", a_max={amax_v:.2f}"
            print(f"{tag}   iter {i:3d}: L_det={loss.item():.6f}, "
                  f"T_Corr={corr:.4f}, T_MAE={mae:.4f}{extras}")

    # Final forward pass to get trajectories
    with th.no_grad():
        T_est = sigmoid_reparam(u_T, T_min, T_max)
        v0_final = sigmoid_reparam(u_v0, v0_lo, v0_hi) if learn_v0 else th.tensor(cfg.v_free, device=device, dtype=dtype)
        amax_final = sigmoid_reparam(u_amax, amax_lo, amax_hi) if learn_amax else th.tensor(cfg.a_max, device=device, dtype=dtype)

        S_pred, V_pred, _ = rollout_idm_multilane(
            s0_init=s0, v0_init=v0, leader_idx=leader_idx,
            num_steps=steps, dt=cfg.dt,
            a_max=amax_final.expand(N), b=b_vec,
            v_target=v0_final.expand(N),
            s0=cfg.s0, T_headway=T_est,
            ghost_v=cfg.ghost_v_base, ghost_gap0=cfg.ghost_gap0,
        )

    return {
        "T_est": T_est,
        "v0_est": float(v0_final.item()),
        "amax_est": float(amax_final.item()),
        "S_pred": S_pred,
        "V_pred": V_pred,
        "history": history,
        "label": label,
    }


# ---------------------------------------------------------------------------
# Visualization: Step 0 plots
# ---------------------------------------------------------------------------

def create_speed_heatmap(S, V, cfg, x_bins, t_bins):
    """Create time-space speed heatmap from micro trajectories."""
    S_np = S.cpu().numpy()
    V_np = V.cpu().numpy()
    heatmap = np.full((len(t_bins) - 1, len(x_bins) - 1), np.nan)
    counts = np.zeros_like(heatmap)

    for t_idx in range(0, S_np.shape[0], max(1, S_np.shape[0] // 200)):
        t = t_idx * cfg.dt
        t_bin = np.searchsorted(t_bins, t) - 1
        if 0 <= t_bin < len(t_bins) - 1:
            for n in range(S_np.shape[1]):
                x = S_np[t_idx, n]
                x_bin = np.searchsorted(x_bins, x) - 1
                if 0 <= x_bin < len(x_bins) - 1:
                    if np.isnan(heatmap[t_bin, x_bin]):
                        heatmap[t_bin, x_bin] = 0.0
                    heatmap[t_bin, x_bin] += V_np[t_idx, n]
                    counts[t_bin, x_bin] += 1

    valid = counts > 0
    heatmap[valid] /= counts[valid]
    return heatmap


def plot_step0(data, result, cfg, output_dir):
    """
    Step 0 plots:
    A) Time-space speed heatmap: ground truth vs sim, side by side.
    B) Simulated trajectories overlaid on the sim speed field.
    """
    S_true = data["S"]
    V_true = data["V"]
    S_pred = result["S_pred"]
    V_pred = result["V_pred"]

    x_bins = np.linspace(cfg.x_min, cfg.x_max, 80)
    t_bins = np.linspace(0, cfg.duration_s, 100)

    hm_true = create_speed_heatmap(S_true, V_true, cfg, x_bins, t_bins)
    hm_sim = create_speed_heatmap(S_pred, V_pred, cfg, x_bins, t_bins)

    vmin, vmax = 0, 30

    # --- Plot A: Side-by-side speed heatmap ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    im = ax.imshow(hm_true, aspect="auto", origin="lower",
                   extent=[x_bins[0], x_bins[-1], t_bins[0], t_bins[-1]],
                   cmap="RdYlGn", vmin=vmin, vmax=vmax)
    ax.set_xlabel("Position (m)", fontsize=12)
    ax.set_ylabel("Time (s)", fontsize=12)
    ax.set_title("Ground Truth (event-driven ghost)", fontsize=13)
    plt.colorbar(im, ax=ax, label="Speed (m/s)")

    ax = axes[1]
    im = ax.imshow(hm_sim, aspect="auto", origin="lower",
                   extent=[x_bins[0], x_bins[-1], t_bins[0], t_bins[-1]],
                   cmap="RdYlGn", vmin=vmin, vmax=vmax)
    ax.set_xlabel("Position (m)", fontsize=12)
    ax.set_ylabel("Time (s)", fontsize=12)
    ax.set_title(f"Reconstructed ({result['label']})", fontsize=13)
    plt.colorbar(im, ax=ax, label="Speed (m/s)")

    h = result["history"]
    fig.suptitle(f"Time–Space Speed Field — T_Corr={h['T_Corr'][-1]:.3f}, "
                 f"L_det={h['L_det'][-1]:.4f}", fontsize=14)
    plt.tight_layout()
    path = os.path.join(output_dir, f"step0_heatmap_{result['label']}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")

    # --- Plot B: Trajectories on speed field ---
    fig, ax = plt.subplots(figsize=(12, 7))

    im = ax.imshow(hm_sim, aspect="auto", origin="lower",
                   extent=[x_bins[0], x_bins[-1], t_bins[0], t_bins[-1]],
                   cmap="RdYlGn", vmin=vmin, vmax=vmax, alpha=0.6)
    plt.colorbar(im, ax=ax, label="Speed (m/s)")

    # Overlay trajectories (subsample for clarity)
    S_np = S_pred.cpu().numpy()
    N = S_np.shape[1]
    time = np.arange(S_np.shape[0]) * cfg.dt
    stride = max(1, S_np.shape[0] // 500)  # subsample time

    for n in range(0, N, 3):  # every 3rd vehicle
        ax.plot(S_np[::stride, n], time[::stride],
                color="k", linewidth=0.3, alpha=0.5)

    ax.set_xlabel("Position (m)", fontsize=12)
    ax.set_ylabel("Time (s)", fontsize=12)
    ax.set_title(f"Reconstructed Trajectories on Speed Field ({result['label']})", fontsize=13)
    ax.set_xlim(cfg.x_min, cfg.x_max)
    ax.set_ylim(0, cfg.duration_s)
    plt.tight_layout()
    path = os.path.join(output_dir, f"step0_trajectories_{result['label']}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_step0_observations(data, result, cfg, obs, time_indices, obs_interval, output_dir):
    """Step 0 supplementary: detector speed timeseries (sim vs obs)."""
    x_dets = data["x_dets"]
    Nd = len(x_dets)

    obs_speed = obs["speed"].cpu().numpy()
    obs_times = time_indices.cpu().numpy() * cfg.dt

    # Compute sim detector outputs
    with th.no_grad():
        pred = detector_crosslane_at_times(
            S=result["S_pred"], V=result["V_pred"],
            lane_id=data["lane_id"], xq=data["xq"], x_dets=x_dets,
            time_indices=time_indices, sigma=cfg.sigma, half_window=10.0,
        )
    sim_speed = pred["speed"].cpu().numpy()

    # Select subset of detectors to show
    show_idx = list(range(0, Nd, max(1, Nd // 6)))
    n_show = len(show_idx)

    fig, axes = plt.subplots(1, n_show, figsize=(4 * n_show, 4), squeeze=False)
    for i, j in enumerate(show_idx):
        ax = axes[0, i]
        ax.plot(obs_times, obs_speed[:, j], "b-", linewidth=2, label="Obs")
        ax.plot(obs_times, sim_speed[:, j], "r--", linewidth=2, label="Sim")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Speed (m/s)")
        ax.set_title(f"x={x_dets[j]}m")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Detector Speed: Obs vs Sim ({result['label']})", fontsize=13)
    plt.tight_layout()
    path = os.path.join(output_dir, f"step0_detectors_{result['label']}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Step 2 Plots: Headway timeseries (C) + Fundamental diagram (D)
# ---------------------------------------------------------------------------

def plot_headway_timeseries(data, result, cfg, output_dir):
    """Plot C: Headway time series for representative vehicles (safety check)."""
    S = result["S_pred"]
    V = result["V_pred"]
    leader_idx = data["leader_idx"]
    lane_id = data["lane_id"]

    N = S.shape[1]
    T_steps = S.shape[0]
    time = np.arange(T_steps) * cfg.dt

    has_leader = leader_idx >= 0
    safe_idx = th.clamp(leader_idx, min=0)
    spacing = (S[:, safe_idx] - S).cpu().numpy()  # [T, N]

    # Pick 5 vehicles with leaders, one per lane if possible
    candidates = []
    for lane in range(cfg.num_lanes):
        lane_vehs = th.where((lane_id == lane) & has_leader)[0].cpu().numpy()
        if len(lane_vehs) > 0:
            # Pick vehicle near the middle of the platoon
            mid = lane_vehs[len(lane_vehs) // 2]
            candidates.append(mid)
    # Fill up to 5
    all_with_leader = th.where(has_leader)[0].cpu().numpy()
    for v in all_with_leader:
        if len(candidates) >= 5:
            break
        if v not in candidates:
            candidates.append(v)

    n_show = len(candidates)
    fig, axes = plt.subplots(2, n_show, figsize=(4 * n_show, 7), squeeze=False)

    for i, veh in enumerate(candidates):
        lane = int(lane_id[veh].item())

        # Headway (gap)
        ax = axes[0, i]
        ax.plot(time, spacing[:, veh], color="steelblue", linewidth=1)
        ax.axhline(cfg.s0, color="r", ls="--", alpha=0.5, label=f"s0={cfg.s0}m")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Gap (m)")
        ax.set_title(f"Veh {veh} (lane {lane})")
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        # Speed
        ax = axes[1, i]
        v_np = V[:, veh].cpu().numpy()
        leader = int(leader_idx[veh].item())
        v_leader_np = V[:, leader].cpu().numpy() if leader >= 0 else np.full_like(v_np, np.nan)
        ax.plot(time, v_np, color="steelblue", linewidth=1, label="Vehicle")
        ax.plot(time, v_leader_np, color="darkorange", linewidth=1, alpha=0.7, label="Leader")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Speed (m/s)")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Headway & Speed Time Series ({result['label']})", fontsize=13)
    plt.tight_layout()
    path = os.path.join(output_dir, f"plot_C_headway_{result['label']}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_fundamental_diagram(metrics, result, output_dir):
    """Plot D: Fundamental diagram scatter (sim vs obs)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    rho_obs = np.array(metrics.get("_fd_rho_obs", []))
    flow_obs = np.array(metrics.get("_fd_flow_obs", []))
    rho_sim = np.array(metrics.get("_fd_rho_sim", []))
    flow_sim = np.array(metrics.get("_fd_flow_sim", []))

    ax = axes[0]
    if len(rho_obs) > 0:
        ax.scatter(rho_obs, flow_obs, alpha=0.4, s=15, color="steelblue", label="Obs")
    if len(rho_sim) > 0:
        ax.scatter(rho_sim, flow_sim, alpha=0.4, s=15, color="darkorange", label="Sim")
    ax.set_xlabel("Density (veh/m)", fontsize=12)
    ax.set_ylabel("Flow (veh/s)", fontsize=12)
    ax.set_title("Fundamental Diagram", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Speed vs density
    speed_obs = np.array(flow_obs) / (np.array(rho_obs) + 1e-8) if len(rho_obs) > 0 else np.array([])
    speed_sim = np.array(flow_sim) / (np.array(rho_sim) + 1e-8) if len(rho_sim) > 0 else np.array([])

    ax = axes[1]
    if len(rho_obs) > 0:
        ax.scatter(rho_obs, speed_obs, alpha=0.4, s=15, color="steelblue", label="Obs")
    if len(rho_sim) > 0:
        ax.scatter(rho_sim, speed_sim, alpha=0.4, s=15, color="darkorange", label="Sim")
    ax.set_xlabel("Density (veh/m)", fontsize=12)
    ax.set_ylabel("Speed (m/s)", fontsize=12)
    ax.set_title("Speed–Density", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Fundamental Diagram ({result['label']})", fontsize=14)
    plt.tight_layout()
    path = os.path.join(output_dir, f"plot_D_fd_{result['label']}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Evaluation metrics (Step 2, lightweight inline version)
# ---------------------------------------------------------------------------

def compute_trajectory_metrics(data, result, cfg, obs, time_indices):
    """Compute trajectory quality metrics (Step 2)."""
    S = result["S_pred"]
    V = result["V_pred"]
    leader_idx = data["leader_idx"]
    lane_id = data["lane_id"]
    x_dets = data["x_dets"]
    xq = data["xq"]

    metrics = {}

    # 1. Collision rate / minimum gap
    has_leader = leader_idx >= 0
    safe_idx = th.clamp(leader_idx, min=0)
    spacing = S[:, safe_idx] - S
    spacing_valid = spacing[:, has_leader]
    min_gap = float(spacing_valid.min().item()) if spacing_valid.numel() > 0 else float("inf")
    collision_frac = float((spacing_valid < 2.0).sum().item()) / max(spacing_valid.numel(), 1)
    metrics["min_gap"] = min_gap
    metrics["collision_frac"] = collision_frac

    # 2. Acceleration statistics
    accel = (V[1:] - V[:-1]) / cfg.dt
    metrics["accel_rms"] = float(th.sqrt((accel ** 2).mean()).item())
    metrics["accel_p99"] = float(accel.abs().quantile(0.99).item())
    metrics["accel_max"] = float(accel.abs().max().item())

    # 3. Macro speed field error (RMSE at detector locations)
    with th.no_grad():
        pred = detector_crosslane_at_times(
            S=S, V=V, lane_id=lane_id, xq=xq, x_dets=x_dets,
            time_indices=time_indices, sigma=cfg.sigma, half_window=10.0,
        )
    speed_err = pred["speed"] - obs["speed"]
    valid = th.isfinite(speed_err)
    metrics["speed_rmse"] = float(th.sqrt((speed_err[valid] ** 2).mean()).item())
    metrics["speed_mae"] = float(speed_err[valid].abs().mean().item())

    flow_err = pred["flow"] - obs["flow"]
    valid_f = th.isfinite(flow_err)
    metrics["flow_rmse"] = float(th.sqrt((flow_err[valid_f] ** 2).mean()).item())

    # 4. Fundamental diagram plausibility
    # Compute density and flow at detector locations using the macro fields
    rho_sim = pred["flow"] / (pred["speed"] + 1e-6)  # density = flow / speed
    rho_obs = obs["flow"] / (obs["speed"] + 1e-6)
    # Check if sim (rho, flow) falls in plausible range
    rho_sim_np = rho_sim.cpu().numpy().flatten()
    flow_sim_np = pred["flow"].cpu().numpy().flatten()
    rho_obs_np = rho_obs.cpu().numpy().flatten()
    flow_obs_np = obs["flow"].cpu().numpy().flatten()
    # Filter valid
    valid_sim = np.isfinite(rho_sim_np) & np.isfinite(flow_sim_np) & (rho_sim_np > 0) & (rho_sim_np < 1.0)
    valid_obs = np.isfinite(rho_obs_np) & np.isfinite(flow_obs_np) & (rho_obs_np > 0) & (rho_obs_np < 1.0)
    metrics["fd_rho_sim_mean"] = float(rho_sim_np[valid_sim].mean()) if valid_sim.sum() > 0 else 0.0
    metrics["fd_flow_sim_mean"] = float(flow_sim_np[valid_sim].mean()) if valid_sim.sum() > 0 else 0.0
    # Store for plotting
    metrics["_fd_rho_sim"] = rho_sim_np[valid_sim].tolist() if valid_sim.sum() > 0 else []
    metrics["_fd_flow_sim"] = flow_sim_np[valid_sim].tolist() if valid_sim.sum() > 0 else []
    metrics["_fd_rho_obs"] = rho_obs_np[valid_obs].tolist() if valid_obs.sum() > 0 else []
    metrics["_fd_flow_obs"] = flow_obs_np[valid_obs].tolist() if valid_obs.sum() > 0 else []

    # 5. Wave propagation speed via cross-correlation of speed drops
    obs_speed_np = obs["speed"].cpu().numpy()  # [K, J]
    sim_speed_np = pred["speed"].cpu().numpy()
    x_dets_arr = np.array(x_dets)
    dx_det = float(x_dets_arr[1] - x_dets_arr[0]) if len(x_dets_arr) > 1 else 50.0

    def _estimate_wave_speed(speed_field, obs_int, dx):
        """Cross-correlation of speed between adjacent detectors."""
        K_, J_ = speed_field.shape
        if J_ < 2 or K_ < 4:
            return np.nan
        lags = []
        for j in range(J_ - 1):
            s1 = speed_field[:, j] - speed_field[:, j].mean()
            s2 = speed_field[:, j + 1] - speed_field[:, j + 1].mean()
            if s1.std() < 1e-6 or s2.std() < 1e-6:
                continue
            corr = np.correlate(s1, s2, mode="full")
            mid = len(corr) // 2
            # Search for peak lag in [1, K//2] (positive lag = downstream lags behind)
            search = corr[mid + 1: mid + K_ // 2]
            if len(search) == 0:
                continue
            best_lag = np.argmax(search) + 1  # in time steps
            lags.append(best_lag)
        if len(lags) == 0:
            return np.nan
        median_lag = np.median(lags)
        if median_lag < 0.5:
            return np.nan
        return dx / (median_lag * obs_int)  # m/s

    metrics["wave_speed_obs"] = float(_estimate_wave_speed(obs_speed_np, time_indices.cpu().numpy()[1] * cfg.dt - time_indices.cpu().numpy()[0] * cfg.dt if len(time_indices) > 1 else cfg.dt, dx_det))
    metrics["wave_speed_sim"] = float(_estimate_wave_speed(sim_speed_np, time_indices.cpu().numpy()[1] * cfg.dt - time_indices.cpu().numpy()[0] * cfg.dt if len(time_indices) > 1 else cfg.dt, dx_det))
    ws_obs = metrics["wave_speed_obs"]
    ws_sim = metrics["wave_speed_sim"]
    metrics["wave_speed_err"] = float(abs(ws_sim - ws_obs)) if np.isfinite(ws_sim) and np.isfinite(ws_obs) else float("nan")

    # T recovery (diagnostic only)
    T_est = result["T_est"]
    T_true = data["T_true"]
    T_est_np = T_est.cpu().numpy()
    T_true_np = T_true.cpu().numpy()
    metrics["T_Corr"] = float(np.corrcoef(T_true_np, T_est_np)[0, 1]) if T_est_np.std() > 1e-8 else 0.0
    metrics["T_MAE"] = float(np.abs(T_true_np - T_est_np).mean())

    # Learned global params
    metrics["v0_est"] = result["v0_est"]
    metrics["amax_est"] = result["amax_est"]

    return metrics


def print_metrics(metrics: dict, label: str = ""):
    """Print metrics in a formatted table."""
    tag = f"[{label}] " if label else ""
    print(f"\n{tag}Trajectory Quality Metrics:")
    print(f"  1. Collisions:    frac={metrics['collision_frac']:.6f}, min_gap={metrics['min_gap']:.2f}m")
    print(f"  2. Acceleration:  RMS={metrics['accel_rms']:.3f}, p99={metrics['accel_p99']:.3f}, "
          f"max={metrics['accel_max']:.3f} m/s²")
    print(f"  3. Speed RMSE:    {metrics['speed_rmse']:.4f} m/s  |  MAE: {metrics['speed_mae']:.4f}")
    print(f"     Flow RMSE:     {metrics['flow_rmse']:.4f}")
    ws_obs = metrics.get('wave_speed_obs', float('nan'))
    ws_sim = metrics.get('wave_speed_sim', float('nan'))
    ws_err = metrics.get('wave_speed_err', float('nan'))
    print(f"  4. FD: rho_mean={metrics.get('fd_rho_sim_mean', 0):.4f}, flow_mean={metrics.get('fd_flow_sim_mean', 0):.4f}")
    print(f"  5. Wave speed:    obs={ws_obs:.2f}, sim={ws_sim:.2f}, err={ws_err:.2f} m/s")
    print(f"  ---")
    print(f"  T_Corr:        {metrics['T_Corr']:.4f}  (diagnostic)")
    print(f"  T_MAE:         {metrics['T_MAE']:.4f}")
    print(f"  v0_est:        {metrics['v0_est']:.2f} m/s")
    print(f"  a_max_est:     {metrics['amax_est']:.2f} m/s²")


# ---------------------------------------------------------------------------
# Step 0: Look at trajectories
# ---------------------------------------------------------------------------

def do_step0(args):
    """Step 0: Run 2s det-only reconstruction and visualize."""
    device = args.device
    dtype = th.float32
    output_dir = os.path.join(args.output_dir, "step0")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Step 0: Look at the Trajectories")
    print("=" * 60)

    # Generate ground truth
    cfg = WaveScenarioConfig(detector_spacing=50)
    data = create_wave_scenario(cfg, device=device, dtype=dtype, seed=args.seed)
    N = data["s0"].shape[0]
    print(f"  Vehicles: {N}, True T: mean={data['T_true'].mean():.3f}, std={data['T_true'].std():.3f}")

    # Observations at chosen resolution
    obs_interval = args.obs_interval
    obs_data = generate_detector_observations(
        data["S"], data["V"], data["lane_id"], cfg, obs_interval=obs_interval,
    )
    obs = {"flow": obs_data["flow"], "speed": obs_data["speed"]}
    time_idx = obs_data["time_indices"]
    print(f"  Observations: dt_obs={obs_interval}s, K={time_idx.shape[0]}")

    # Run reconstruction (T only, baseline)
    result = run_reconstruction(
        data=data, cfg=cfg, obs=obs, time_indices=time_idx,
        obs_interval=obs_interval,
        learn_v0=False, learn_amax=False,
        iters=args.iters, lr=args.lr,
        label="baseline_T",
    )

    # Metrics
    metrics = compute_trajectory_metrics(data, result, cfg, obs, time_idx)
    print_metrics(metrics, "baseline_T")

    # Plots
    plot_step0(data, result, cfg, output_dir)
    plot_step0_observations(data, result, cfg, obs, time_idx, obs_interval, output_dir)

    # Save
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics to {output_dir}/metrics.json")

    # Write observation notes
    print("\n" + "=" * 60)
    print("Step 0: What to look for in the plots:")
    print("=" * 60)
    print("  1. Systematic speed bias? → v0 is wrong → go to Step 1")
    print("  2. Wave exists but wrong timing? → ghost/boundary issue")
    print("  3. Wave shape correct but too smooth? → T distribution issue")
    print("  4. Wave missing entirely? → fundamental model problem")
    print(f"\n  Plots saved to: {output_dir}/")


# ---------------------------------------------------------------------------
# Step 1: Release frozen global parameters
# ---------------------------------------------------------------------------

def do_step1(args):
    """Step 1: Ablation on learnable global params (A/B/C)."""
    device = args.device
    dtype = th.float32
    output_dir = os.path.join(args.output_dir, "step1")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Step 1: Release Frozen Global Parameters")
    print("=" * 60)

    # Generate ground truth
    cfg = WaveScenarioConfig(detector_spacing=50)
    data = create_wave_scenario(cfg, device=device, dtype=dtype, seed=args.seed)

    obs_interval = args.obs_interval
    obs_data = generate_detector_observations(
        data["S"], data["V"], data["lane_id"], cfg, obs_interval=obs_interval,
    )
    obs = {"flow": obs_data["flow"], "speed": obs_data["speed"]}
    time_idx = obs_data["time_indices"]

    configs = [
        {"label": "A_T_only",     "learn_v0": False, "learn_amax": False},
        {"label": "B_T+v0",       "learn_v0": True,  "learn_amax": False},
        {"label": "C_T+v0+amax",  "learn_v0": True,  "learn_amax": True},
    ]

    all_results = []
    all_metrics = []

    for c in configs:
        print(f"\n{'='*60}")
        print(f"Config {c['label']}")
        print(f"{'='*60}")

        result = run_reconstruction(
            data=data, cfg=cfg, obs=obs, time_indices=time_idx,
            obs_interval=obs_interval,
            learn_v0=c["learn_v0"], learn_amax=c["learn_amax"],
            iters=args.iters, lr=args.lr,
            label=c["label"],
        )

        metrics = compute_trajectory_metrics(data, result, cfg, obs, time_idx)
        print_metrics(metrics, c["label"])

        plot_step0(data, result, cfg, output_dir)
        plot_step0_observations(data, result, cfg, obs, time_idx, obs_interval, output_dir)

        all_results.append(result)
        all_metrics.append({"label": c["label"], **metrics})

    # Summary comparison
    print("\n" + "=" * 70)
    print("Step 1: Ablation Summary")
    print("=" * 70)
    header = f"{'Config':>15} | {'L_det':>8} | {'Speed RMSE':>10} | {'T_Corr':>7} | {'v0':>6} | {'a_max':>6} | {'Collisions':>10}"
    print(header)
    print("-" * 85)
    for m in all_metrics:
        print(f"{m['label']:>15} | {m['speed_rmse']:8.4f} | {m['speed_rmse']:10.4f} | "
              f"{m['T_Corr']:7.4f} | {m['v0_est']:6.2f} | {m['amax_est']:6.2f} | "
              f"{m['collision_frac']:10.6f}")
    print("=" * 70)

    # Plot comparison: L_det convergence across configs
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    colors = {"A_T_only": "steelblue", "B_T+v0": "darkorange", "C_T+v0+amax": "seagreen"}

    # L_det vs iter
    ax = axes[0]
    for r in all_results:
        ax.plot(r["history"]["L_det"], color=colors[r["label"]], linewidth=2, label=r["label"])
    ax.set_xlabel("Iteration")
    ax.set_ylabel("L_det")
    ax.set_title("Detector Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # T_Corr vs iter
    ax = axes[1]
    for r in all_results:
        ax.plot(r["history"]["T_Corr"], color=colors[r["label"]], linewidth=2, label=r["label"])
    ax.set_xlabel("Iteration")
    ax.set_ylabel("T_Corr")
    ax.set_title("T Correlation (diagnostic)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # v0 convergence
    ax = axes[2]
    for r in all_results:
        ax.plot(r["history"]["v0_est"], color=colors[r["label"]], linewidth=2, label=r["label"])
    ax.axhline(cfg.v_free, color="k", ls="--", alpha=0.5, label=f"cfg.v_free={cfg.v_free}")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("v0 (m/s)")
    ax.set_title("Learned v0")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "step1_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")

    # Save metrics
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Saved: {output_dir}/metrics.json")


# ---------------------------------------------------------------------------
# Step 2: Build minimal trajectory evaluation
# ---------------------------------------------------------------------------

def do_step2(args):
    """Step 2: Full eval suite on a single reconstruction (best config from Step 1)."""
    device = args.device
    dtype = th.float32
    output_dir = os.path.join(args.output_dir, "step2")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Step 2: Minimal Trajectory Evaluation")
    print("=" * 60)

    cfg = WaveScenarioConfig(detector_spacing=50)
    data = create_wave_scenario(cfg, device=device, dtype=dtype, seed=args.seed)

    obs_interval = args.obs_interval
    obs_data = generate_detector_observations(
        data["S"], data["V"], data["lane_id"], cfg, obs_interval=obs_interval,
    )
    obs = {"flow": obs_data["flow"], "speed": obs_data["speed"]}
    time_idx = obs_data["time_indices"]

    # Run best config (T + v0 + a_max from Step 1)
    result = run_reconstruction(
        data=data, cfg=cfg, obs=obs, time_indices=time_idx,
        obs_interval=obs_interval,
        learn_v0=True, learn_amax=True,
        iters=args.iters, lr=args.lr,
        label="best_C",
    )

    # Full metrics
    metrics = compute_trajectory_metrics(data, result, cfg, obs, time_idx)
    print_metrics(metrics, "best_C")

    # All 4 plots
    plot_step0(data, result, cfg, output_dir)                          # A + B
    plot_headway_timeseries(data, result, cfg, output_dir)             # C
    plot_fundamental_diagram(metrics, result, output_dir)              # D

    # Save
    metrics_save = {k: v for k, v in metrics.items() if not k.startswith("_")}
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics_save, f, indent=2)
    print(f"\nSaved: {output_dir}/metrics.json")


# ---------------------------------------------------------------------------
# Step 3: Reassess L_arr with trajectory metrics
# ---------------------------------------------------------------------------

def _run_recovery_with_arr(
    data, cfg, obs, time_indices, obs_interval, *,
    use_arrspread, beta, iters, lr, warmup_frac, L_det_guard,
    dv_thresh, tau, label,
):
    """Recovery loop with optional L_arr (reuses run_2s_recovery pattern)."""
    device = data["s0"].device
    dtype = data["s0"].dtype
    N = data["s0"].shape[0]

    s0 = data["s0"]
    v0 = data["v0"]
    lane_id = data["lane_id"]
    leader_idx = data["leader_idx"]
    xq = data["xq"]
    x_dets = data["x_dets"]
    T_true = data["T_true"]

    steps = int(cfg.duration_s / cfg.dt)
    T_min, T_max = cfg.T_min, cfg.T_max

    # Learnable params: T + v0 + a_max (best config from Step 1)
    u_T = th.nn.Parameter(inverse_sigmoid_reparam(
        th.full((N,), cfg.T_mean, device=device, dtype=dtype), T_min, T_max,
    ))
    v0_lo, v0_hi = 15.0, 40.0
    u_v0 = th.nn.Parameter(inverse_sigmoid_reparam(
        th.tensor(cfg.v_free, device=device, dtype=dtype), v0_lo, v0_hi,
    ))
    amax_lo, amax_hi = 0.5, 3.0
    u_amax = th.nn.Parameter(inverse_sigmoid_reparam(
        th.tensor(cfg.a_max, device=device, dtype=dtype), amax_lo, amax_hi,
    ))
    opt = th.optim.Adam([u_T, u_v0, u_amax], lr=lr)
    b_vec = th.full((N,), cfg.b_comfort, device=device, dtype=dtype)

    if use_arrspread:
        with th.no_grad():
            arr_obs = compute_arrival_spread(obs["speed"], obs_interval, cfg, dv_thresh, tau)
        std_arr_obs = arr_obs["std_arr"].detach()

    warmup_iters = int(iters * warmup_frac)
    L_det_baseline = None
    current_beta = beta

    history = {"loss": [], "L_det": [], "L_arrspread": [],
               "T_Corr": [], "T_MAE": [], "T_std": [],
               "v0_est": [], "amax_est": []}

    tag = f"[{label}]"
    print(f"{tag} Recovery: obs_int={obs_interval}s, arrspread={use_arrspread}, iters={iters}")

    T_true_np = T_true.cpu().numpy()

    for i in range(iters):
        opt.zero_grad()
        T = sigmoid_reparam(u_T, T_min, T_max)
        v0_val = sigmoid_reparam(u_v0, v0_lo, v0_hi)
        amax_val = sigmoid_reparam(u_amax, amax_lo, amax_hi)

        S, V, _ = rollout_idm_multilane(
            s0_init=s0, v0_init=v0, leader_idx=leader_idx,
            num_steps=steps, dt=cfg.dt,
            a_max=amax_val.expand(N), b=b_vec, v_target=v0_val.expand(N),
            s0=cfg.s0, T_headway=T,
            ghost_v=cfg.ghost_v_base, ghost_gap0=cfg.ghost_gap0,
        )
        pred = detector_crosslane_at_times(
            S=S, V=V, lane_id=lane_id, xq=xq, x_dets=x_dets,
            time_indices=time_indices, sigma=cfg.sigma, half_window=10.0,
        )
        L_det = detector_loss(
            pred=pred, obs=obs, weights={"flow": 1.0, "speed": 0.1},
            loss_type="huber", huber_delta=1.0,
        )

        if i == warmup_iters - 1:
            L_det_baseline = L_det.item()

        use_arr_now = use_arrspread and (i >= warmup_iters)
        if use_arr_now:
            arr_sim = compute_arrival_spread(pred["speed"], obs_interval, cfg, dv_thresh, tau)
            L_arr = (arr_sim["std_arr"] - std_arr_obs) ** 2
            if L_det_baseline is not None and L_det.item() > L_det_guard * L_det_baseline:
                current_beta *= 0.5
        else:
            L_arr = th.tensor(0.0, device=device, dtype=dtype)

        loss = L_det + current_beta * L_arr
        if not th.isfinite(loss):
            break
        loss.backward()
        opt.step()

        with th.no_grad():
            T_np = T.cpu().numpy()
            corr = float(np.corrcoef(T_true_np, T_np)[0, 1]) if T_np.std() > 1e-8 else 0.0
            mae = float(np.abs(T_true_np - T_np).mean())

        history["loss"].append(float(loss.item()))
        history["L_det"].append(float(L_det.item()))
        history["L_arrspread"].append(float(L_arr.item()))
        history["T_Corr"].append(corr)
        history["T_MAE"].append(mae)
        history["T_std"].append(float(T.std().item()))
        history["v0_est"].append(float(v0_val.item()))
        history["amax_est"].append(float(amax_val.item()))

        if i % 20 == 0 or i == iters - 1:
            print(f"{tag}   iter {i:3d}: L_det={L_det.item():.6f}, T_Corr={corr:.4f}, v0={v0_val.item():.2f}")

    with th.no_grad():
        T_est = sigmoid_reparam(u_T, T_min, T_max)
        v0_final = sigmoid_reparam(u_v0, v0_lo, v0_hi)
        amax_final = sigmoid_reparam(u_amax, amax_lo, amax_hi)
        S_pred, V_pred, _ = rollout_idm_multilane(
            s0_init=s0, v0_init=v0, leader_idx=leader_idx,
            num_steps=steps, dt=cfg.dt,
            a_max=amax_final.expand(N), b=b_vec, v_target=v0_final.expand(N),
            s0=cfg.s0, T_headway=T_est,
            ghost_v=cfg.ghost_v_base, ghost_gap0=cfg.ghost_gap0,
        )

    return {
        "T_est": T_est, "v0_est": float(v0_final.item()),
        "amax_est": float(amax_final.item()),
        "S_pred": S_pred, "V_pred": V_pred,
        "history": history, "label": label,
    }


def do_step3(args):
    """Step 3: Reassess L_arr — does it improve trajectory quality?"""
    device = args.device
    dtype = th.float32
    output_dir = os.path.join(args.output_dir, "step3")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Step 3: Reassess L_arr with Trajectory Metrics")
    print("=" * 60)

    cfg = WaveScenarioConfig(detector_spacing=50)
    data = create_wave_scenario(cfg, device=device, dtype=dtype, seed=args.seed)

    # Generate observations at both resolutions
    obs_5s = generate_detector_observations(data["S"], data["V"], data["lane_id"], cfg, obs_interval=5.0)
    obs_2s = generate_detector_observations(data["S"], data["V"], data["lane_id"], cfg, obs_interval=2.0)

    conditions = [
        {"label": "5s_det",      "obs": {"flow": obs_5s["flow"], "speed": obs_5s["speed"]},
         "time_idx": obs_5s["time_indices"], "obs_interval": 5.0, "use_arrspread": False},
        {"label": "5s_det+arr",  "obs": {"flow": obs_5s["flow"], "speed": obs_5s["speed"]},
         "time_idx": obs_5s["time_indices"], "obs_interval": 5.0, "use_arrspread": True},
        {"label": "2s_det",      "obs": {"flow": obs_2s["flow"], "speed": obs_2s["speed"]},
         "time_idx": obs_2s["time_indices"], "obs_interval": 2.0, "use_arrspread": False},
        {"label": "2s_det+arr",  "obs": {"flow": obs_2s["flow"], "speed": obs_2s["speed"]},
         "time_idx": obs_2s["time_indices"], "obs_interval": 2.0, "use_arrspread": True},
    ]

    all_results = []
    all_metrics = []

    for c in conditions:
        print(f"\n{'='*60}")
        print(f"Condition: {c['label']}")
        print(f"{'='*60}")

        result = _run_recovery_with_arr(
            data, cfg, c["obs"], c["time_idx"], c["obs_interval"],
            use_arrspread=c["use_arrspread"],
            beta=args.beta, iters=args.iters, lr=args.lr,
            warmup_frac=0.3, L_det_guard=1.5,
            dv_thresh=5.0, tau=1.0,
            label=c["label"],
        )
        metrics = compute_trajectory_metrics(data, result, cfg, c["obs"], c["time_idx"])
        print_metrics(metrics, c["label"])

        # Plots C and D for each condition
        plot_headway_timeseries(data, result, cfg, output_dir)
        plot_fundamental_diagram(metrics, result, output_dir)

        all_results.append(result)
        metrics_clean = {k: v for k, v in metrics.items() if not k.startswith("_")}
        metrics_clean["label"] = c["label"]
        all_metrics.append(metrics_clean)

    # Summary table
    print("\n" + "=" * 90)
    print("Step 3: L_arr Impact on Trajectory Quality")
    print("=" * 90)
    header = (f"{'Condition':>14} | {'Speed RMSE':>10} | {'Flow RMSE':>9} | "
              f"{'Wave Spd Err':>12} | {'T_Corr':>7} | {'v0':>6} | {'Accel p99':>9}")
    print(header)
    print("-" * 90)
    for m in all_metrics:
        ws_err = m.get("wave_speed_err", float("nan"))
        ws_str = f"{ws_err:12.2f}" if np.isfinite(ws_err) else f"{'N/A':>12}"
        print(f"{m['label']:>14} | {m['speed_rmse']:10.4f} | {m['flow_rmse']:9.4f} | "
              f"{ws_str} | {m['T_Corr']:7.4f} | {m['v0_est']:6.2f} | {m['accel_p99']:9.3f}")
    print("=" * 90)

    # Interpretation
    def _get(label, key):
        for m in all_metrics:
            if m["label"] == label:
                return m.get(key, float("nan"))
        return float("nan")

    print("\n--- Interpretation ---")
    # Does L_arr improve trajectory quality at 5s?
    srmse_5s = _get("5s_det", "speed_rmse")
    srmse_5s_arr = _get("5s_det+arr", "speed_rmse")
    print(f"  5s: Speed RMSE {srmse_5s:.4f} → {srmse_5s_arr:.4f} (Δ={srmse_5s_arr - srmse_5s:+.4f})")

    srmse_2s = _get("2s_det", "speed_rmse")
    srmse_2s_arr = _get("2s_det+arr", "speed_rmse")
    print(f"  2s: Speed RMSE {srmse_2s:.4f} → {srmse_2s_arr:.4f} (Δ={srmse_2s_arr - srmse_2s:+.4f})")

    # Decision
    if srmse_5s_arr < srmse_5s - 0.01 or srmse_2s_arr < srmse_2s - 0.01:
        print("  → L_arr IMPROVES trajectory quality. Keep it.")
    elif abs(srmse_5s_arr - srmse_5s) < 0.01 and abs(srmse_2s_arr - srmse_2s) < 0.01:
        print("  → L_arr has NO trajectory-level impact. Drop it for simplicity.")
    else:
        print("  → L_arr WORSENS trajectory quality. Drop it.")

    # Save
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nSaved: {output_dir}/metrics.json")


# ---------------------------------------------------------------------------
# Step 4: Resolution scaling curve
# ---------------------------------------------------------------------------

def do_step4(args):
    """Step 4: Resolution scaling curve."""
    device = args.device
    dtype = th.float32
    output_dir = os.path.join(args.output_dir, "step4")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Step 4: Resolution Scaling Curve")
    print("=" * 60)

    dt_obs_list = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0]

    # Generate ground truth once
    cfg = WaveScenarioConfig(detector_spacing=50)
    data = create_wave_scenario(cfg, device=device, dtype=dtype, seed=args.seed)

    all_metrics = []

    for dt_obs in dt_obs_list:
        print(f"\n{'='*60}")
        print(f"dt_obs = {dt_obs:.0f}s")
        print(f"{'='*60}")

        obs_data = generate_detector_observations(
            data["S"], data["V"], data["lane_id"], cfg, obs_interval=dt_obs,
        )
        obs = {"flow": obs_data["flow"], "speed": obs_data["speed"]}
        time_idx = obs_data["time_indices"]

        result = run_reconstruction(
            data=data, cfg=cfg, obs=obs, time_indices=time_idx,
            obs_interval=dt_obs,
            learn_v0=args.learn_v0, learn_amax=args.learn_amax,
            iters=args.iters, lr=args.lr,
            label=f"dt{dt_obs:.0f}s",
        )

        metrics = compute_trajectory_metrics(data, result, cfg, obs, time_idx)
        metrics["dt_obs"] = dt_obs
        metrics["K"] = int(time_idx.shape[0])
        print_metrics(metrics, f"dt_obs={dt_obs:.0f}s")

        all_metrics.append(metrics)

    # Plot resolution scaling curve
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    dt_vals = [m["dt_obs"] for m in all_metrics]

    metric_plots = [
        (axes[0, 0], "speed_rmse", "Speed RMSE (m/s)"),
        (axes[0, 1], "speed_mae", "Speed MAE (m/s)"),
        (axes[0, 2], "flow_rmse", "Flow RMSE"),
        (axes[1, 0], "T_Corr", "T_Corr (diagnostic)"),
        (axes[1, 1], "accel_rms", "Accel RMS (m/s²)"),
        (axes[1, 2], "collision_frac", "Collision Fraction"),
    ]

    for ax, key, ylabel in metric_plots:
        vals = [m[key] for m in all_metrics]
        ax.plot(dt_vals, vals, "o-", color="steelblue", linewidth=2, markersize=8)
        ax.set_xlabel("dt_obs (s)", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(ylabel, fontsize=13)
        ax.set_xticks(dt_vals)
        ax.grid(True, alpha=0.3)

    learn_str = "T"
    if args.learn_v0:
        learn_str += "+v0"
    if args.learn_amax:
        learn_str += "+amax"
    fig.suptitle(f"Resolution Scaling Curve (learnable: {learn_str})", fontsize=14)
    plt.tight_layout()
    path = os.path.join(output_dir, "step4_resolution_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\nSaved: {path}")

    # Save
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Saved: {output_dir}/metrics.json")

    # Summary table
    print("\n" + "=" * 80)
    print("Resolution Scaling: Summary")
    print("=" * 80)
    header = f"{'dt_obs':>6} | {'K':>4} | {'Speed RMSE':>10} | {'Flow RMSE':>9} | {'T_Corr':>7} | {'Accel RMS':>9} | {'Collisions':>10}"
    print(header)
    print("-" * 80)
    for m in all_metrics:
        print(f"{m['dt_obs']:6.0f} | {m['K']:4d} | {m['speed_rmse']:10.4f} | "
              f"{m['flow_rmse']:9.4f} | {m['T_Corr']:7.4f} | "
              f"{m['accel_rms']:9.3f} | {m['collision_frac']:10.6f}")
    print("=" * 80)


# ---------------------------------------------------------------------------
# Step 5: Per-lane vs cross-lane observation experiment
# ---------------------------------------------------------------------------

def _run_reconstruction_perlane(
    data, cfg, obs_perlane, time_indices, obs_interval, *,
    iters, lr, label,
):
    """Recovery loop using per-lane observations (upper-bound reference)."""
    device = data["s0"].device
    dtype = data["s0"].dtype
    N = data["s0"].shape[0]

    s0 = data["s0"]
    v0 = data["v0"]
    lane_id = data["lane_id"]
    leader_idx = data["leader_idx"]
    xq = data["xq"]
    x_dets = data["x_dets"]
    T_true = data["T_true"]

    steps = int(cfg.duration_s / cfg.dt)
    T_min, T_max = cfg.T_min, cfg.T_max

    # Learnable: T + v0 + a_max (best config)
    u_T = th.nn.Parameter(inverse_sigmoid_reparam(
        th.full((N,), cfg.T_mean, device=device, dtype=dtype), T_min, T_max,
    ))
    v0_lo, v0_hi = 15.0, 40.0
    u_v0 = th.nn.Parameter(inverse_sigmoid_reparam(
        th.tensor(cfg.v_free, device=device, dtype=dtype), v0_lo, v0_hi,
    ))
    amax_lo, amax_hi = 0.5, 3.0
    u_amax = th.nn.Parameter(inverse_sigmoid_reparam(
        th.tensor(cfg.a_max, device=device, dtype=dtype), amax_lo, amax_hi,
    ))
    opt = th.optim.Adam([u_T, u_v0, u_amax], lr=lr)
    b_vec = th.full((N,), cfg.b_comfort, device=device, dtype=dtype)

    history = {"loss": [], "L_det": [], "T_Corr": [], "T_MAE": [], "T_std": [],
               "v0_est": [], "amax_est": []}

    tag = f"[{label}]"
    print(f"{tag} Per-lane reconstruction: obs_int={obs_interval}s, iters={iters}")

    T_true_np = T_true.cpu().numpy()

    for i in range(iters):
        opt.zero_grad()
        T = sigmoid_reparam(u_T, T_min, T_max)
        v0_val = sigmoid_reparam(u_v0, v0_lo, v0_hi)
        amax_val = sigmoid_reparam(u_amax, amax_lo, amax_hi)

        S, V, _ = rollout_idm_multilane(
            s0_init=s0, v0_init=v0, leader_idx=leader_idx,
            num_steps=steps, dt=cfg.dt,
            a_max=amax_val.expand(N), b=b_vec, v_target=v0_val.expand(N),
            s0=cfg.s0, T_headway=T,
            ghost_v=cfg.ghost_v_base, ghost_gap0=cfg.ghost_gap0,
        )

        # Per-lane detector prediction
        pred = detector_perlane_at_times(
            S=S, V=V, lane_id=lane_id, xq=xq, x_dets=x_dets,
            time_indices=time_indices, sigma=cfg.sigma, half_window=10.0,
        )

        loss = detector_loss(
            pred=pred, obs=obs_perlane,
            weights={"flow": 1.0, "speed": 0.1},
            loss_type="huber", huber_delta=1.0,
        )

        if not th.isfinite(loss):
            print(f"{tag}   NaN at iter {i}, stopping")
            break
        loss.backward()
        opt.step()

        with th.no_grad():
            T_np = T.cpu().numpy()
            corr = float(np.corrcoef(T_true_np, T_np)[0, 1]) if T_np.std() > 1e-8 else 0.0
            mae = float(np.abs(T_true_np - T_np).mean())

        history["loss"].append(float(loss.item()))
        history["L_det"].append(float(loss.item()))
        history["T_Corr"].append(corr)
        history["T_MAE"].append(mae)
        history["T_std"].append(float(T.std().item()))
        history["v0_est"].append(float(v0_val.item()))
        history["amax_est"].append(float(amax_val.item()))

        if i % 20 == 0 or i == iters - 1:
            print(f"{tag}   iter {i:3d}: L_det={loss.item():.6f}, "
                  f"T_Corr={corr:.4f}, v0={v0_val.item():.2f}, a_max={amax_val.item():.2f}")

    with th.no_grad():
        T_est = sigmoid_reparam(u_T, T_min, T_max)
        v0_final = sigmoid_reparam(u_v0, v0_lo, v0_hi)
        amax_final = sigmoid_reparam(u_amax, amax_lo, amax_hi)
        S_pred, V_pred, _ = rollout_idm_multilane(
            s0_init=s0, v0_init=v0, leader_idx=leader_idx,
            num_steps=steps, dt=cfg.dt,
            a_max=amax_final.expand(N), b=b_vec, v_target=v0_final.expand(N),
            s0=cfg.s0, T_headway=T_est,
            ghost_v=cfg.ghost_v_base, ghost_gap0=cfg.ghost_gap0,
        )

    return {
        "T_est": T_est, "v0_est": float(v0_final.item()),
        "amax_est": float(amax_final.item()),
        "S_pred": S_pred, "V_pred": V_pred,
        "history": history, "label": label,
    }


def _generate_perlane_obs(S, V, lane_id, cfg, obs_interval):
    """Generate per-lane detector observations from ground-truth trajectories."""
    T_steps, N = S.shape
    dt = cfg.dt
    total_time = (T_steps - 1) * dt
    obs_times = th.arange(0, total_time, obs_interval)
    time_indices = (obs_times / dt).long()
    time_indices = time_indices[time_indices < T_steps]

    obs = detector_perlane_at_times(
        S=S, V=V, lane_id=lane_id,
        xq=th.linspace(cfg.x_min, cfg.x_max, 100, device=S.device, dtype=S.dtype),
        x_dets=cfg.detector_positions,
        time_indices=time_indices,
        sigma=cfg.sigma, half_window=10.0,
    )
    return obs, time_indices


def do_step5(args):
    """Step 5: Per-lane vs cross-lane observation (information ceiling diagnostic)."""
    device = args.device
    dtype = th.float32
    output_dir = os.path.join(args.output_dir, "step5")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Step 5: Per-lane vs Cross-lane Observation")
    print("=" * 60)

    cfg = WaveScenarioConfig(detector_spacing=50)
    data = create_wave_scenario(cfg, device=device, dtype=dtype, seed=args.seed)
    N = data["s0"].shape[0]
    num_lanes = cfg.num_lanes
    Nd = len(cfg.detector_positions)
    print(f"  Vehicles: {N}, Detectors: {Nd}, Lanes: {num_lanes}")
    print(f"  Cross-lane obs shape: [K, {Nd}]")
    print(f"  Per-lane obs shape:   [K, {Nd * num_lanes}]")

    conditions = []

    # Test at both 2s and 5s
    for obs_interval in [2.0, 5.0]:
        tag_t = f"{int(obs_interval)}s"

        # --- Cross-lane (baseline, same as Steps 1-4) ---
        obs_cl = generate_detector_observations(
            data["S"], data["V"], data["lane_id"], cfg, obs_interval=obs_interval,
        )
        obs_crosslane = {"flow": obs_cl["flow"], "speed": obs_cl["speed"]}
        time_idx_cl = obs_cl["time_indices"]

        print(f"\n{'='*60}")
        print(f"{tag_t} cross-lane")
        print(f"{'='*60}")
        result_cl = run_reconstruction(
            data=data, cfg=cfg, obs=obs_crosslane, time_indices=time_idx_cl,
            obs_interval=obs_interval,
            learn_v0=True, learn_amax=True,
            iters=args.iters, lr=args.lr,
            label=f"{tag_t}_crosslane",
        )
        metrics_cl = compute_trajectory_metrics(data, result_cl, cfg, obs_crosslane, time_idx_cl)
        print_metrics(metrics_cl, f"{tag_t}_crosslane")

        # --- Per-lane ---
        obs_perlane, time_idx_pl = _generate_perlane_obs(
            data["S"], data["V"], data["lane_id"], cfg, obs_interval,
        )

        print(f"\n{'='*60}")
        print(f"{tag_t} per-lane")
        print(f"{'='*60}")
        result_pl = _run_reconstruction_perlane(
            data, cfg, obs_perlane, time_idx_pl, obs_interval,
            iters=args.iters, lr=args.lr,
            label=f"{tag_t}_perlane",
        )
        # Evaluate with cross-lane metrics for fair comparison
        metrics_pl = compute_trajectory_metrics(data, result_pl, cfg, obs_crosslane, time_idx_cl)
        print_metrics(metrics_pl, f"{tag_t}_perlane")

        conditions.append({**{k: v for k, v in metrics_cl.items() if not k.startswith("_")},
                           "label": f"{tag_t}_crosslane"})
        conditions.append({**{k: v for k, v in metrics_pl.items() if not k.startswith("_")},
                           "label": f"{tag_t}_perlane"})

        # Plots for per-lane result
        plot_step0(data, result_pl, cfg, output_dir)
        plot_headway_timeseries(data, result_pl, cfg, output_dir)

    # Summary table
    print("\n" + "=" * 90)
    print("Step 5: Per-lane vs Cross-lane — Trajectory Quality")
    print("=" * 90)
    header = (f"{'Condition':>18} | {'Speed RMSE':>10} | {'Flow RMSE':>9} | "
              f"{'T_Corr':>7} | {'T_MAE':>7} | {'v0':>6} | {'a_max':>6}")
    print(header)
    print("-" * 90)
    for m in conditions:
        print(f"{m['label']:>18} | {m['speed_rmse']:10.4f} | {m['flow_rmse']:9.4f} | "
              f"{m['T_Corr']:7.4f} | {m['T_MAE']:7.4f} | {m['v0_est']:6.2f} | {m['amax_est']:6.2f}")
    print("=" * 90)

    # Interpretation
    def _get(label, key):
        for m in conditions:
            if m["label"] == label:
                return m.get(key, float("nan"))
        return float("nan")

    print("\n--- Interpretation ---")
    for t_tag in ["2s", "5s"]:
        cl = _get(f"{t_tag}_crosslane", "speed_rmse")
        pl = _get(f"{t_tag}_perlane", "speed_rmse")
        tcorr_cl = _get(f"{t_tag}_crosslane", "T_Corr")
        tcorr_pl = _get(f"{t_tag}_perlane", "T_Corr")
        print(f"  {t_tag}: Speed RMSE  {cl:.4f} → {pl:.4f} (Δ={pl - cl:+.4f})")
        print(f"  {t_tag}: T_Corr      {tcorr_cl:.4f} → {tcorr_pl:.4f} (Δ={tcorr_pl - tcorr_cl:+.4f})")

    cl2 = _get("2s_crosslane", "speed_rmse")
    pl2 = _get("2s_perlane", "speed_rmse")
    if pl2 < cl2 - 0.02:
        print("\n  → Per-lane observation IMPROVES reconstruction.")
        print("    Cross-lane aggregation is an information bottleneck.")
    elif abs(pl2 - cl2) < 0.02:
        print("\n  → Per-lane observation has MINIMAL impact.")
        print("    Cross-lane aggregation is NOT the bottleneck.")
    else:
        print("\n  → Per-lane observation WORSENS reconstruction.")
        print("    More observation channels = harder optimization without proportional info gain.")

    # Save
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(conditions, f, indent=2)
    print(f"\nSaved: {output_dir}/metrics.json")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Trajectory Reconstruction")
    parser.add_argument("--step", type=int, required=True, choices=[0, 1, 2, 3, 4, 5],
                        help="Which step to run (0-5)")
    parser.add_argument("--device", default="cuda" if th.cuda.is_available() else "cpu")
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=1.0,
                        help="L_arr weight for step 3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--obs-interval", type=float, default=2.0,
                        help="Observation interval in seconds (default: 2s)")
    parser.add_argument("--output-dir", default="outputs/reconstruct")
    parser.add_argument("--learn-v0", action="store_true",
                        help="Learn global v0 (for step 4)")
    parser.add_argument("--learn-amax", action="store_true",
                        help="Learn global a_max (for step 4)")
    args = parser.parse_args()

    if args.step == 0:
        do_step0(args)
    elif args.step == 1:
        do_step1(args)
    elif args.step == 2:
        do_step2(args)
    elif args.step == 3:
        do_step3(args)
    elif args.step == 4:
        do_step4(args)
    elif args.step == 5:
        do_step5(args)


if __name__ == "__main__":
    main()
