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
from src.pde_constraint import aggregate_macro_fields, continuity_loss
from src.aggregation import GraphAggregator
from src.aggregation.wave_energy import (
    compute_wave_energy_direct,
    detect_wave_active_windows,
    wave_energy_loss_correlation,
)
from src.aggregation.lag_speed import (
    detect_wave_windows,
    estimate_wave_speed,
    lag_loss,
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


def fit_T_with_pde(
    data: dict,
    cfg: WaveScenarioConfig,
    lambda_T: float = 0.01,
    beta_cont: float = 0.001,
    iters: int = 100,
    lr: float = 0.05,
    boundary_cells: int = 5,
    warmup_frac: float = 0.3,
):
    """Fit T parameters with PDE (continuity) constraint."""
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

    # Compute dx from spatial grid
    dx = float(xq[1] - xq[0])

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

    history = {
        "loss": [], "L_det": [], "L_graph": [], "L_cont": [],
        "T_std": [],
    }

    warmup_iters = int(iters * warmup_frac)
    print(f"Fitting T with PDE constraint: lambda_T={lambda_T}, beta_cont={beta_cont}")
    print(f"  Stage A (L_det only): iters 0-{warmup_iters-1}")
    print(f"  Stage B (+ continuity): iters {warmup_iters}-{iters-1}")
    print(f"  dx={dx:.2f}m, dt={cfg.dt}s, boundary_cells={boundary_cells}")

    for i in range(iters):
        opt.zero_grad()

        use_pde = (i >= warmup_iters)

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

        # Detector loss
        L_det = detector_loss(
            pred=pred, obs=obs,
            weights={"flow": 1.0, "speed": 0.1},
            loss_type="huber", huber_delta=1.0,
        )

        # PDE (continuity) constraint (after warmup)
        if use_pde:
            macro = aggregate_macro_fields(S, V, lane_id, xq, sigma=cfg.sigma)
            L_cont = continuity_loss(
                macro["rho"], macro["q"],
                dt=cfg.dt, dx=dx, boundary_cells=boundary_cells,
            )
        else:
            L_cont = th.tensor(0.0, device=device, dtype=dtype)

        # Graph regularization
        L_graph = laplacian_penalty(T, leader_idx, lane_id)

        # Combined loss
        loss = L_det + beta_cont * L_cont + lambda_T * L_graph

        if not th.isfinite(loss):
            print(f"  NaN at iter {i}, stopping")
            break

        loss.backward()
        opt.step()

        # Record history
        history["loss"].append(float(loss.item()))
        history["L_det"].append(float(L_det.item()))
        history["L_graph"].append(float(L_graph.item()))
        history["L_cont"].append(float(L_cont.item()) if th.is_tensor(L_cont) else 0.0)
        history["T_std"].append(float(T.std().item()))

        if i % 20 == 0:
            stage = "B" if use_pde else "A"
            L_cont_val = float(L_cont.item()) if th.is_tensor(L_cont) else 0.0
            print(f"  [{stage}] iter {i:3d}: loss={loss.item():.4f}, L_det={L_det.item():.4f}, "
                  f"L_cont={L_cont_val:.4f}, L_graph={L_graph.item():.4f}, T_std={T.std().item():.3f}")

    # Final T
    with th.no_grad():
        T_est = sigmoid_reparam(u_T, T_min, T_max)

    return T_est, history


def fit_T_with_graph(
    data: dict,
    cfg: WaveScenarioConfig,
    lambda_T: float = 0.0,  # Default OFF per Phase 8
    beta_E: float = 0.1,    # Wave energy loss weight
    iters: int = 100,
    lr: float = 0.05,
    epsilon: float = 0.1,   # Graph filter smoothing
    warmup_frac: float = 0.3,
    L_det_guard: float = 2.0,  # Guardrail: reduce beta if L_det > guard * baseline
):
    """Fit T parameters with Graph aggregation + Wave Energy supervision."""
    device = data["s0"].device
    dtype = data["s0"].dtype
    N = data["s0"].shape[0]

    s0 = data["s0"]
    v0 = data["v0"]
    lane_id = data["lane_id"]
    leader_idx = data["leader_idx"]
    x_dets = data["x_dets"]
    time_idx = data["time_idx_5s"]
    obs = data["obs_5s"]

    steps = int(cfg.duration_s / cfg.dt)
    dx = cfg.detector_spacing

    # Build graph aggregator
    x_dets_t = th.tensor(x_dets, device=device, dtype=dtype)
    agg = GraphAggregator(x_dets_t, dx, epsilon=epsilon, device=device, dtype=dtype)

    # Precompute observed wave energy
    V_obs = obs["speed"]  # [K, J]
    E_obs = compute_wave_energy_direct(V_obs)
    wave_mask = detect_wave_active_windows(E_obs, quantile=0.7)
    print(f"  Wave-active windows: {wave_mask.sum().item()}/{len(wave_mask)} time points")

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

    history = {
        "loss": [], "L_det": [], "L_E": [], "L_graph": [],
        "T_std": [], "E_corr": [],
    }

    warmup_iters = int(iters * warmup_frac)
    L_det_baseline = None
    current_beta_E = beta_E

    print(f"Fitting T with Graph aggregation + Wave Energy:")
    print(f"  epsilon={epsilon}, beta_E={beta_E}, lambda_T={lambda_T}")
    print(f"  Stage A (L_det only): iters 0-{warmup_iters-1}")
    print(f"  Stage B (+ energy): iters {warmup_iters}-{iters-1}")

    for i in range(iters):
        opt.zero_grad()

        use_energy = (i >= warmup_iters)

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

        # Graph aggregation
        macro = agg(S, V)
        V_sim_full = macro["speed"]  # [T, J]
        V_sim = V_sim_full[time_idx]  # [K, J]

        # Detector loss (Huber on speed, same as KDE baseline)
        diff = V_sim - V_obs
        huber_delta = 1.0
        abs_diff = th.abs(diff)
        L_det = th.where(
            abs_diff <= huber_delta,
            0.5 * diff ** 2,
            huber_delta * (abs_diff - 0.5 * huber_delta)
        ).mean()

        # Record baseline L_det for guardrail
        if i == warmup_iters - 1:
            L_det_baseline = L_det.item()

        # Wave energy loss (after warmup)
        if use_energy:
            E_sim = compute_wave_energy_direct(V_sim)
            L_E = wave_energy_loss_correlation(E_sim, E_obs, wave_mask)

            # Guardrail: reduce beta if L_det degrades
            if L_det_baseline is not None and L_det.item() > L_det_guard * L_det_baseline:
                current_beta_E = current_beta_E * 0.5
                if i % 20 == 0:
                    print(f"  [GUARD] L_det={L_det.item():.4f} > {L_det_guard}x baseline, reducing beta_E to {current_beta_E:.4f}")
        else:
            L_E = th.tensor(0.0, device=device, dtype=dtype)
            E_sim = compute_wave_energy_direct(V_sim)

        # Graph regularization on T (weak or off)
        if lambda_T > 0:
            L_graph = laplacian_penalty(T, leader_idx, lane_id)
        else:
            L_graph = th.tensor(0.0, device=device, dtype=dtype)

        # Combined loss
        loss = L_det + current_beta_E * L_E + lambda_T * L_graph

        if not th.isfinite(loss):
            print(f"  NaN at iter {i}, stopping")
            break

        loss.backward()
        opt.step()

        # Compute energy correlation for monitoring
        with th.no_grad():
            E_corr = th.corrcoef(th.stack([E_sim.detach(), E_obs]))[0, 1].item()

        # Record history
        history["loss"].append(float(loss.item()))
        history["L_det"].append(float(L_det.item()))
        history["L_E"].append(float(L_E.item()) if use_energy else 0.0)
        history["L_graph"].append(float(L_graph.item()) if lambda_T > 0 else 0.0)
        history["T_std"].append(float(T.std().item()))
        history["E_corr"].append(E_corr)

        if i % 20 == 0:
            stage = "B" if use_energy else "A"
            print(f"  [{stage}] iter {i:3d}: loss={loss.item():.4f}, L_det={L_det.item():.4f}, "
                  f"L_E={L_E.item():.4f}, E_corr={E_corr:.3f}, T_std={T.std().item():.3f}")

    # Final T
    with th.no_grad():
        T_est = sigmoid_reparam(u_T, T_min, T_max)

    return T_est, history


def fit_T_with_lag_speed(
    data: dict,
    cfg: WaveScenarioConfig,
    lambda_T: float = 0.0,
    beta_lag: float = 0.1,
    iters: int = 100,
    lr: float = 0.05,
    warmup_frac: float = 0.3,
    lag_max: int = 6,
    skip: int = 3,
    L_det_guard: float = 2.0,
):
    """Fit T parameters with Lag-Based Wave Speed supervision."""
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

    # Precompute observed lag/speed
    V_obs = obs["speed"]
    # Use early window where vehicles are present
    wave_mask = th.zeros(V_obs.shape[0], dtype=th.bool, device=device)
    wave_mask[:8] = True  # First 40s

    result_obs = estimate_wave_speed(
        V_obs, wave_mask,
        dx=cfg.detector_spacing, dt_obs=cfg.obs_interval,
        lag_max=lag_max, skip=skip, conf_threshold=0.0,
    )
    lag_obs = result_obs["lag"]
    valid_obs = th.isfinite(result_obs["speed"]) & (result_obs["lag"] > 0.5)
    print(f"  Observed wave speeds: {valid_obs.sum().item()} valid pairs")
    if valid_obs.sum() > 0:
        speeds_obs = result_obs["speed"][valid_obs]
        print(f"  Mean obs speed: {speeds_obs.mean():.2f} m/s")

    # Initialize T
    T_min, T_max = cfg.T_min, cfg.T_max
    u_T = th.nn.Parameter(inverse_sigmoid_reparam(
        th.full((N,), cfg.T_mean, device=device, dtype=dtype),
        T_min, T_max
    ))

    opt = th.optim.Adam([u_T], lr=lr)

    a_max = th.full((N,), cfg.a_max, device=device, dtype=dtype)
    b = th.full((N,), cfg.b_comfort, device=device, dtype=dtype)
    v_target = th.full((N,), cfg.v_free, device=device, dtype=dtype)

    history = {
        "loss": [], "L_det": [], "L_lag": [], "L_graph": [],
        "T_std": [], "speed_err": [],
    }

    warmup_iters = int(iters * warmup_frac)
    L_det_baseline = None
    current_beta = beta_lag

    print(f"Fitting T with Lag-Speed supervision:")
    print(f"  beta_lag={beta_lag}, lambda_T={lambda_T}, skip={skip}")
    print(f"  Stage A (L_det only): iters 0-{warmup_iters-1}")
    print(f"  Stage B (+ lag): iters {warmup_iters}-{iters-1}")

    for i in range(iters):
        opt.zero_grad()

        use_lag = (i >= warmup_iters)

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

        # Detector observations
        pred = detector_crosslane_at_times(
            S=S, V=V, lane_id=lane_id, xq=xq, x_dets=x_dets,
            time_indices=time_idx, sigma=cfg.sigma, half_window=10.0,
        )
        V_sim = pred["speed"]

        # Detector loss
        L_det = detector_loss(
            pred=pred, obs=obs,
            weights={"flow": 1.0, "speed": 0.1},
            loss_type="huber", huber_delta=1.0,
        )

        # Record baseline
        if i == warmup_iters - 1:
            L_det_baseline = L_det.item()

        # Lag-speed loss (after warmup)
        if use_lag and valid_obs.sum() > 0:
            result_sim = estimate_wave_speed(
                V_sim, wave_mask,
                dx=cfg.detector_spacing, dt_obs=cfg.obs_interval,
                lag_max=lag_max, skip=skip, conf_threshold=0.0,
            )
            lag_sim = result_sim["lag"]

            # Loss on lag values (more stable than speed)
            L_lag = lag_loss(lag_sim, lag_obs, valid_obs)

            # Guardrail
            if L_det_baseline is not None and L_det.item() > L_det_guard * L_det_baseline:
                current_beta = current_beta * 0.5
                if i % 20 == 0:
                    print(f"  [GUARD] reducing beta to {current_beta:.4f}")

            # Speed error for monitoring
            with th.no_grad():
                valid_both = valid_obs & th.isfinite(result_sim["speed"])
                if valid_both.sum() > 0:
                    speed_err = (result_sim["speed"][valid_both] - result_obs["speed"][valid_both]).abs().mean().item()
                else:
                    speed_err = 0.0
        else:
            L_lag = th.tensor(0.0, device=device, dtype=dtype)
            speed_err = 0.0

        # Graph regularization
        if lambda_T > 0:
            L_graph = laplacian_penalty(T, leader_idx, lane_id)
        else:
            L_graph = th.tensor(0.0, device=device, dtype=dtype)

        # Combined loss
        loss = L_det + current_beta * L_lag + lambda_T * L_graph

        if not th.isfinite(loss):
            print(f"  NaN at iter {i}, stopping")
            break

        loss.backward()
        opt.step()

        # Record history
        history["loss"].append(float(loss.item()))
        history["L_det"].append(float(L_det.item()))
        history["L_lag"].append(float(L_lag.item()) if use_lag else 0.0)
        history["L_graph"].append(float(L_graph.item()) if lambda_T > 0 else 0.0)
        history["T_std"].append(float(T.std().item()))
        history["speed_err"].append(speed_err)

        if i % 20 == 0:
            stage = "B" if use_lag else "A"
            print(f"  [{stage}] iter {i:3d}: loss={loss.item():.4f}, L_det={L_det.item():.4f}, "
                  f"L_lag={L_lag.item():.4f}, speed_err={speed_err:.2f}, T_std={T.std().item():.3f}")

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
    # PDE constraint arguments
    parser.add_argument("--pde", action="store_true", help="Use PDE (continuity) constraint")
    parser.add_argument("--beta-cont", type=float, default=0.001, help="Continuity loss weight")
    parser.add_argument("--boundary-cells", type=int, default=5, help="Boundary cells to exclude from PDE loss")
    # Graph aggregation + wave energy arguments
    parser.add_argument("--graph", action="store_true", help="Use Graph aggregation + Wave Energy supervision")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Graph filter smoothing parameter")
    parser.add_argument("--beta-E", type=float, default=0.1, help="Wave energy loss weight")
    # Lag-speed arguments
    parser.add_argument("--lag-speed", action="store_true", help="Use Lag-Based Wave Speed supervision")
    parser.add_argument("--beta-lag", type=float, default=0.1, help="Lag loss weight")
    parser.add_argument("--lag-max", type=int, default=6, help="Maximum lag to search (steps)")
    parser.add_argument("--skip", type=int, default=3, help="Detector skip for lag estimation")
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
    if args.lag_speed:
        print("   Using Lag-Based Wave Speed supervision")
        T_est, history = fit_T_with_lag_speed(
            data, cfg,
            lambda_T=args.lambda_T,
            beta_lag=args.beta_lag,
            iters=args.iters,
            lr=args.lr,
            warmup_frac=args.warmup_frac,
            lag_max=args.lag_max,
            skip=args.skip,
        )
    elif args.graph:
        print("   Using Graph aggregation + Wave Energy supervision")
        T_est, history = fit_T_with_graph(
            data, cfg,
            lambda_T=args.lambda_T,
            beta_E=args.beta_E,
            iters=args.iters,
            lr=args.lr,
            epsilon=args.epsilon,
            warmup_frac=args.warmup_frac,
        )
    elif args.pde:
        print("   Using PDE (continuity) constraint")
        T_est, history = fit_T_with_pde(
            data, cfg,
            lambda_T=args.lambda_T,
            beta_cont=args.beta_cont,
            iters=args.iters,
            lr=args.lr,
            boundary_cells=args.boundary_cells,
            warmup_frac=args.warmup_frac,
        )
    elif args.wavefront:
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
