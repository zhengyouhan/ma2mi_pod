#!/usr/bin/env python
"""
2s Recovery Experiment — Close the Resolution–Identifiability Loop.

Demonstrates that the heterogeneity channel (arrival-time dispersion)
is actionable: when detector sampling improves from 5s → 2s,
micro-parameter recovery (T) improves.

Factorial design: {5s, 2s} resolution × {det-only, det+arrival-spread} loss.

Usage:
    python -m src.synthetic.run_2s_recovery --device cuda
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
from src.obs.detector_operator import detector_crosslane_at_times
from src.loss import detector_loss


# ---------------------------------------------------------------------------
# A. Differentiable Arrival-Spread Loss
# ---------------------------------------------------------------------------

def compute_arrival_spread(
    V: th.Tensor,
    obs_interval: float,
    cfg: WaveScenarioConfig,
    dv_thresh: float = 5.0,
    tau: float = 1.0,
    eps: float = 1e-6,
) -> dict:
    """
    Compute soft arrival times and their spread across detectors.

    Args:
        V: [K, J] detector speed over time
        obs_interval: seconds between observations
        cfg: scenario config (for wave timing)
        dv_thresh: speed drop threshold (m/s)
        tau: softness parameter (m/s)
        eps: numerical stability

    Returns:
        dict with mu_arr [J], std_arr scalar, P [K_w1, J], S_act [K_w1, J]
    """
    K, J = V.shape
    device, dtype = V.device, V.dtype

    # Pre-event baseline: observations before wave1_time
    k_pre = max(1, int(cfg.wave1_time / obs_interval))
    V_base = V[:k_pre, :].mean(dim=0)  # [J]

    # W1 window: [0, wave1_time + wave1_duration + 30s]
    w1_end_time = cfg.wave1_time + cfg.wave1_duration + 30.0
    w1_end_idx = min(int(w1_end_time / obs_interval) + 1, K)

    V_w1 = V[:w1_end_idx, :]  # [K_w1, J]
    K_w1 = V_w1.shape[0]

    # Drop signal: positive when speed is below baseline
    D = V_base.unsqueeze(0) - V_w1  # [K_w1, J]

    # Soft event activation
    S_act = th.sigmoid((D - dv_thresh) / tau)  # [K_w1, J]

    # Normalized temporal distribution
    S_sum = S_act.sum(dim=0) + eps  # [J]
    P = S_act / S_sum.unsqueeze(0)  # [K_w1, J]

    # Time values in seconds for each W1 step
    t_seconds = th.arange(K_w1, device=device, dtype=dtype) * obs_interval  # [K_w1]

    # Soft arrival time per detector
    mu_arr = (t_seconds.unsqueeze(1) * P).sum(dim=0)  # [J]

    # Spread across detectors (numerically stable)
    std_arr = th.sqrt(th.var(mu_arr) + eps)

    return {
        "mu_arr": mu_arr,
        "std_arr": std_arr,
        "P": P,
        "S_act": S_act,
    }


def compute_arrival_spread_loss(
    V_sim: th.Tensor,
    V_obs: th.Tensor,
    obs_interval: float,
    cfg: WaveScenarioConfig,
    dv_thresh: float = 5.0,
    tau: float = 1.0,
    eps: float = 1e-6,
) -> dict:
    """
    Differentiable arrival-spread loss: (std_arr_sim - std_arr_obs)^2.

    Gradients flow through V_sim only; V_obs is detached.
    """
    arr_sim = compute_arrival_spread(V_sim, obs_interval, cfg, dv_thresh, tau, eps)
    with th.no_grad():
        arr_obs = compute_arrival_spread(V_obs, obs_interval, cfg, dv_thresh, tau, eps)

    std_sim = arr_sim["std_arr"]
    std_obs = arr_obs["std_arr"].detach()

    L_arrspread = (std_sim - std_obs) ** 2

    return {
        "L_arrspread": L_arrspread,
        "std_arr_sim": std_sim,
        "std_arr_obs": std_obs,
    }


# ---------------------------------------------------------------------------
# B. Generic Recovery Loop
# ---------------------------------------------------------------------------

def run_recovery(
    data: dict,
    cfg: WaveScenarioConfig,
    obs: dict,
    time_indices: th.Tensor,
    obs_interval: float,
    *,
    use_arrspread: bool = False,
    beta: float = 1.0,
    iters: int = 100,
    lr: float = 0.05,
    warmup_frac: float = 0.3,
    L_det_guard: float = 1.5,
    dv_thresh: float = 5.0,
    tau: float = 1.0,
    label: str = "",
) -> dict:
    """
    Gradient-based T recovery with optional arrival-spread loss.

    Returns dict with T_est [N] and per-iteration history.
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

    # Initialize T from prior mean
    u_T = th.nn.Parameter(inverse_sigmoid_reparam(
        th.full((N,), cfg.T_mean, device=device, dtype=dtype),
        T_min, T_max,
    ))
    opt = th.optim.Adam([u_T], lr=lr)

    a_max = th.full((N,), cfg.a_max, device=device, dtype=dtype)
    b = th.full((N,), cfg.b_comfort, device=device, dtype=dtype)
    v_target = th.full((N,), cfg.v_free, device=device, dtype=dtype)

    # Precompute observed arrival spread (fixed target)
    if use_arrspread:
        with th.no_grad():
            arr_obs = compute_arrival_spread(
                obs["speed"], obs_interval, cfg, dv_thresh, tau,
            )
        std_arr_obs = arr_obs["std_arr"].detach()

    warmup_iters = int(iters * warmup_frac)
    L_det_baseline = None
    current_beta = beta

    history = {
        "loss": [], "L_det": [], "L_arrspread": [],
        "T_Corr": [], "T_MAE": [], "T_std": [],
        "std_arr_sim": [],
    }

    tag = f"[{label}]" if label else ""
    print(f"{tag} Recovery: obs_interval={obs_interval}s, arrspread={use_arrspread}, "
          f"beta={beta}, iters={iters}, lr={lr}")
    if use_arrspread:
        print(f"{tag}   Warmup iters 0-{warmup_iters-1} (L_det only), "
              f"then +L_arrspread iters {warmup_iters}-{iters-1}")
        print(f"{tag}   std_arr_obs = {std_arr_obs.item():.4f}")

    T_true_np = T_true.cpu().numpy()

    for i in range(iters):
        opt.zero_grad()

        T = sigmoid_reparam(u_T, T_min, T_max)

        S, V, _ = rollout_idm_multilane(
            s0_init=s0, v0_init=v0, leader_idx=leader_idx,
            num_steps=steps, dt=cfg.dt,
            a_max=a_max, b=b, v_target=v_target,
            s0=cfg.s0, T_headway=T,
            ghost_v=cfg.ghost_v_base, ghost_gap0=cfg.ghost_gap0,
        )

        pred = detector_crosslane_at_times(
            S=S, V=V, lane_id=lane_id, xq=xq, x_dets=x_dets,
            time_indices=time_indices, sigma=cfg.sigma, half_window=10.0,
        )

        L_det = detector_loss(
            pred=pred, obs=obs,
            weights={"flow": 1.0, "speed": 0.1},
            loss_type="huber", huber_delta=1.0,
        )

        # Record baseline at end of warmup
        if i == warmup_iters - 1:
            L_det_baseline = L_det.item()

        # Arrival-spread loss (after warmup, if enabled)
        use_arr_this_iter = use_arrspread and (i >= warmup_iters)
        if use_arr_this_iter:
            arr_sim = compute_arrival_spread(
                pred["speed"], obs_interval, cfg, dv_thresh, tau,
            )
            std_arr_sim = arr_sim["std_arr"]
            L_arr = (std_arr_sim - std_arr_obs) ** 2

            # Guardrail: reduce beta if L_det degrades
            if L_det_baseline is not None and L_det.item() > L_det_guard * L_det_baseline:
                current_beta = current_beta * 0.5
                if i % 20 == 0:
                    print(f"{tag}   [GUARD] iter {i}: L_det={L_det.item():.4f} > "
                          f"{L_det_guard}x baseline, beta→{current_beta:.4f}")
        else:
            L_arr = th.tensor(0.0, device=device, dtype=dtype)
            std_arr_sim = None

        loss = L_det + current_beta * L_arr

        if not th.isfinite(loss):
            print(f"{tag}   NaN at iter {i}, stopping")
            break

        loss.backward()
        opt.step()

        # Compute metrics (no grad)
        with th.no_grad():
            T_np = T.cpu().numpy()
            if T_np.std() < 1e-8:
                corr = 0.0  # T is still homogeneous (early iterations)
            else:
                corr = float(np.corrcoef(T_true_np, T_np)[0, 1])
            mae = float(np.abs(T_true_np - T_np).mean())
            t_std = float(T.std().item())

        history["loss"].append(float(loss.item()))
        history["L_det"].append(float(L_det.item()))
        history["L_arrspread"].append(float(L_arr.item()))
        history["T_Corr"].append(corr)
        history["T_MAE"].append(mae)
        history["T_std"].append(t_std)
        history["std_arr_sim"].append(
            float(std_arr_sim.item()) if std_arr_sim is not None else 0.0
        )

        if i % 20 == 0 or i == iters - 1:
            arr_str = f", L_arr={L_arr.item():.6f}" if use_arr_this_iter else ""
            print(f"{tag}   iter {i:3d}: loss={loss.item():.6f}, L_det={L_det.item():.6f}"
                  f"{arr_str}, T_Corr={corr:.4f}, T_MAE={mae:.4f}")

    # Final T
    with th.no_grad():
        T_est = sigmoid_reparam(u_T, T_min, T_max)

    return {
        "T_est": T_est,
        "history": history,
        "label": label,
        "obs_interval": obs_interval,
        "use_arrspread": use_arrspread,
    }


# ---------------------------------------------------------------------------
# C. Reporting: Table, Plots, JSON
# ---------------------------------------------------------------------------

def generate_summary_table(results: list[dict]) -> str:
    """Format the 4-row summary table."""
    lines = []
    lines.append("=" * 70)
    lines.append("2s Recovery Experiment: Summary")
    lines.append("=" * 70)
    header = f"{'dt_obs':>6} | {'loss':>13} | {'T_Corr':>7} | {'T_std':>7} | {'T_MAE':>7} | {'L_det':>8}"
    lines.append(header)
    lines.append("-" * 70)

    for r in results:
        dt = f"{r['obs_interval']:.0f}s"
        loss_name = "L_det+L_arr" if r["use_arrspread"] else "L_det"
        h = r["history"]
        T_Corr = h["T_Corr"][-1] if h["T_Corr"] else float("nan")
        T_std = h["T_std"][-1] if h["T_std"] else float("nan")
        T_MAE = h["T_MAE"][-1] if h["T_MAE"] else float("nan")
        L_det = h["L_det"][-1] if h["L_det"] else float("nan")

        line = f"{dt:>6} | {loss_name:>13} | {T_Corr:7.4f} | {T_std:7.4f} | {T_MAE:7.4f} | {L_det:8.4f}"
        lines.append(line)

    lines.append("=" * 70)
    return "\n".join(lines)


def plot_T_corr_vs_iter(results: list[dict], output_dir: str):
    """Plot T_Corr over iterations for all 4 conditions."""
    fig, ax = plt.subplots(figsize=(10, 6))

    styles = {
        "5s_det":     {"color": "steelblue",  "ls": "-",  "label": "5s, L_det only"},
        "5s_det+arr": {"color": "steelblue",  "ls": "--", "label": "5s, L_det + L_arr"},
        "2s_det":     {"color": "darkorange", "ls": "-",  "label": "2s, L_det only"},
        "2s_det+arr": {"color": "darkorange", "ls": "--", "label": "2s, L_det + L_arr"},
    }

    for r in results:
        lbl = r["label"]
        s = styles.get(lbl, {"color": "gray", "ls": "-", "label": lbl})
        iters = list(range(len(r["history"]["T_Corr"])))
        ax.plot(iters, r["history"]["T_Corr"],
                color=s["color"], ls=s["ls"], linewidth=2, label=s["label"])

    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("T_Corr (Pearson)", fontsize=12)
    ax.set_title("T Parameter Recovery: Correlation vs Iteration", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(output_dir, "fig_T_corr_vs_iter.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_std_arr_convergence(results: list[dict], output_dir: str):
    """Plot std_arr_sim convergence for arrival-spread conditions."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {"5s_det+arr": "steelblue", "2s_det+arr": "darkorange"}

    for r in results:
        if not r["use_arrspread"]:
            continue
        lbl = r["label"]
        h = r["history"]
        iters = list(range(len(h["std_arr_sim"])))
        ax.plot(iters, h["std_arr_sim"],
                color=colors.get(lbl, "gray"), linewidth=2,
                label=f"{lbl}: std_arr_sim")

    # Plot observed targets
    for r in results:
        if not r["use_arrspread"]:
            continue
        # Get std_arr_obs from the last non-zero L_arrspread step
        # (it's constant, stored in the loss computation)
        # We'll compute it from the observation data
        pass

    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("std_arr (arrival time spread) [s]", fontsize=12)
    ax.set_title("Arrival-Spread Convergence", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(output_dir, "fig_std_arr_convergence.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_L_det_vs_iter(results: list[dict], output_dir: str):
    """Plot L_det over iterations to verify stability."""
    fig, ax = plt.subplots(figsize=(10, 6))

    styles = {
        "5s_det":     {"color": "steelblue",  "ls": "-",  "label": "5s, L_det only"},
        "5s_det+arr": {"color": "steelblue",  "ls": "--", "label": "5s, L_det + L_arr"},
        "2s_det":     {"color": "darkorange", "ls": "-",  "label": "2s, L_det only"},
        "2s_det+arr": {"color": "darkorange", "ls": "--", "label": "2s, L_det + L_arr"},
    }

    for r in results:
        lbl = r["label"]
        s = styles.get(lbl, {"color": "gray", "ls": "-", "label": lbl})
        iters = list(range(len(r["history"]["L_det"])))
        ax.plot(iters, r["history"]["L_det"],
                color=s["color"], ls=s["ls"], linewidth=2, label=s["label"])

    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("L_det", fontsize=12)
    ax.set_title("Detector Loss Stability", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(output_dir, "fig_L_det_vs_iter.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def generate_interpretation(results: list[dict]) -> str:
    """Generate interpretation text based on results."""
    # Extract final metrics
    metrics = {}
    for r in results:
        h = r["history"]
        metrics[r["label"]] = {
            "T_Corr": h["T_Corr"][-1] if h["T_Corr"] else float("nan"),
            "T_MAE": h["T_MAE"][-1] if h["T_MAE"] else float("nan"),
            "L_det": h["L_det"][-1] if h["L_det"] else float("nan"),
        }

    lines = []
    lines.append("\n--- Interpretation ---")

    # Check 1: Does 2s (det-only) improve vs 5s?
    corr_5s = metrics.get("5s_det", {}).get("T_Corr", float("nan"))
    corr_2s = metrics.get("2s_det", {}).get("T_Corr", float("nan"))
    delta_res = corr_2s - corr_5s
    lines.append(f"1. Resolution effect (det-only): T_Corr 5s={corr_5s:.4f} → 2s={corr_2s:.4f} "
                 f"(Δ={delta_res:+.4f})")
    if delta_res > 0.02:
        lines.append("   → YES: 2s improves T recovery over 5s.")
    elif delta_res > -0.02:
        lines.append("   → MARGINAL: Resolution effect is small.")
    else:
        lines.append("   → NO: 2s does not improve (or hurts) T recovery.")

    # Check 2: Does arrival-spread help more at 2s than 5s?
    corr_5s_arr = metrics.get("5s_det+arr", {}).get("T_Corr", float("nan"))
    corr_2s_arr = metrics.get("2s_det+arr", {}).get("T_Corr", float("nan"))
    benefit_5s = corr_5s_arr - corr_5s
    benefit_2s = corr_2s_arr - corr_2s
    lines.append(f"2. Arrival-spread benefit at 5s: Δ={benefit_5s:+.4f}")
    lines.append(f"   Arrival-spread benefit at 2s: Δ={benefit_2s:+.4f}")
    if benefit_2s > benefit_5s + 0.01:
        lines.append("   → YES: Arrival-spread helps more at 2s than 5s.")
    else:
        lines.append("   → NO: Arrival-spread benefit is not stronger at 2s.")

    # Check 3: L_det stability
    Ldet_5s = metrics.get("5s_det", {}).get("L_det", float("nan"))
    Ldet_5s_arr = metrics.get("5s_det+arr", {}).get("L_det", float("nan"))
    Ldet_2s = metrics.get("2s_det", {}).get("L_det", float("nan"))
    Ldet_2s_arr = metrics.get("2s_det+arr", {}).get("L_det", float("nan"))
    lines.append(f"3. L_det stability: 5s={Ldet_5s:.4f}/{Ldet_5s_arr:.4f}, "
                 f"2s={Ldet_2s:.4f}/{Ldet_2s_arr:.4f}")

    # Overall verdict
    lines.append("")
    if delta_res > 0.02 and benefit_2s > benefit_5s + 0.01:
        lines.append("VERDICT: Strong success — resolution and arrival-spread both help.")
    elif delta_res > 0.02:
        lines.append("VERDICT: Weak success — resolution helps, arrival-spread effect unclear.")
    elif corr_2s_arr > corr_5s:
        lines.append("VERDICT: Partial success — combined 2s+arr beats 5s baseline.")
    else:
        lines.append("VERDICT: Failure — no improvement from resolution or arrival-spread.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="2s Recovery Experiment")
    parser.add_argument("--device", default="cuda" if th.cuda.is_available() else "cpu")
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=1.0, help="Arrival-spread loss weight")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs/recovery_2s")
    parser.add_argument("--warmup-frac", type=float, default=0.3)
    parser.add_argument("--dv-thresh", type=float, default=5.0)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--L-det-guard", type=float, default=1.5)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = args.device
    dtype = th.float32

    print("=" * 60)
    print("2s Recovery Experiment")
    print("=" * 60)

    # --- 1. Generate ground truth ---
    print("\n1. Generating ground truth...")
    cfg = WaveScenarioConfig(detector_spacing=50)
    data = create_wave_scenario(cfg, device=device, dtype=dtype, seed=args.seed)

    N = data["s0"].shape[0]
    print(f"   Vehicles: {N}")
    print(f"   True T: mean={data['T_true'].mean():.3f}, std={data['T_true'].std():.3f}")
    print(f"   Detectors: {len(cfg.detector_positions)} @ {cfg.detector_spacing}m spacing")

    # --- 2. Generate observation sets at both resolutions ---
    print("\n2. Generating observations at 5s and 2s...")
    obs_5s = generate_detector_observations(
        data["S"], data["V"], data["lane_id"], cfg, obs_interval=5.0,
    )
    obs_5s_dict = {"flow": obs_5s["flow"], "speed": obs_5s["speed"]}
    time_idx_5s = obs_5s["time_indices"]
    print(f"   5s: {time_idx_5s.shape[0]} time points, speed shape {obs_5s['speed'].shape}")

    obs_2s = generate_detector_observations(
        data["S"], data["V"], data["lane_id"], cfg, obs_interval=2.0,
    )
    obs_2s_dict = {"flow": obs_2s["flow"], "speed": obs_2s["speed"]}
    time_idx_2s = obs_2s["time_indices"]
    print(f"   2s: {time_idx_2s.shape[0]} time points, speed shape {obs_2s['speed'].shape}")

    # --- 3. Run 4 recovery conditions ---
    conditions = [
        {"label": "5s_det",      "obs": obs_5s_dict, "time_idx": time_idx_5s,
         "obs_interval": 5.0, "use_arrspread": False},
        {"label": "5s_det+arr",  "obs": obs_5s_dict, "time_idx": time_idx_5s,
         "obs_interval": 5.0, "use_arrspread": True},
        {"label": "2s_det",      "obs": obs_2s_dict, "time_idx": time_idx_2s,
         "obs_interval": 2.0, "use_arrspread": False},
        {"label": "2s_det+arr",  "obs": obs_2s_dict, "time_idx": time_idx_2s,
         "obs_interval": 2.0, "use_arrspread": True},
    ]

    all_results = []
    for idx, cond in enumerate(conditions):
        print(f"\n{'='*60}")
        print(f"Condition {idx+1}/4: {cond['label']}")
        print(f"{'='*60}")

        result = run_recovery(
            data=data, cfg=cfg,
            obs=cond["obs"], time_indices=cond["time_idx"],
            obs_interval=cond["obs_interval"],
            use_arrspread=cond["use_arrspread"],
            beta=args.beta,
            iters=args.iters,
            lr=args.lr,
            warmup_frac=args.warmup_frac,
            L_det_guard=args.L_det_guard,
            dv_thresh=args.dv_thresh,
            tau=args.tau,
            label=cond["label"],
        )
        all_results.append(result)

    # --- 4. Summary table ---
    print("\n" + "=" * 60)
    table = generate_summary_table(all_results)
    print(table)

    with open(os.path.join(args.output_dir, "summary_table.txt"), "w") as f:
        f.write(table)
    print(f"Saved: {args.output_dir}/summary_table.txt")

    # --- 5. Interpretation ---
    interp = generate_interpretation(all_results)
    print(interp)

    with open(os.path.join(args.output_dir, "interpretation.txt"), "w") as f:
        f.write(table + "\n" + interp)
    print(f"Saved: {args.output_dir}/interpretation.txt")

    # --- 6. Plots ---
    print("\n6. Generating plots...")
    plot_T_corr_vs_iter(all_results, args.output_dir)
    plot_std_arr_convergence(all_results, args.output_dir)
    plot_L_det_vs_iter(all_results, args.output_dir)

    # --- 7. Save full results as JSON ---
    json_results = []
    for r in all_results:
        json_results.append({
            "label": r["label"],
            "obs_interval": r["obs_interval"],
            "use_arrspread": r["use_arrspread"],
            "history": r["history"],
            "T_est_mean": float(r["T_est"].mean().item()),
            "T_est_std": float(r["T_est"].std().item()),
        })

    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump({
            "config": {
                "iters": args.iters,
                "lr": args.lr,
                "beta": args.beta,
                "seed": args.seed,
                "warmup_frac": args.warmup_frac,
                "dv_thresh": args.dv_thresh,
                "tau": args.tau,
                "L_det_guard": args.L_det_guard,
                "T_true_mean": float(data["T_true"].mean().item()),
                "T_true_std": float(data["T_true"].std().item()),
            },
            "conditions": json_results,
        }, f, indent=2)
    print(f"Saved: {args.output_dir}/results.json")

    print("\nDone!")


if __name__ == "__main__":
    main()
