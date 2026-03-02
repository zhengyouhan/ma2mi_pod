#!/usr/bin/env python
"""
Route A (Refined): Forward-only sweep to test whether detector-level damping
is sensitive to std(T) when mean(T) is fixed.

Replaces naive global attenuation slope with:
- Transition-region proxies (front thickness W_tr, center X_tr)
- Arrival-time proxies (reach fraction p_reach, arrival spread W_time,
  arrival gradient c_arr)

No training, no gradients.

Usage:
    python -m src.synthetic.run_forward_sweep
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch as th
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sp_stats

from src.synthetic.wave_scenario import (
    WaveScenarioConfig,
    create_initial_state,
    generate_heterogeneous_T,
    simulate_with_waves,
    generate_detector_observations,
)


# ---------------------------------------------------------------------------
# Phase 0: Frozen scenario
# ---------------------------------------------------------------------------

def frozen_scenario_config(obs_interval: float = 5.0) -> WaveScenarioConfig:
    """Return frozen scenario: 19 detectors at 50m spacing."""
    return WaveScenarioConfig(detector_spacing=50, obs_interval=obs_interval)


def create_frozen_geometry(cfg: WaveScenarioConfig, device="cpu", dtype=th.float32) -> dict:
    """Create initial state once. T_true will be overwritten per run."""
    return create_initial_state(cfg, device, dtype)


# ---------------------------------------------------------------------------
# Single forward run
# ---------------------------------------------------------------------------

@th.no_grad()
def run_single(init: dict, cfg: WaveScenarioConfig, T: th.Tensor,
               device="cpu", dtype=th.float32) -> dict:
    """Run one forward simulation and return detector speed observations."""
    run_init = {**init, "T_true": T}
    S, V, A = simulate_with_waves(run_init, cfg, device, dtype)
    obs = generate_detector_observations(S, V, init["lane_id"], cfg,
                                         obs_interval=cfg.obs_interval)
    return {
        "speed": obs["speed"],       # [K, J]
        "flow": obs["flow"],
        "x_dets": obs["x_dets"],
        "time_indices": obs["time_indices"],
    }


# ---------------------------------------------------------------------------
# Phase 1: Amplitude profile A(j) (kept from v1)
# ---------------------------------------------------------------------------

def compute_amplitude_profile(speed: th.Tensor, cfg: WaveScenarioConfig) -> dict:
    """
    Compute per-detector amplitude drop for Wave 1.

    Returns V_base [J], V_min [J], A [J] as numpy arrays, plus index ranges.
    """
    dt_obs = cfg.obs_interval
    K, J = speed.shape

    # W0: pre-event baseline [0, wave1_time)
    w0_end = max(1, int(cfg.wave1_time / dt_obs))
    # W1: event + propagation [0, wave1_time + duration + 30]
    w1_end = min(int((cfg.wave1_time + cfg.wave1_duration + 30.0) / dt_obs) + 1, K)

    V_base = speed[:w0_end, :].mean(dim=0).cpu().numpy().astype(np.float64)
    V_min = speed[:w1_end, :].min(dim=0).values.cpu().numpy().astype(np.float64)
    A = V_base - V_min

    return {"V_base": V_base, "V_min": V_min, "A": A,
            "w0_end": w0_end, "w1_end": w1_end}


# ---------------------------------------------------------------------------
# Phase 2: Transition-region proxies
# ---------------------------------------------------------------------------

def compute_transition_proxies(A: np.ndarray, x_dets: np.ndarray,
                               alpha: float = 0.2) -> dict:
    """
    Extract transition width W_tr and center X_tr from amplitude profile.

    J_tr = { j | alpha*A_max < A(j) < (1-alpha)*A_max }
    W_tr = (max(J_tr) - min(J_tr)) * dx
    X_tr = mean(x[J_tr])
    """
    A_max = A.max()
    if A_max < 1e-6:
        return {"W_tr": float("nan"), "X_tr": float("nan"),
                "J_tr_size": 0, "A_max": float(A_max)}

    lo = alpha * A_max
    hi = (1.0 - alpha) * A_max
    J_tr = np.where((A > lo) & (A < hi))[0]

    if len(J_tr) < 2:
        return {"W_tr": float("nan"), "X_tr": float("nan"),
                "J_tr_size": len(J_tr), "A_max": float(A_max)}

    W_tr = float(x_dets[J_tr[-1]] - x_dets[J_tr[0]])
    X_tr = float(x_dets[J_tr].mean())

    return {"W_tr": W_tr, "X_tr": X_tr,
            "J_tr_size": len(J_tr), "A_max": float(A_max)}


# ---------------------------------------------------------------------------
# Phase 3: Arrival-time proxies
# ---------------------------------------------------------------------------

def compute_arrival_proxies(speed: th.Tensor, x_dets: np.ndarray,
                            cfg: WaveScenarioConfig, V_base: np.ndarray,
                            dv_thresh: float = 5.0) -> dict:
    """
    Compute arrival time t_arr(j), reach fraction, arrival spread, arrival gradient.

    t_arr(j) = earliest t in W1 where V_base(j) - V(t,j) >= dv_thresh
    """
    dt_obs = cfg.obs_interval
    K, J = speed.shape
    speed_np = speed.cpu().numpy().astype(np.float64)

    # W1 range
    w1_start = 0
    w1_end = min(int((cfg.wave1_time + cfg.wave1_duration + 30.0) / dt_obs) + 1, K)

    t_arr = np.full(J, np.nan)
    for j in range(J):
        for k in range(w1_start, w1_end):
            if V_base[j] - speed_np[k, j] >= dv_thresh:
                t_arr[j] = k * dt_obs
                break

    # (A) Reach fraction
    finite_mask = np.isfinite(t_arr)
    p_reach = float(finite_mask.sum() / J)

    # (B) Arrival spread: IQR of t_arr on finite entries
    if finite_mask.sum() >= 4:
        q75, q25 = np.percentile(t_arr[finite_mask], [75, 25])
        W_time = float(q75 - q25)
        std_t_arr = float(np.std(t_arr[finite_mask], ddof=1))
    else:
        W_time = float("nan")
        std_t_arr = float("nan")

    # (C) Arrival gradient: t_arr(j) ~ b0 + b1 * x[j]
    J_arr = np.where(finite_mask)[0]
    if len(J_arr) >= 5:
        X_reg = np.column_stack([np.ones(len(J_arr)), x_dets[J_arr]])
        beta, _, _, _ = np.linalg.lstsq(X_reg, t_arr[J_arr], rcond=None)
        b1 = beta[1]
        c_arr = float(1.0 / b1) if abs(b1) > 1e-12 else float("nan")
        # R^2
        ss_res = np.sum((t_arr[J_arr] - X_reg @ beta) ** 2)
        ss_tot = np.sum((t_arr[J_arr] - t_arr[J_arr].mean()) ** 2)
        r2_arr = float(1.0 - ss_res / (ss_tot + 1e-12))
    else:
        c_arr = float("nan")
        r2_arr = float("nan")

    return {
        "t_arr": t_arr.tolist(),
        "p_reach": p_reach,
        "W_time": W_time,
        "std_t_arr": std_t_arr,
        "c_arr": c_arr,
        "r2_arr": r2_arr,
        "n_reached": int(finite_mask.sum()),
    }


# ---------------------------------------------------------------------------
# All proxies for one run
# ---------------------------------------------------------------------------

def compute_all_proxies(speed: th.Tensor, x_dets_list: list,
                        cfg: WaveScenarioConfig, alpha: float = 0.2,
                        dv_thresh: float = 5.0) -> dict:
    """Compute amplitude, transition, and arrival proxies for a single run."""
    x_dets = np.array(x_dets_list, dtype=np.float64)

    # Phase 1: amplitude
    amp = compute_amplitude_profile(speed, cfg)

    # Phase 2: transition
    trans = compute_transition_proxies(amp["A"], x_dets, alpha=alpha)

    # Phase 3: arrival
    arrival = compute_arrival_proxies(speed, x_dets, cfg, amp["V_base"],
                                      dv_thresh=dv_thresh)

    # Legacy: global slope (for comparison)
    J = len(x_dets)
    X = np.column_stack([np.ones(J), x_dets])
    beta, _, _, _ = np.linalg.lstsq(X, amp["A"], rcond=None)
    slope = float(-beta[1])

    return {
        "A_profile": amp["A"].tolist(),
        "slope": slope,
        # transition
        "W_tr": trans["W_tr"],
        "X_tr": trans["X_tr"],
        "J_tr_size": trans["J_tr_size"],
        "A_max": trans["A_max"],
        # arrival
        "t_arr": arrival["t_arr"],
        "p_reach": arrival["p_reach"],
        "W_time": arrival["W_time"],
        "std_t_arr": arrival["std_t_arr"],
        "c_arr": arrival["c_arr"],
        "r2_arr": arrival["r2_arr"],
        "n_reached": arrival["n_reached"],
    }


# ---------------------------------------------------------------------------
# Sweep executor
# ---------------------------------------------------------------------------

def run_sweep(sweep_name: str, mean_T_values: list, std_T_values: list,
              n_seeds: int, cfg: WaveScenarioConfig,
              alpha: float = 0.2, dv_thresh: float = 5.0,
              device="cpu", dtype=th.float32) -> list[dict]:
    """Run forward sweep over T distributions."""
    N = cfg.num_lanes * cfg.vehicles_per_lane
    init = create_frozen_geometry(cfg, device, dtype)

    results = []
    total = len(mean_T_values) * len(std_T_values) * n_seeds
    idx = 0

    for mean_T in mean_T_values:
        for std_T in std_T_values:
            for seed in range(n_seeds):
                idx += 1
                print(f"  [{idx}/{total}] {sweep_name}: "
                      f"mean_T={mean_T}, std_T={std_T}, seed={seed}")

                T = generate_heterogeneous_T(
                    N, mean_T, std_T, cfg.T_min, cfg.T_max, seed=seed
                ).to(device=device, dtype=dtype)

                out = run_single(init, cfg, T, device, dtype)
                proxies = compute_all_proxies(out["speed"], out["x_dets"], cfg,
                                              alpha=alpha, dv_thresh=dv_thresh)

                results.append({
                    "sweep": sweep_name,
                    "mean_T": mean_T,
                    "std_T": std_T,
                    "seed": seed,
                    "T_actual_mean": float(T.mean()),
                    "T_actual_std": float(T.std()),
                    **proxies,
                })

    return results


# ---------------------------------------------------------------------------
# Phase 4: Statistical analysis (generic for any metric)
# ---------------------------------------------------------------------------

def _aggregate_metric(results: list[dict], sweep_name: str,
                      group_key: str, metric_key: str) -> dict:
    """Aggregate a metric by group_key, compute mean/std/CI/effect-size."""
    by_group: dict[float, list[float]] = defaultdict(list)
    for r in results:
        if r["sweep"] == sweep_name:
            val = r[metric_key]
            if val is not None and np.isfinite(val):
                by_group[r[group_key]].append(val)

    if not by_group:
        return {"per_group": {}, "pooled_std": 0.0, "max_delta": 0.0,
                "threshold": 0.0, "is_flat": True}

    sorted_keys = sorted(by_group.keys())
    ref_vals = np.array(by_group[sorted_keys[0]])
    ref_mean = float(ref_vals.mean()) if len(ref_vals) > 0 else 0.0

    stats = {}
    all_vals = []

    for gk in sorted_keys:
        vals = np.array(by_group[gk])
        all_vals.extend(vals.tolist())
        n = len(vals)
        mean_v = float(vals.mean())
        std_v = float(vals.std(ddof=1)) if n > 1 else 0.0

        if n > 1:
            t_crit = sp_stats.t.ppf(0.975, df=n - 1)
            ci95 = float(t_crit * std_v / np.sqrt(n))
        else:
            ci95 = 0.0

        stats[gk] = {
            "mean": mean_v,
            "std": std_v,
            "ci95": ci95,
            "n": n,
            "delta": mean_v - ref_mean,
            "values": vals.tolist(),
        }

    pooled_std = float(np.std(all_vals, ddof=1)) if len(all_vals) > 1 else 0.0
    max_delta = max(abs(s["delta"]) for s in stats.values())
    is_flat = max_delta < 0.5 * pooled_std if pooled_std > 0 else True

    # Relative effect: max change / reference mean
    rel_effect = max_delta / abs(ref_mean) if abs(ref_mean) > 1e-12 else float("nan")
    is_flat_rel = rel_effect < 0.01 if np.isfinite(rel_effect) else True

    return {
        "per_group": {str(k): v for k, v in stats.items()},
        "pooled_std": pooled_std,
        "max_delta": max_delta,
        "threshold": 0.5 * pooled_std,
        "is_flat": is_flat,
        "rel_effect": rel_effect,
        "is_flat_rel": is_flat_rel,
    }


def analyze_sweep(results: list[dict], sweep_name: str,
                  group_key: str) -> dict:
    """Analyze all proxy metrics for a sweep."""
    metrics = ["slope", "W_tr", "X_tr", "W_time", "p_reach",
               "std_t_arr", "c_arr"]
    analysis = {}
    for m in metrics:
        analysis[m] = _aggregate_metric(results, sweep_name, group_key, m)
    return analysis


def check_monotonicity(analysis: dict, metric_key: str) -> bool:
    """Check if a metric shows monotonic trend across groups."""
    per_group = analysis[metric_key]["per_group"]
    if len(per_group) < 2:
        return True
    sorted_keys = sorted(float(k) for k in per_group.keys())
    means = [per_group[str(k)]["mean"] for k in sorted_keys]
    diffs = [means[i + 1] - means[i] for i in range(len(means) - 1)]
    return all(d > 0 for d in diffs) or all(d < 0 for d in diffs)


# ---------------------------------------------------------------------------
# Phase 6: Plots
# ---------------------------------------------------------------------------

def _plot_metric_vs_group(analysis_metric: dict, group_label: str,
                          metric_label: str, title: str,
                          output_path: str, color: str = "steelblue"):
    """Generic: errorbar plot of one metric vs one grouping variable."""
    per_group = analysis_metric["per_group"]
    if not per_group:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    keys = sorted(float(k) for k in per_group.keys())
    means = [per_group[str(k)]["mean"] for k in keys]
    ci95s = [per_group[str(k)]["ci95"] for k in keys]

    ax.errorbar(keys, means, yerr=ci95s, fmt="o-", capsize=5,
                color=color, linewidth=2, markersize=8)
    for k in keys:
        vals = per_group[str(k)]["values"]
        ax.scatter([k] * len(vals), vals, alpha=0.3, color=color, s=20)

    ax.set_xlabel(group_label, fontsize=12)
    ax.set_ylabel(metric_label, fontsize=12)
    ax.set_title(title, fontsize=13)

    verdict = "FLAT" if analysis_metric["is_flat"] else "SENSITIVE"
    rel = analysis_metric.get("rel_effect", float("nan"))
    ax.text(0.02, 0.98,
            f"Decision: {verdict}\n"
            f"max|delta|={analysis_metric['max_delta']:.4f}, "
            f"threshold={analysis_metric['threshold']:.4f}\n"
            f"rel. effect={rel:.2%}",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_main_sweep(main_analysis: dict, output_dir: str, n_seeds: int):
    """Phase 6.1: Main negative-result plots for std(T) sweep."""
    for metric, label in [
        ("W_tr", "Transition width W_tr [m]"),
        ("W_time", "Arrival time IQR [s]"),
        ("p_reach", "Reach fraction"),
        ("X_tr", "Transition center X_tr [m]"),
        ("std_t_arr", "Arrival time std [s]"),
        ("c_arr", "Arrival speed c_arr [m/s]"),
        ("slope", "Attenuation slope s [1/m] (legacy)"),
    ]:
        a = main_analysis[metric]
        if not a["per_group"]:
            continue
        _plot_metric_vs_group(
            a,
            group_label="std(T) [s]",
            metric_label=label,
            title=f"{label} vs T Heterogeneity\n"
                  f"(Fixed mean(T)=1.5, N={n_seeds} seeds per std)",
            output_path=os.path.join(output_dir, f"main_{metric}.png"),
            color="steelblue",
        )


def plot_sanity_sweep(sanity_analysis: dict, output_dir: str, n_seeds: int):
    """Phase 6.3: Sanity plots for mean(T) sweep."""
    for metric, label in [
        ("W_tr", "Transition width W_tr [m]"),
        ("W_time", "Arrival time IQR [s]"),
        ("p_reach", "Reach fraction"),
        ("X_tr", "Transition center X_tr [m]"),
        ("std_t_arr", "Arrival time std [s]"),
        ("c_arr", "Arrival speed c_arr [m/s]"),
        ("slope", "Attenuation slope s [1/m] (legacy)"),
    ]:
        a = sanity_analysis[metric]
        if not a["per_group"]:
            continue
        _plot_metric_vs_group(
            a,
            group_label="mean(T) [s]",
            metric_label=label,
            title=f"Sanity: {label} vs mean(T)\n"
                  f"(Fixed std(T)=0.2, N={n_seeds} seeds per mean)",
            output_path=os.path.join(output_dir, f"sanity_{metric}.png"),
            color="darkorange",
        )


def plot_diagnostic(results: list[dict], cfg: WaveScenarioConfig,
                    output_dir: str):
    """Phase 6.2: Diagnostic plot for one representative run (seed=0, std=0.2)."""
    # Pick seed=0, std_T=0.2, mean_T=1.5
    run = None
    for r in results:
        if (r["sweep"] == "main" and r["seed"] == 0
                and r["std_T"] == 0.2):
            run = r
            break
    if run is None:
        return

    x_dets = np.arange(1, len(run["A_profile"]) + 1) * cfg.detector_spacing
    A = np.array(run["A_profile"])
    t_arr = np.array(run["t_arr"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: A(j) profile with transition band
    ax = axes[0]
    ax.plot(x_dets, A, "o-", color="steelblue", markersize=5, linewidth=2)
    A_max = A.max()
    if A_max > 0:
        ax.axhline(0.2 * A_max, color="red", linestyle="--", alpha=0.5,
                    label=f"alpha=0.2 bounds")
        ax.axhline(0.8 * A_max, color="red", linestyle="--", alpha=0.5)
        # Shade transition region
        J_tr = np.where((A > 0.2 * A_max) & (A < 0.8 * A_max))[0]
        if len(J_tr) >= 2:
            ax.axvspan(x_dets[J_tr[0]], x_dets[J_tr[-1]], alpha=0.15,
                       color="red", label=f"J_tr (W_tr={run['W_tr']:.0f}m)")
    ax.set_xlabel("Detector position (m)", fontsize=11)
    ax.set_ylabel("Amplitude drop A(j) [m/s]", fontsize=11)
    ax.set_title("Amplitude Profile (cliff + transition)", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Right: t_arr(j) with censored markers
    ax = axes[1]
    finite = np.isfinite(t_arr)
    ax.plot(x_dets[finite], t_arr[finite], "o-", color="darkorange",
            markersize=5, linewidth=2, label="Arrived")
    ax.scatter(x_dets[~finite],
               np.full((~finite).sum(), t_arr[finite].max() + 5 if finite.any() else 40),
               marker="x", color="gray", s=60, label="Censored (no arrival)")
    ax.set_xlabel("Detector position (m)", fontsize=11)
    ax.set_ylabel("Arrival time t_arr [s]", fontsize=11)
    ax.set_title("Wave Arrival Time per Detector", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.suptitle(f"Diagnostic: mean(T)=1.5, std(T)=0.2, seed=0, "
                 f"dv_thresh=5 m/s", fontsize=13)
    plt.tight_layout()
    path = os.path.join(output_dir, "diagnostic.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_amplitude_profiles(results: list[dict], cfg: WaveScenarioConfig,
                            output_dir: str):
    """Overlay A(j) for seed=0 across std(T) values."""
    fig, ax = plt.subplots(figsize=(10, 5))
    std_vals = [0.0, 0.1, 0.2, 0.3, 0.4]
    colors = plt.cm.viridis(np.linspace(0, 1, len(std_vals)))

    for r in results:
        if r["sweep"] == "main" and r["seed"] == 0:
            A = r["A_profile"]
            x = np.arange(1, len(A) + 1) * cfg.detector_spacing
            cidx = std_vals.index(r["std_T"])
            ax.plot(x, A, "o-", color=colors[cidx],
                    label=f"std(T)={r['std_T']:.1f}", markersize=4)

    ax.set_xlabel("Detector position (m)", fontsize=12)
    ax.set_ylabel("Amplitude drop A(j) [m/s]", fontsize=12)
    ax.set_title("Wave 1 Amplitude Profile (seed=0)", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "amplitude_profiles.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Phase 7: Conclusion
# ---------------------------------------------------------------------------

def generate_conclusion(main_analysis: dict, sanity_analysis: dict) -> str:
    """Generate interpretation text based on results."""
    # Headline metrics: W_tr and W_time
    headline_metrics = ["W_tr", "W_time", "p_reach"]
    flat_count = sum(1 for m in headline_metrics
                     if main_analysis[m]["is_flat"])
    total_m = len(headline_metrics)

    # Sanity monotonicity
    sanity_mono = {m: check_monotonicity(sanity_analysis, m)
                   for m in headline_metrics}
    any_mono = any(sanity_mono.values())

    lines = []
    lines.append("=== Route A Refined Results ===\n")

    # Per-metric summary
    for m in ["W_tr", "W_time", "p_reach", "std_t_arr", "c_arr", "slope"]:
        a = main_analysis[m]
        verdict = "FLAT" if a["is_flat"] else "SENSITIVE"
        rel = a.get("rel_effect", float("nan"))
        lines.append(f"  {m:12s}: {verdict:10s}  max|delta|={a['max_delta']:.4f}, "
                     f"rel={rel:.2%}, pooled_std={a['pooled_std']:.4f}")

    lines.append("")

    if flat_count >= 2:
        lines.append(
            "After accounting for partial reach and isolating the transition "
            "region, the dependence of damping/arrival proxies on std(T) is "
            f"bounded by small absolute/relative effects ({flat_count}/{total_m} "
            "headline metrics flat). This supports near-insensitivity of macro "
            "detector observations to micro heterogeneity under strong exogenous "
            "forcing at 50m/5s resolution."
        )
    else:
        lines.append(
            "Transition-width and/or arrival-spread proxies reveal a weak but "
            "consistent dependence on std(T), suggesting heterogeneity signatures "
            "exist but are confined to front thickness/delay statistics."
        )

    lines.append("")
    sanity_str = ", ".join(f"{m}={'YES' if v else 'NO'}"
                           for m, v in sanity_mono.items())
    lines.append(f"Sanity (monotonic wrt mean(T)): {sanity_str}")

    if not any_mono:
        lines.append(
            "WARNING: No headline proxy is monotonic wrt mean(T). "
            "Consider extending W1 window or adjusting threshold."
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Route A Refined: Forward-only T heterogeneity sweep")
    parser.add_argument("--output-dir", default="outputs/route_a_sweep")
    parser.add_argument("--n-seeds-main", type=int, default=10)
    parser.add_argument("--n-seeds-sanity", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.2,
                        help="Transition band parameter")
    parser.add_argument("--dv-thresh", type=float, default=5.0,
                        help="Arrival threshold (m/s)")
    parser.add_argument("--obs-interval", type=float, default=5.0,
                        help="Detector observation interval (s)")
    parser.add_argument("--device",
                        default="cuda" if th.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = args.device
    dtype = th.float32

    print("=" * 60)
    print("Route A (Refined): Forward-Only T Heterogeneity Sweep")
    print("=" * 60)

    # Phase 0: Frozen scenario
    cfg = frozen_scenario_config(obs_interval=args.obs_interval)
    print(f"\nScenario: {cfg.x_max:.0f}m road, {cfg.num_lanes} lanes, "
          f"{cfg.vehicles_per_lane} veh/lane")
    print(f"Detectors: {len(cfg.detector_positions)} at "
          f"{cfg.detector_spacing}m spacing")
    print(f"Ghost: v_base={cfg.ghost_v_base} m/s")
    print(f"Wave 1: t={cfg.wave1_time}s, drop={cfg.wave1_v_drop} m/s, "
          f"dur={cfg.wave1_duration}s")
    print(f"Params: alpha={args.alpha}, dv_thresh={args.dv_thresh} m/s")

    # Main sweep
    print(f"\n--- Main Sweep: mean(T)=1.5, std(T) in [0.0..0.4], "
          f"{args.n_seeds_main} seeds ---")
    main_results = run_sweep(
        "main",
        mean_T_values=[1.5],
        std_T_values=[0.0, 0.1, 0.2, 0.3, 0.4],
        n_seeds=args.n_seeds_main,
        cfg=cfg, alpha=args.alpha, dv_thresh=args.dv_thresh,
        device=device, dtype=dtype,
    )

    # Sanity sweep
    print(f"\n--- Sanity Sweep: std(T)=0.2, mean(T) in [1.0, 1.5, 2.0], "
          f"{args.n_seeds_sanity} seeds ---")
    sanity_results = run_sweep(
        "sanity",
        mean_T_values=[1.0, 1.5, 2.0],
        std_T_values=[0.2],
        n_seeds=args.n_seeds_sanity,
        cfg=cfg, alpha=args.alpha, dv_thresh=args.dv_thresh,
        device=device, dtype=dtype,
    )

    # Phase 4: Statistical analysis
    print("\n--- Statistical Analysis ---")
    main_analysis = analyze_sweep(main_results, "main", "std_T")
    sanity_analysis = analyze_sweep(sanity_results, "sanity", "mean_T")

    for m in ["W_tr", "W_time", "p_reach", "std_t_arr", "c_arr", "slope"]:
        a = main_analysis[m]
        verdict = "FLAT" if a["is_flat"] else "SENSITIVE"
        print(f"  {m:12s}: {verdict:10s}  "
              f"max|delta|={a['max_delta']:.4f}, "
              f"rel={a.get('rel_effect', float('nan')):.2%}")

    # Phase 6: Plots
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\n--- Generating Plots -> {args.output_dir}/ ---")
    plot_main_sweep(main_analysis, args.output_dir, args.n_seeds_main)
    plot_sanity_sweep(sanity_analysis, args.output_dir, args.n_seeds_sanity)
    plot_diagnostic(main_results, cfg, args.output_dir)
    plot_amplitude_profiles(main_results, cfg, args.output_dir)

    # Save JSON
    conclusion = generate_conclusion(main_analysis, sanity_analysis)
    all_out = {
        "main_results": main_results,
        "sanity_results": sanity_results,
        "main_analysis": main_analysis,
        "sanity_analysis": sanity_analysis,
        "conclusion": conclusion,
        "config": {
            "detector_spacing": cfg.detector_spacing,
            "obs_interval": cfg.obs_interval,
            "num_detectors": len(cfg.detector_positions),
            "duration_s": cfg.duration_s,
            "alpha": args.alpha,
            "dv_thresh": args.dv_thresh,
            "wave1": {"time": cfg.wave1_time, "v_drop": cfg.wave1_v_drop,
                      "duration": cfg.wave1_duration},
            "wave2": {"time": cfg.wave2_time, "v_drop": cfg.wave2_v_drop,
                      "duration": cfg.wave2_duration},
        },
    }
    json_path = os.path.join(args.output_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(all_out, f, indent=2)
    print(f"Saved: {json_path}")

    # Print conclusion
    print("\n" + "=" * 60)
    print(conclusion)
    print("=" * 60)


if __name__ == "__main__":
    main()
