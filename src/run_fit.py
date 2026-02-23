from __future__ import annotations

import argparse
import csv
import os

import torch as th

from src.data.ngsim_loader import load_ngsim_detector_dataset
from src.eval.metrics import (
    wave_arrival_time, wave_arrival_error,
    trajectory_collision_count, detector_reconstruction_error, acceleration_stats
)
from src.graph_prior import laplacian_penalty, compute_platooning_weights
from src.loss import detector_loss
from src.obs.detector_operator import detector_crosslane_at_times
from src.sim.rollout import rollout_idm_multilane
from src.sim.reparam import sigmoid_reparam, inverse_sigmoid_reparam


def _smoothness_stat(T: th.Tensor, leader_idx: th.Tensor, lane_id: th.Tensor) -> float:
    with th.no_grad():
        valid = leader_idx >= 0
        valid = valid & (lane_id == lane_id[th.clamp(leader_idx, min=0)])
        if valid.sum() == 0:
            return 0.0
        i = th.where(valid)[0]
        j = leader_idx[i]
        return float((T[i] - T[j]).abs().mean().item())


def fit_once(data: dict, cfg: argparse.Namespace, lambda_T: float) -> dict:
    device = data["s0"].device
    dtype = data["s0"].dtype
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

    # Use sigmoid reparameterization: u_T ∈ ℝ → T ∈ [T_min, T_max]
    u_T = th.nn.Parameter(inverse_sigmoid_reparam(
        th.full((N,), cfg.T_init, device=device, dtype=dtype),
        cfg.T_min, cfg.T_max
    ))

    params = [u_T]
    v0_param = None
    unique_lanes = None
    lane_to_idx = None
    if cfg.learn_v0 == "vector":
        v0_param = th.nn.Parameter(v0_obs.clone())
        params.append(v0_param)
    elif cfg.learn_v0 == "scalar":
        v0_param = th.nn.Parameter(th.tensor(float(v0_obs.mean().item()), device=device, dtype=dtype))
        params.append(v0_param)
    elif cfg.learn_v0 == "lane":
        # Lane-wise v0: one parameter per lane
        unique_lanes = th.unique(lane_id)
        lane_to_idx = {int(lane.item()): i for i, lane in enumerate(unique_lanes)}
        lane_indices = th.tensor([lane_to_idx[int(l.item())] for l in lane_id], device=device)
        # Initialize with mean v0 per lane
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

    history = {
        "loss": [], "L_det": [], "L_graph": [], "L_prior_T": [], "L_prior_v0": [],
        "grad_T": [], "min_spacing": [], "has_nan": []
    }

    for _ in range(cfg.iters):
        opt.zero_grad()

        T = sigmoid_reparam(u_T, cfg.T_min, cfg.T_max)
        if cfg.learn_v0 == "vector":
            v0 = th.clamp(v0_param, 0.0, cfg.v0_max)
        elif cfg.learn_v0 == "scalar":
            v0 = th.ones_like(v0_obs) * th.clamp(v0_param, 0.0, cfg.v0_max)
        elif cfg.learn_v0 == "lane":
            # Broadcast lane-wise v0 to per-vehicle
            v0_clamped = th.clamp(v0_param, 0.0, cfg.v0_max)
            v0 = v0_clamped[lane_indices]
        else:
            v0 = v0_obs

        S, V, _ = rollout_idm_multilane(
            s0_init=s0,
            v0_init=v0,
            leader_idx=leader_idx,
            num_steps=steps,
            dt=cfg.dt,
            a_max=a_max,
            b=b,
            v_target=v_target,
            s0=cfg.s0_idm,
            T_headway=T,
            ghost_v=cfg.ghost_v,
            ghost_gap0=cfg.ghost_gap0,
        )

        # Compute min spacing for stability monitoring
        with th.no_grad():
            valid_leaders = leader_idx >= 0
            if valid_leaders.any():
                follower_pos = S[:, valid_leaders]
                leader_pos = S[:, leader_idx[valid_leaders]]
                spacing = leader_pos - follower_pos
                min_spacing = float(spacing.min().item())
            else:
                min_spacing = float("inf")

        pred = detector_crosslane_at_times(
            S=S,
            V=V,
            lane_id=lane_id,
            xq=xq,
            x_dets=x_dets,
            time_indices=time_idx,
            sigma=cfg.sigma,
            half_window=10.0,
        )

        L_det = detector_loss(
            pred=pred,
            obs=obs,
            weights={"flow": 1.0, "speed": cfg.speed_weight},
            loss_type="huber",
            huber_delta=1.0,
        )

        if cfg.state_weighted_graph:
            weights = compute_platooning_weights(
                S, V, leader_idx, lane_id,
                s0=cfg.s0_platooning, vs=cfg.vs_platooning
            )
        else:
            weights = None

        L_graph = laplacian_penalty(T, leader_idx, lane_id, weights=weights)

        # Weak priors to prevent runaway
        L_prior_T = ((T - cfg.T_bar) ** 2).mean()
        L_prior_v0 = th.tensor(0.0, device=device, dtype=dtype)
        if cfg.learn_v0 != "none" and v0_param is not None:
            if cfg.learn_v0 == "lane":
                L_prior_v0 = ((v0_param - cfg.v0_bar) ** 2).mean()
            else:
                L_prior_v0 = ((v0 - cfg.v0_bar) ** 2).mean()

        loss = L_det + lambda_T * L_graph + cfg.beta_T * L_prior_T + cfg.beta_v0 * L_prior_v0

        # NaN/Inf check
        has_nan = not th.isfinite(loss)
        if has_nan:
            history["has_nan"].append(True)
            history["loss"].append(float("nan"))
            history["L_det"].append(float("nan"))
            history["L_graph"].append(float("nan"))
            history["L_prior_T"].append(float("nan"))
            history["L_prior_v0"].append(float("nan"))
            history["grad_T"].append(float("nan"))
            history["min_spacing"].append(min_spacing)
            break  # Early stop on divergence

        loss.backward()
        grad_T = float(u_T.grad.norm().item()) if u_T.grad is not None else 0.0
        opt.step()

        history["loss"].append(float(loss.item()))
        history["L_det"].append(float(L_det.item()))
        history["L_graph"].append(float(L_graph.item()))
        history["L_prior_T"].append(float(L_prior_T.item()))
        history["L_prior_v0"].append(float(L_prior_v0.item()))
        history["grad_T"].append(grad_T)
        history["min_spacing"].append(min_spacing)
        history["has_nan"].append(False)

    with th.no_grad():
        T_final = sigmoid_reparam(u_T, cfg.T_min, cfg.T_max).detach().clone()
        smooth = _smoothness_stat(T_final, leader_idx, lane_id)
        return {
            "T": T_final,
            "history": history,
            "smoothness": smooth,
            "L_det_final": history["L_det"][-1],
            "L_graph_final": history["L_graph"][-1],
            "grad_T_last": history["grad_T"][-1],
        }


def evaluate_fit(data: dict, cfg: argparse.Namespace, T: th.Tensor) -> dict:
    """Run full evaluation on fitted parameters."""
    device = data["s0"].device
    dtype = data["s0"].dtype
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

    a_max = th.full((N,), cfg.a_max, device=device, dtype=dtype)
    b = th.full((N,), cfg.b_comfort, device=device, dtype=dtype)
    v_target = th.full((N,), cfg.v_target, device=device, dtype=dtype)

    with th.no_grad():
        # Run simulation with fitted T
        S, V, A = rollout_idm_multilane(
            s0_init=s0,
            v0_init=v0_obs,
            leader_idx=leader_idx,
            num_steps=steps,
            dt=cfg.dt,
            a_max=a_max,
            b=b,
            v_target=v_target,
            s0=cfg.s0_idm,
            T_headway=T,
            ghost_v=cfg.ghost_v,
            ghost_gap0=cfg.ghost_gap0,
        )

        # Detector predictions
        pred = detector_crosslane_at_times(
            S=S, V=V, lane_id=lane_id, xq=xq, x_dets=x_dets,
            time_indices=time_idx, sigma=cfg.sigma, half_window=10.0,
        )

        # 1. Detector reconstruction error
        recon_err = detector_reconstruction_error(pred, obs)

        # 2. Trajectory plausibility
        collision_stats = trajectory_collision_count(S, leader_idx, s_min=cfg.s0_idm)
        accel_stats = acceleration_stats(V, cfg.dt)

        # 3. Wave arrival time (if duration is long enough)
        time_s = time_idx.float() * cfg.dt
        wave_metrics = {}
        if cfg.duration_s >= cfg.wave_sustained_s + 10:
            t_pred = wave_arrival_time(
                pred["speed"], time_s, cfg.wave_v_threshold, cfg.wave_sustained_s, dt=5.0
            )
            t_obs = wave_arrival_time(
                obs["speed"], time_s, cfg.wave_v_threshold, cfg.wave_sustained_s, dt=5.0
            )
            wave_err = wave_arrival_error(t_pred, t_obs)
            wave_metrics = {
                "wave_mae": float(wave_err["mae"].item()),
                "wave_rmse": float(wave_err["rmse"].item()),
                "wave_valid_count": int(wave_err["valid_count"].item()),
            }

        return {
            "flow_mse": float(recon_err.get("flow_mse", th.tensor(float("nan"))).item()),
            "speed_mse": float(recon_err.get("speed_mse", th.tensor(float("nan"))).item()),
            "flow_mae": float(recon_err.get("flow_mae", th.tensor(float("nan"))).item()),
            "speed_mae": float(recon_err.get("speed_mae", th.tensor(float("nan"))).item()),
            "collision_count": int(collision_stats["collision_count"].item()),
            "collision_fraction": float(collision_stats["collision_fraction"].item()),
            "min_spacing": float(collision_stats["min_spacing"].item()),
            "accel_mean": float(accel_stats["accel_mean"].item()),
            "accel_std": float(accel_stats["accel_std"].item()),
            "accel_max": float(accel_stats["accel_max"].item()),
            "jerk_max": float(accel_stats["jerk_max"].item()),
            **wave_metrics,
        }


def _pick_t0_base(csv_path: str) -> int:
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
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
    if min_t is None:
        raise RuntimeError("Could not infer base timestamp from csv")
    return min_t


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Minimal graph-Laplacian DA fit for IDM T (and optional v0)")
    ap.add_argument("--csv", type=str, default="data/0820am-0835am/trajectories-0820am-0835am.csv")
    ap.add_argument("--t0-ms", type=int, default=None)
    ap.add_argument("--offset-s", type=float, default=120.0)
    ap.add_argument("--duration-s", type=float, default=10.0)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--x-center", type=float, default=340.0)
    ap.add_argument("--x-half", type=float, default=150.0)
    ap.add_argument("--sigma", type=float, default=10.0)

    ap.add_argument("--iters", type=int, default=80)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--lambda-T", type=float, default=0.01)
    ap.add_argument("--lambda-T-high", type=float, default=1.0)
    ap.add_argument("--run-lambda-scan", action="store_true")
    ap.add_argument("--lambda-sweep", action="store_true",
                    help="Run full sweep: lambda_T in {0, 1e-4, 1e-3, 1e-2, 1e-1, 1}")

    ap.add_argument("--learn-v0", type=str, default="none", choices=["none", "scalar", "vector", "lane"])
    ap.add_argument("--v0-max", type=float, default=45.0)
    ap.add_argument("--T-init", type=float, default=1.5)
    ap.add_argument("--T-min", type=float, default=0.5)
    ap.add_argument("--T-max", type=float, default=3.0)
    ap.add_argument("--T-bar", type=float, default=1.5,
                    help="Prior mean for T (time headway)")
    ap.add_argument("--v0-bar", type=float, default=25.0,
                    help="Prior mean for v0 (desired speed)")
    ap.add_argument("--beta-T", type=float, default=0.0,
                    help="Weight for T prior loss")
    ap.add_argument("--beta-v0", type=float, default=0.0,
                    help="Weight for v0 prior loss")

    ap.add_argument("--a-max", type=float, default=2.0)
    ap.add_argument("--b-comfort", type=float, default=2.0)
    ap.add_argument("--v-target", type=float, default=25.0)
    ap.add_argument("--s0-idm", type=float, default=2.0)
    ap.add_argument("--ghost-v", type=float, default=15.0)
    ap.add_argument("--ghost-gap0", type=float, default=50.0)

    ap.add_argument("--speed-weight", type=float, default=0.1)
    ap.add_argument("--state-weighted-graph", action="store_true")
    ap.add_argument("--s0-platooning", type=float, default=5.0,
                    help="Spacing scale for platooning weights (m)")
    ap.add_argument("--vs-platooning", type=float, default=2.0,
                    help="Velocity scale for platooning weights (m/s)")

    # Evaluation options
    ap.add_argument("--eval", action="store_true",
                    help="Run full evaluation metrics after fitting")
    ap.add_argument("--wave-v-threshold", type=float, default=15.0,
                    help="Speed threshold for wave detection (m/s)")
    ap.add_argument("--wave-sustained-s", type=float, default=20.0,
                    help="Sustained duration for wave detection (s)")

    return ap.parse_args()


def main() -> None:
    cfg = parse_args()
    if not os.path.exists(cfg.csv):
        raise FileNotFoundError(f"CSV not found: {cfg.csv}")

    device = "cuda" if th.cuda.is_available() else "cpu"
    dtype = th.float32

    t0_base = _pick_t0_base(cfg.csv)
    t0_ms = cfg.t0_ms if cfg.t0_ms is not None else t0_base + int(round(cfg.offset_s * 1000.0))

    x_min = cfg.x_center - cfg.x_half
    x_max = cfg.x_center + cfg.x_half

    data = load_ngsim_detector_dataset(
        csv_path=cfg.csv,
        t0_ms=t0_ms,
        duration_s=cfg.duration_s,
        x_min=x_min,
        x_max=x_max,
        dt=cfg.dt,
        sigma=cfg.sigma,
        device=device,
        dtype=dtype,
    )
    if data is None:
        raise RuntimeError("No usable vehicles found for requested slice")

    print(f"Device: {device}")
    print(f"t0_ms: {t0_ms}, vehicles: {data['s0'].shape[0]}, lanes: {th.unique(data['lane_id']).tolist()}")
    print(f"detectors: {len(data['x_dets'])}, 5s points: {data['time_idx_5s'].numel()}")

    if cfg.lambda_sweep:
        # Full sweep: lambda_T in {0, 1e-4, 1e-3, 1e-2, 1e-1, 1}
        lambdas = [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
        results = []
        print("\nRunning lambda sweep...")
        for lam in lambdas:
            out = fit_once(data, cfg, lambda_T=lam)
            results.append({
                "lambda": lam,
                "L_det": out["L_det_final"],
                "L_graph": out["L_graph_final"],
                "smoothness": out["smoothness"],
                "T_mean": float(out["T"].mean().item()),
                "T_std": float(out["T"].std().item()),
            })
            print(f"  lambda={lam:.1e}: L_det={out['L_det_final']:.6f}, smooth={out['smoothness']:.6f}")

        print("\n" + "=" * 70)
        print("Lambda Sweep Summary")
        print("=" * 70)
        print(f"{'lambda':>10} {'L_det':>12} {'L_graph':>12} {'smooth':>10} {'T_mean':>8} {'T_std':>8}")
        print("-" * 70)
        for r in results:
            print(f"{r['lambda']:>10.1e} {r['L_det']:>12.6f} {r['L_graph']:>12.6f} {r['smoothness']:>10.6f} {r['T_mean']:>8.4f} {r['T_std']:>8.4f}")

    elif cfg.run_lambda_scan:
        low = fit_once(data, cfg, lambda_T=cfg.lambda_T)
        high = fit_once(data, cfg, lambda_T=cfg.lambda_T_high)

        print("\nLambda scan summary")
        print(f"lambda={cfg.lambda_T:.4f}:   L_det={low['L_det_final']:.6f}, smooth={low['smoothness']:.6f}, grad_T_last={low['grad_T_last']:.6f}")
        print(f"lambda={cfg.lambda_T_high:.4f}: L_det={high['L_det_final']:.6f}, smooth={high['smoothness']:.6f}, grad_T_last={high['grad_T_last']:.6f}")
    else:
        out = fit_once(data, cfg, lambda_T=cfg.lambda_T)
        print("\nSingle run summary")
        print(f"L_det_final={out['L_det_final']:.6f}")
        print(f"L_graph_final={out['L_graph_final']:.6f}")
        print(f"T_mean={out['T'].mean().item():.6f}, T_std={out['T'].std().item():.6f}")
        print(f"smoothness={out['smoothness']:.6f}")
        print(f"grad_T_last={out['grad_T_last']:.6f}")

        if cfg.eval:
            print("\n" + "=" * 50)
            print("Evaluation Metrics")
            print("=" * 50)
            eval_metrics = evaluate_fit(data, cfg, out["T"])

            print("\nDetector Reconstruction:")
            print(f"  flow_mse={eval_metrics['flow_mse']:.6f}, flow_mae={eval_metrics['flow_mae']:.6f}")
            print(f"  speed_mse={eval_metrics['speed_mse']:.6f}, speed_mae={eval_metrics['speed_mae']:.6f}")

            print("\nTrajectory Plausibility:")
            print(f"  collision_count={eval_metrics['collision_count']}, collision_frac={eval_metrics['collision_fraction']:.4f}")
            print(f"  min_spacing={eval_metrics['min_spacing']:.2f} m")
            print(f"  accel: mean={eval_metrics['accel_mean']:.3f}, std={eval_metrics['accel_std']:.3f}, max={eval_metrics['accel_max']:.3f} m/s²")
            print(f"  jerk_max={eval_metrics['jerk_max']:.3f} m/s³")

            if "wave_mae" in eval_metrics:
                print("\nWave Arrival Time:")
                print(f"  MAE={eval_metrics['wave_mae']:.2f} s, RMSE={eval_metrics['wave_rmse']:.2f} s")
                print(f"  valid_detectors={eval_metrics['wave_valid_count']}")


if __name__ == "__main__":
    main()
