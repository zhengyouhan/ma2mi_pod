from __future__ import annotations

import argparse
import csv
import os

import torch as th

from new_project.src.data.ngsim_loader import load_ngsim_detector_dataset
from new_project.src.graph_prior import laplacian_penalty
from new_project.src.obs.detector_operator import detector_outputs_at_times
from new_project.src.sim.rollout import rollout_idm_multilane


def _smoothness_stat(T: th.Tensor, leader_idx: th.Tensor, lane_id: th.Tensor) -> float:
    with th.no_grad():
        valid = leader_idx >= 0
        valid = valid & (lane_id == lane_id[th.clamp(leader_idx, min=0)])
        if valid.sum() == 0:
            return 0.0
        i = th.where(valid)[0]
        j = leader_idx[i]
        return float((T[i] - T[j]).abs().mean().item())


def _masked_mse(a: th.Tensor, b: th.Tensor) -> th.Tensor:
    m = th.isfinite(a) & th.isfinite(b)
    if m.sum() == 0:
        return th.zeros((), device=a.device, dtype=a.dtype)
    return ((a[m] - b[m]) ** 2).mean()


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

    T_param = th.nn.Parameter(th.full((N,), cfg.T_init, device=device, dtype=dtype))

    params = [T_param]
    v0_param = None
    if cfg.learn_v0 == "vector":
        v0_param = th.nn.Parameter(v0_obs.clone())
        params.append(v0_param)
    elif cfg.learn_v0 == "scalar":
        v0_param = th.nn.Parameter(th.tensor(float(v0_obs.mean().item()), device=device, dtype=dtype))
        params.append(v0_param)

    opt = th.optim.Adam(params, lr=cfg.lr)

    a_max = th.full((N,), cfg.a_max, device=device, dtype=dtype)
    b = th.full((N,), cfg.b_comfort, device=device, dtype=dtype)
    v_target = th.full((N,), cfg.v_target, device=device, dtype=dtype)

    history = {"loss": [], "L_det": [], "L_graph": [], "grad_T": []}

    for _ in range(cfg.iters):
        opt.zero_grad()

        T = th.clamp(T_param, cfg.T_min, cfg.T_max)
        if cfg.learn_v0 == "vector":
            v0 = th.clamp(v0_param, 0.0, cfg.v0_max)
        elif cfg.learn_v0 == "scalar":
            v0 = th.ones_like(v0_obs) * th.clamp(v0_param, 0.0, cfg.v0_max)
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

        pred = detector_outputs_at_times(
            S=S,
            V=V,
            xq=xq,
            x_dets=x_dets,
            time_indices=time_idx,
            sigma=cfg.sigma,
            half_window=10.0,
        )

        L_k = _masked_mse(pred["density"], obs["density"])
        L_v = _masked_mse(pred["speed"], obs["speed"])
        L_det = L_k + cfg.speed_weight * L_v

        if cfg.state_weighted_graph:
            weights = (1.0 / (1.0 + V[0].detach())).to(dtype)
        else:
            weights = None

        L_graph = laplacian_penalty(T, leader_idx, lane_id, weights=weights)
        loss = L_det + lambda_T * L_graph

        loss.backward()
        grad_T = float(T_param.grad.norm().item()) if T_param.grad is not None else 0.0
        opt.step()

        history["loss"].append(float(loss.item()))
        history["L_det"].append(float(L_det.item()))
        history["L_graph"].append(float(L_graph.item()))
        history["grad_T"].append(grad_T)

    with th.no_grad():
        T_final = th.clamp(T_param, cfg.T_min, cfg.T_max).detach().clone()
        smooth = _smoothness_stat(T_final, leader_idx, lane_id)
        return {
            "T": T_final,
            "history": history,
            "smoothness": smooth,
            "L_det_final": history["L_det"][-1],
            "L_graph_final": history["L_graph"][-1],
            "grad_T_last": history["grad_T"][-1],
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

    ap.add_argument("--learn-v0", type=str, default="none", choices=["none", "scalar", "vector"])
    ap.add_argument("--v0-max", type=float, default=45.0)
    ap.add_argument("--T-init", type=float, default=1.5)
    ap.add_argument("--T-min", type=float, default=0.5)
    ap.add_argument("--T-max", type=float, default=3.0)

    ap.add_argument("--a-max", type=float, default=2.0)
    ap.add_argument("--b-comfort", type=float, default=2.0)
    ap.add_argument("--v-target", type=float, default=25.0)
    ap.add_argument("--s0-idm", type=float, default=2.0)
    ap.add_argument("--ghost-v", type=float, default=15.0)
    ap.add_argument("--ghost-gap0", type=float, default=50.0)

    ap.add_argument("--speed-weight", type=float, default=0.1)
    ap.add_argument("--state-weighted-graph", action="store_true")

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

    if cfg.run_lambda_scan:
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


if __name__ == "__main__":
    main()
