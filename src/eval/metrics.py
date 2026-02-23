"""
Evaluation metrics for traffic wave reconstruction.

Phase 5 metrics:
- Wave arrival time error (primary KPI)
- Detector reconstruction error (secondary KPI)
- Trajectory physical plausibility
"""
from __future__ import annotations

import torch as th


def wave_arrival_time(
    speed: th.Tensor,
    time_s: th.Tensor,
    v_threshold: float,
    sustained_s: float = 20.0,
    dt: float = 5.0,
) -> th.Tensor:
    """
    Detect wave arrival time at each detector.

    Wave arrival = first time when speed drops below threshold
    and remains below for sustained_s seconds.

    Args:
        speed: [T, Nd] speed at each detector over time
        time_s: [T] time in seconds
        v_threshold: speed threshold (m/s)
        sustained_s: required sustained duration (s)
        dt: time step between observations (s)

    Returns:
        t_arrival: [Nd] arrival time per detector (NaN if no arrival)
    """
    T, Nd = speed.shape
    sustained_steps = int(sustained_s / dt)

    t_arrival = th.full((Nd,), float("nan"), device=speed.device, dtype=speed.dtype)

    below_threshold = speed < v_threshold  # [T, Nd]

    for j in range(Nd):
        for t in range(T - sustained_steps + 1):
            if below_threshold[t : t + sustained_steps, j].all():
                t_arrival[j] = time_s[t]
                break

    return t_arrival


def wave_arrival_error(
    t_pred: th.Tensor,
    t_obs: th.Tensor,
) -> dict[str, th.Tensor]:
    """
    Compute wave arrival time error statistics.

    Args:
        t_pred: [Nd] predicted arrival times
        t_obs: [Nd] observed arrival times

    Returns:
        dict with: mae, rmse, valid_count
    """
    valid = th.isfinite(t_pred) & th.isfinite(t_obs)
    if valid.sum() == 0:
        return {
            "mae": th.tensor(float("nan")),
            "rmse": th.tensor(float("nan")),
            "valid_count": th.tensor(0),
        }

    diff = th.abs(t_pred[valid] - t_obs[valid])
    return {
        "mae": diff.mean(),
        "rmse": th.sqrt((diff**2).mean()),
        "valid_count": valid.sum(),
    }


def trajectory_collision_count(
    S: th.Tensor,
    leader_idx: th.Tensor,
    s_min: float = 2.0,
) -> dict[str, th.Tensor]:
    """
    Count collision events (spacing below minimum).

    Args:
        S: [T, N] positions over time
        leader_idx: [N] leader index per vehicle (-1 if none)
        s_min: minimum safe spacing (m)

    Returns:
        dict with: collision_count, collision_fraction, min_spacing
    """
    T, N = S.shape
    has_leader = leader_idx >= 0
    safe_idx = th.clamp(leader_idx, min=0)

    # Compute spacing to leader
    s_lead = S[:, safe_idx]  # [T, N]
    spacing = s_lead - S  # [T, N]

    # Only consider vehicles with leaders
    spacing_valid = spacing[:, has_leader]  # [T, N_with_leader]

    collisions = spacing_valid < s_min
    collision_count = collisions.sum()
    total_pairs = spacing_valid.numel()

    return {
        "collision_count": collision_count,
        "collision_fraction": collision_count.float() / max(total_pairs, 1),
        "min_spacing": spacing_valid.min() if spacing_valid.numel() > 0 else th.tensor(float("inf")),
    }


def parameter_smoothness(
    param: th.Tensor,
    leader_idx: th.Tensor,
    lane_id: th.Tensor,
) -> dict[str, th.Tensor]:
    """
    Compute smoothness statistics for parameters on the leader graph.

    Args:
        param: [N] parameter values (e.g., T or v0)
        leader_idx: [N] leader index per vehicle (-1 if none)
        lane_id: [N] lane ID per vehicle

    Returns:
        dict with: mean_diff, std_diff, max_diff
    """
    valid = leader_idx >= 0
    safe_leader = th.clamp(leader_idx, min=0)
    same_lane = lane_id == lane_id[safe_leader]
    valid = valid & same_lane

    if valid.sum() == 0:
        return {
            "mean_diff": th.tensor(0.0),
            "std_diff": th.tensor(0.0),
            "max_diff": th.tensor(0.0),
        }

    i = th.where(valid)[0]
    j = leader_idx[i]
    diff = th.abs(param[i] - param[j])

    return {
        "mean_diff": diff.mean(),
        "std_diff": diff.std() if diff.numel() > 1 else th.tensor(0.0),
        "max_diff": diff.max(),
    }


def detector_reconstruction_error(
    pred: dict[str, th.Tensor],
    obs: dict[str, th.Tensor],
) -> dict[str, th.Tensor]:
    """
    Compute per-channel reconstruction error.

    Args:
        pred: dict with 'flow', 'speed' tensors [T, Nd]
        obs: dict with 'flow', 'speed' tensors [T, Nd]

    Returns:
        dict with: flow_mse, speed_mse, flow_mae, speed_mae
    """
    results = {}
    for key in ["flow", "speed"]:
        if key in pred and key in obs:
            p, o = pred[key], obs[key]
            valid = th.isfinite(p) & th.isfinite(o)
            if valid.sum() > 0:
                diff = p[valid] - o[valid]
                results[f"{key}_mse"] = (diff**2).mean()
                results[f"{key}_mae"] = diff.abs().mean()
            else:
                results[f"{key}_mse"] = th.tensor(float("nan"))
                results[f"{key}_mae"] = th.tensor(float("nan"))
    return results


def acceleration_stats(
    V: th.Tensor,
    dt: float,
) -> dict[str, th.Tensor]:
    """
    Compute acceleration and jerk statistics for trajectory plausibility.

    Args:
        V: [T, N] velocity trajectories
        dt: time step (s)

    Returns:
        dict with: accel_mean, accel_std, accel_max, jerk_max
    """
    # Acceleration: dv/dt
    accel = (V[1:] - V[:-1]) / dt  # [T-1, N]

    # Jerk: da/dt
    jerk = (accel[1:] - accel[:-1]) / dt if accel.shape[0] > 1 else th.zeros(1)

    return {
        "accel_mean": accel.mean(),
        "accel_std": accel.std(),
        "accel_max": accel.abs().max(),
        "jerk_max": jerk.abs().max() if jerk.numel() > 0 else th.tensor(0.0),
    }
