from __future__ import annotations

import torch as th


def compute_platooning_weights(
    S: th.Tensor,
    V: th.Tensor,
    leader_idx: th.Tensor,
    lane_id: th.Tensor,
    s0: float = 5.0,
    vs: float = 2.0,
    time_idx: int = 0,
) -> th.Tensor:
    """
    Compute soft platooning weights: w_i = exp(-s_i/s0) * exp(-|dv_i|/vs)

    Vehicles close together with similar speeds get higher weight,
    encouraging parameter similarity within platoons.

    Args:
        S: [T, N] position trajectories
        V: [T, N] velocity trajectories
        leader_idx: [N] leader indices (-1 if no leader)
        lane_id: [N] lane assignments
        s0: spacing scale (default 5.0 m)
        vs: velocity scale (default 2.0 m/s)
        time_idx: which timestep to evaluate (default 0)

    Returns:
        [N] detached weights (no gradient through weights)
    """
    N = S.shape[1]
    device, dtype = S.device, S.dtype

    valid = leader_idx >= 0
    safe_leader = th.clamp(leader_idx, min=0)
    same_lane = lane_id == lane_id[safe_leader]
    valid = valid & same_lane

    # Extract state at specified time
    s = S[time_idx]
    v = V[time_idx]

    # Compute spacing and relative velocity
    spacing = th.zeros(N, device=device, dtype=dtype)
    rel_speed = th.zeros(N, device=device, dtype=dtype)

    spacing[valid] = s[safe_leader[valid]] - s[valid]
    rel_speed[valid] = v[safe_leader[valid]] - v[valid]

    # Compute weights (only for valid edges)
    w = th.exp(-spacing.abs() / s0) * th.exp(-rel_speed.abs() / vs)
    w = w * valid.float()

    return w.detach()


def laplacian_penalty(
    param: th.Tensor,
    leader_idx: th.Tensor,
    lane_id: th.Tensor,
    weights: th.Tensor | None = None,
) -> th.Tensor:
    """
    Leader-edge graph penalty: sum_i w_i * (p_i - p_leader(i))^2.

    Only same-lane valid leader edges are used.
    """
    if param.ndim != 1:
        raise ValueError("param must be 1D")
    if leader_idx.shape != param.shape or lane_id.shape != param.shape:
        raise ValueError("leader_idx and lane_id must match param shape")

    valid = leader_idx >= 0
    safe_leader = th.clamp(leader_idx, min=0)
    same_lane = lane_id == lane_id[safe_leader]
    valid = valid & same_lane

    if valid.sum() == 0:
        return th.zeros((), device=param.device, dtype=param.dtype)

    i = th.where(valid)[0]
    j = leader_idx[i]
    diff_sq = (param[i] - param[j]) ** 2

    if weights is None:
        w = th.ones_like(diff_sq)
    else:
        if weights.shape == param.shape:
            w = weights[i]
        elif weights.shape == diff_sq.shape:
            w = weights
        else:
            raise ValueError("weights must have shape [N] or [num_valid_edges]")

    return (w * diff_sq).sum()
