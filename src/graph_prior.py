from __future__ import annotations

import torch as th


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
