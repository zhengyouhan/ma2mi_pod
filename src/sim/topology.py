from __future__ import annotations

import torch as th


def build_lane_masks(lane_id: th.Tensor) -> dict[int, th.Tensor]:
    """Return boolean mask per lane id."""
    lane_id = lane_id.to(th.long)
    masks: dict[int, th.Tensor] = {}
    for lane in th.unique(lane_id).tolist():
        masks[int(lane)] = lane_id == int(lane)
    return masks


def compute_leader_idx_per_lane(positions: th.Tensor, lane_id: th.Tensor) -> th.Tensor:
    """
    Leader index for each vehicle in same lane (no lane changes).

    leader_idx[i] = j where j is nearest vehicle ahead in same lane; -1 if none.
    """
    if positions.ndim != 1 or lane_id.ndim != 1 or positions.shape[0] != lane_id.shape[0]:
        raise ValueError("positions and lane_id must be 1D with same length")

    n = positions.shape[0]
    leader_idx = th.full((n,), -1, dtype=th.long, device=positions.device)

    for lane, mask in build_lane_masks(lane_id).items():
        idx = th.where(mask)[0]
        if idx.numel() == 0:
            continue
        lane_pos = positions[idx]
        order = th.argsort(lane_pos)
        sorted_idx = idx[order]
        if sorted_idx.numel() > 1:
            leader_idx[sorted_idx[:-1]] = sorted_idx[1:]
    return leader_idx
