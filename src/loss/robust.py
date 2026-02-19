"""
Robust loss functions for traffic parameter optimization.

All losses are NaN-safe and support masking for missing/invalid data.
"""
from __future__ import annotations

import torch as th


def _nan_to_zero(x: th.Tensor) -> th.Tensor:
    """Replace NaN/Inf with zeros for safe masking operations."""
    return th.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def masked_mse(
    pred: th.Tensor,
    obs: th.Tensor,
    mask: th.Tensor,
) -> th.Tensor:
    """
    NaN-safe masked MSE loss.

    Args:
        pred: predictions
        obs: observations
        mask: bool mask (True = valid)

    Returns:
        Scalar loss (never NaN)
    """
    pred = _nan_to_zero(pred)
    obs = _nan_to_zero(obs)
    diff = pred - obs
    diff = th.where(mask, diff, th.zeros_like(diff))
    denom = mask.to(diff.dtype).sum().clamp_min(1.0)
    return (diff * diff).sum() / denom


def masked_huber(
    pred: th.Tensor,
    obs: th.Tensor,
    mask: th.Tensor,
    delta: float = 1.0,
) -> th.Tensor:
    """
    NaN-safe masked Huber loss.

    L(x) = 0.5 * x^2           if |x| <= delta
           delta * (|x| - 0.5 * delta)  otherwise

    Args:
        pred: predictions
        obs: observations
        mask: bool mask (True = valid)
        delta: threshold for quadratic vs linear regime

    Returns:
        Scalar loss (never NaN)
    """
    pred = _nan_to_zero(pred)
    obs = _nan_to_zero(obs)
    diff = th.abs(pred - obs)
    diff = th.where(mask, diff, th.zeros_like(diff))

    quadratic = 0.5 * diff * diff
    linear = delta * (diff - 0.5 * delta)
    loss_per_elem = th.where(diff <= delta, quadratic, linear)

    denom = mask.to(diff.dtype).sum().clamp_min(1.0)
    return loss_per_elem.sum() / denom


def masked_log1p_mse(
    pred: th.Tensor,
    obs: th.Tensor,
    mask: th.Tensor,
) -> th.Tensor:
    """
    NaN-safe masked MSE on log1p transformed values.

    Useful for density/flow with heavy tails.

    Args:
        pred: predictions (non-negative)
        obs: observations (non-negative)
        mask: bool mask (True = valid)

    Returns:
        Scalar loss (never NaN)
    """
    pred = _nan_to_zero(pred).clamp_min(0.0)
    obs = _nan_to_zero(obs).clamp_min(0.0)
    diff = th.log1p(pred) - th.log1p(obs)
    diff = th.where(mask, diff, th.zeros_like(diff))
    denom = mask.to(diff.dtype).sum().clamp_min(1.0)
    return (diff * diff).sum() / denom


def detector_loss(
    pred: dict[str, th.Tensor],
    obs: dict[str, th.Tensor],
    weights: dict[str, float] | None = None,
    loss_type: str = "mse",
    huber_delta: float = 1.0,
) -> th.Tensor:
    """
    Compute detector reconstruction loss.

    Args:
        pred: {"speed": [K, Nd], "flow": [K, Nd], ...}
        obs: {"speed": [K, Nd], "flow": [K, Nd], ...}
        weights: channel weights, default {"speed": 1.0, "flow": 1.0}
        loss_type: "mse" or "huber"
        huber_delta: delta for Huber loss

    Returns:
        Scalar loss
    """
    if weights is None:
        weights = {"speed": 1.0, "flow": 1.0}

    loss = th.tensor(0.0, device=next(iter(pred.values())).device)

    for key in weights:
        if key not in pred or key not in obs:
            continue

        p = pred[key]
        o = obs[key]
        mask = th.isfinite(p) & th.isfinite(o)

        if mask.sum() == 0:
            continue

        if loss_type == "mse":
            loss = loss + weights[key] * masked_mse(p, o, mask)
        elif loss_type == "huber":
            loss = loss + weights[key] * masked_huber(p, o, mask, delta=huber_delta)
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

    return loss
