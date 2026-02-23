"""
Reparameterization utilities for bounded parameters.

Sigmoid reparameterization maps unbounded u ∈ ℝ to bounded x ∈ [low, high]:
    x = low + (high - low) * sigmoid(u)

This ensures gradients flow smoothly even at parameter boundaries,
unlike hard clamp which has zero gradient at boundaries.
"""
from __future__ import annotations

import torch as th


def sigmoid_reparam(u: th.Tensor, low: float, high: float) -> th.Tensor:
    """
    Map unbounded parameter u to bounded [low, high] via sigmoid.

    Args:
        u: unbounded parameter tensor
        low: lower bound
        high: upper bound

    Returns:
        x = low + (high - low) * sigmoid(u)
    """
    return low + (high - low) * th.sigmoid(u)


def inverse_sigmoid_reparam(x: th.Tensor, low: float, high: float, eps: float = 1e-6) -> th.Tensor:
    """
    Compute u such that sigmoid_reparam(u, low, high) ≈ x.

    Used to initialize u from a desired initial value x.

    Args:
        x: target value in [low, high]
        low: lower bound
        high: upper bound
        eps: small value to avoid log(0)

    Returns:
        u = logit((x - low) / (high - low))
    """
    x_norm = (x - low) / (high - low)
    x_norm = th.clamp(x_norm, eps, 1.0 - eps)
    return th.log(x_norm / (1.0 - x_norm))
