"""
PDE Constraints for IDM-Based Inference.

Implements continuity equation (∂t ρ + ∂x q = 0) as soft regularization.

Math:
    Continuity equation: ∂ρ/∂t + ∂q/∂x = 0
    Residual: r_cont = ∂ρ/∂t + ∂q/∂x
    Loss: L_cont = mean(r_cont^2) over interior region
"""
from __future__ import annotations

import torch as th

from src.obs.detector_operator import micro_to_macro_gaussian


def _soft_conditional(
    condition_value: th.Tensor,
    threshold: float,
    safe_value: th.Tensor,
    computed_value: th.Tensor,
    beta: float = 20.0,
) -> th.Tensor:
    """Smooth conditional: returns computed_value where condition > threshold."""
    mask = th.sigmoid(beta * (condition_value - threshold))
    return mask * computed_value + (1.0 - mask) * safe_value


def aggregate_macro_fields(
    S: th.Tensor,
    V: th.Tensor,
    lane_id: th.Tensor,
    xq: th.Tensor,
    sigma: float = 10.0,
    eps: float = 1e-6,
) -> dict[str, th.Tensor]:
    """
    Aggregate micro trajectories to macro fields on full space-time grid.

    Cross-lane aggregation:
        rho_tot = Σ_ℓ rho_ℓ
        q_tot = Σ_ℓ q_ℓ
        v_tot = (Σ_ℓ q_ℓ · v_ℓ) / (q_tot + eps)

    Args:
        S: [T, N] positions
        V: [T, N] velocities
        lane_id: [N] lane ID per vehicle
        xq: [M] spatial query grid
        sigma: Gaussian kernel width
        eps: small value to avoid division by zero

    Returns:
        dict with:
            rho: [T, M] density
            v: [T, M] velocity
            q: [T, M] flow
    """
    if S.shape != V.shape or S.ndim != 2:
        raise ValueError("S and V must have shape [T, N]")
    if lane_id.shape[0] != S.shape[1]:
        raise ValueError("lane_id must have length N")

    T, _ = S.shape
    M = xq.shape[0]
    device, dtype = S.device, S.dtype

    unique_lanes = th.unique(lane_id)

    rho_per_lane = []
    u_per_lane = []
    q_per_lane = []

    for lane in unique_lanes:
        mask = lane_id == lane
        n_lane = mask.sum().item()

        if n_lane == 0:
            rho_per_lane.append(th.zeros((T, M), device=device, dtype=dtype))
            u_per_lane.append(th.zeros((T, M), device=device, dtype=dtype))
            q_per_lane.append(th.zeros((T, M), device=device, dtype=dtype))
        else:
            S_lane = S[:, mask]
            V_lane = V[:, mask]
            rho_l, u_l, q_l = micro_to_macro_gaussian(S_lane, V_lane, xq, sigma=sigma)
            rho_per_lane.append(rho_l)
            u_per_lane.append(u_l)
            q_per_lane.append(q_l)

    # Stack: [num_lanes, T, M]
    rho_stack = th.stack(rho_per_lane, dim=0)
    u_stack = th.stack(u_per_lane, dim=0)
    q_stack = th.stack(q_per_lane, dim=0)

    # Aggregate across lanes: [T, M]
    rho_tot = rho_stack.sum(dim=0)
    q_tot = q_stack.sum(dim=0)

    # Flow-weighted speed
    qv_sum = (q_stack * u_stack).sum(dim=0)
    v_tot = _soft_conditional(
        condition_value=q_tot,
        threshold=eps,
        safe_value=th.zeros_like(q_tot),
        computed_value=qv_sum / (q_tot + eps),
        beta=20.0,
    )

    return {"rho": rho_tot, "v": v_tot, "q": q_tot}


def compute_continuity_residual(
    rho: th.Tensor,
    q: th.Tensor,
    dt: float,
    dx: float,
) -> th.Tensor:
    """
    Compute continuity equation residual: r = ∂ρ/∂t + ∂q/∂x

    Uses central differences for interior points.

    Args:
        rho: [T, M] density
        q: [T, M] flow
        dt: time step
        dx: spatial step

    Returns:
        residual: [T-2, M-2] residual at interior points
    """
    # Temporal derivative: (rho[t+1] - rho[t-1]) / (2*dt)
    drho_dt = (rho[2:, :] - rho[:-2, :]) / (2 * dt)  # [T-2, M]

    # Spatial derivative: (q[:, j+1] - q[:, j-1]) / (2*dx)
    dq_dx = (q[:, 2:] - q[:, :-2]) / (2 * dx)  # [T, M-2]

    # Align to interior: [T-2, M-2]
    drho_dt_interior = drho_dt[:, 1:-1]
    dq_dx_interior = dq_dx[1:-1, :]

    residual = drho_dt_interior + dq_dx_interior
    return residual


def continuity_loss(
    rho: th.Tensor,
    q: th.Tensor,
    dt: float,
    dx: float,
    boundary_cells: int = 5,
) -> th.Tensor:
    """
    Continuity equation loss: L = mean(r^2) over interior region.

    Args:
        rho: [T, M] density
        q: [T, M] flow
        dt: time step
        dx: spatial step
        boundary_cells: number of cells to exclude from each spatial boundary

    Returns:
        scalar loss
    """
    residual = compute_continuity_residual(rho, q, dt, dx)

    # Mask additional boundary cells if requested
    if boundary_cells > 0 and residual.shape[1] > 2 * boundary_cells:
        residual = residual[:, boundary_cells:-boundary_cells]

    return (residual ** 2).mean()
