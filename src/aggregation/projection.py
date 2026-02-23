"""
Projection Operator: Micro states → Raw detector signal.

Minimal smoothing projection that assigns vehicles to detector cells.
Laplacian filtering will handle smoothing separately.

Math:
    For each detector j at position x_j with cell width dx:
    - count: c[n,j] = number of vehicles in cell
    - speed: v_raw[n,j] = mean velocity of vehicles in cell
    - density: rho_raw[n,j] = c[n,j] / dx
    - flow: q_raw[n,j] = rho_raw[n,j] * v_raw[n,j]
"""
from __future__ import annotations

import torch as th


def hard_binning_projection(
    S: th.Tensor,
    V: th.Tensor,
    x_dets: th.Tensor,
    dx: float,
    eps: float = 1e-6,
) -> dict[str, th.Tensor]:
    """
    Hard binning projection: assign each vehicle to nearest detector cell.

    Args:
        S: [T, N] positions
        V: [T, N] velocities
        x_dets: [J] detector positions
        dx: cell width (detector spacing)
        eps: small value for numerical stability

    Returns:
        dict with:
            count: [T, J] vehicle count per cell
            v_raw: [T, J] mean speed per cell (NaN where count=0)
            rho_raw: [T, J] density per cell
            q_raw: [T, J] flow per cell
    """
    T, N = S.shape
    J = x_dets.shape[0]
    device, dtype = S.device, S.dtype

    # Cell boundaries: [x_j - dx/2, x_j + dx/2]
    half_dx = dx / 2.0

    count = th.zeros((T, J), device=device, dtype=dtype)
    v_sum = th.zeros((T, J), device=device, dtype=dtype)

    for j in range(J):
        x_lo = x_dets[j] - half_dx
        x_hi = x_dets[j] + half_dx

        # Mask: [T, N] - which vehicles are in this cell at each time
        in_cell = (S >= x_lo) & (S < x_hi)

        # Count vehicles in cell
        count[:, j] = in_cell.sum(dim=1).float()

        # Sum of velocities in cell (for mean calculation)
        v_masked = V * in_cell.float()
        v_sum[:, j] = v_masked.sum(dim=1)

    # Mean speed: v_raw = v_sum / count (NaN where count=0)
    v_raw = th.where(
        count > eps,
        v_sum / (count + eps),
        th.full_like(count, float('nan'))
    )

    # Density and flow
    rho_raw = count / dx
    q_raw = rho_raw * th.nan_to_num(v_raw, nan=0.0)

    return {
        "count": count,
        "v_raw": v_raw,
        "rho_raw": rho_raw,
        "q_raw": q_raw,
    }


def soft_assignment_projection(
    S: th.Tensor,
    V: th.Tensor,
    x_dets: th.Tensor,
    dx: float,
    sigma: float = None,
    eps: float = 1e-6,
) -> dict[str, th.Tensor]:
    """
    Soft assignment projection with small-support kernel.

    Uses a narrow Gaussian kernel (sigma = dx/4 by default) for
    minimal smoothing while maintaining differentiability.

    Args:
        S: [T, N] positions
        V: [T, N] velocities
        x_dets: [J] detector positions
        dx: cell width (detector spacing)
        sigma: kernel width (default: dx/4 for minimal smoothing)
        eps: small value for numerical stability

    Returns:
        dict with v_raw, rho_raw, q_raw: [T, J]
    """
    if sigma is None:
        sigma = dx / 4.0  # Minimal smoothing

    T, N = S.shape
    J = x_dets.shape[0]
    device, dtype = S.device, S.dtype

    # Compute distances: [T, N, J]
    # S[:, :, None] - x_dets[None, None, :] would be [T, N, J]
    dist = S.unsqueeze(-1) - x_dets.view(1, 1, -1)  # [T, N, J]

    # Gaussian weights: [T, N, J]
    W = th.exp(-0.5 * (dist / sigma) ** 2)

    # Sum weights per detector: [T, J]
    W_sum = W.sum(dim=1)  # [T, J]

    # Weighted velocity sum: [T, J]
    V_expanded = V.unsqueeze(-1)  # [T, N, 1]
    WV_sum = (W * V_expanded).sum(dim=1)  # [T, J]

    # Mean speed
    v_raw = th.where(
        W_sum > eps,
        WV_sum / (W_sum + eps),
        th.full_like(W_sum, float('nan'))
    )

    # Density (normalized by cell width and kernel normalization)
    norm = (2.0 * 3.14159265359) ** 0.5 * sigma
    rho_raw = W_sum / (norm * dx)

    # Flow
    q_raw = rho_raw * th.nan_to_num(v_raw, nan=0.0)

    return {
        "v_raw": v_raw,
        "rho_raw": rho_raw,
        "q_raw": q_raw,
        "W_sum": W_sum,
    }


def fill_empty_cells(
    v_raw: th.Tensor,
    method: str = "neighbor",
) -> th.Tensor:
    """
    Fill NaN values in raw signal.

    Args:
        v_raw: [T, J] raw signal with NaN for empty cells
        method: "neighbor" (spatial interpolation) or "forward" (time-forward fill)

    Returns:
        v_filled: [T, J] filled signal
    """
    v_filled = v_raw.clone()

    if method == "neighbor":
        # Simple neighbor interpolation: average of left and right
        T, J = v_raw.shape
        for j in range(J):
            nan_mask = th.isnan(v_filled[:, j])
            if nan_mask.any():
                # Get left and right values
                left_val = v_filled[:, j - 1] if j > 0 else v_filled[:, j + 1]
                right_val = v_filled[:, j + 1] if j < J - 1 else v_filled[:, j - 1]

                # Average (handle NaN in neighbors)
                fill_val = th.nanmean(th.stack([left_val, right_val], dim=-1), dim=-1)
                v_filled[:, j] = th.where(nan_mask, fill_val, v_filled[:, j])

    elif method == "forward":
        # Time-forward fill
        T, J = v_raw.shape
        for t in range(1, T):
            nan_mask = th.isnan(v_filled[t])
            v_filled[t] = th.where(nan_mask, v_filled[t - 1], v_filled[t])

    # Final fallback: replace remaining NaN with 0
    v_filled = th.nan_to_num(v_filled, nan=0.0)

    return v_filled
