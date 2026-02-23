"""
Graph Wave-Energy Supervision.

Measures wave activity via quadratic form: E[k] = V[k]^T L V[k]

This captures spatial variation (wave structure) without fragile front localization.

Math:
    E[k] = Σ_j (V[k,j+1] - V[k,j])²

    For 1D chain Laplacian:
    E = V^T L V = Σ_{edges} w_{ij} (V_i - V_j)²
"""
from __future__ import annotations

import torch as th

from src.aggregation.laplacian_builder import build_chain_laplacian


def compute_graph_energy(
    V: th.Tensor,
    L: th.Tensor = None,
) -> th.Tensor:
    """
    Compute graph energy E[k] = V[k]^T L V[k] for each time step.

    Args:
        V: [K, J] velocity at observation times
        L: [J, J] graph Laplacian (if None, uses chain Laplacian)

    Returns:
        E: [K] energy per time step
    """
    K, J = V.shape

    if L is None:
        L = build_chain_laplacian(J, device=V.device, dtype=V.dtype)

    # E[k] = V[k] @ L @ V[k]
    # Efficient: (V @ L) * V summed over J
    LV = V @ L  # [K, J]
    E = (V * LV).sum(dim=1)  # [K]

    return E


def compute_wave_energy_direct(V: th.Tensor) -> th.Tensor:
    """
    Compute wave energy directly as sum of squared differences.

    E[k] = Σ_j (V[k,j+1] - V[k,j])²

    This is equivalent to V^T L V for chain Laplacian with w=1.
    """
    dV = V[:, 1:] - V[:, :-1]  # [K, J-1]
    E = (dV ** 2).sum(dim=1)  # [K]
    return E


def detect_wave_active_windows(
    E_obs: th.Tensor,
    quantile: float = 0.7,
) -> th.Tensor:
    """
    Detect wave-active time windows based on observed energy.

    Args:
        E_obs: [K] observed energy
        quantile: threshold quantile (e.g., 0.7 = top 30%)

    Returns:
        mask: [K] boolean mask for wave-active windows
    """
    threshold = th.quantile(E_obs, quantile)
    mask = E_obs >= threshold
    return mask


def wave_energy_loss_mse(
    E_sim: th.Tensor,
    E_obs: th.Tensor,
    wave_mask: th.Tensor = None,
) -> th.Tensor:
    """
    MSE loss between simulated and observed wave energy.

    Args:
        E_sim: [K] simulated energy
        E_obs: [K] observed energy
        wave_mask: [K] optional mask for wave-active windows

    Returns:
        scalar loss
    """
    if wave_mask is not None:
        E_sim = E_sim[wave_mask]
        E_obs = E_obs[wave_mask]

    return th.mean((E_sim - E_obs) ** 2)


def wave_energy_loss_correlation(
    E_sim: th.Tensor,
    E_obs: th.Tensor,
    wave_mask: th.Tensor = None,
) -> th.Tensor:
    """
    Correlation-based wave energy loss: L = 1 - Corr(E_sim, E_obs)

    This avoids scale hijacking (pushing global speed levels).

    Args:
        E_sim: [K] simulated energy
        E_obs: [K] observed energy
        wave_mask: [K] optional mask for wave-active windows

    Returns:
        scalar loss in [0, 2]
    """
    if wave_mask is not None:
        E_sim = E_sim[wave_mask]
        E_obs = E_obs[wave_mask]

    # Normalize to zero mean
    E_sim_c = E_sim - E_sim.mean()
    E_obs_c = E_obs - E_obs.mean()

    # Correlation
    num = (E_sim_c * E_obs_c).sum()
    denom = th.sqrt((E_sim_c ** 2).sum() * (E_obs_c ** 2).sum() + 1e-8)
    corr = num / denom

    # Loss: 1 - corr (minimizing pushes corr toward 1)
    return 1.0 - corr


def wave_energy_loss_normalized(
    E_sim: th.Tensor,
    E_obs: th.Tensor,
    wave_mask: th.Tensor = None,
) -> th.Tensor:
    """
    Z-score normalized MSE loss.

    Normalizes both signals to zero mean, unit variance before comparing.
    """
    if wave_mask is not None:
        E_sim = E_sim[wave_mask]
        E_obs = E_obs[wave_mask]

    # Z-score normalize
    E_sim_z = (E_sim - E_sim.mean()) / (E_sim.std() + 1e-8)
    E_obs_z = (E_obs - E_obs.mean()) / (E_obs.std() + 1e-8)

    return th.mean((E_sim_z - E_obs_z) ** 2)


def compute_wave_sharpness(V: th.Tensor) -> th.Tensor:
    """
    Compute wave sharpness proxy: max negative spatial gradient.

    sharpness[k] = max_j |-(V[k,j+1] - V[k,j])|  where gradient is negative

    Higher values indicate sharper wavefronts.
    """
    dV = V[:, 1:] - V[:, :-1]  # [K, J-1]
    neg_grad = -dV  # Negative gradient (positive where speed drops)
    sharpness = neg_grad.max(dim=1).values  # [K]
    return sharpness
