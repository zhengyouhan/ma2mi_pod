"""
Wavefront-Aware Loss Functions

Implements differentiable wavefront detection and loss components for
improving parameter identifiability in traffic simulation.

Key idea: Wavefront statistics (position, speed, sharpness) are dynamically
sensitive to car-following parameters like T, while aggregate detector
observations are low-pass filtered.
"""
from __future__ import annotations

import torch as th
from typing import Tuple, Dict, Optional


# =============================================================================
# Phase 1: Differentiable Wavefront Detection on Detector Grid
# =============================================================================

def compute_spatial_gradient(
    V: th.Tensor,
    x_det: th.Tensor,
) -> Tuple[th.Tensor, th.Tensor]:
    """
    Compute spatial velocity gradient along detector positions.

    Phase 1.2: G[k,j] = V[k,j+1] - V[k,j]

    Args:
        V: [K, J] speed at each time k and detector j
        x_det: [J] detector positions in meters

    Returns:
        G: [K, J-1] spatial gradient (negative = wavefront)
        x_grad: [J-1] gradient positions (midpoints between detectors)
    """
    # Spatial gradient: difference between adjacent detectors
    G = V[:, 1:] - V[:, :-1]  # [K, J-1]

    # Gradient positions are midpoints between detectors
    x_grad = 0.5 * (x_det[:-1] + x_det[1:])  # [J-1]

    return G, x_grad


def soft_argmin_weights(
    G: th.Tensor,
    tau: float = 1.0,
) -> th.Tensor:
    """
    Compute soft-argmin weights for differentiable front localization.

    Phase 1.3: w[k,j] = softmax(-G[k,j] / tau)

    Wavefront corresponds to strongest NEGATIVE gradient (speed drop).
    We use -G so that minimum (most negative) gets highest weight.

    Args:
        G: [K, J-1] spatial gradient
        tau: temperature parameter (lower = sharper localization)

    Returns:
        w: [K, J-1] soft attention weights summing to 1 per time step
    """
    # Softmax over -G to find minimum (most negative gradient)
    w = th.softmax(-G / tau, dim=-1)  # [K, J-1]
    return w


def compute_wavefront_position(
    w: th.Tensor,
    x_grad: th.Tensor,
) -> th.Tensor:
    """
    Compute expected wavefront position using soft attention.

    Phase 1.4: x_front[k] = sum_j w[k,j] * x_grad[j]

    Args:
        w: [K, J-1] soft attention weights
        x_grad: [J-1] gradient positions

    Returns:
        x_front: [K] expected wavefront position at each time
    """
    x_front = (w * x_grad.unsqueeze(0)).sum(dim=-1)  # [K]
    return x_front


def compute_wavefront_speed(
    x_front: th.Tensor,
    dt: float,
) -> th.Tensor:
    """
    Compute wavefront propagation speed.

    Phase 2.2: c[k] = (x_front[k+1] - x_front[k]) / dt

    Args:
        x_front: [K] wavefront positions
        dt: observation time interval (seconds)

    Returns:
        c: [K-1] wavefront speed (m/s), negative = backward propagation
    """
    c = (x_front[1:] - x_front[:-1]) / dt  # [K-1]
    return c


def compute_wavefront_amplitude(
    w: th.Tensor,
    G: th.Tensor,
) -> th.Tensor:
    """
    Compute wavefront sharpness/amplitude.

    Phase 2.3: a[k] = sum_j w[k,j] * (-G[k,j])

    This is the weighted average of the negative gradient at the front.
    Larger values = sharper/steeper wavefront.

    Args:
        w: [K, J-1] soft attention weights
        G: [K, J-1] spatial gradient

    Returns:
        a: [K] wavefront amplitude (positive = speed drop)
    """
    a = (w * (-G)).sum(dim=-1)  # [K]
    return a


# =============================================================================
# Phase 3: Wave-Active Time Windows
# =============================================================================

def compute_wave_activity_score(
    G: th.Tensor,
) -> th.Tensor:
    """
    Compute wave activity score at each time step.

    Phase 3.1: S[k] = sum_j max(-G[k,j], 0)

    High score indicates strong negative gradients (active wavefront).

    Args:
        G: [K, J-1] spatial gradient

    Returns:
        S: [K] wave activity score
    """
    S = th.clamp(-G, min=0).sum(dim=-1)  # [K]
    return S


def get_wave_active_mask(
    S: th.Tensor,
    quantile: float = 0.8,
) -> th.Tensor:
    """
    Get mask for wave-active time steps.

    Phase 3.2: K_wave = {k | S[k] >= quantile(S, q)}

    Args:
        S: [K] wave activity score
        quantile: threshold quantile (e.g., 0.8 = top 20%)

    Returns:
        mask: [K] boolean mask for wave-active times
    """
    threshold = th.quantile(S, quantile)
    mask = S >= threshold
    return mask


# =============================================================================
# Phase 2: Wavefront Loss Components
# =============================================================================

def wavefront_position_loss(
    x_front_sim: th.Tensor,
    x_front_obs: th.Tensor,
    mask: Optional[th.Tensor] = None,
) -> th.Tensor:
    """
    Front position alignment loss.

    Phase 2.1: L_pos = mean((x_sim - x_obs)^2) over wave-active times

    Args:
        x_front_sim: [K] simulated wavefront positions
        x_front_obs: [K] observed wavefront positions
        mask: [K] optional mask for wave-active times

    Returns:
        L_pos: scalar loss
    """
    diff_sq = (x_front_sim - x_front_obs) ** 2

    if mask is not None and mask.sum() > 0:
        L_pos = diff_sq[mask].mean()
    else:
        L_pos = diff_sq.mean()

    return L_pos


def wavefront_speed_loss(
    c_sim: th.Tensor,
    c_obs: th.Tensor,
    mask: Optional[th.Tensor] = None,
) -> th.Tensor:
    """
    Front speed alignment loss.

    Phase 2.2: L_speed = mean((c_sim - c_obs)^2) over wave-active times

    Args:
        c_sim: [K-1] simulated wavefront speeds
        c_obs: [K-1] observed wavefront speeds
        mask: [K-1] optional mask for wave-active times

    Returns:
        L_speed: scalar loss
    """
    diff_sq = (c_sim - c_obs) ** 2

    if mask is not None and mask.sum() > 0:
        L_speed = diff_sq[mask].mean()
    else:
        L_speed = diff_sq.mean()

    return L_speed


def wavefront_amplitude_loss(
    a_sim: th.Tensor,
    a_obs: th.Tensor,
    mask: Optional[th.Tensor] = None,
) -> th.Tensor:
    """
    Front sharpness/amplitude alignment loss.

    Phase 2.3: L_amp = mean((a_sim - a_obs)^2) over wave-active times

    Args:
        a_sim: [K] simulated wavefront amplitude
        a_obs: [K] observed wavefront amplitude
        mask: [K] optional mask for wave-active times

    Returns:
        L_amp: scalar loss
    """
    diff_sq = (a_sim - a_obs) ** 2

    if mask is not None and mask.sum() > 0:
        L_amp = diff_sq[mask].mean()
    else:
        L_amp = diff_sq.mean()

    return L_amp


# =============================================================================
# Phase 4: Wavefront-Weighted Detector Loss
# =============================================================================

def wavefront_weighted_detector_loss(
    V_sim: th.Tensor,
    V_obs: th.Tensor,
    w_obs: th.Tensor,
    lambda_0: float = 1.0,
    lambda_1: float = 1.0,
) -> th.Tensor:
    """
    Detector loss with wavefront weighting.

    Phase 4.1: L_det_wf = sum_{k,j} (lambda_0 + lambda_1 * w_obs[k,j]) * (V_sim - V_obs)^2

    Prioritizes fitting wavefront regions over uniform averaging.

    Args:
        V_sim: [K, J] simulated speed
        V_obs: [K, J] observed speed
        w_obs: [K, J-1] observed wavefront weights
        lambda_0: base weight
        lambda_1: wavefront weight multiplier

    Returns:
        L_det_wf: scalar loss
    """
    # Expand weights to match detector positions
    # w_obs is [K, J-1], we need [K, J]
    # Assign weight to both detectors at each gradient position
    K, J = V_sim.shape
    weights = th.full((K, J), lambda_0, device=V_sim.device, dtype=V_sim.dtype)

    # Add wavefront weight to detectors adjacent to gradient positions
    weights[:, :-1] += lambda_1 * w_obs
    weights[:, 1:] += lambda_1 * w_obs

    diff_sq = (V_sim - V_obs) ** 2
    L_det_wf = (weights * diff_sq).mean()

    return L_det_wf


# =============================================================================
# Phase 5: Wavefront-Aware Laplacian Regularization
# =============================================================================

def compute_vehicle_wavefront_distance(
    S: th.Tensor,
    x_front: th.Tensor,
    time_indices: th.Tensor,
    dt_sim: float,
    dt_obs: float,
) -> th.Tensor:
    """
    Compute average distance of each vehicle to wavefront over time.

    Phase 5.1: d_i = mean_t |x_i(t) - x_front(t)|

    Args:
        S: [T, N] vehicle positions over simulation time
        x_front: [K] wavefront positions at observation times
        time_indices: [K] simulation time indices for observations
        dt_sim: simulation timestep
        dt_obs: observation interval

    Returns:
        d: [N] average distance to wavefront per vehicle
    """
    N = S.shape[1]
    K = len(time_indices)

    distances = th.zeros(K, N, device=S.device, dtype=S.dtype)

    for k, t_idx in enumerate(time_indices):
        t_idx = int(t_idx.item()) if isinstance(t_idx, th.Tensor) else int(t_idx)
        if t_idx < S.shape[0]:
            distances[k] = th.abs(S[t_idx] - x_front[k])

    # Average over time
    d = distances.mean(dim=0)  # [N]
    return d


def compute_wavefront_gating(
    d: th.Tensor,
    r: float = 50.0,
    delta_r: float = 20.0,
) -> th.Tensor:
    """
    Compute gating coefficient for wavefront-aware Laplacian.

    Phase 5.2: eta_i = sigmoid((d_i - r) / delta_r)

    Vehicles far from wavefront (d > r) get eta -> 1 (full smoothing)
    Vehicles near wavefront (d < r) get eta -> 0 (allow heterogeneity)

    Args:
        d: [N] average distance to wavefront
        r: distance threshold (meters)
        delta_r: transition width

    Returns:
        eta: [N] gating coefficients in [0, 1]
    """
    eta = th.sigmoid((d - r) / delta_r)
    return eta


def wavefront_gated_laplacian_penalty(
    T: th.Tensor,
    leader_idx: th.Tensor,
    lane_id: th.Tensor,
    eta: th.Tensor,
) -> th.Tensor:
    """
    Laplacian penalty with wavefront gating.

    Phase 5.3: L_lap_wf = sum_i eta_i * (T_i - T_leader(i))^2

    Args:
        T: [N] time headway parameters
        leader_idx: [N] leader indices (-1 for no leader)
        lane_id: [N] lane assignments
        eta: [N] gating coefficients

    Returns:
        L_lap_wf: scalar penalty
    """
    N = T.shape[0]

    valid = leader_idx >= 0
    safe_leader = th.clamp(leader_idx, min=0)

    # Same-lane check
    same_lane = lane_id == lane_id[safe_leader]
    valid = valid & same_lane

    if valid.sum() == 0:
        return th.tensor(0.0, device=T.device, dtype=T.dtype)

    diff_sq = (T - T[safe_leader]) ** 2
    L_lap_wf = (eta * diff_sq * valid.float()).sum() / valid.sum()

    return L_lap_wf


# =============================================================================
# Combined Wavefront Loss
# =============================================================================

def compute_all_wavefront_features(
    V: th.Tensor,
    x_det: th.Tensor,
    dt_obs: float,
    tau: float = 1.0,
) -> Dict[str, th.Tensor]:
    """
    Compute all wavefront features from detector observations.

    Args:
        V: [K, J] speed at each time and detector
        x_det: [J] detector positions
        dt_obs: observation interval
        tau: softmax temperature

    Returns:
        dict with: G, x_grad, w, x_front, c, a, S, mask
    """
    # Convert x_det to tensor if list
    if isinstance(x_det, list):
        x_det = th.tensor(x_det, device=V.device, dtype=V.dtype)

    G, x_grad = compute_spatial_gradient(V, x_det)
    w = soft_argmin_weights(G, tau=tau)
    x_front = compute_wavefront_position(w, x_grad)
    c = compute_wavefront_speed(x_front, dt_obs)
    a = compute_wavefront_amplitude(w, G)
    S = compute_wave_activity_score(G)
    mask = get_wave_active_mask(S, quantile=0.8)

    return {
        "G": G,
        "x_grad": x_grad,
        "w": w,
        "x_front": x_front,
        "c": c,
        "a": a,
        "S": S,
        "mask": mask,
    }


def wavefront_loss(
    V_sim: th.Tensor,
    V_obs: th.Tensor,
    x_det: th.Tensor,
    dt_obs: float,
    tau: float = 1.0,
    alpha_pos: float = 1.0,
    alpha_speed: float = 1.0,
    alpha_amp: float = 1.0,
    use_mask: bool = True,
) -> Dict[str, th.Tensor]:
    """
    Compute combined wavefront loss.

    Args:
        V_sim: [K, J] simulated speed
        V_obs: [K, J] observed speed
        x_det: [J] detector positions
        dt_obs: observation interval
        tau: softmax temperature
        alpha_pos: weight for position loss
        alpha_speed: weight for speed loss
        alpha_amp: weight for amplitude loss
        use_mask: whether to apply wave-active mask

    Returns:
        dict with: L_pos, L_speed, L_amp, L_total, features_obs, features_sim
    """
    # Compute features for both obs and sim
    feat_obs = compute_all_wavefront_features(V_obs, x_det, dt_obs, tau)
    feat_sim = compute_all_wavefront_features(V_sim, x_det, dt_obs, tau)

    mask = feat_obs["mask"] if use_mask else None
    mask_speed = mask[:-1] if mask is not None else None

    L_pos = wavefront_position_loss(feat_sim["x_front"], feat_obs["x_front"], mask)
    L_speed = wavefront_speed_loss(feat_sim["c"], feat_obs["c"], mask_speed)
    L_amp = wavefront_amplitude_loss(feat_sim["a"], feat_obs["a"], mask)

    L_total = alpha_pos * L_pos + alpha_speed * L_speed + alpha_amp * L_amp

    return {
        "L_pos": L_pos,
        "L_speed": L_speed,
        "L_amp": L_amp,
        "L_total": L_total,
        "features_obs": feat_obs,
        "features_sim": feat_sim,
    }
