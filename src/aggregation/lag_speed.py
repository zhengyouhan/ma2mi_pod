"""
Lag-Based Wave Speed Estimation from Detector Time Series.

Estimates wave propagation speed using cross-correlation between
adjacent detector signals.

Math:
    For detectors j and j+1 with spacing dx:
    - lag[j] = argmax_ℓ Corr(V[:,j], V[:,j+1] shifted by ℓ)
    - wave_speed[j] = dx / (lag[j] * dt_obs)

    Backward propagating waves: downstream (j+1) leads upstream (j),
    so expected lag is positive.
"""
from __future__ import annotations

import torch as th


def normalize_signal(x: th.Tensor, eps: float = 1e-8) -> th.Tensor:
    """
    Normalize signal to zero mean, unit variance.

    Args:
        x: [K] time series
        eps: numerical stability

    Returns:
        x_norm: [K] normalized signal
    """
    return (x - x.mean()) / (x.std() + eps)


def compute_drop_signal(V: th.Tensor) -> th.Tensor:
    """
    Compute deceleration indicator (speed drop signal).

    z[k] = ReLU(-dV[k]) where dV[k] = V[k] - V[k-1]

    Args:
        V: [K, J] speed time series

    Returns:
        z: [K-1, J] drop signal
    """
    dV = V[1:] - V[:-1]  # [K-1, J]
    z = th.relu(-dV)  # Positive where speed drops
    return z


def compute_wave_activity(V: th.Tensor) -> th.Tensor:
    """
    Compute wave activity score per time step.

    S[k] = sum_j ReLU(-G[k,j]) where G[k,j] = V[k,j+1] - V[k,j]

    Args:
        V: [K, J] speed

    Returns:
        S: [K] activity score
    """
    G = V[:, 1:] - V[:, :-1]  # Spatial gradient [K, J-1]
    S = th.relu(-G).sum(dim=1)  # [K]
    return S


def detect_wave_windows(
    V_obs: th.Tensor,
    quantile: float = 0.7,
) -> th.Tensor:
    """
    Detect wave-active time windows from observed speeds.

    Args:
        V_obs: [K, J] observed speeds
        quantile: threshold quantile (e.g., 0.7 = top 30%)

    Returns:
        mask: [K] boolean mask for wave-active windows
    """
    S = compute_wave_activity(V_obs)
    threshold = th.quantile(S, quantile)
    mask = S >= threshold
    return mask


def cross_correlation_at_lag(
    a: th.Tensor,
    b: th.Tensor,
    lag: int,
) -> th.Tensor:
    """
    Compute normalized cross-correlation at specific lag.

    Corr(a[t], b[t-lag]) - shifting b backward by lag

    Args:
        a: [K] signal at detector j
        b: [K] signal at detector j+1
        lag: lag in discrete steps (>=0)

    Returns:
        scalar correlation
    """
    if lag < 0:
        raise ValueError("lag must be >= 0")

    K = a.shape[0]
    if lag >= K:
        return th.tensor(0.0, device=a.device, dtype=a.dtype)

    # a[lag:] vs b[:K-lag]
    a_slice = a[lag:]
    b_slice = b[:K - lag]

    # Normalize
    a_norm = normalize_signal(a_slice)
    b_norm = normalize_signal(b_slice)

    # Correlation
    corr = (a_norm * b_norm).mean()
    return corr


def estimate_lag_per_edge(
    signal: th.Tensor,
    wave_mask: th.Tensor,
    lag_max: int = 5,
    subbin: bool = True,
) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
    """
    Estimate lag between adjacent detectors using cross-correlation.

    Args:
        signal: [K, J] preprocessed signal (speed or drop)
        wave_mask: [K] boolean mask for wave-active windows
        lag_max: maximum lag to search (in discrete steps)
        subbin: use parabolic refinement for sub-step precision

    Returns:
        lag: [J-1] estimated lag per edge (in discrete steps, possibly fractional)
        conf: [J-1] confidence score per edge
        corr_peak: [J-1] peak correlation value
    """
    K, J = signal.shape
    device, dtype = signal.device, signal.dtype

    # Extract wave-active portion
    signal_wave = signal[wave_mask]  # [K_wave, J]
    K_wave = signal_wave.shape[0]

    if K_wave < lag_max + 2:
        # Not enough data
        return (
            th.zeros(J - 1, device=device, dtype=dtype),
            th.zeros(J - 1, device=device, dtype=dtype),
            th.zeros(J - 1, device=device, dtype=dtype),
        )

    lag_out = th.zeros(J - 1, device=device, dtype=dtype)
    conf_out = th.zeros(J - 1, device=device, dtype=dtype)
    corr_peak_out = th.zeros(J - 1, device=device, dtype=dtype)

    for j in range(J - 1):
        a = signal_wave[:, j]      # Upstream detector
        b = signal_wave[:, j + 1]  # Downstream detector

        # Compute correlation for each lag
        corrs = []
        for lag in range(lag_max + 1):
            corr = cross_correlation_at_lag(a, b, lag)
            corrs.append(corr)
        corrs = th.stack(corrs)  # [lag_max+1]

        # Find peak
        peak_idx = corrs.argmax().item()
        peak_val = corrs[peak_idx]

        # Confidence: peak margin over second best
        sorted_corrs = corrs.sort(descending=True).values
        if len(sorted_corrs) > 1:
            margin = sorted_corrs[0] - sorted_corrs[1]
        else:
            margin = sorted_corrs[0]

        # Subbin refinement (parabolic interpolation)
        if subbin and 0 < peak_idx < lag_max:
            y0 = corrs[peak_idx - 1]
            y1 = corrs[peak_idx]
            y2 = corrs[peak_idx + 1]
            # Parabola: y = a*x^2 + b*x + c
            # Peak at x = -b/(2a) = (y0 - y2) / (2*(y0 - 2*y1 + y2))
            denom = 2 * (y0 - 2 * y1 + y2)
            if abs(denom) > 1e-8:
                offset = (y0 - y2) / denom
                offset = max(-0.5, min(0.5, offset))
                refined_lag = peak_idx + offset.item()
            else:
                refined_lag = float(peak_idx)
        else:
            refined_lag = float(peak_idx)

        lag_out[j] = refined_lag
        conf_out[j] = margin
        corr_peak_out[j] = peak_val

    return lag_out, conf_out, corr_peak_out


def estimate_wave_speed(
    V: th.Tensor,
    wave_mask: th.Tensor,
    dx: float,
    dt_obs: float,
    lag_max: int = 5,
    mode: str = "speed",
    subbin: bool = True,
    conf_threshold: float = 0.1,
    skip: int = 1,
) -> dict:
    """
    Estimate wave propagation speed from detector time series.

    Args:
        V: [K, J] speed time series
        wave_mask: [K] boolean mask for wave-active windows
        dx: detector spacing (m)
        dt_obs: observation interval (s)
        lag_max: maximum lag to search
        mode: "speed" (raw V) or "drop" (deceleration signal)
        subbin: use parabolic refinement
        conf_threshold: minimum confidence for valid edge
        skip: detector skip for wider pairs (1=adjacent, 2=skip-1, etc.)

    Returns:
        dict with:
            lag: [num_pairs] estimated lag (discrete steps)
            lag_time: [num_pairs] lag in seconds
            speed: [num_pairs] wave speed (m/s)
            conf: [num_pairs] confidence scores
            valid_mask: [num_pairs] edges above confidence threshold
            pair_indices: [num_pairs, 2] detector indices for each pair
    """
    K, J = V.shape
    device, dtype = V.device, V.dtype

    # Preprocess signal
    if mode == "drop":
        # Use drop signal (need to adjust wave_mask for shorter signal)
        signal = compute_drop_signal(V)  # [K-1, J]
        wave_mask_adj = wave_mask[1:]  # Align with drop signal
    else:
        signal = V
        wave_mask_adj = wave_mask

    # Normalize each detector
    signal_norm = th.zeros_like(signal)
    for j in range(J):
        signal_norm[:, j] = normalize_signal(signal[:, j])

    # Build detector pairs with skip
    pairs = []
    for j in range(J - skip):
        pairs.append((j, j + skip))
    num_pairs = len(pairs)

    if num_pairs == 0:
        return {
            "lag": th.tensor([], device=device, dtype=dtype),
            "lag_time": th.tensor([], device=device, dtype=dtype),
            "speed": th.tensor([], device=device, dtype=dtype),
            "conf": th.tensor([], device=device, dtype=dtype),
            "corr_peak": th.tensor([], device=device, dtype=dtype),
            "valid_mask": th.tensor([], dtype=th.bool, device=device),
            "pair_indices": th.tensor([], dtype=th.long, device=device),
        }

    # Estimate lag per pair
    lag_out = th.zeros(num_pairs, device=device, dtype=dtype)
    conf_out = th.zeros(num_pairs, device=device, dtype=dtype)
    corr_peak_out = th.zeros(num_pairs, device=device, dtype=dtype)

    signal_wave = signal_norm[wave_mask_adj]
    K_wave = signal_wave.shape[0]

    if K_wave < lag_max + 2:
        # Not enough data
        return {
            "lag": lag_out,
            "lag_time": lag_out * dt_obs,
            "speed": th.full_like(lag_out, float('inf')),
            "conf": conf_out,
            "corr_peak": corr_peak_out,
            "valid_mask": conf_out >= conf_threshold,
            "pair_indices": th.tensor(pairs, dtype=th.long, device=device),
        }

    for i, (j_up, j_down) in enumerate(pairs):
        a = signal_wave[:, j_up]   # Upstream
        b = signal_wave[:, j_down] # Downstream

        # Compute correlation for each lag
        corrs = []
        for lag in range(lag_max + 1):
            corr = cross_correlation_at_lag(a, b, lag)
            corrs.append(corr)
        corrs = th.stack(corrs)

        # Find peak
        peak_idx = corrs.argmax().item()
        peak_val = corrs[peak_idx]

        # Confidence: peak margin over second best
        sorted_corrs = corrs.sort(descending=True).values
        if len(sorted_corrs) > 1:
            margin = sorted_corrs[0] - sorted_corrs[1]
        else:
            margin = sorted_corrs[0]

        # Subbin refinement
        if subbin and 0 < peak_idx < lag_max:
            y0 = corrs[peak_idx - 1]
            y1 = corrs[peak_idx]
            y2 = corrs[peak_idx + 1]
            denom = 2 * (y0 - 2 * y1 + y2)
            if abs(denom) > 1e-8:
                offset = (y0 - y2) / denom
                offset = max(-0.5, min(0.5, offset.item()))
                refined_lag = peak_idx + offset
            else:
                refined_lag = float(peak_idx)
        else:
            refined_lag = float(peak_idx)

        lag_out[i] = refined_lag
        conf_out[i] = margin
        corr_peak_out[i] = peak_val

    # Convert to time and speed (using actual distance = skip * dx)
    actual_dx = skip * dx
    lag_time = lag_out * dt_obs
    speed = th.where(
        lag_time > 1e-6,
        actual_dx / lag_time,
        th.full_like(lag_time, float('inf'))
    )

    # Valid edges (above confidence threshold)
    valid_mask = conf_out >= conf_threshold

    return {
        "lag": lag_out,
        "lag_time": lag_time,
        "speed": speed,
        "conf": conf_out,
        "corr_peak": corr_peak_out,
        "valid_mask": valid_mask,
        "pair_indices": th.tensor(pairs, dtype=th.long, device=device),
    }


def wave_speed_loss(
    speed_sim: th.Tensor,
    speed_obs: th.Tensor,
    valid_mask: th.Tensor,
) -> th.Tensor:
    """
    MSE loss on wave speeds for valid (high-confidence) edges.

    Args:
        speed_sim: [J-1] simulated wave speeds
        speed_obs: [J-1] observed wave speeds
        valid_mask: [J-1] boolean mask for valid edges

    Returns:
        scalar loss
    """
    if valid_mask.sum() == 0:
        return th.tensor(0.0, device=speed_sim.device, dtype=speed_sim.dtype)

    # Filter finite speeds
    finite_mask = th.isfinite(speed_sim) & th.isfinite(speed_obs) & valid_mask

    if finite_mask.sum() == 0:
        return th.tensor(0.0, device=speed_sim.device, dtype=speed_sim.dtype)

    diff = speed_sim[finite_mask] - speed_obs[finite_mask]
    return (diff ** 2).mean()


def lag_loss(
    lag_sim: th.Tensor,
    lag_obs: th.Tensor,
    valid_mask: th.Tensor,
) -> th.Tensor:
    """
    MSE loss on lag values for valid edges.

    Args:
        lag_sim: [J-1] simulated lags
        lag_obs: [J-1] observed lags
        valid_mask: [J-1] boolean mask

    Returns:
        scalar loss
    """
    if valid_mask.sum() == 0:
        return th.tensor(0.0, device=lag_sim.device, dtype=lag_sim.dtype)

    diff = lag_sim[valid_mask] - lag_obs[valid_mask]
    return (diff ** 2).mean()
