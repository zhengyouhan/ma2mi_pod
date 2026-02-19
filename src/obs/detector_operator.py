from __future__ import annotations

import torch as th


def _soft_mask(x: th.Tensor, threshold: float, beta: float = 20.0) -> th.Tensor:
    return th.sigmoid(beta * (x - threshold))


def _soft_conditional(
    condition_value: th.Tensor,
    threshold: float,
    safe_value: th.Tensor,
    computed_value: th.Tensor,
    beta: float = 20.0,
) -> th.Tensor:
    mask = _soft_mask(condition_value, threshold, beta)
    return mask * computed_value + (1.0 - mask) * safe_value


def micro_to_macro_gaussian(
    S: th.Tensor,
    V: th.Tensor,
    xq: th.Tensor,
    sigma: float,
    chunk_T: int = 256,
    eps: float = 1e-6,
) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
    """
    Gaussian micro->macro operator.

    Args:
      S, V: [T, N]
      xq: [M]
    Returns:
      rho, u, q: [T, M]
    """
    if S.shape != V.shape or S.ndim != 2:
        raise ValueError("S and V must have shape [T, N]")

    T, _ = S.shape
    M = xq.shape[0]
    sigma_t = th.as_tensor(sigma, device=S.device, dtype=S.dtype)
    norm = th.sqrt(th.as_tensor(2.0 * th.pi, device=S.device, dtype=S.dtype)) * sigma_t

    rho_out = th.empty((T, M), device=S.device, dtype=S.dtype)
    u_out = th.empty((T, M), device=S.device, dtype=S.dtype)

    for t0 in range(0, T, chunk_T):
        t1 = min(t0 + chunk_T, T)
        S_blk = S[t0:t1]
        V_blk = V[t0:t1]

        dx = S_blk[:, :, None] - xq[None, None, :]
        W = th.exp(-0.5 * (dx / sigma_t).clamp(-20, 20) ** 2)

        Wsum = W.sum(dim=1)
        WV = (W * V_blk[:, :, None]).sum(dim=1)

        rho_blk = Wsum / norm
        u_blk = _soft_conditional(
            condition_value=Wsum,
            threshold=eps,
            safe_value=th.zeros_like(Wsum),
            computed_value=WV / (Wsum + eps),
            beta=20.0,
        )

        rho_out[t0:t1] = rho_blk
        u_out[t0:t1] = u_blk

    q_out = rho_out * u_out
    return rho_out, u_out, q_out


def detector_timeseries_from_micro(
    S: th.Tensor,
    V: th.Tensor,
    xq: th.Tensor,
    x_dets: list[float],
    *,
    sigma: float,
    half_window: float = 10.0,
    L_eff: float = 7.5,
    eps_rho: float = 1e-3,
) -> dict[str, th.Tensor]:
    """
    Detector operator with same aggregation logic as existing code.

    Returns dict with tensors of shape [T, Nd]:
      - density, speed, flow, occ
    """
    rho, u, q = micro_to_macro_gaussian(S, V, xq, sigma=sigma)

    T, _ = rho.shape
    Nd = len(x_dets)

    density = th.zeros((T, Nd), device=rho.device, dtype=rho.dtype)
    speed = th.zeros((T, Nd), device=rho.device, dtype=rho.dtype)
    flow = th.zeros((T, Nd), device=rho.device, dtype=rho.dtype)
    occ = th.zeros((T, Nd), device=rho.device, dtype=rho.dtype)

    for j, xd in enumerate(x_dets):
        win = (xq >= xd - half_window) & (xq <= xd + half_window)
        idx = th.where(win)[0]
        if idx.numel() == 0:
            raise ValueError(f"Detector window at x={xd:.2f} has no grid points")

        rho_w = rho[:, idx].mean(dim=1)
        q_w = q[:, idx].mean(dim=1)

        speed_w = _soft_conditional(
            condition_value=rho_w,
            threshold=eps_rho,
            safe_value=th.zeros_like(rho_w),
            computed_value=q_w / rho_w.clamp_min(eps_rho),
            beta=20.0,
        )

        density[:, j] = rho_w
        speed[:, j] = speed_w
        flow[:, j] = q_w
        occ[:, j] = th.clamp(rho_w * th.as_tensor(L_eff, device=rho.device, dtype=rho.dtype), 0.0, 1.0)

    return {"density": density, "speed": speed, "flow": flow, "occ": occ}


def detector_outputs_at_times(
    S: th.Tensor,
    V: th.Tensor,
    xq: th.Tensor,
    x_dets: list[float],
    time_indices: th.Tensor,
    *,
    sigma: float,
    half_window: float = 10.0,
    L_eff: float = 7.5,
) -> dict[str, th.Tensor]:
    """Sample detector outputs at selected time indices.

    Returned shapes are [K, Nd], where K = len(time_indices).
    """
    det_ts = detector_timeseries_from_micro(
        S=S,
        V=V,
        xq=xq,
        x_dets=x_dets,
        sigma=sigma,
        half_window=half_window,
        L_eff=L_eff,
    )
    return {k: v[time_indices] for k, v in det_ts.items()}
