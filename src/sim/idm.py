import torch as th
import torch.nn.functional as F

IDM_DELTA = 4.0


def _soft_min(x: th.Tensor, min_val: th.Tensor, beta: float = 10.0) -> th.Tensor:
    return min_val + F.softplus(beta * (x - min_val)) / beta


def _soft_max(x: th.Tensor, max_val: th.Tensor, beta: float = 10.0) -> th.Tensor:
    return max_val - F.softplus(beta * (max_val - x)) / beta


def _soft_clamp(
    x: th.Tensor,
    min_val: th.Tensor | None,
    max_val: th.Tensor | None,
    beta: float = 10.0,
) -> th.Tensor:
    if min_val is not None:
        x = _soft_min(x, min_val, beta=beta)
    if max_val is not None:
        x = _soft_max(x, max_val, beta=beta)
    return x


def idm_acceleration(
    *,
    a_max: th.Tensor,
    b: th.Tensor,
    v_curr: th.Tensor,
    v_target: th.Tensor,
    pos_delta: th.Tensor,
    vel_delta: th.Tensor,
    s0: th.Tensor,
    T: th.Tensor,
    delta: float = IDM_DELTA,
    eps: float = 1e-6,
    min_gap: float | None = 2.0,
    gap_beta: float = 20.0,
    accel_beta: float = 20.0,
    prevent_negative_speed: bool = True,
    delta_time: th.Tensor | None = None,
) -> th.Tensor:
    s = pos_delta
    if min_gap is not None:
        s = _soft_min(s, th.as_tensor(min_gap, device=s.device, dtype=s.dtype), beta=gap_beta)
    s = th.clamp(s, min=eps)

    b = th.clamp(b, min=eps)
    sqrt_ab = th.sqrt(th.clamp(a_max * b, min=eps))

    s_star = s0 + v_curr * T + (v_curr * vel_delta) / (2.0 * sqrt_ab + eps)
    s_star = th.clamp(s_star, min=0.0)

    free = (v_curr / (v_target + eps)) ** delta
    interact = (s_star / s) ** 2
    acc = a_max * (1.0 - free - interact)
    acc = _soft_clamp(acc, min_val=-b, max_val=a_max, beta=accel_beta)

    if prevent_negative_speed and delta_time is not None:
        min_acc_nonneg = -v_curr / (delta_time + eps)
        acc = th.maximum(acc, min_acc_nonneg)

    return acc
