from __future__ import annotations

import torch as th

from .idm import IDM_DELTA, idm_acceleration


def rollout_idm_multilane(
    *,
    s0_init: th.Tensor,
    v0_init: th.Tensor,
    leader_idx: th.Tensor,
    num_steps: int,
    dt: float,
    a_max: th.Tensor,
    b: th.Tensor,
    v_target: th.Tensor,
    s0: th.Tensor | float,
    T_headway: th.Tensor,
    delta: float = IDM_DELTA,
    min_gap: float = 2.0,
    gap_beta: float = 20.0,
    accel_beta: float = 20.0,
    prevent_negative_speed: bool = True,
    ghost_v: float = 15.0,
    ghost_gap0: float = 50.0,
) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
    """
    Multi-lane IDM rollout using explicit leader mapping.

    Returns:
      S: [steps+1, N], V: [steps+1, N], A: [steps, N]
    """
    n = s0_init.shape[0]
    dt_t = th.as_tensor(dt, device=s0_init.device, dtype=s0_init.dtype)

    s_list = [s0_init]
    v_list = [v0_init]
    a_list = []

    has_leader = leader_idx >= 0
    safe_idx = th.clamp(leader_idx, min=0)

    ghost_v_t = th.as_tensor(ghost_v, device=s0_init.device, dtype=s0_init.dtype)
    ghost_gap0_t = th.as_tensor(ghost_gap0, device=s0_init.device, dtype=s0_init.dtype)
    ghost_s0 = s0_init + ghost_gap0_t

    for t in range(num_steps):
        s = s_list[t]
        v = v_list[t]

        ghost_s_t = ghost_s0 + ghost_v_t * (t * dt_t)
        s_lead_real = s[safe_idx]
        v_lead_real = v[safe_idx]

        s_lead = th.where(has_leader, s_lead_real, ghost_s_t)
        v_lead = th.where(has_leader, v_lead_real, ghost_v_t.expand(n))

        gap = s_lead - s
        dv = v - v_lead

        acc = idm_acceleration(
            a_max=a_max,
            b=b,
            v_curr=v,
            v_target=v_target,
            pos_delta=gap,
            vel_delta=dv,
            s0=th.as_tensor(s0, device=s.device, dtype=s.dtype),
            T=T_headway,
            delta=delta,
            min_gap=min_gap,
            gap_beta=gap_beta,
            accel_beta=accel_beta,
            prevent_negative_speed=prevent_negative_speed,
            delta_time=dt_t,
        )

        a_list.append(acc)

        v_next = v + acc * dt_t
        if prevent_negative_speed:
            v_next = th.clamp(v_next, min=0.0)
        s_next = s + v_next * dt_t

        v_list.append(v_next)
        s_list.append(s_next)

    return th.stack(s_list, dim=0), th.stack(v_list, dim=0), th.stack(a_list, dim=0)
