from __future__ import annotations

import csv

import torch as th

from src.obs.detector_operator import detector_outputs_at_times
from src.sim.topology import compute_leader_idx_per_lane

FT2M = 0.3048


class _Row:
    __slots__ = ("veh", "time_ms", "x_m", "lane", "v_ms")

    def __init__(self, veh: int, time_ms: int, x_m: float, lane: int, v_ms: float):
        self.veh = veh
        self.time_ms = time_ms
        self.x_m = x_m
        self.lane = lane
        self.v_ms = v_ms


def align_indices_to_grid(num_steps: int, dt: float, grid_sec: float = 5.0) -> th.Tensor:
    idx = []
    t = 0.0
    last = num_steps - 1
    while True:
        i = int(round(t / dt))
        if i > last:
            break
        idx.append(i)
        t += grid_sec
    if not idx:
        idx = [0]
    idx = sorted(set(idx))
    return th.tensor(idx, dtype=th.long)


def _read_rows(csv_path: str, t0_ms: int, t1_ms: int) -> list[_Row]:
    rows: list[_Row] = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        h = {name: i for i, name in enumerate(header)}

        i_veh = h["Vehicle_ID"]
        i_time = h["Global_Time"]
        i_y = h["Local_Y"]
        i_lane = h["Lane_ID"]
        i_v = h["v_Vel"]

        for r in reader:
            try:
                t_ms = int(float(r[i_time]))
            except ValueError:
                continue
            if t_ms < t0_ms or t_ms > t1_ms:
                continue

            try:
                rows.append(
                    _Row(
                        veh=int(float(r[i_veh])),
                        time_ms=t_ms,
                        x_m=float(r[i_y]) * FT2M,
                        lane=int(float(r[i_lane])),
                        v_ms=float(r[i_v]) * FT2M,
                    )
                )
            except ValueError:
                continue
    return rows


def _pick_t0_records(rows: list[_Row], t0_ms: int) -> list[_Row]:
    exact = [r for r in rows if r.time_ms == t0_ms]
    if len(exact) >= 5:
        return exact

    # fallback: first appearance within 500 ms window
    first: dict[int, _Row] = {}
    for r in rows:
        if t0_ms <= r.time_ms <= t0_ms + 500 and r.veh not in first:
            first[r.veh] = r
    return list(first.values())


def _build_observation_matrix(
    rows: list[_Row],
    veh_ids: list[int],
    t0_ms: int,
    duration_s: float,
    dt: float,
    *,
    device: str,
    dtype: th.dtype,
) -> tuple[th.Tensor, th.Tensor]:
    T = int(round(duration_s / dt)) + 1
    N = len(veh_ids)

    S_real = th.full((T, N), float("nan"), device=device, dtype=dtype)
    V_real = th.full((T, N), float("nan"), device=device, dtype=dtype)

    id_to_col = {vid: i for i, vid in enumerate(veh_ids)}

    for r in rows:
        i = id_to_col.get(r.veh)
        if i is None:
            continue
        t_idx = int(round((r.time_ms - t0_ms) / 1000.0 / dt))
        if 0 <= t_idx < T:
            S_real[t_idx, i] = r.x_m
            V_real[t_idx, i] = r.v_ms

    return S_real, V_real


def _default_detector_positions(x_min: float, x_max: float) -> list[float]:
    step = (x_max - x_min) / 6.0
    return [x_min + step * i for i in range(7)]


def _detector_lane_association(t0_records: list[_Row], x_dets: list[float], radius_m: float = 25.0) -> th.Tensor:
    out = []
    for xd in x_dets:
        counts: dict[int, int] = {}
        for r in t0_records:
            if abs(r.x_m - xd) <= radius_m:
                counts[r.lane] = counts.get(r.lane, 0) + 1
        if not counts:
            out.append(-1)
        else:
            out.append(max(counts, key=counts.get))
    return th.tensor(out, dtype=th.long)


def load_ngsim_detector_dataset(
    csv_path: str,
    t0_ms: int,
    duration_s: float,
    x_min: float,
    x_max: float,
    *,
    dt: float = 0.1,
    sigma: float = 10.0,
    detector_positions_m: list[float] | None = None,
    device: str = "cpu",
    dtype: th.dtype = th.float32,
) -> dict | None:
    t1_ms = t0_ms + int(round(duration_s * 1000.0))
    rows = _read_rows(csv_path, t0_ms, t1_ms)
    if len(rows) == 0:
        return None

    t0_records = _pick_t0_records(rows, t0_ms)
    t0_records = [r for r in t0_records if x_min <= r.x_m <= x_max]
    if len(t0_records) < 5:
        return None

    t0_records.sort(key=lambda r: r.x_m)

    veh_ids = [r.veh for r in t0_records]
    s0 = th.tensor([r.x_m for r in t0_records], device=device, dtype=dtype)
    v0 = th.tensor([r.v_ms for r in t0_records], device=device, dtype=dtype)
    lane_id = th.tensor([r.lane for r in t0_records], device=device, dtype=th.long)

    leader_idx = compute_leader_idx_per_lane(s0, lane_id)

    S_real, V_real = _build_observation_matrix(
        rows=rows,
        veh_ids=veh_ids,
        t0_ms=t0_ms,
        duration_s=duration_s,
        dt=dt,
        device=device,
        dtype=dtype,
    )

    S_real_clean = th.where(th.isnan(S_real), th.tensor(1e6, device=device, dtype=dtype), S_real)
    V_real_clean = th.where(th.isnan(V_real), th.tensor(0.0, device=device, dtype=dtype), V_real)

    xq = th.linspace(x_min - 20.0, x_max + 20.0, 201, device=device, dtype=dtype)
    x_dets = detector_positions_m if detector_positions_m is not None else _default_detector_positions(x_min, x_max)

    time_idx_5s = align_indices_to_grid(num_steps=S_real.shape[0], dt=dt, grid_sec=5.0).to(device)
    obs_5s = detector_outputs_at_times(
        S=S_real_clean,
        V=V_real_clean,
        xq=xq,
        x_dets=x_dets,
        time_indices=time_idx_5s,
        sigma=sigma,
        half_window=10.0,
    )

    detector_lane_id = _detector_lane_association(t0_records, x_dets).to(device)

    return {
        "veh_ids": veh_ids,
        "s0": s0,
        "v0": v0,
        "lane_id": lane_id,
        "leader_idx": leader_idx,
        "xq": xq,
        "x_dets": x_dets,
        "detector_lane_id": detector_lane_id,
        "S_real": S_real,
        "V_real": V_real,
        "obs_5s": obs_5s,
        "time_idx_5s": time_idx_5s,
        "time_s_5s": time_idx_5s.to(dtype) * th.as_tensor(dt, device=device, dtype=dtype),
        "duration_s": duration_s,
        "dt": dt,
    }
