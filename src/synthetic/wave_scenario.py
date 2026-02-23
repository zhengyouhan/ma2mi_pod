"""
Synthetic stop-and-go wave scenario with heterogeneous IDM parameters.

Creates a realistic highway segment with:
- Open boundary conditions (vehicles enter at upstream, exit downstream)
- Heterogeneous T (time headway) values
- Triggered stop-and-go waves via leader deceleration events
"""
from __future__ import annotations

import torch as th
from dataclasses import dataclass
from typing import Tuple


@dataclass
class WaveScenarioConfig:
    """Configuration for synthetic wave scenario."""
    # Domain
    x_min: float = 0.0
    x_max: float = 1000.0  # 1km road segment
    num_lanes: int = 3

    # Vehicles per lane (initial)
    vehicles_per_lane: int = 30
    initial_spacing: float = 25.0  # meters (denser traffic)

    # Time
    duration_s: float = 120.0  # 2 minutes
    dt: float = 0.1

    # IDM parameters (base)
    v_free: float = 25.0  # m/s (~90 km/h) - slower for clearer waves
    a_max: float = 1.5
    b_comfort: float = 2.0
    s0: float = 2.0

    # Heterogeneous T distribution
    T_mean: float = 1.5
    T_std: float = 0.4
    T_min: float = 0.8
    T_max: float = 2.5

    # Ghost driver (controls leading vehicle behavior)
    ghost_v_base: float = 20.0      # base ghost velocity (m/s)
    ghost_gap0: float = 30.0        # initial gap to ghost

    # Wave events via ghost slowdown
    wave1_time: float = 5.0         # when ghost starts slowing
    wave1_v_drop: float = 10.0      # velocity drop (m/s)
    wave1_duration: float = 5.0     # how long ghost stays slow

    wave2_time: float = 30.0
    wave2_v_drop: float = 8.0
    wave2_duration: float = 6.0

    # Detector settings
    detector_positions: list = None
    detector_spacing: float = 200.0  # meters between detectors
    obs_interval: float = 5.0  # seconds between observations
    sigma: float = 10.0

    def __post_init__(self):
        if self.detector_positions is None:
            # Place detectors based on spacing
            self.detector_positions = [
                x for x in range(int(self.detector_spacing), int(self.x_max), int(self.detector_spacing))
            ]


def generate_heterogeneous_T(
    N: int,
    mean: float,
    std: float,
    T_min: float,
    T_max: float,
    seed: int = 42,
) -> th.Tensor:
    """Generate heterogeneous time headway values."""
    th.manual_seed(seed)
    T = th.randn(N) * std + mean
    T = th.clamp(T, T_min, T_max)
    return T


def create_initial_state(cfg: WaveScenarioConfig, device: str = "cpu", dtype: th.dtype = th.float32):
    """Create initial vehicle positions and velocities."""
    N = cfg.num_lanes * cfg.vehicles_per_lane

    # Lane assignments
    lane_id = th.zeros(N, dtype=th.long, device=device)
    for i in range(N):
        lane_id[i] = i % cfg.num_lanes

    # Initial positions: staggered by lane, starting from upstream
    s0 = th.zeros(N, device=device, dtype=dtype)
    for i in range(N):
        lane = lane_id[i].item()
        lane_idx = i // cfg.num_lanes
        # Start from upstream (x_min), vehicles spread toward downstream
        # First vehicle at 50m, last at 50 + (N_per_lane-1)*spacing
        s0[i] = cfg.x_min + 50.0 + lane_idx * cfg.initial_spacing + lane * 3.0

    # Initial velocities: near free flow
    v0 = th.full((N,), cfg.v_free - 2.0, device=device, dtype=dtype)

    # Leader indices: within same lane (leader is ahead = higher position)
    leader_idx = th.full((N,), -1, dtype=th.long, device=device)
    for lane in range(cfg.num_lanes):
        lane_mask = lane_id == lane
        lane_vehicles = th.where(lane_mask)[0]
        sorted_idx = lane_vehicles[th.argsort(s0[lane_vehicles])]
        for k in range(len(sorted_idx) - 1):
            leader_idx[sorted_idx[k]] = sorted_idx[k + 1]

    # Heterogeneous T
    T_true = generate_heterogeneous_T(
        N, cfg.T_mean, cfg.T_std, cfg.T_min, cfg.T_max
    ).to(device=device, dtype=dtype)

    return {
        "s0": s0,
        "v0": v0,
        "lane_id": lane_id,
        "leader_idx": leader_idx,
        "T_true": T_true,
    }


def idm_acceleration(
    v: th.Tensor,
    s: th.Tensor,
    dv: th.Tensor,
    v_free: float,
    a_max: th.Tensor,
    b: th.Tensor,
    s0: float,
    T: th.Tensor,
    eps: float = 1e-6,
) -> th.Tensor:
    """Compute IDM acceleration."""
    # Desired gap
    s_star = s0 + T * v + v * dv / (2 * th.sqrt(a_max * b))
    s_star = th.clamp(s_star, min=s0)

    # Safe spacing
    s_safe = th.clamp(s, min=eps)

    # IDM acceleration
    a = a_max * (1.0 - (v / v_free) ** 4 - (s_star / s_safe) ** 2)
    return a


def simulate_with_waves(
    init: dict,
    cfg: WaveScenarioConfig,
    device: str = "cpu",
    dtype: th.dtype = th.float32,
) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
    """
    Simulate traffic with wave-triggering events.

    Returns:
        S: [T, N] positions
        V: [T, N] velocities
        A: [T, N] accelerations
    """
    N = init["s0"].shape[0]
    steps = int(cfg.duration_s / cfg.dt)

    s = init["s0"].clone()
    v = init["v0"].clone()
    lane_id = init["lane_id"]
    leader_idx = init["leader_idx"]
    T = init["T_true"]

    a_max = th.full((N,), cfg.a_max, device=device, dtype=dtype)
    b = th.full((N,), cfg.b_comfort, device=device, dtype=dtype)

    S_list = [s.clone()]
    V_list = [v.clone()]
    A_list = []

    # Ghost driver state (starts ahead of all vehicles)
    ghost_pos = s.max() + cfg.ghost_gap0
    ghost_v = cfg.ghost_v_base

    for step in range(steps):
        t = step * cfg.dt

        # Ghost velocity profile (creates waves by slowing down)
        ghost_v = cfg.ghost_v_base
        # Wave 1: ghost slows down
        if cfg.wave1_time <= t < cfg.wave1_time + cfg.wave1_duration:
            ghost_v = cfg.ghost_v_base - cfg.wave1_v_drop
        # Wave 2: ghost slows down again
        if cfg.wave2_time <= t < cfg.wave2_time + cfg.wave2_duration:
            ghost_v = cfg.ghost_v_base - cfg.wave2_v_drop

        ghost_v = max(ghost_v, 5.0)  # minimum ghost speed

        # Update ghost position
        ghost_pos = ghost_pos + ghost_v * cfg.dt

        # Compute spacing to leader
        has_leader = leader_idx >= 0
        safe_leader = th.clamp(leader_idx, min=0)

        spacing = th.full((N,), cfg.ghost_gap0, device=device, dtype=dtype)
        dv = th.zeros(N, device=device, dtype=dtype)

        spacing[has_leader] = s[safe_leader[has_leader]] - s[has_leader]
        dv[has_leader] = v[has_leader] - v[safe_leader[has_leader]]

        # Leading vehicles (no leader) follow the ghost driver
        no_leader = ~has_leader
        spacing[no_leader] = ghost_pos - s[no_leader]
        dv[no_leader] = v[no_leader] - ghost_v

        # IDM acceleration
        a = idm_acceleration(v, spacing, dv, cfg.v_free, a_max, b, cfg.s0, T)

        # Clamp acceleration
        a = th.clamp(a, -cfg.b_comfort * 2, cfg.a_max)

        A_list.append(a.clone())

        # Euler integration
        v = v + a * cfg.dt
        v = th.clamp(v, min=0.0)  # No negative velocities
        s = s + v * cfg.dt

        S_list.append(s.clone())
        V_list.append(v.clone())

    S = th.stack(S_list, dim=0)
    V = th.stack(V_list, dim=0)
    A = th.stack(A_list, dim=0)

    return S, V, A


def generate_detector_observations(
    S: th.Tensor,
    V: th.Tensor,
    lane_id: th.Tensor,
    cfg: WaveScenarioConfig,
    obs_interval: float = 5.0,
) -> dict:
    """
    Generate detector observations from simulated trajectories.

    Returns observations at specified time intervals.
    """
    from src.obs.detector_operator import detector_crosslane_aggregate

    T_steps, N = S.shape
    dt = cfg.dt
    total_time = (T_steps - 1) * dt

    # Observation time indices (every obs_interval seconds)
    obs_times = th.arange(0, total_time, obs_interval)
    time_indices = (obs_times / dt).long()
    time_indices = time_indices[time_indices < T_steps]

    # Spatial grid for aggregation
    xq = th.linspace(cfg.x_min, cfg.x_max, 100, device=S.device, dtype=S.dtype)
    x_dets = cfg.detector_positions

    # Aggregate at each observation time
    flow_list = []
    speed_list = []

    for t_idx in time_indices:
        # Use a small window around t_idx
        t_start = max(0, t_idx - 5)
        t_end = min(T_steps, t_idx + 5)

        S_window = S[t_start:t_end]
        V_window = V[t_start:t_end]

        obs = detector_crosslane_aggregate(
            S_window, V_window, lane_id, xq, x_dets, sigma=cfg.sigma
        )

        # Take middle time point
        mid = (t_end - t_start) // 2
        flow_list.append(obs["flow"][mid])
        speed_list.append(obs["speed"][mid])

    return {
        "flow": th.stack(flow_list, dim=0),
        "speed": th.stack(speed_list, dim=0),
        "time_indices": time_indices,
        "x_dets": x_dets,
    }


def create_wave_scenario(
    cfg: WaveScenarioConfig = None,
    device: str = "cpu",
    dtype: th.dtype = th.float32,
    seed: int = 42,
) -> dict:
    """
    Create complete wave scenario with ground truth and observations.

    Returns:
        dict with:
            - s0, v0, lane_id, leader_idx: initial conditions
            - T_true: ground truth heterogeneous T values
            - S, V, A: full trajectories
            - obs_5s: detector observations
            - time_idx_5s: observation time indices
            - x_dets: detector positions
            - xq: spatial grid
            - cfg: scenario configuration
    """
    if cfg is None:
        cfg = WaveScenarioConfig()

    th.manual_seed(seed)

    # Create initial state
    init = create_initial_state(cfg, device, dtype)

    # Simulate with waves
    S, V, A = simulate_with_waves(init, cfg, device, dtype)

    # Generate detector observations
    obs = generate_detector_observations(S, V, init["lane_id"], cfg, obs_interval=cfg.obs_interval)

    # Spatial grid
    xq = th.linspace(cfg.x_min, cfg.x_max, 100, device=device, dtype=dtype)

    return {
        "s0": init["s0"],
        "v0": init["v0"],
        "lane_id": init["lane_id"],
        "leader_idx": init["leader_idx"],
        "T_true": init["T_true"],
        "S": S,
        "V": V,
        "A": A,
        "obs_5s": obs,  # kept for backward compatibility
        "time_idx_5s": obs["time_indices"],
        "x_dets": obs["x_dets"],
        "xq": xq,
        "cfg": cfg,
    }


if __name__ == "__main__":
    # Quick test
    print("Creating wave scenario...")
    cfg = WaveScenarioConfig()
    data = create_wave_scenario(cfg)

    print(f"Vehicles: {data['s0'].shape[0]}")
    print(f"Lanes: {data['lane_id'].unique().tolist()}")
    print(f"T_true range: [{data['T_true'].min():.2f}, {data['T_true'].max():.2f}]")
    print(f"T_true mean: {data['T_true'].mean():.2f}, std: {data['T_true'].std():.2f}")
    print(f"Trajectory shape: {data['S'].shape}")
    print(f"Observation times: {data['time_idx_5s'].shape[0]}")
    print(f"Detectors: {data['x_dets']}")

    # Check for waves in velocity
    V = data["V"]
    print(f"\nVelocity range over time:")
    print(f"  t=0:   [{V[0].min():.1f}, {V[0].max():.1f}] m/s")
    print(f"  t=20s: [{V[200].min():.1f}, {V[200].max():.1f}] m/s")
    print(f"  t=60s: [{V[600].min():.1f}, {V[600].max():.1f}] m/s")
    print(f"  t=120s:[{V[-1].min():.1f}, {V[-1].max():.1f}] m/s")
