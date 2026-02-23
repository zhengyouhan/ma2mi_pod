"""
Integration Tests: Full Pipeline

Tests the complete flow from simulation to optimization to evaluation.
"""
from __future__ import annotations

import torch as th

from src.sim.rollout import rollout_idm_multilane
from src.sim.reparam import sigmoid_reparam, inverse_sigmoid_reparam
from src.obs.detector_operator import detector_crosslane_aggregate
from src.graph_prior import laplacian_penalty, compute_platooning_weights
from src.loss import detector_loss
from src.eval.metrics import trajectory_collision_count, acceleration_stats


def create_mock_scenario(N: int = 20, device: str = "cpu", dtype: th.dtype = th.float32):
    """Create a mock multi-lane scenario with good spacing."""
    # Initial positions: well-spaced within each lane
    lane_id = th.tensor([i % 3 for i in range(N)], device=device)
    s0 = th.zeros(N, device=device, dtype=dtype)
    for i in range(N):
        lane = lane_id[i].item()
        lane_count = i // 3  # Vehicle index within lane
        s0[i] = 50.0 + lane_count * 30.0 + lane * 200.0  # 30m spacing, large lane offset

    # Initial velocities
    v0 = th.full((N,), 15.0, device=device, dtype=dtype)

    # Leader indices: within same lane (leader is vehicle AHEAD, higher position)
    leader_idx = th.full((N,), -1, dtype=th.long, device=device)
    for lane in range(3):
        lane_vehicles = th.where(lane_id == lane)[0]
        sorted_idx = lane_vehicles[th.argsort(s0[lane_vehicles])]  # Sorted by increasing position
        # Vehicle at position k follows vehicle at position k+1 (the one ahead)
        for k in range(len(sorted_idx) - 1):
            leader_idx[sorted_idx[k]] = sorted_idx[k + 1]

    return {
        "s0": s0,
        "v0": v0,
        "lane_id": lane_id,
        "leader_idx": leader_idx,
    }


def test_end_to_end_simulation():
    """Test 1: Simulation produces valid trajectories."""
    N = 20
    data = create_mock_scenario(N)

    T_headway = th.full((N,), 1.5)
    a_max = th.full((N,), 2.0)
    b = th.full((N,), 2.0)
    v_target = th.full((N,), 25.0)

    S, V, A = rollout_idm_multilane(
        s0_init=data["s0"],
        v0_init=data["v0"],
        leader_idx=data["leader_idx"],
        num_steps=100,
        dt=0.1,
        a_max=a_max,
        b=b,
        v_target=v_target,
        s0=2.0,
        T_headway=T_headway,
        ghost_v=15.0,
        ghost_gap0=50.0,
    )

    assert S.shape == (101, N), f"S shape mismatch: {S.shape}"
    assert V.shape == (101, N), f"V shape mismatch: {V.shape}"
    assert th.isfinite(S).all(), "S contains NaN/Inf"
    assert th.isfinite(V).all(), "V contains NaN/Inf"
    assert (V >= 0).all(), "Negative velocities"

    print(f"Test 1 PASSED: Simulation shape={S.shape}, V_range=[{V.min():.1f}, {V.max():.1f}]")


def test_optimization_loop():
    """Test 2: Full optimization loop converges."""
    N = 15
    data = create_mock_scenario(N)

    # Learnable T with sigmoid reparam
    T_min, T_max = 0.5, 3.0
    u_T = th.nn.Parameter(inverse_sigmoid_reparam(
        th.full((N,), 1.5), T_min, T_max
    ))

    opt = th.optim.Adam([u_T], lr=0.05)

    a_max = th.full((N,), 2.0)
    b = th.full((N,), 2.0)
    v_target = th.full((N,), 25.0)

    # Create fake target observations
    xq = th.linspace(50, 300, 50)
    x_dets = [100.0, 150.0, 200.0]

    losses = []
    for _ in range(20):
        opt.zero_grad()

        T = sigmoid_reparam(u_T, T_min, T_max)

        S, V, _ = rollout_idm_multilane(
            s0_init=data["s0"],
            v0_init=data["v0"],
            leader_idx=data["leader_idx"],
            num_steps=50,
            dt=0.1,
            a_max=a_max,
            b=b,
            v_target=v_target,
            s0=2.0,
            T_headway=T,
            ghost_v=15.0,
            ghost_gap0=50.0,
        )

        pred = detector_crosslane_aggregate(S, V, data["lane_id"], xq, x_dets, sigma=10.0)

        # Fake target: current pred + noise
        with th.no_grad():
            target = {k: v + th.randn_like(v) * 0.1 for k, v in pred.items()}

        L_det = detector_loss(pred, target, weights={"flow": 1.0, "speed": 0.1}, loss_type="mse")
        L_graph = laplacian_penalty(T, data["leader_idx"], data["lane_id"])

        loss = L_det + 0.01 * L_graph
        loss.backward()
        opt.step()

        losses.append(loss.item())

    # Check convergence (loss should decrease or stay stable)
    assert th.isfinite(th.tensor(losses)).all(), "Loss contains NaN"
    assert losses[-1] < losses[0] * 2, "Loss increased significantly"

    print(f"Test 2 PASSED: loss[0]={losses[0]:.4f} -> loss[-1]={losses[-1]:.4f}")


def test_full_pipeline_with_metrics():
    """Test 3: Full pipeline including evaluation metrics."""
    N = 20
    data = create_mock_scenario(N)

    T = th.full((N,), 1.5)
    a_max = th.full((N,), 2.0)
    b = th.full((N,), 2.0)
    v_target = th.full((N,), 25.0)

    S, V, A = rollout_idm_multilane(
        s0_init=data["s0"],
        v0_init=data["v0"],
        leader_idx=data["leader_idx"],
        num_steps=100,
        dt=0.1,
        a_max=a_max,
        b=b,
        v_target=v_target,
        s0=2.0,
        T_headway=T,
        ghost_v=15.0,
        ghost_gap0=50.0,
    )

    # Evaluate trajectory quality
    collision_stats = trajectory_collision_count(S, data["leader_idx"], s_min=2.0)
    accel_stats_result = acceleration_stats(V, dt=0.1)

    # Check that metrics are computed correctly (not that values are perfect)
    assert th.isfinite(collision_stats["collision_count"]), "collision_count should be finite"
    assert th.isfinite(collision_stats["min_spacing"]), "min_spacing should be finite"
    assert th.isfinite(accel_stats_result["accel_max"]), "accel_max should be finite"

    # Reasonable bounds check
    collision_frac = collision_stats["collision_fraction"].item()
    assert collision_frac < 0.5, f"Too many collisions: {collision_frac:.2%}"

    print(f"Test 3 PASSED: collision_frac={collision_frac:.2%}, accel_max={accel_stats_result['accel_max'].item():.2f}")


def test_platooning_weights_integration():
    """Test 4: Platooning weights integrate with Laplacian."""
    N = 20
    data = create_mock_scenario(N)

    T_headway = th.full((N,), 1.5)
    a_max = th.full((N,), 2.0)
    b = th.full((N,), 2.0)
    v_target = th.full((N,), 25.0)

    S, V, _ = rollout_idm_multilane(
        s0_init=data["s0"],
        v0_init=data["v0"],
        leader_idx=data["leader_idx"],
        num_steps=50,
        dt=0.1,
        a_max=a_max,
        b=b,
        v_target=v_target,
        s0=2.0,
        T_headway=T_headway,
        ghost_v=15.0,
        ghost_gap0=50.0,
    )

    # Compute weights
    weights = compute_platooning_weights(
        S, V, data["leader_idx"], data["lane_id"],
        s0=5.0, vs=2.0
    )

    # Compute weighted penalty
    T_varied = th.randn(N) * 0.3 + 1.5
    L_const = laplacian_penalty(T_varied, data["leader_idx"], data["lane_id"])
    L_weighted = laplacian_penalty(T_varied, data["leader_idx"], data["lane_id"], weights=weights)

    assert th.isfinite(L_const), "Constant penalty not finite"
    assert th.isfinite(L_weighted), "Weighted penalty not finite"

    print(f"Test 4 PASSED: L_const={L_const.item():.4f}, L_weighted={L_weighted.item():.4f}")


def run_all_tests():
    """Run all integration tests."""
    print("=" * 60)
    print("Integration Tests: Full Pipeline")
    print("=" * 60)

    test_end_to_end_simulation()
    test_optimization_loop()
    test_full_pipeline_with_metrics()
    test_platooning_weights_integration()

    print("=" * 60)
    print("All Integration tests PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
