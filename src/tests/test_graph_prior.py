"""
Phase 3 Tests: Graph Laplacian and Platooning Weights

Tests:
1. Laplacian penalty: correct computation with known values
2. Same-lane filtering: only same-lane edges contribute
3. Platooning weights: close vehicles get higher weight
4. Gradient flow through penalty
"""
from __future__ import annotations

import torch as th

from src.graph_prior import laplacian_penalty, compute_platooning_weights


def test_laplacian_penalty_computation():
    """Test 1: Verify Laplacian penalty formula."""
    # Simple case: 3 vehicles in a chain
    T = th.tensor([1.0, 1.2, 1.5])
    leader_idx = th.tensor([-1, 0, 1])
    lane_id = th.tensor([0, 0, 0])

    L = laplacian_penalty(T, leader_idx, lane_id)

    # Expected: (T[1]-T[0])^2 + (T[2]-T[1])^2 = 0.04 + 0.09 = 0.13
    expected = (1.2 - 1.0) ** 2 + (1.5 - 1.2) ** 2
    assert abs(L.item() - expected) < 1e-6, f"Expected {expected}, got {L.item()}"

    print(f"Test 1 PASSED: L={L.item():.6f}, expected={expected:.6f}")


def test_same_lane_filtering():
    """Test 2: Only same-lane leader edges contribute."""
    T = th.tensor([1.0, 1.5, 1.0, 1.5])
    # Vehicle 1 follows 0 (same lane), Vehicle 3 follows 2 (different lane)
    leader_idx = th.tensor([-1, 0, -1, 2])
    lane_id = th.tensor([0, 0, 1, 0])  # Vehicle 3 is lane 0, vehicle 2 is lane 1

    L = laplacian_penalty(T, leader_idx, lane_id)

    # Only edge (1,0) should contribute: (1.5-1.0)^2 = 0.25
    # Edge (3,2) crosses lanes, should be excluded
    expected = (1.5 - 1.0) ** 2
    assert abs(L.item() - expected) < 1e-6, f"Expected {expected}, got {L.item()}"

    print(f"Test 2 PASSED: Same-lane filtering works, L={L.item():.6f}")


def test_platooning_weights_spacing():
    """Test 3: Close vehicles get higher platooning weight."""
    N = 4
    # Two pairs: one close (5m), one far (30m)
    S = th.tensor([[0.0, 5.0, 100.0, 130.0]])  # [1, N]
    V = th.tensor([[15.0, 15.0, 15.0, 15.0]])  # Same speeds
    leader_idx = th.tensor([-1, 0, -1, 2])
    lane_id = th.tensor([0, 0, 1, 1])

    w = compute_platooning_weights(S, V, leader_idx, lane_id, s0=5.0, vs=2.0)

    # Vehicle 1 is 5m from leader 0: w = exp(-5/5) * exp(0) = exp(-1) ≈ 0.368
    # Vehicle 3 is 30m from leader 2: w = exp(-30/5) * exp(0) = exp(-6) ≈ 0.0025
    assert w[1] > w[3], f"Close vehicle should have higher weight: w[1]={w[1]:.4f}, w[3]={w[3]:.4f}"
    assert abs(w[1].item() - 0.368) < 0.01, f"Expected w[1] ≈ 0.368, got {w[1].item():.4f}"

    print(f"Test 3 PASSED: w_close={w[1].item():.4f}, w_far={w[3].item():.6f}")


def test_platooning_weights_velocity():
    """Test 4: Similar speeds get higher weight."""
    N = 4
    S = th.tensor([[0.0, 10.0, 100.0, 110.0]])  # Same spacing
    # Pair 1: same speed, Pair 2: different speeds
    V = th.tensor([[15.0, 15.0, 15.0, 20.0]])
    leader_idx = th.tensor([-1, 0, -1, 2])
    lane_id = th.tensor([0, 0, 1, 1])

    w = compute_platooning_weights(S, V, leader_idx, lane_id, s0=10.0, vs=2.0)

    # Vehicle 1: dv=0, w = exp(-10/10) * exp(0) = exp(-1) ≈ 0.368
    # Vehicle 3: dv=5, w = exp(-10/10) * exp(-5/2) = exp(-1) * exp(-2.5) ≈ 0.030
    assert w[1] > w[3], f"Same-speed vehicle should have higher weight"

    print(f"Test 4 PASSED: w_same_speed={w[1].item():.4f}, w_diff_speed={w[3].item():.4f}")


def test_weighted_laplacian_penalty():
    """Test 5: Weighted penalty uses provided weights."""
    T = th.tensor([1.0, 1.5, 1.0, 2.0])
    leader_idx = th.tensor([-1, 0, -1, 2])
    lane_id = th.tensor([0, 0, 1, 1])

    # Without weights
    L_const = laplacian_penalty(T, leader_idx, lane_id, weights=None)

    # With custom weights
    weights = th.tensor([0.0, 1.0, 0.0, 0.1])  # Low weight for second edge
    L_weighted = laplacian_penalty(T, leader_idx, lane_id, weights=weights)

    # L_const = 0.25 + 1.0 = 1.25
    # L_weighted = 1.0 * 0.25 + 0.1 * 1.0 = 0.35
    assert L_weighted < L_const, "Weighted penalty should be smaller"

    print(f"Test 5 PASSED: L_const={L_const.item():.4f}, L_weighted={L_weighted.item():.4f}")


def test_gradient_flow():
    """Test 6: Gradients flow through Laplacian penalty."""
    T = th.nn.Parameter(th.tensor([1.0, 1.5, 1.2]))
    leader_idx = th.tensor([-1, 0, 1])
    lane_id = th.tensor([0, 0, 0])

    L = laplacian_penalty(T, leader_idx, lane_id)
    L.backward()

    assert T.grad is not None, "T.grad is None"
    assert T.grad.norm() > 0, "Gradient is zero"

    print(f"Test 6 PASSED: ||grad_T||={T.grad.norm().item():.6f}")


def run_all_tests():
    """Run all graph prior tests."""
    print("=" * 60)
    print("Phase 3 Tests: Graph Laplacian and Platooning Weights")
    print("=" * 60)

    test_laplacian_penalty_computation()
    test_same_lane_filtering()
    test_platooning_weights_spacing()
    test_platooning_weights_velocity()
    test_weighted_laplacian_penalty()
    test_gradient_flow()

    print("=" * 60)
    print("All Phase 3 tests PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
