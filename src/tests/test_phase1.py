"""
Phase 1 Sanity Tests: Cross-lane Macro-Micro Bridge

Tests:
1. detector_crosslane_aggregate returns finite values
2. y_hat changes when micro trajectories change
3. loss_det.backward() produces nonzero gradients
"""
from __future__ import annotations

import torch as th

from src.obs.detector_operator import detector_crosslane_aggregate
from src.loss import detector_loss


def test_crosslane_returns_finite():
    """Test 1: detector_crosslane_aggregate returns finite values."""
    T, N, M = 100, 20, 50
    S = th.cumsum(th.rand(T, N) * 0.5, dim=0)
    V = th.rand(T, N) * 20 + 5
    lane_id = th.randint(0, 3, (N,))
    xq = th.linspace(0, 100, M)
    x_dets = [25.0, 50.0, 75.0]

    out = detector_crosslane_aggregate(S, V, lane_id, xq, x_dets, sigma=10.0)

    assert th.isfinite(out["flow"]).all(), "flow contains NaN/Inf"
    assert th.isfinite(out["speed"]).all(), "speed contains NaN/Inf"
    assert th.isfinite(out["density"]).all(), "density contains NaN/Inf"
    assert out["flow"].shape == (T, len(x_dets)), f"flow shape mismatch: {out['flow'].shape}"
    assert out["speed"].shape == (T, len(x_dets)), f"speed shape mismatch: {out['speed'].shape}"

    print("Test 1 PASSED: detector_crosslane_aggregate returns finite values")


def test_yhat_changes_with_trajectory():
    """Test 2: y_hat changes when micro trajectories change."""
    T, N, M = 50, 10, 30
    S1 = th.cumsum(th.rand(T, N) * 0.5, dim=0)
    V1 = th.rand(T, N) * 20 + 5

    # Perturb trajectories
    S2 = S1 + 10.0
    V2 = V1 + 5.0

    lane_id = th.randint(0, 2, (N,))
    xq = th.linspace(0, 50, M)
    x_dets = [15.0, 35.0]

    out1 = detector_crosslane_aggregate(S1, V1, lane_id, xq, x_dets, sigma=10.0)
    out2 = detector_crosslane_aggregate(S2, V2, lane_id, xq, x_dets, sigma=10.0)

    flow_diff = (out1["flow"] - out2["flow"]).abs().mean()
    speed_diff = (out1["speed"] - out2["speed"]).abs().mean()

    assert flow_diff > 1e-6, f"flow did not change: diff={flow_diff}"
    assert speed_diff > 1e-6, f"speed did not change: diff={speed_diff}"

    print(f"Test 2 PASSED: y_hat changes with trajectory (flow_diff={flow_diff:.4f}, speed_diff={speed_diff:.4f})")


def test_loss_backward_produces_gradients():
    """Test 3: loss_det.backward() produces nonzero gradients."""
    T, N, M = 50, 10, 30

    # Make S require gradients (simulating learnable parameters affecting trajectory)
    S = th.nn.Parameter(th.cumsum(th.rand(T, N) * 0.5, dim=0))
    V = th.rand(T, N) * 20 + 5

    lane_id = th.randint(0, 2, (N,))
    xq = th.linspace(0, 50, M)
    x_dets = [15.0, 35.0]

    pred = detector_crosslane_aggregate(S, V, lane_id, xq, x_dets, sigma=10.0)

    # Create fake observations
    obs = {
        "flow": th.rand_like(pred["flow"]),
        "speed": th.rand_like(pred["speed"]),
    }

    L_det = detector_loss(
        pred=pred,
        obs=obs,
        weights={"flow": 1.0, "speed": 0.1},
        loss_type="huber",
        huber_delta=1.0,
    )

    L_det.backward()

    assert S.grad is not None, "S.grad is None"
    grad_norm = S.grad.norm().item()
    assert grad_norm > 1e-10, f"gradient too small: {grad_norm}"

    print(f"Test 3 PASSED: loss_det.backward() produces gradients (||grad||={grad_norm:.6f})")


def run_all_tests():
    """Run all Phase 1 sanity tests."""
    print("=" * 60)
    print("Phase 1 Sanity Tests")
    print("=" * 60)

    test_crosslane_returns_finite()
    test_yhat_changes_with_trajectory()
    test_loss_backward_produces_gradients()

    print("=" * 60)
    print("All Phase 1 tests PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
