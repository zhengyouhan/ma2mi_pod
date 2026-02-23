"""
Phase 5 Tests: Evaluation Metrics

Tests:
1. Wave arrival time detection
2. Wave arrival error computation
3. Trajectory collision count
4. Acceleration statistics
5. Detector reconstruction error
"""
from __future__ import annotations

import torch as th

from src.eval.metrics import (
    wave_arrival_time, wave_arrival_error,
    trajectory_collision_count, detector_reconstruction_error, acceleration_stats
)


def test_wave_arrival_detection():
    """Test 1: Wave arrival time is correctly detected."""
    # Speed drops below 15 at t=10, stays below for 15s (3 steps)
    speed = th.tensor([
        [20.0, 22.0],  # t=0
        [18.0, 21.0],  # t=5
        [12.0, 19.0],  # t=10 - det 0 drops
        [10.0, 14.0],  # t=15 - det 1 drops
        [9.0, 13.0],   # t=20
        [8.0, 12.0],   # t=25
        [7.0, 11.0],   # t=30
    ])
    time_s = th.tensor([0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0])

    t_arrival = wave_arrival_time(speed, time_s, v_threshold=15.0, sustained_s=15.0, dt=5.0)

    assert t_arrival[0].item() == 10.0, f"Det 0 arrival should be 10.0, got {t_arrival[0].item()}"
    assert t_arrival[1].item() == 15.0, f"Det 1 arrival should be 15.0, got {t_arrival[1].item()}"

    print(f"Test 1 PASSED: arrivals={t_arrival.tolist()}")


def test_wave_arrival_no_event():
    """Test 2: NaN returned when no wave arrives."""
    speed = th.tensor([
        [20.0, 22.0],
        [18.0, 21.0],
        [17.0, 19.0],
    ])
    time_s = th.tensor([0.0, 5.0, 10.0])

    t_arrival = wave_arrival_time(speed, time_s, v_threshold=15.0, sustained_s=10.0, dt=5.0)

    assert th.isnan(t_arrival[0]), "Should be NaN when no wave"
    assert th.isnan(t_arrival[1]), "Should be NaN when no wave"

    print("Test 2 PASSED: NaN returned for no wave event")


def test_wave_arrival_error():
    """Test 3: Wave arrival error computation."""
    t_pred = th.tensor([10.0, 15.0, float("nan")])
    t_obs = th.tensor([12.0, 18.0, 20.0])

    err = wave_arrival_error(t_pred, t_obs)

    # Valid: (10-12, 15-18) = (2, 3), MAE = 2.5, RMSE = sqrt((4+9)/2) = 2.55
    assert err["valid_count"].item() == 2, f"Valid count should be 2"
    assert abs(err["mae"].item() - 2.5) < 0.01, f"MAE should be 2.5"

    print(f"Test 3 PASSED: MAE={err['mae'].item():.2f}, valid={err['valid_count'].item()}")


def test_collision_count():
    """Test 4: Collision detection."""
    S = th.tensor([
        [0.0, 10.0, 25.0],   # t=0: 10m, 15m gaps
        [5.0, 14.0, 28.0],   # t=1: 9m, 14m gaps
        [10.0, 11.5, 30.0],  # t=2: 1.5m gap (collision!)
    ])
    leader_idx = th.tensor([-1, 0, 1])

    stats = trajectory_collision_count(S, leader_idx, s_min=2.0)

    # Only t=2 for vehicle 1 is below 2m
    assert stats["collision_count"].item() >= 1, "Should detect at least 1 collision"
    assert stats["min_spacing"].item() < 2.0, f"Min spacing should be < 2m"

    print(f"Test 4 PASSED: collisions={stats['collision_count'].item()}, min_spacing={stats['min_spacing'].item():.2f}m")


def test_acceleration_stats():
    """Test 5: Acceleration and jerk statistics."""
    V = th.tensor([
        [10.0, 12.0],
        [12.0, 14.0],  # accel = 20, 20 m/s² (with dt=0.1)
        [13.0, 13.0],  # accel = 10, -10 m/s²
    ])

    stats = acceleration_stats(V, dt=0.1)

    # Accelerations: [[20,20], [10,-10]]
    # Mean = (20+20+10-10)/4 = 10
    assert stats["accel_max"].item() == 20.0, f"Max accel should be 20"

    print(f"Test 5 PASSED: accel_max={stats['accel_max'].item():.1f}, accel_mean={stats['accel_mean'].item():.1f}")


def test_detector_reconstruction_error():
    """Test 6: Detector reconstruction error."""
    pred = {
        "flow": th.tensor([[10.0, 12.0], [11.0, 13.0]]),
        "speed": th.tensor([[20.0, 22.0], [19.0, 21.0]]),
    }
    obs = {
        "flow": th.tensor([[9.0, 11.0], [10.0, 12.0]]),
        "speed": th.tensor([[21.0, 23.0], [20.0, 22.0]]),
    }

    err = detector_reconstruction_error(pred, obs)

    # Flow diff: all +1, MSE = 1.0
    # Speed diff: all -1, MSE = 1.0
    assert abs(err["flow_mse"].item() - 1.0) < 0.01, f"Flow MSE should be 1.0"
    assert abs(err["speed_mse"].item() - 1.0) < 0.01, f"Speed MSE should be 1.0"

    print(f"Test 6 PASSED: flow_mse={err['flow_mse'].item():.4f}, speed_mse={err['speed_mse'].item():.4f}")


def test_nan_handling():
    """Test 7: Metrics handle NaN values correctly."""
    pred = {
        "flow": th.tensor([[10.0, float("nan")], [11.0, 13.0]]),
        "speed": th.tensor([[20.0, 22.0], [float("nan"), 21.0]]),
    }
    obs = {
        "flow": th.tensor([[9.0, 11.0], [10.0, 12.0]]),
        "speed": th.tensor([[21.0, 23.0], [20.0, 22.0]]),
    }

    err = detector_reconstruction_error(pred, obs)

    # Should compute only on valid entries
    assert th.isfinite(err["flow_mse"]), "flow_mse should be finite"
    assert th.isfinite(err["speed_mse"]), "speed_mse should be finite"

    print(f"Test 7 PASSED: NaN handling works")


def run_all_tests():
    """Run all metrics tests."""
    print("=" * 60)
    print("Phase 5 Tests: Evaluation Metrics")
    print("=" * 60)

    test_wave_arrival_detection()
    test_wave_arrival_no_event()
    test_wave_arrival_error()
    test_collision_count()
    test_acceleration_stats()
    test_detector_reconstruction_error()
    test_nan_handling()

    print("=" * 60)
    print("All Phase 5 tests PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
