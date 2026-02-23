"""
Phase 2 Tests: Sigmoid Reparameterization

Tests:
1. Round-trip accuracy: inverse_sigmoid_reparam -> sigmoid_reparam
2. Bounded output: sigmoid_reparam always in [low, high]
3. Gradient at boundaries: nonzero gradient near bounds
4. Gradient flow through simulation
"""
from __future__ import annotations

import torch as th

from src.sim.reparam import sigmoid_reparam, inverse_sigmoid_reparam


def test_round_trip_accuracy():
    """Test 1: Round-trip conversion accuracy."""
    low, high = 0.5, 3.0
    x_orig = th.tensor([0.6, 1.0, 1.5, 2.0, 2.9])

    u = inverse_sigmoid_reparam(x_orig, low, high)
    x_back = sigmoid_reparam(u, low, high)

    error = (x_orig - x_back).abs().max().item()
    assert error < 1e-5, f"Round-trip error too large: {error}"

    print(f"Test 1 PASSED: Round-trip error = {error:.2e}")


def test_bounded_output():
    """Test 2: Output is always within [low, high]."""
    low, high = 0.5, 3.0

    # Test extreme values of u
    u = th.tensor([-100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0])
    x = sigmoid_reparam(u, low, high)

    assert (x >= low).all(), f"Values below lower bound: {x.min().item()}"
    assert (x <= high).all(), f"Values above upper bound: {x.max().item()}"

    print(f"Test 2 PASSED: All outputs in [{low}, {high}], range=[{x.min().item():.4f}, {x.max().item():.4f}]")


def test_gradient_at_boundaries():
    """Test 3: Nonzero gradient near boundaries (unlike hard clamp)."""
    low, high = 0.5, 3.0

    # Near upper bound
    u_high = th.nn.Parameter(th.tensor([5.0]))
    x_high = sigmoid_reparam(u_high, low, high)
    x_high.backward()
    grad_high = u_high.grad.item()

    # Near lower bound
    u_low = th.nn.Parameter(th.tensor([-5.0]))
    x_low = sigmoid_reparam(u_low, low, high)
    x_low.backward()
    grad_low = u_low.grad.item()

    assert abs(grad_high) > 1e-6, f"Gradient at upper boundary too small: {grad_high}"
    assert abs(grad_low) > 1e-6, f"Gradient at lower boundary too small: {grad_low}"

    print(f"Test 3 PASSED: grad_high={grad_high:.6f}, grad_low={grad_low:.6f}")


def test_compare_with_hard_clamp():
    """Test 4: Sigmoid has gradient where hard clamp doesn't."""
    low, high = 0.5, 3.0

    # Value exceeding bounds
    x_raw = 3.5

    # Hard clamp: gradient is 0 when clamped
    u_hard = th.nn.Parameter(th.tensor([x_raw]))
    x_hard = th.clamp(u_hard, low, high)
    x_hard.backward()
    grad_hard = u_hard.grad.item()

    # Sigmoid: always has gradient
    u_soft = th.nn.Parameter(inverse_sigmoid_reparam(th.tensor([high - 0.01]), low, high))
    x_soft = sigmoid_reparam(u_soft, low, high)
    x_soft.backward()
    grad_soft = u_soft.grad.item()

    assert grad_hard == 0.0, f"Hard clamp should have 0 gradient at boundary: {grad_hard}"
    assert grad_soft > 0.0, f"Sigmoid should have positive gradient: {grad_soft}"

    print(f"Test 4 PASSED: hard_clamp_grad={grad_hard}, sigmoid_grad={grad_soft:.6f}")


def run_all_tests():
    """Run all reparameterization tests."""
    print("=" * 60)
    print("Phase 2 Tests: Sigmoid Reparameterization")
    print("=" * 60)

    test_round_trip_accuracy()
    test_bounded_output()
    test_gradient_at_boundaries()
    test_compare_with_hard_clamp()

    print("=" * 60)
    print("All Phase 2 tests PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
