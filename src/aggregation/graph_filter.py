"""
Laplacian Graph Filter F_L.

Applies controlled graph diffusion via Tikhonov filtering.

Math:
    Filter: v_macro = (I + εL)^{-1} v_raw

    This is equivalent to minimizing:
        ||v_macro - v_raw||^2 + ε * v_macro^T L v_macro

    Properties:
    - ε=0: no smoothing (v_macro = v_raw)
    - ε→∞: maximal smoothing (v_macro → constant)
    - Differentiable end-to-end via linear solve
"""
from __future__ import annotations

import torch as th

from src.aggregation.laplacian_builder import build_chain_laplacian


def tikhonov_filter(
    v_raw: th.Tensor,
    L: th.Tensor,
    epsilon: float = 0.1,
    delta: float = 1e-6,
) -> th.Tensor:
    """
    Apply Tikhonov graph filter: v_macro = (I + εL)^{-1} v_raw

    Args:
        v_raw: [T, J] raw signal (may contain NaN, will be filled)
        L: [J, J] graph Laplacian
        epsilon: smoothing parameter (larger = more smooth)
        delta: diagonal jitter for numerical stability

    Returns:
        v_macro: [T, J] filtered signal
    """
    T, J = v_raw.shape
    device, dtype = v_raw.device, v_raw.dtype

    # Fill NaN values first (required for linear solve)
    v_input = th.nan_to_num(v_raw, nan=0.0)

    # Build filter matrix: A = I + εL + δI
    I = th.eye(J, device=device, dtype=dtype)
    A = I + epsilon * L + delta * I

    # Solve A @ v_macro = v_raw for each time step
    # PyTorch's solve: A @ X = B → X = solve(A, B)
    # v_macro: [T, J]
    v_macro = th.linalg.solve(A, v_input.T).T

    return v_macro


def tikhonov_filter_2d(
    v_raw: th.Tensor,
    L_space: th.Tensor,
    epsilon_space: float = 0.1,
    epsilon_time: float = 0.01,
    delta: float = 1e-6,
) -> th.Tensor:
    """
    Apply 2D Tikhonov filter with separate space and time smoothing.

    Solves: (I + ε_x L_x ⊗ I_t + ε_t I_x ⊗ L_t) v = v_raw

    For efficiency, applies sequential 1D filters:
    1. Spatial filter per time
    2. Temporal filter per detector

    Args:
        v_raw: [T, J] raw signal
        L_space: [J, J] spatial Laplacian
        epsilon_space: spatial smoothing parameter
        epsilon_time: temporal smoothing parameter
        delta: numerical stability

    Returns:
        v_macro: [T, J] filtered signal
    """
    T, J = v_raw.shape
    device, dtype = v_raw.device, v_raw.dtype

    # Step 1: Spatial filter
    v_space = tikhonov_filter(v_raw, L_space, epsilon_space, delta)

    if epsilon_time > 0:
        # Step 2: Temporal filter (build temporal Laplacian)
        L_time = build_chain_laplacian(T, w=1.0, device=device, dtype=dtype)

        # Apply temporal filter per detector (transpose)
        v_time = tikhonov_filter(v_space.T, L_time, epsilon_time, delta).T
        return v_time
    else:
        return v_space


class GraphAggregator:
    """
    Complete aggregation pipeline: Projection + Graph Filter.

    Replaces Gaussian KDE with:
    1. Soft assignment projection (minimal smoothing)
    2. Tikhonov graph filter (controlled smoothing)
    """

    def __init__(
        self,
        x_dets: th.Tensor,
        dx: float,
        epsilon: float = 0.1,
        sigma_proj: float = None,
        epsilon_time: float = 0.0,
        device: str = "cpu",
        dtype: th.dtype = th.float32,
    ):
        """
        Initialize aggregator.

        Args:
            x_dets: [J] detector positions
            dx: detector spacing
            epsilon: spatial smoothing parameter
            sigma_proj: projection kernel width (default: dx/4)
            epsilon_time: temporal smoothing parameter (0 = no time smoothing)
            device: torch device
            dtype: torch dtype
        """
        self.x_dets = x_dets
        self.dx = dx
        self.epsilon = epsilon
        self.sigma_proj = sigma_proj if sigma_proj is not None else dx / 4.0
        self.epsilon_time = epsilon_time
        self.device = device
        self.dtype = dtype

        J = len(x_dets)
        self.L = build_chain_laplacian(J, w=1.0, device=device, dtype=dtype)

    def __call__(
        self,
        S: th.Tensor,
        V: th.Tensor,
    ) -> dict[str, th.Tensor]:
        """
        Aggregate micro states to macro detector signals.

        Args:
            S: [T, N] positions
            V: [T, N] velocities

        Returns:
            dict with:
                speed: [T, J] filtered speed
                flow: [T, J] filtered flow
                density: [T, J] filtered density
        """
        from src.aggregation.projection import soft_assignment_projection

        # Step 1: Projection
        proj = soft_assignment_projection(
            S, V, self.x_dets, self.dx, sigma=self.sigma_proj
        )

        v_raw = proj["v_raw"]
        rho_raw = proj["rho_raw"]
        q_raw = proj["q_raw"]

        # Step 2: Graph filter
        if self.epsilon_time > 0:
            v_macro = tikhonov_filter_2d(
                v_raw, self.L, self.epsilon, self.epsilon_time
            )
            rho_macro = tikhonov_filter_2d(
                rho_raw, self.L, self.epsilon, self.epsilon_time
            )
        else:
            v_macro = tikhonov_filter(v_raw, self.L, self.epsilon)
            rho_macro = tikhonov_filter(rho_raw, self.L, self.epsilon)

        # Flow from filtered fields
        q_macro = rho_macro * v_macro

        return {
            "speed": v_macro,
            "flow": q_macro,
            "density": rho_macro,
            # Also return raw for comparison
            "speed_raw": v_raw,
            "flow_raw": q_raw,
            "density_raw": rho_raw,
        }
