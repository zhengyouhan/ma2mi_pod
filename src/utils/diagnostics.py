"""
Diagnostics utilities for optimization tracking.

Provides loss component analysis, convergence monitoring, and issue detection.
"""
from __future__ import annotations

import torch as th
from dataclasses import dataclass, field


@dataclass
class TrainingDiagnostics:
    """Container for diagnostics at a single iteration."""

    iteration: int

    # Loss components
    L_total: float = 0.0
    L_det: float = 0.0
    L_graph: float = 0.0

    # Gradient info
    grad_norm_T: float = 0.0
    grad_norm_v0: float = 0.0

    # Parameter stats
    T_mean: float = 0.0
    T_std: float = 0.0
    T_min: float = 0.0
    T_max: float = 0.0

    v0_mean: float = 0.0
    v0_std: float = 0.0

    # Smoothness
    T_smoothness: float = 0.0  # mean |T_i - T_leader(i)|

    def summary_line(self) -> str:
        """One-line summary for logging."""
        return (
            f"it={self.iteration:04d} "
            f"L={self.L_total:.4f} "
            f"L_det={self.L_det:.4f} "
            f"L_graph={self.L_graph:.6f} "
            f"T=[{self.T_min:.2f},{self.T_max:.2f}] "
            f"smooth={self.T_smoothness:.3f}"
        )

    def detailed_report(self) -> str:
        """Multi-line detailed report."""
        lines = [
            f"=== Iteration {self.iteration} ===",
            f"Loss: total={self.L_total:.6f}, det={self.L_det:.6f}, graph={self.L_graph:.6f}",
            f"Gradients: ||∇T||={self.grad_norm_T:.4f}, ||∇v0||={self.grad_norm_v0:.4f}",
            f"T: mean={self.T_mean:.3f}, std={self.T_std:.3f}, range=[{self.T_min:.3f}, {self.T_max:.3f}]",
            f"v0: mean={self.v0_mean:.2f}, std={self.v0_std:.2f}",
            f"Smoothness: |T_i - T_leader|={self.T_smoothness:.4f}",
        ]
        return "\n".join(lines)


class DiagnosticsTracker:
    """Tracks diagnostics over training iterations."""

    def __init__(self, lambda_T: float = 0.01):
        self.lambda_T = lambda_T
        self.history: list[TrainingDiagnostics] = []

    def compute(
        self,
        iteration: int,
        L_det: th.Tensor,
        L_graph: th.Tensor,
        L_total: th.Tensor,
        T_param: th.Tensor,
        leader_idx: th.Tensor,
        lane_id: th.Tensor,
        v0_param: th.Tensor | None = None,
    ) -> TrainingDiagnostics:
        """Compute all diagnostics for current iteration."""
        diag = TrainingDiagnostics(iteration=iteration)

        # Loss values
        diag.L_total = L_total.item()
        diag.L_det = L_det.item()
        diag.L_graph = L_graph.item()

        with th.no_grad():
            # Gradient norms
            if T_param.grad is not None:
                diag.grad_norm_T = T_param.grad.norm().item()
            if v0_param is not None and v0_param.grad is not None:
                diag.grad_norm_v0 = v0_param.grad.norm().item()

            # T statistics
            diag.T_mean = T_param.mean().item()
            diag.T_std = T_param.std().item()
            diag.T_min = T_param.min().item()
            diag.T_max = T_param.max().item()

            # v0 statistics
            if v0_param is not None:
                diag.v0_mean = v0_param.mean().item()
                diag.v0_std = v0_param.std().item() if v0_param.numel() > 1 else 0.0

            # T smoothness
            valid = leader_idx >= 0
            safe_leader = th.clamp(leader_idx, min=0)
            same_lane = lane_id == lane_id[safe_leader]
            valid = valid & same_lane

            if valid.sum() > 0:
                i = th.where(valid)[0]
                j = leader_idx[i]
                diff = th.abs(T_param[i] - T_param[j])
                diag.T_smoothness = diff.mean().item()

        self.history.append(diag)
        return diag

    def check_issues(self, diag: TrainingDiagnostics) -> list[str]:
        """Check for potential issues and return warnings."""
        warnings = []

        if diag.T_min < 0.3:
            warnings.append(f"WARNING: T_min={diag.T_min:.2f} < 0.3s, unrealistically small!")

        if diag.T_max > 5.0:
            warnings.append(f"WARNING: T_max={diag.T_max:.2f} > 5.0s, unrealistically large!")

        if diag.L_graph < 1e-10 and self.lambda_T > 0:
            warnings.append("WARNING: L_graph ~ 0, graph regularization not contributing!")

        if diag.grad_norm_T > 100:
            warnings.append(f"WARNING: ||∇T||={diag.grad_norm_T:.1f} > 100, gradient explosion!")

        return warnings

    def get_summary(self) -> dict:
        """Get summary statistics over all iterations."""
        if not self.history:
            return {}

        return {
            "n_iterations": len(self.history),
            "final_L_total": self.history[-1].L_total,
            "final_L_det": self.history[-1].L_det,
            "final_T_mean": self.history[-1].T_mean,
            "final_T_smoothness": self.history[-1].T_smoothness,
        }


def print_diagnostics_header():
    """Print header for diagnostics output."""
    print("=" * 70)
    print("OPTIMIZATION DIAGNOSTICS")
    print("=" * 70)
    print("Monitoring: L_det, L_graph, T range, smoothness, gradients")
    print("=" * 70)
