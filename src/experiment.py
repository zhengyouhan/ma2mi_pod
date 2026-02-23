"""
Reproducible experiment protocol for traffic wave reconstruction.

Phase 6: Paper-ready experiment definitions.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Any

import torch as th


@dataclass
class ExperimentConfig:
    """Fixed protocol for reproducible experiments."""

    # Data settings
    csv: str = "data/0820am-0835am/trajectories-0820am-0835am.csv"
    t0_ms: int | None = None
    offset_s: float = 120.0
    duration_s: float = 60.0  # Longer for wave detection
    x_center: float = 340.0
    x_half: float = 150.0

    # Simulation settings
    dt: float = 0.1
    sigma: float = 10.0

    # IDM parameters (fixed)
    a_max: float = 2.0
    b_comfort: float = 2.0
    v_target: float = 25.0
    s0_idm: float = 2.0
    ghost_v: float = 15.0
    ghost_gap0: float = 50.0

    # Optimization settings
    iters: int = 100
    lr: float = 0.03
    seed: int = 42

    # Decision variable settings
    T_init: float = 1.5
    T_min: float = 0.5
    T_max: float = 3.0
    T_bar: float = 1.5
    learn_v0: str = "none"
    v0_max: float = 45.0
    v0_bar: float = 25.0

    # Regularization
    lambda_T: float = 0.01
    beta_T: float = 0.0
    beta_v0: float = 0.0
    state_weighted_graph: bool = False
    s0_platooning: float = 5.0
    vs_platooning: float = 2.0

    # Loss settings
    speed_weight: float = 0.1

    # Evaluation settings
    wave_v_threshold: float = 15.0
    wave_sustained_s: float = 20.0

    # Experiment metadata
    name: str = "default"
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ExperimentConfig":
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)


# Predefined experiment configurations
BASELINE_NO_REG = ExperimentConfig(
    name="baseline_no_reg",
    description="Baseline: no regularization (lambda_T=0)",
    lambda_T=0.0,
    beta_T=0.0,
)

BASELINE_L2_PRIOR = ExperimentConfig(
    name="baseline_l2_prior",
    description="Baseline: weak L2 prior only (no graph)",
    lambda_T=0.0,
    beta_T=0.001,
)

PROPOSED_GRAPH_LAPLACIAN = ExperimentConfig(
    name="proposed_graph_laplacian",
    description="Proposed: graph Laplacian on T",
    lambda_T=0.01,
    beta_T=0.0,
)

PROPOSED_SOFT_WEIGHTS = ExperimentConfig(
    name="proposed_soft_weights",
    description="Proposed: graph Laplacian with soft platooning weights",
    lambda_T=0.01,
    beta_T=0.0,
    state_weighted_graph=True,
)

ABLATION_CONFIGS = {
    "no_reg": BASELINE_NO_REG,
    "l2_prior": BASELINE_L2_PRIOR,
    "graph_const": PROPOSED_GRAPH_LAPLACIAN,
    "graph_soft": PROPOSED_SOFT_WEIGHTS,
}


@dataclass
class ExperimentResult:
    """Container for experiment results."""

    config: dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Fitted parameters
    T: list[float] | None = None
    v0: list[float] | None = None

    # Training history
    history: dict[str, list[float]] | None = None

    # Final metrics
    L_det_final: float | None = None
    L_graph_final: float | None = None
    smoothness: float | None = None

    # Evaluation metrics
    flow_mse: float | None = None
    speed_mse: float | None = None
    collision_count: int | None = None
    collision_fraction: float | None = None
    min_spacing: float | None = None
    accel_max: float | None = None
    jerk_max: float | None = None
    wave_mae: float | None = None
    wave_rmse: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ExperimentResult":
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    th.manual_seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(seed)


def create_output_dir(base_dir: str = "results") -> str:
    """Create timestamped output directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def save_trajectories(
    S: th.Tensor,
    V: th.Tensor,
    output_dir: str,
    prefix: str = "traj",
) -> None:
    """Save trajectories to files."""
    th.save(S, os.path.join(output_dir, f"{prefix}_positions.pt"))
    th.save(V, os.path.join(output_dir, f"{prefix}_velocities.pt"))


def save_predictions(
    pred: dict[str, th.Tensor],
    obs: dict[str, th.Tensor],
    output_dir: str,
) -> None:
    """Save detector predictions and observations."""
    th.save(pred, os.path.join(output_dir, "detector_pred.pt"))
    th.save(obs, os.path.join(output_dir, "detector_obs.pt"))
