# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Environment

Use conda environment `ngsim` for all Python execution:
```bash
conda activate ngsim
```

## Development Workflow

Before writing any code:
1. Start with how you will verify this change works
2. Write the test or verification step first
3. Then implement the code
4. Run verification and iterate until it passes

## Running the Simulation

Main entry point for optimization:
```bash
python src/run_fit.py --csv <path_to_ngsim_csv> [options]
```

Key command-line arguments:
- `--csv`: Path to NGSIM trajectory CSV file
- `--duration-s`: Simulation window in seconds (default: 10)
- `--dt`: Time step in seconds (default: 0.1)
- `--iters`: Optimization iterations (default: 80)
- `--lr`: Learning rate (default: 0.03)
- `--lambda-T`: Graph Laplacian regularization weight (default: 0.01)
- `--learn-v0`: Whether to optimize desired speed (`none`, `scalar`, or `vector`)
- `--run-lambda-scan`: Run comparison between lambda values

## Architecture Overview

This is a differentiable traffic simulation system using PyTorch for gradient-based parameter optimization. It bridges microscopic vehicle dynamics with macroscopic traffic observations.

### Core Pipeline

1. **Data Loading** (`src/data/`): Loads NGSIM CSV trajectories, extracts initial conditions, builds leader-follower topology, creates detector observations at 5-second intervals

2. **IDM Simulation** (`src/sim/`): Intelligent Driver Model with differentiable parameters (T, v0), multi-lane rollout, per-lane leader-follower topology

3. **Observation Operator** (`src/obs/`): Micro-to-macro mapping using Gaussian kernel aggregation, converts trajectories to detector outputs

4. **Graph Regularization** (`src/graph_prior.py`): Laplacian penalty enforcing parameter smoothness between leader-follower pairs

5. **Loss Functions** (`src/loss/`): Robust losses (MSE, Huber, log1p) with NaN-safe masking for detector reconstruction

6. **Evaluation Metrics** (`src/eval/`): Wave arrival time detection, trajectory collision checks, parameter smoothness

7. **Utilities** (`src/utils/`): Training diagnostics, convergence monitoring, issue detection

### Decision Variables

- **T (time headway)**: Per-vehicle, bounded via sigmoid reparameterization
- **v0 (desired speed)**: Optional, can be scalar or per-vehicle
- **v_init**: Fixed from data (not optimized)

### Loss Components

- `L_det`: Detector reconstruction loss (predicted vs observed)
- `L_graph`: Graph Laplacian penalty for parameter smoothness
- Total: `loss = L_det + λ_T * L_graph_T`

### Graph Laplacian Penalty (`src/graph_prior.py`)

```
R(p) = Σᵢ wᵢ · (pᵢ - p_leader(i))²
```
Only same-lane valid leader edges contribute. Default `wᵢ = 1`.

## Project Phases

See `project_overview.md` for the detailed 6-phase development plan covering:
- Phase 1: Cross-lane aggregated detector implementation
- Phase 2: Differentiable parameter injection
- Phase 3: Graph Laplacian regularization
- Phase 4: Hyperparameter sweep
- Phase 5: Evaluation metrics (wave arrival time, reconstruction error)
- Phase 6: Reproducible experiment protocol

## Key Assumptions

- Multi-lane simulation without lane changes
- Vehicles stay in assigned lanes with per-lane leader-follower relationships
- Cross-lane aggregation mitigates lane-wise mismatch in NGSIM data
- All operations support gradient computation via PyTorch autograd

## Project Structure Note

- The `archive/` directory contains a previous Fourier-basis implementation for reference. Active development is in `src/`.
- Keep codes in a form of module, and keep them organized in src/. 
- Maintain a CLAUDE.md in each subdirectory in src/ to keep information consice and sturctured. These CLAUDE.md should contain the math formulation and notes for the implmentation in a minimalist way.
