# Ma2Mi — Macroscopic to Microscopic Trajectory Reconstruction

Reconstruct vehicle trajectories from macroscopic sensor data using differentiable microscopic simulation.

## Research Track

This is **Track 2** of a two-track research program:
- **Track 1 (diffLC)**: Differentiable simulation with lane-changing — the forward engine
- **Track 2 (Ma2Mi)**: Data assimilation, macro→micro reconstruction — the inverse problem

## Approach

- **Forward model**: Differentiable IDM (car-following), PyTorch
- **Parameterization**: Per-vehicle T (time headway), optional global v0
- **Micro→macro bridge**: Gaussian kernel aggregation → density/speed/flow fields
- **Observations**: Cross-lane aggregated loop detectors (realistic setup)
- **Regularization**: Graph Laplacian (leader-follower smoothness)
- **Optimization**: Gradient descent through the full pipeline

## History

- `archive/` contains the original Fourier-basis version (master thesis, `ma2mi_final`)
  - Fourier parameterization, identifiability analysis, NGSIM experiments
  - PINN baseline, probe vehicle experiments
- Active `src/` is the successor with graph regularization, robust losses, wave front analysis

## Environment

```bash
conda activate ngsim
```

## Status

See `project_overview.md` for the detailed research plan.
