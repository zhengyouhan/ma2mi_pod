# aggregation/ - Graph-Laplacian Micro→Macro Aggregation

## Overview

Replaces Gaussian KDE with two-step pipeline:
1. **Projection** (micro → raw detector signal)
2. **Graph Filter** (raw → macro signal via Laplacian)

## Projection (`projection.py`)

### Hard Binning
```
c[n,j] = count of vehicles in cell j at time n
v_raw[n,j] = mean velocity of vehicles in cell j
```

### Soft Assignment (default)
```
W[i,j] = exp(-½((s_i - x_j)/σ)²)
v_raw[n,j] = Σ_i W[i,j] v_i / Σ_i W[i,j]
```
Uses narrow kernel σ = dx/4 for minimal smoothing.

## Laplacian Builder (`laplacian_builder.py`)

1D chain graph:
```
Nodes: detectors j = 0..J-1
Edges: (j, j+1) with weight w

L = D - W = tridiag(-1, 2, -1)
```

## Graph Filter (`graph_filter.py`)

### Tikhonov Filter
```
v_macro = (I + εL)^{-1} v_raw
```

Equivalent to minimizing:
```
||v_macro - v_raw||² + ε · v_macro^T L v_macro
```

Properties:
- ε=0: no smoothing
- ε→∞: constant output
- Differentiable via linear solve

### 2D Filter (optional)
Sequential space-time filtering:
```
v = F_space(F_time(v_raw))
```

## GraphAggregator Class

Complete pipeline:
```python
agg = GraphAggregator(x_dets, dx, epsilon=0.1)
macro = agg(S, V)  # returns {speed, flow, density}
```

## Key Difference from KDE

| Aspect | Gaussian KDE | Graph Laplacian |
|--------|-------------|-----------------|
| Smoothing | Global convolution | Topology-aware |
| Control | Single σ | ε (tunable) |
| Wavefront | Over-smoothed | Preserved |
| Math | Kernel density | Graph diffusion |
