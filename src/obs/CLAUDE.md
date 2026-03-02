# obs/ - Observation Operator (Micro-to-Macro)

## Gaussian Aggregation (`detector_operator.py`)

Density field via Gaussian kernel:
```
ρ(x, t) = (1/√(2π)σ) Σᵢ exp(-½((x - sᵢ(t))/σ)²)
```

Weighted speed:
```
u(x, t) = Σᵢ wᵢ · vᵢ / Σᵢ wᵢ
where wᵢ = exp(-½((x - sᵢ)/σ)²)
```

Flow:
```
q = ρ · u
```

## Detector Outputs

At each detector location `x_det`, aggregate over spatial window `[x_det - Δx, x_det + Δx]`:
- `density`: mean of `ρ` in window
- `speed`: `q/ρ` with soft threshold for low density
- `flow`: mean of `q` in window
- `occ`: `clamp(ρ · L_eff, 0, 1)` where `L_eff` is effective vehicle length

## Cross-lane Aggregation (`detector_crosslane_aggregate`)

Per-lane processing then aggregation:
1. For each lane ℓ: compute `ρ_ℓ, u_ℓ, q_ℓ` via Gaussian kernel
2. Aggregate:
```
q_tot = Σ_ℓ q_ℓ
ρ_tot = Σ_ℓ ρ_ℓ
v_tot = (Σ_ℓ q_ℓ · v_ℓ) / (q_tot + ε)   [flow-weighted]
```

Returns: `{"flow", "speed", "density"}` at each detector

## Soft Conditionals

For division-safe operations when `ρ → 0`:
```
_soft_conditional(ρ, threshold, 0, q/ρ)
```
Uses sigmoid-based blending to avoid NaN gradients.
