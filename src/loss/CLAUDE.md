# loss/ - Robust Loss Functions

## Masked MSE (`robust.py`)

```
L_mse = Σ mask_i · (pred_i - obs_i)² / Σ mask_i
```

NaN-safe: replaces NaN/Inf with zeros before computation.

## Huber Loss (`robust.py`)

```
L(x) = 0.5·x²                    if |x| ≤ δ
       δ·(|x| - 0.5·δ)           otherwise
```

Robust to outliers: quadratic near zero, linear for large errors.

## Log1p MSE (`robust.py`)

```
L = Σ mask_i · (log(1+pred_i) - log(1+obs_i))² / Σ mask_i
```

Scale-invariant for density/flow with heavy tails.

## Detector Loss (`robust.py`)

Weighted sum over channels (speed, flow, occupancy):
```
L_det = Σ_k w_k · L_k(pred_k, obs_k)
```

Supports both MSE and Huber loss types.
