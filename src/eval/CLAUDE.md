# eval/ - Evaluation Metrics

## Wave Arrival Time (`metrics.py`)

Primary KPI for Phase 5.

Detection criterion:
```
t_arrival[j] = first t where v(t:t+T_w, j) < v_th  ∀ sustained T_w
```

Error metrics:
```
MAE  = mean |t_pred - t_obs|
RMSE = sqrt(mean (t_pred - t_obs)²)
```

## Trajectory Collision Count (`metrics.py`)

Physical plausibility check:
```
collision = (s_lead - s) < s_min
```

Reports: count, fraction, minimum spacing observed.

## Parameter Smoothness (`metrics.py`)

Graph-based smoothness on leader edges:
```
mean_diff = mean |p_i - p_leader(i)|
```

Used to evaluate effect of graph Laplacian regularization.
