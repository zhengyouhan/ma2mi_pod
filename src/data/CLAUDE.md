# data/ - NGSIM Data Loading

## Data Format

NGSIM CSV columns used:
- `Vehicle_ID`: unique vehicle identifier
- `Global_Time`: timestamp in milliseconds
- `Local_Y`: longitudinal position in feet
- `Lane_ID`: lane number
- `v_Vel`: velocity in ft/s

Unit conversion: `FT2M = 0.3048`

## Loading Pipeline (`ngsim_loader.py`)

1. Read rows within time window `[t0_ms, t0_ms + duration_s * 1000]`
2. Extract initial conditions from first frame (or first 500ms fallback)
3. Filter vehicles within spatial window `[x_min, x_max]`
4. Build leader indices via `compute_leader_idx_per_lane`
5. Construct observation matrix `S_real, V_real: [T, N]`
6. Compute detector observations at 5-second intervals

## Output Dictionary

```python
{
    "s0", "v0": initial conditions [N]
    "lane_id", "leader_idx": topology [N]
    "xq": spatial grid [M]
    "x_dets": detector positions
    "obs_5s": {"density", "speed", "flow", "occ"} at 5s intervals
    "time_idx_5s": time indices for observations
    "S_real", "V_real": ground truth trajectories [T, N]
}
```

