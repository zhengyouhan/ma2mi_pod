# sim/ - Microscopic Simulation

## IDM Acceleration (`idm.py`)

Intelligent Driver Model:
```
a = a_max * [1 - (v/v0)^δ - (s*/s)^2]
```
where desired gap:
```
s* = s0 + v·T + (v·Δv)/(2·√(a_max·b))
```

Parameters:
- `T`: time headway (learnable)
- `v0` (v_target): desired speed
- `s0`: minimum gap
- `a_max`: max acceleration
- `b`: comfortable deceleration
- `δ = 4`: acceleration exponent

Numerical stability:
- Soft clamping via `softplus` for differentiability
- `min_gap` enforcement with soft thresholding
- Prevents negative speed: `a ≥ -v/dt`

## Rollout (`rollout.py`)

Euler integration:
```
v_{t+1} = v_t + a_t · dt
s_{t+1} = s_t + v_{t+1} · dt
```

Ghost vehicles for leaders without a leading vehicle:
```
s_ghost(t) = s0 + gap0 + v_ghost · t
```

Returns `(S, V, A)` with shapes `[steps+1, N]`, `[steps+1, N]`, `[steps, N]`

## Topology (`topology.py`)

Per-lane leader assignment:
- `leader_idx[i] = j` if vehicle `j` is nearest ahead in same lane
- `leader_idx[i] = -1` if no leader exists
