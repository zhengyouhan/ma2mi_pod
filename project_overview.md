# Project TODO (Phase 1–6): Cross-lane Macro–Micro Bridge + Graph-Laplacian DA on (T, v_desire)

> Assumptions:
> - Multi-lane simulation **without lane-change**
> - Macro–micro bridge uses **cross-lane aggregated detectors**
> - Decision variables: **T (time headway)** and **v_desire** (notation `v0`)
> - Initial speed `v_init` is **not** optimized (fixed/initialized from data)
> - Data: NGSIM → lane-wise mismatch exists; cross-lane aggregation mitigates it

---

## Phase 1 — Macro–Micro Bridge (Cross-lane Aggregated Detectors) ✅

### 1.1 Choose observation channels (start minimal)
- [x] Decide which detector outputs to fit first:
  - [x] **Option A (chosen):** `q_tot(t)` (flow) + `v_tot(t)` (speed)

### 1.2 Implement cross-lane aggregation at each detector location
For each detector position `x_j` and time bin `t_k` (5s):
- [x] Implement lane-wise detector operator per lane: `y_hat_lane[ℓ] = detector_lane(micro_traj_lane[ℓ])`
- [x] Aggregate across lanes:
  - [x] Flow: `q_tot = \sum_ℓ q_ℓ`
  - [x] Density: `rho_tot = \sum_ℓ rho_ℓ`
  - [x] Speed (flow-weighted): `v_tot = (\sum_ℓ q_ℓ * v_ℓ) / (\sum_ℓ q_ℓ + eps)`
- [x] Ensure returned tensor shape is consistent: `y_hat: [T_obs, J_det, d]`

> Implementation: `src/obs/detector_operator.py::detector_crosslane_aggregate()`

### 1.3 Align simulation outputs to detector sampling grid (5s)
- [x] Ensure micro rollout time step `dt_sim` is compatible (0.1s default)
- [x] Implement binning/integration over 5s windows via `align_indices_to_grid()`

### 1.4 Normalize and robustify the detector loss
- [x] Use robust loss: Huber loss via `src/loss/robust.py::detector_loss()`
- [x] Verify that loss scale is stable across scenarios

### 1.5 Sanity tests for Phase 1
- [x] Run a short rollout and confirm:
  - [x] `detector_operator(micro_traj)` returns finite values
  - [x] `y_hat` changes when micro trajectories change
  - [x] `loss_det.backward()` produces nonzero gradients

> Tests: `src/tests/test_phase1.py`

---

## Phase 2 — Inject (T, v_desire) into Micro Simulation (Differentiable)

### 2.1 Standardize notation & variable meanings (code + paper)
- [ ] Use `T` for time headway
- [ ] Use `v0` for **v_desire** (desired speed)
- [ ] Use `v_init` for initial speeds (fixed, not optimized)

### 2.2 Choose parameterization and constraints (avoid runaway)
- [ ] Decide bounds:
  - [ ] `T ∈ [T_min, T_max]` (e.g., [0.5, 3.0] s)
  - [ ] `v0 ∈ [v0_min, v0_max]` (e.g., [5, 40] m/s)
- [ ] Implement reparameterization:
  - [ ] `T = T_min + (T_max - T_min) * sigmoid(u_T)`
  - [ ] `v0 = v0_min + (v0_max - v0_min) * sigmoid(u_v0)`

### 2.3 Choose dimensionality (start conservative)
- [ ] **Recommended start (more identifiable):**
  - [ ] `T`: per-vehicle `T[i]`
  - [ ] `v0`: **lane-wise scalar** `v0_lane[ℓ]` (upgrade to per-vehicle later)
- [ ] Initialize:
  - [ ] `u_T` so that `T` starts near typical values (1.2–1.8 s)
  - [ ] `u_v0` so that `v0` starts near free-flow estimate
  - [ ] `v_init` from data (first frame), not optimized

### 2.4 Verify differentiability & numerical stability in rollout
- [ ] Ensure no hard non-differentiable ops in the computational graph:
  - [ ] avoid `argmax/argmin` in forward (or `detach` decisions)
- [ ] Add epsilons to avoid division by zero
- [ ] Collision/spacing handling:
  - [ ] enforce minimum spacing with smooth clamp if needed
- [ ] Validate:
  - [ ] `loss_det.backward()` yields gradients for `u_T` and `u_v0` (when enabled)

---

## Phase 3 — Graph Construction + Laplacian Regularization (Graph-based Platooning)

### 3.1 Define the graph (first version)
- [ ] Nodes: vehicles
- [ ] Edges: same-lane leader–follower pairs
  - [ ] For each vehicle `i`, edge `(i, leader(i))` if leader exists
- [ ] Obtain from rollout/topology:
  - [ ] `leader_idx[i]` (`-1` if none)
  - [ ] `lane_id[i]`

### 3.2 Implement Laplacian penalty without building L explicitly
For param vector `p` (either `T` or `v0`):
- [ ] `mask = leader_idx >= 0`
- [ ] `diff = p[mask] - p[leader_idx[mask]]`
- [ ] Constant weights baseline: `w = 1`
- [ ] Penalty: `R(p) = sum(w * diff^2)`

Add both terms (start with T only):
- [ ] `loss_graph_T = R(T)`
- [ ] Optional later: `loss_graph_v0 = R(v0)` (if v0 is per-vehicle)

### 3.3 (Optional upgrade) Soft “platooning weights” (second version)
- [ ] Define weights using state features (detach):
  - [ ] spacing `s_i`, relative speed `dv_i`
  - [ ] `w_i = exp(-s_i/s0) * exp(-abs(dv_i)/vs)`
  - [ ] `w = w.detach()` to stabilize optimization
- [ ] Decide at what time to evaluate weights:
  - [ ] start-of-window or window-average (avoid time-varying weights per step)

---

## Phase 4 — Total Objective + Hyperparameter Sweep

### 4.1 Total loss definition
- [ ] Detector loss (cross-lane):
  - [ ] `loss_det = robust_loss(y_hat_tot, y_obs_tot)`
- [ ] Graph regularization:
  - [ ] `loss_graph_T = R(T)`
  - [ ] `loss_graph_v0 = R(v0)` (optional / later)
- [ ] Weak priors to prevent runaway (small weight):
  - [ ] `loss_prior_T = ||T - T_bar||^2`
  - [ ] `loss_prior_v0 = ||v0 - v0_bar||^2`

Total:
- [ ] `loss = loss_det + λ_T * loss_graph_T + λ_v0 * loss_graph_v0 + β_T * loss_prior_T + β_v0 * loss_prior_v0`

### 4.2 Train in stages (recommended)
- [ ] Stage A: baseline fit
  - [ ] set `λ_T=0`
  - [ ] optimize `T` (and optional lane-wise `v0`)
  - [ ] record metrics
- [ ] Stage B: enable Laplacian
  - [ ] sweep `λ_T` on log-scale:
    - [ ] `λ_T ∈ {0, 1e-4, 1e-3, 1e-2, 1e-1, 1}`
  - [ ] keep `β` small (e.g., 1e-4–1e-3 relative to detector loss scale)
- [ ] Stage C: enable v_desire
  - [ ] start with **lane-wise** `v0_lane[ℓ]`
  - [ ] then consider per-vehicle `v0[i]` with graph regularization

### 4.3 Numerical stability checks during optimization
- [ ] Monitor:
  - [ ] NaNs / infs
  - [ ] gradient norms
  - [ ] min spacing (no collisions)
- [ ] Early-stop / backtracking if divergence occurs

---

## Phase 5 — Evaluation Metrics (Wave + Trajectory Quality)

### 5.1 Primary KPI: wave arrival time error (phase accuracy)
- [ ] Define event detection on cross-lane speed:
  - [ ] choose threshold `v_th`
  - [ ] require sustained condition over `T_w` (e.g., 20–30s)
- [ ] For each detector `j`:
  - [ ] compute `t_obs[j]` and `t_pred[j]`
- [ ] Report:
  - [ ] MAE/RMSE over detectors: `|t_pred - t_obs|`
  - [ ] Optional: infer wave speed from `t(x)` slope and compare

### 5.2 Secondary KPI: detector reconstruction error
- [ ] Report per-channel MSE (or robust loss):
  - [ ] `q_tot`, `v_tot` (and `rho_tot` if used)
- [ ] Report ablations:
  - [ ] no-graph vs graph
  - [ ] constant weights vs soft weights

### 5.3 Trajectory physical plausibility
- [ ] Collisions: count of negative spacing or below `s_min`
- [ ] Acceleration / jerk distribution sanity
- [ ] Optional: travel time / queue length consistency

---

## Phase 6 — Reproducible Experiment Protocol (Paper-ready)

### 6.1 Fix the protocol
- [ ] Fix:
  - [ ] road segment
  - [ ] time interval(s)
  - [ ] detector locations / aggregation definition
  - [ ] simulation dt and observation interval (5s)
  - [ ] optimization schedule (iters, lr, optimizer)
  - [ ] random seeds

### 6.2 Logging and outputs
- [ ] Save:
  - [ ] optimized `T`, `v0`
  - [ ] detector predictions `y_hat_tot`
  - [ ] trajectories `x_i(t), v_i(t)`
  - [ ] leader graph stats (optional)
- [ ] Plot:
  - [ ] cross-lane detector signals: obs vs pred
  - [ ] space–time diagram (speed or density)
  - [ ] wave arrival time plot / fitted wave speed
  - [ ] parameter distributions (T, v0) and their smoothness on the graph

### 6.3 Baselines & ablations to include
- [ ] Baseline 1: no regularization (`λ_T=0`)
- [ ] Baseline 2: weak L2 prior only (no graph)
- [ ] Proposed: graph Laplacian on `T` (and optionally `v0`)
- [ ] Ablation: constant vs soft graph weights
- [ ] Optional comparison: Fourier basis (as “failure mode” reference), if you want a narrative

---