# utils/ - Utilities and Diagnostics

## TrainingDiagnostics (`diagnostics.py`)

Per-iteration snapshot containing:
- Loss components: `L_total`, `L_det`, `L_graph`
- Gradient norms: `||∇T||`, `||∇v0||`
- Parameter stats: mean, std, min, max of T and v0
- Smoothness: `mean |T_i - T_leader(i)|`

## DiagnosticsTracker (`diagnostics.py`)

Maintains history and provides:
- `compute()`: Create diagnostics for current iteration
- `check_issues()`: Detect anomalies (gradient explosion, unrealistic parameters)
- `get_summary()`: Final statistics

## Issue Detection

Warnings triggered for:
- `T_min < 0.3s` or `T_max > 5.0s` (unrealistic)
- `L_graph ~ 0` when regularization enabled
- `||∇T|| > 100` (gradient explosion)
