# Revised Plan: Trajectory Reconstruction via Differentiable IDM

## Guiding Principle

We are doing **trajectory reconstruction**, not parameter identification.
T_Corr is a diagnostic, not a success criterion.
Every step below must be justified by its impact on **trajectory quality**.

---

## Pre-Decision: Lock Temporal Resolution

Before any experiment, decide once:

- Is 2s observation interval realistic for your target application?
- If yes → all subsequent work uses 2s as mainline.
- If no → mainline is 5s; 2s results serve as upper-bound reference only.

This affects every downstream choice. Do not defer.

---

## Step 0 — Look at the Trajectories (Day 1)

**What:** Take your current best run (2s det-only, T_Corr ≈ 0.44). Produce exactly two plots:

1. **Time–space speed heatmap** (position × time, color = speed): sim vs obs side by side.
2. **Simulated trajectories overlaid on speed field**: x_sim(i,t) drawn on top of the heatmap.

**Why:** You have never directly inspected the reconstructed trajectories. Every decision so far has been based on aggregate numbers (T_Corr, L_det). One heatmap will immediately reveal the dominant failure mode:

- Systematic speed bias across all space → global param (v0) is wrong → go to Step 1
- Wave exists but arrives at wrong time/place → ghost driver timing or boundary issue
- Wave shape roughly correct but internal structure too smooth → T distribution or aggregation issue
- Wave missing entirely → fundamental model/loss problem

**Do not proceed to Step 1 until you have looked at this plot and written down what you see.**

---

## Step 1 — Release Frozen Global Parameters (Days 2–3)

**Premise:** You currently freeze v0, a_max, s0, b and only learn per-vehicle T. Any misspecification in the frozen values is silently absorbed into T estimates, distorting both T and trajectories.

**What:**

- Make v0 a single global learnable scalar (shared across all vehicles), with a soft prior to a plausible range (e.g., N(30, 3) m/s or simple clamping).
- Optionally: also make a_max global learnable.
- Keep per-vehicle T_i as before.

This adds 1–2 degrees of freedom. Ill-posedness is negligibly affected.

**Ablation (at your locked resolution):**

| Config       | Learnable params                         |
| ------------ | ---------------------------------------- |
| A (baseline) | per-vehicle T only                       |
| B            | per-vehicle T + global v0                |
| C            | per-vehicle T + global v0 + global a_max |

**Evaluate using Step 0 plots**, not just T_Corr. Check:

- Does the speed field bias disappear?
- Does L_det improve?
- Does the learned v0 settle at a plausible value?
- Do trajectories look qualitatively better?

**Expected:** If v0 was misspecified, you will see immediate improvement in trajectory realism and possibly a shift in the T distribution (T_std may increase because T no longer compensates for v0 error).

---

## Step 2 — Build Minimal Trajectory Evaluation (Days 3–4)

After Step 0 shows you the failure modes and Step 1 addresses the most likely systematic error, build a lightweight eval script. Not a full suite — just enough to quantify what you see in the plots.

**Metrics (5 total, all cheap):**

1. **Collision rate / minimum gap** — hard constraint, must be zero or near-zero
2. **Acceleration statistics** — RMS and 99th percentile; flag if outside plausible range (e.g., |a| > 5 m/s²)
3. **Macro speed field error** — RMSE of sim vs obs speed heatmap (this is essentially your L_det but reported as a metric)
4. **Fundamental diagram plausibility** — scatter (density, flow) from sim; check it falls within a reasonable envelope
5. **Wave propagation speed** — estimated from cross-correlation of speed drops between adjacent detectors; compare sim vs obs

**Plots (4 total):**

- A: Time–space speed heatmap (sim vs obs) — you already have this from Step 0
- B: Simulated trajectories on heatmap — already have from Step 0
- C: Headway time series for 3–5 representative vehicles (safety check)
- D: Fundamental diagram scatter (sim vs obs)

Package as a single `eval_reconstruction.py` that takes a run directory and outputs everything.

---

## Step 3 — Reassess L_arr with Trajectory Metrics (Day 5)

**Context:** You showed L_arr helps T_Corr at 5s but not at 2s. But T_Corr is no longer your primary metric.

**What:** Re-evaluate your existing 4 runs (5s/2s × with/without L_arr) using the Step 2 evaluation suite. Ask a different question: does L_arr improve **trajectory quality** even if it doesn't improve T_Corr?

Possible outcomes:

- L_arr improves wave structure in trajectories → keep it, tune β with trajectory metrics as the target
- L_arr makes no trajectory-level difference → drop it, simplify the pipeline
- L_arr improves some aspects but worsens others → understand the tradeoff, then decide

**Do not do a β sweep unless Step 3 shows L_arr has trajectory-level impact worth optimizing.**

---

## Step 4 — Resolution Scaling Curve (Days 5–7, parallelizable)

**What:** Run reconstruction at dt_obs ∈ {1s, 2s, 3s, 5s, 7s, 10s} using the best configuration from Steps 1–3 (with global v0, with or without L_arr as decided).

**Evaluate each with the full Step 2 suite.**

**Output:** A single figure showing trajectory quality metrics as a function of dt_obs. This is the centerpiece of your paper's empirical contribution: a quantitative characterization of how observation resolution controls reconstruction quality under realistic (cross-lane aggregated, short window) constraints.

**What to look for:**

- A clear transition region where quality degrades rapidly (your earlier data suggests 2–3s)
- Whether the transition is sharp (phase-transition-like) or gradual
- Which metrics degrade first as resolution coarsens

---

## What This Plan Does NOT Include (and why)

| Dropped item                   | Reason                                                       |
| ------------------------------ | ------------------------------------------------------------ |
| Laplacian aggregation rewrite  | Not justified until you prove KDE smoothing is the bottleneck (you haven't) |
| Graph wave energy loss         | Likely redundant with L_det; no evidence of information increment |
| Damping-based supervision      | Interesting but premature; fix the basics first              |
| Self-excited wave experiments  | Valuable research but a different paper                      |
| Full 3-axis ablation (27 runs) | Too expensive; sequential elimination is more efficient      |
| Per-lane detector loss         | Incompatible with your realistic cross-lane aggregated setup |

---

## Decision Points

After Step 0:

> If trajectories look reasonable → proceed as planned.
> If trajectories are catastrophically wrong → diagnose (ghost driver? initial conditions? model?) before continuing.

After Step 1:

> If global v0 helps significantly → keep it; consider whether other globals are worth adding.
> If no change → frozen params were already reasonable; the problem is elsewhere.

After Step 3:

> If L_arr has trajectory-level value → include in mainline config.
> If not → drop it permanently; your pipeline becomes simpler.

After Step 4:

> You have a complete empirical story: "differentiable IDM reconstruction under realistic macro observations, with resolution as the key design variable."
> Decide whether this is sufficient for a paper or whether you need mechanism extensions (damping, endogenous waves, ramp merging).