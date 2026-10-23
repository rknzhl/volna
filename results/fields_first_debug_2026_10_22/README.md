# Fields-first validation and baselines

## Flow correction

The noisy one-sensor example failed because the estimated ambient flow drifted away from the fitted window surface. Reducing the RK4 step left an 8.02% full-turn defect. Normal correction using the same frozen local graphs reduced it to 0.218%, checked from eight starting windows. No target endpoint, true phases, or regression decoder is used; the original acceptance thresholds are retained.

This correction defines a geometry-constrained numerical inverse. It does not establish exact closure of the original ambient ODE. [Diagnosis and fixed-candidate checks](loop_diagnosis.json).

## Current validation

The same pipeline selects its lag, neighbourhood and phase count from training measurements. Predictions are frozen before the synthetic phases are used for evaluation. Each query block contains 320 snapshots; 95 snapshots at each boundary are excluded.

| Scenario | Inferred phases | Noise std / clean training std | Query reading NRMSE, % | Phase MAE, rad |
|---|---:|---:|---:|---:|
| One nonlinear sensor | 1 | 0 | 1.150 | 0.00426 |
| One nonlinear sensor, linear trend | 1 | 0.10 | 2.548 | 0.01210 |
| Two waves, quadratic trend | 2 | 0.05 | 0.887 | 0.00498 |
| Nonlinear sensors, quadratic trend | 2 | 0.05 | 2.705 | 0.01405 |
| Layered optical signal, linear trend | 2 | 0.05 | 1.041 | 0.00536 |
| Three phases, six sensors | 3 | 0 | 1.302 | 0.00387 |

[Individual results](validation/) include the training checks and held-out checks. Acceptance is conditional on these finite diagnostics, not a guarantee for arbitrary sensor data.

At this validation snapshot the curved three-phase case had not passed. The original persistence-gap rule selected two phases even with 768 landmarks: [historical probe](curved_three_phase_resolution_probe.json). A [subsequent dimension correction](../fields_first_curved_r_fix_2026_10_22/README.md) selected three phases but left a field/coordinate consistency failure. The [later geometry correction](../fields_first_curved_fields_fix_2026_10_22/README.md) passes the complete noiseless pipeline. The [noise follow-up](../fields_first_curved_noise_2026_10_22/README.md) fails the successful clean configuration at 1%, 5% and 10%; its search limits are explicit. The earlier chirped example violates the constant phase-increment model and remains unresolved.

### Diagnosis of the two unresolved cases

**Curved torus:** the noiseless four-sensor map is an embedding of a three-dimensional torus. Its explicit inverse recovers all three source phases to a maximum wrapped error of 2.3e-13 radians. MSSA retains the full numerical signal rank, and the filtered snapshot remains an embedding: [analytic certificate](remaining_cases_audit/curved_filter_analytics.json). The original dimension rule failed on these observations. The corrected rule cross-checks local geometry against persistence and resolves three classes with 1024 landmarks: [dimension-stage audit](../fields_first_curved_r_fix_2026_10_22/README.md). The later complete noiseless result is linked above.

**Changing frequency:** even exact generator phases have training increment RMS 0.05252 radians against the best fixed increment, above the 0.05 threshold. On the query block the error against the frozen training increment is 0.19562 radians. A delayed window depends on phase and current frequency; its natural state is a cylinder, not the assumed phase-only torus. This case requires a nonstationary model extension. Raising the threshold would not restore the missing phase-only inverse.

No arithmetic or indexing defect explaining either failure was found. The diagnostic used one numerical thread, two small persistence probes and no field fits or inverse flows. [Reproducible probes and exact diagnostics](remaining_cases_audit/diagnostics.json) · [Probe script](remaining_cases_audit/diagnose_remaining.py).

## Baselines

[Comparison table](baselines/COMPARISON.md): SSA, MSSA, circular coordinates and toroidal coordinates. Eleven saved successful runs cover six scenarios, including repeated scenarios with different query lengths. This evaluates accepted runs; it is not a failure-rate benchmark.

All methods receive identical raw measurements and train/query splits and are scored at the same timestamps. Fitters receive neither true phases nor their number. CC/TC share our training-selected delay length. [Input equivalence evidence](generator_equivalence.json) verifies that the historical generators produce bitwise identical arrays.

MSSA reconstructs the multichannel readings more accurately here. Our phases are closer to the uniform generator phases than native CC/TC. CC/TC recover the correct phase count and primitive topology; their coordinates need not have uniform angular speed.

## Stronger phase controls

The additional controls fit a periodic empirical CDF to each native coordinate using training samples only. Query coordinates use this frozen calibration. Raw-input hashes, splits, lags and evaluation indices match the original comparison. No fields-first model was refitted.

On the scalar cases, calibration removes the original advantage: query phase MAE is 0.004193 versus ours 0.004265 for clean data, and 0.009829 versus ours 0.012097 at 10% noise. The small clean difference does not establish statistical superiority. [Scalar control, code and frozen coordinates](phase_advantage_audit/circular_cdf_diagnostic.json).

For the four multivariate cases, the advantage survives both native CC/TC and their calibrated variants. The table uses the common generator phase basis and the best of the four controls. Errors are in radians.

| Scenario | Our phase MAE | Best control phase MAE | Minimum advantage factor |
|---|---:|---:|---:|
| Nonlinear sensors | 0.01405 | 0.17577 | 12.5 |
| Layered optical signal | at most 0.00722 | 0.06426 | 8.9 |
| Two waves | 0.00498 | 0.08566 | 17.2 |
| Three phases, six sensors | at most 0.00488 | 0.09380 | 19.2 |

The two upper bounds account for integer mixing of the recovered phases; even these bounds are below every control error. They are not estimates of the exact transformed errors. [Results, reproducible code and frozen phase arrays](phase_advantage_audit/multifactor_marginal_cdf_diagnostic.json).

This establishes a phase-accuracy advantage on these four accepted configurations. Coordinatewise CDF is not a complete multidimensional dynamical uniformization. A comparison with SSA/MSSA phase extraction or other dynamical-phase estimators is still missing. There is one seed per configuration; repeated query lengths are not independent replications. [Audit of all 11 original comparisons](phase_advantage_audit/saved_comparison_audit.json).

## Reproduce

Run from the repository root. Numerical libraries use one thread; cases run sequentially.

```bash
.venv/bin/python code/lag_ssa/validate_fields_first.py \
  --output-dir results/fields_first_debug_2026_10_22/validation

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 .venv/bin/python \
  code/lag_ssa/compare_fields_first_baselines.py \
  results/fields_first_e2e_2026_10_20 \
  results/fields_first_debug_2026_10_22 \
  results/fields_first_debug_2026_10_22/validation \
  --output-dir results/fields_first_debug_2026_10_22/baselines \
  --generator-evidence results/fields_first_debug_2026_10_22/generator_equivalence.json \
  --resume
```

The flow, geometry and topology checks in `check_fields_first_safety.py` and `check_fields_first_flow_geometry.py` pass (15 tests). Polynomial evaluation and unused-edge elimination were checked against their original implementations without changing the formulas.
