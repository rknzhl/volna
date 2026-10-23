# Curved torus with noise: corrected pipeline

The clean case and all three paired noise levels pass the public sensor pipeline and the additional flow checks. Phase dimension is estimated from readings; no true phases or noise levels enter fitting.

## What changed

1. **Window span.** Keep the strongest singular subspace only at a clear gap. Preserve a resolved local-dimension hint and the 5% reconstruction budget. The inverse error includes discarded coordinates.
2. **Tangent geometry.** Test exact quadratic relations first; if necessary, choose a near-relation tolerance on separate TRAIN rows. Retain only a spectrally separated tangent estimate. The local cubic fallback remains available.
3. **Topology resolution.** Refine unresolved coverage through 2048 landmarks. At the last resolution, restrict the filtration using the previous TRAIN interval. Surviving bars retain censored death bounds. All integer-closure, primitive-basis, coverage and accuracy checks remain.

The sequence remains windows → tangent overlap transport → coherent fields → integer cycle periods → phases → inverse field flows. No independent circular-coordinate fit supplies the phases.

## Paired experiment

Four sensors observe a three-phase torus. The same orbit, seed 31 and Gaussian realization are used at all noise levels. Noise standard deviation is the stated fraction of each clean TRAIN channel deviation. There are 2400 TRAIN and 320 query samples; after fixed boundary trimming, 130 query snapshots remain. Selection uses TRAIN only; query predictions are frozen before true phases are evaluated.

| Noise | Estimated phases | Window length | Phase MAE, rad | Sensor RMSE / signal std | Result |
|---:|---:|---:|---:|---:|---|
| 0% | 3 | 8 | 0.00439 | 0.46% | Passed |
| 1% | 3 | 4 | 0.00933 | 0.65% | Passed |
| 5% | 3 | 4 | 0.01091 | 1.84% | Passed |
| 10% | 3 | 8 | 0.01393 | 2.29% | Passed |

Sensor error is measured against clean sensor values, normalized by clean TRAIN channel deviations. Reconstruction conditions on an observed query block; it is not a forecast. All matched integer phase bases have absolute determinant one.

| Noise | Retained window coordinates | Relation tolerance | Topology landmarks | Fit + query, s |
|---:|---:|---:|---:|---:|
| 0% | 12 | 1e-10 | 1536 | 116.3 |
| 1% | 12 | 0.0008 | 2048 | 26.6 |
| 5% | 12 | 0.0032 | 2048 | 27.3 |
| 10% | 12 | 0.0064 | 2048 | 105.0 |

The unchanged rule selects the first candidate passing TRAIN checks. With the final coverage refinement, 1% and 5% select length 4. Their earlier length-8 controls remain in `noise01_limit1536.json` and `noise05_limit1536.json`; these are not the final selected runs.

## Independent flow checks

Each frozen model is tested from eight TRAIN windows, with integration steps 0.12 and 0.06. Errors use the full prepared-window RMS; inverse errors include discarded SVD directions.

| Noise | Maximum full-turn defect | Inverse RMS error | Flow-order error | Maximum step-refinement change |
|---:|---:|---:|---:|---:|
| 0% | 2.21% | 0.48% | 0.75% | 6.9e-06 |
| 1% | 1.58% | 0.77% | 0.67% | 8.5e-06 |
| 5% | 4.05% | 2.73% | 1.79% | 8.5e-06 |
| 10% | 4.89% | 3.24% | 1.61% | 6.9e-06 |

All checks pass the original thresholds. At 10% noise the maximum return defect is 4.89% against a 5% limit, so its margin is small. This is one paired synthetic realization, not an estimated success rate across sensors, embeddings or seeds.

## Regression and records

- 65 small implementation checks pass.
- A nonlinear scalar sensor with 10% noise and a linear trend passes: phase MAE 0.01157 rad; sensor RMSE 2.50%.
- Two waves with 5% noise and a quadratic trend pass: phase MAE 0.00431 rad; sensor RMSE 0.70%.
- Numerical jobs ran sequentially with one numerical thread. The final curved runs took 27–116 seconds each.

[Final comparison](final_comparison.json) · [Flow checks](flow_verification.json) · [Regressions](regression/summary.json) · [Original failure](../fields_first_curved_noise_2026_10_22/README.md)

The unbounded 2048-landmark trial was stopped for runtime and is not a failed accuracy test. The alternative prime-2 probe and full-diagram probe were stopped at 60 seconds. They did not change the final coefficient 47. A bounded-range probe completed in 3.94 seconds; the final range is selected from TRAIN, rather than fixed to that probe value.

## Reproduce

Run sequentially from the repository root:

```bash
.venv/bin/python results/fields_first_curved_noise_fix_2026_10_22/run_noise.py --noise 0
.venv/bin/python results/fields_first_curved_noise_fix_2026_10_22/run_noise.py --noise .01
.venv/bin/python results/fields_first_curved_noise_fix_2026_10_22/run_noise.py --noise .05
.venv/bin/python results/fields_first_curved_noise_fix_2026_10_22/run_noise.py --noise .10
.venv/bin/python results/fields_first_curved_noise_fix_2026_10_22/verify_flows.py
```

The flow-check script reads frozen candidate models saved in task scratch storage by the runner. Individual JSON records retain diagnostics and source hashes.
