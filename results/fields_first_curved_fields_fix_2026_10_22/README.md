# Curved torus: fields, phases and inverse reconstruction

The complete pipeline passes on the noiseless curved three-phase, four-sensor example.
Only sensor readings enter the fit. Generator phases and derivatives are used
after fitting for evaluation or in separate diagnostic probes.

## Cause and correction

The local tangent atlas was inaccurate on this curved embedding. Its projected
fields could satisfy the overlap equations approximately while their flows
failed to close. Improving phase interpolation alone did not fix this:
[the local-geometry control](validation_local_geometry/three_phases.json)
still had a full-turn error of 20.9%.

1. Fit quadratic window relations when they hold on separate TRAIN fitting and
   validation rows. Their common gradient kernel gives the tangent plane. If
   these checks fail, retain local polynomial geometry. No circles or phases
   are extracted here.
2. Add a shared affine part to the existing local field dictionary, within a
   fixed size budget. Fit it by the same anchored overlap equations.
3. Choose a short integer period basis; preserve the fitted fields and period
   lattice. Make the phase interpolant reproduce its fitted values and first
   derivatives at landmarks.
4. Select the topology scale and landmark resolution from TRAIN, requiring
   exactly the estimated number of classes and a coverage buffer. Bare TRAIN
   coverage had missed one query window in the [earlier run](validation_cover1024/three_phases.json).

The order remains geometry, overlap transport, fields, periods, phases, flows.
Accuracy thresholds were unchanged.

## Measurements

Training: 2,400 snapshots. Held-out block: 320 snapshots, with 130 interior
snapshots and 123 windows evaluated after the fixed boundary trim. Automatic
selection: phase dimension 3, lag 8, reduced dimension 13, 1,536 landmarks.

| Check | Result |
|---|---:|
| TRAIN / held-out phase-field derivative RMS error | 0.0273 / 0.0333 |
| Held-out phase MAE | 0.00420 rad |
| Held-out sensor RMSE / TRAIN signal standard deviation | 0.00436 |
| Held-out prepared-window reconstruction relative error | 0.00769 |
| Maximum full-turn error, eight starting windows | 0.0217 |
| Full-turn change on halving the ODE step | 0.00000686 |
| Noisy scalar regression: sensor RMSE / signal standard deviation | 0.0247 |

The phase match is a permutation with determinant magnitude one.
[Full pipeline result](validation/three_phases.json),
[eight-start flow check](flow_verification.json),
[noisy regression](regression/scalar_noise10.json).

The main run took 125 seconds with one numerical thread. All 40 small
regression tests passed. Final source edits only corrected diagnostic labels
and added an early refusal of an unsupported implicit-geometry/local-graph
retry. Their accepted numerical path is unchanged; the refusal has its own
regression test. [Source hashes and audit](final_source_audit.json).

These are finite-data checks on this example. Exact quadratic geometry is
accepted only when validated; this result does not establish reliable recovery
for every curved embedding or arbitrary future coverage.
The [subsequent noise check](../fields_first_curved_noise_2026_10_22/README.md)
does not pass the successful clean configuration at 1%, 5% or 10% noise.

## Reproduce

Run sequentially from the repository root:

```bash
.venv/bin/python results/fields_first_curved_fields_fix_2026_10_22/validate_curved.py
.venv/bin/python results/fields_first_curved_fields_fix_2026_10_22/verify_curved_flows.py
.venv/bin/python code/lag_ssa/validate_fields_first.py --only scalar_noise10 --max-candidates 6 --output-dir results/fields_first_curved_fields_fix_2026_10_22/regression
```

Each script limits numerical libraries to one thread. Diagnostic probes in
this directory are separate experiments; they are not extra pipeline stages.
