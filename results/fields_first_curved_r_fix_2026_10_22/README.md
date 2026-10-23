# Curved torus: automatic phase count

**The dimension estimate is corrected; the full reconstruction still fails its training checks.**

## Change

The previous rule selected the largest persistence gap and missed the shorter third class. Local PCA now proposes the phase count; persistence checks that this many classes coexist at a scale covering the training windows. Unresolved cases use 192, 384, 768, then at most 1024 landmarks.

The curved four-sensor example selects $r=3$ automatically. No generator phases, frequencies or phase count enter the fit. Fields, transport, integration, flows and accuracy thresholds are unchanged. The [paper](../../research/phase_ssa_method_v2.pdf) uses the same rule.

## Pipeline check

The pipeline driver completed nine candidates: window lengths 4, 8 and 16, each with 24, 48 and 96 neighbours. Every candidate selected $r=3$ with 1024 landmarks. None passed all training checks.

The best candidate used window length 4 and 48 neighbours:

| Check | Measured | Required |
|---|---:|---:|
| Relative correction needed to make integrated motion coefficients consistent around triangles | 0.0868 | below 0.05 |
| Normalized error in $D\Phi E=I_r$ | 0.4702 | below 0.15 |
| RMS phase-increment error, radians | 0.0293 | below 0.05 |

Thus the remaining failure concerns agreement between the fields and phase coordinates. This run does not isolate whether field estimation or numerical integration causes it. It stopped before inverse reconstruction; no query reconstruction accuracy is claimed. Window length 32 was outside the nine-candidate budget.

[Complete result](three_phases.json) · [Search summary](search_audit.json)

## Regression checks

At the previously accepted window lengths, all six earlier scenarios retained both their phase count and landmark resolution: $1,1,2,2,2,3$. Training inputs match the earlier runs byte for byte. This check repeated dimension selection only; it did not refit the six field models. [Results and input hashes](dimension_regression.json) · [Script](check_dimension_regression.py)

All 21 dimension, geometry and safety tests pass. Computations ran sequentially with one numerical thread. After the pipeline run, only docstrings and rejection logging were edited.

## Reproduce

From the repository root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 .venv/bin/python \
  code/lag_ssa/validate_fields_first.py \
  --output-dir results/fields_first_curved_r_fix_2026_10_22 \
  --only three_phases --max-candidates 9

.venv/bin/python \
  results/fields_first_curved_r_fix_2026_10_22/check_dimension_regression.py
```
