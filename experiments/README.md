# Experiments

This is the single experiment directory for the current article draft.

| File or directory | Contents |
|---|---|
| `../paper/main.tex` | The complete article, including experiment text and tables |
| `data/` | Archived run reports, baseline controls and actual numerical arrays |
| `figures/` | Matplotlib vector PDFs |
| `results.csv`, `all_attempts.csv` | Successful examples and the complete run summary |
| `build_tables.py` | Updates the two tables inside the single article TeX file |
| `plot_experiments.py` | Noise figure and supplementary TRAIN-phase plots |
| `plot_sensors.py` | Sensor layouts for the two successful toys |
| `plot_dimension_selection.py` | Local SVD gaps and cycle lifetimes for the saved one-, two- and three-phase examples |
| `verify_baseline_compatibility.py` | Input, time-index and version checks; no fitting |
| `source_manifest.json`, `experiments_provenance.json` | Source paths and hashes |

## Two reported metrics

- **Phase MAE:** mean absolute wrapped phase difference, in radians. The phase basis and origin are aligned on TRAIN only, after freezing the predictor.
- **Signal NRMSE:** RMS signal error divided channel by channel by the clean oscillatory TRAIN standard deviation. The target includes any added trend. Tables show percentages.

The six archived sensor runs have 320 test samples, of which 130 remain after boundary trimming. Phase errors use complete window starts. Reconstruction uses an observed test block.

## Rebuild figures and tables

From the repository root:

```bash
.venv/bin/python experiments/build_tables.py
.venv/bin/python experiments/plot_experiments.py --saved-arrays
.venv/bin/python experiments/plot_sensors.py
.venv/bin/python experiments/plot_dimension_selection.py
.venv/bin/python experiments/verify_baseline_compatibility.py
.venv/bin/python experiments/verify_giannakis_comparison.py
```

These commands do not train models, calculate persistence or integrate flows. Figure data are preserved locally; the trusted scratch model is needed only to export the TRAIN-phase arrays again.

## Giannakis Koopman comparison

`../code/lag_ssa/giannakis_koopman.py` implements the variable-bandwidth
diffusion basis and diffusion-regularized Galerkin generator of
[Giannakis (2019)](https://arxiv.org/html/1507.02338v4), Sections 4.2--4.3
and 5, Algorithms 1--2. It also provides weighted spectral readouts,
harmonic forecasts and the optional product-basis vector-field readout.
This comparison scores reconstruction of an observed query block.

```bash
.venv/bin/python experiments/run_giannakis_comparison.py --resume
.venv/bin/python code/lag_ssa/check_giannakis_koopman.py --output experiments/data/giannakis/numerical_checks.json
.venv/bin/python experiments/verify_giannakis_comparison.py
.venv/bin/python experiments/build_tables.py
```

All eight reported article cases use their original raw inputs, TRAIN/query
split, common MSSA/highpass preparation, selected window lag and scored
indices. Only TRAIN observations enter the estimator and parameter selection.
Kernel bandwidths use Algorithm 1's scale-function rule; a fixed sample of
32768 off-diagonal pairs approximates this scan on large datasets. The
final kernel is dense and uses every training window. The candidate basis
sizes are 32, 64 and 128; diffusion regularizations are 1e-4, 1e-3 and 1e-2.
A contiguous 75/25 split within TRAIN, with a window-overlap gap, selects
parameters by equally weighted reconstruction and one-step spectral
prediction MSE. The shared preprocessing is fitted on the original TRAIN.

The numerical implementation preserves the constant, uses the exact
reversible normalization from Equation (24), and skew-symmetrizes finite
advection. A frozen Nystrom map encodes query windows. Eigenfunctions are
ordered by normalized Dirichlet energy. Finite phase selection uses the
rounded kernel dimension estimate as a cap (1--4), bounded integer
coefficients (4), modulus-CV and derivative-residual gates (.35 and .5),
and training phase-product coherence (.9). Frequencies for the relation
test are refined from training phase increments using Equation (5), avoiding
the central difference's sine dispersion; raw generator eigenvalues remain
saved. These are explicit finite-sample adaptations, not a verbatim
reproduction of the paper's conceptual phase-selection step.

Signal readout uses a weighted Gram least-squares solve in the full
regularized eigenfunction basis, rather than requiring the selected phase
basis to be complete. It reconstructs highpass windows, averages overlaps
and restores the same query slow component used by the other methods.
This explains why a signal score remains available when phase recovery
is unresolved. The finite primitive-gauge check is not a global certificate.

`data/giannakis/{case}.json` records every candidate, selected parameters,
phase-selection decisions, source/package/input hashes and scored indices.
`{case}.npz` freezes predictions before reference phases or clean signals
enter scoring. `data/giannakis/summary.json` and `giannakis_results.csv` preserve all
eight results. The independent verifier checks original input equivalence,
frozen predictions, parameter selection and recomputed metrics without
fitting an estimator. `numerical_checks.json` contains separate analytic
circle/torus, harmonic-dependence and frozen-extension controls.

With this finite configuration, complete primitive phase matches are
obtained in 3/8 runs. The nonlinear scalar run selects two dependent
characters; the four curved runs produce singular fitted integer gauges.
All five failures remain in the comparison with null phase scores.
The signal reconstruction score is retained for every run. These results
describe this documented configuration, not the full capability of
Koopman methods.

## Successful toys

The sine toy has one inferred phase; phase MAE is 0.000204 rad and signal NRMSE is 0.496%. The clean two-wave toy has two inferred phases; phase MAE is 0.001342 rad and signal NRMSE is 0.110%. Both use the unchanged pipeline.

`figures/toy_walkthrough.pdf` shows actual windows, recovered phases, triangles, estimated fields, a computed path and test reconstruction in one six-panel figure. Regenerate it with `.venv/bin/python experiments/plot_walkthrough.py`.

`figures/toy_sensors.pdf` shows the scalar channel and the twelve sensor positions used by the two-wave generator. The scalar toy has no specified spatial placement.

## What the records establish

- The two successful toys and six final sensor runs match the ten recorded source hashes of the current implementation.
- Saved SSA/MSSA and CC/TC controls match the raw data, split, preprocessing and scored times of the current scalar and two-wave runs. Their recorded dependency versions are preserved. The comparison does not claim that those controls were newly fitted.
- The curved noise study shares one orbit and one scaled Gaussian realization. It is not a success-rate estimate. Its MSSA curve is the shared preprocessing output.
- `figures/experiments_train_phases` is a supplementary TRAIN visualization. Noise errors are evaluated on the test block.
- The initial requested signal `sin(t) + 0.1 sin(0.1t)` has one independent phase. The unchanged procedure left its dimension unresolved in the two-attempt test. Its report and input arrays are retained as `data/toy_signal*`; it is not presented as successful phase recovery.

Exact configurations and the remaining finite diagnostics are in the individual JSON reports. Historical source paths in the manifests identify where each archived record originated.
