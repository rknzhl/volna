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
.venv/bin/python experiments/verify_baseline_compatibility.py
```

These commands do not train models, calculate persistence or integrate flows. Figure data are preserved locally; the trusted scratch model is needed only to export the TRAIN-phase arrays again.

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
