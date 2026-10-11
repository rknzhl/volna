"""Blind phase-factor assay of the cached measured river-level field.

Inference receives eta, station distance, and times only. Constituent metadata
is not loaded by this script. Unknown residual weather/flow is not represented
as a certified finite torus. Model filters and forecasts are frozen on train.
"""

from project_paths import RESULTS_DIR

from dataclasses import asdict
import ast
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from iterative_ssa import Settings, fit_iterative_ssa, predict_field
from nodal_spatial import NodalSpatialModel
from run_actual_wave_iterative import noise_from_training_high_band, serializable
from run_actual_wave_mssa_baseline import fit_joint_baseline
from ssa_character import apply_ssa_character


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / "data/real/tides.npz"
OUT = RESULTS_DIR


def nrmse(prediction, reference):
    return float(np.linalg.norm(prediction-reference) / np.linalg.norm(reference))


def horizontal_mssa_equal_window(y, times, window):
    """Notebook recurrence with L=window; full-record L=Ntrain/2 is not used."""
    source = (ROOT / "code/waves/synthetic/models.py").read_text()
    names = {"hankel", "subspace", "recurrence", "smooth", "extend"}
    nodes = [node for node in ast.parse(source).body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {"np": np}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),"models.py","exec"),namespace)
    n_train = int(.6*y.shape[1])
    n_validation = int(.2*y.shape[1])
    matrices = [namespace["hankel"](row,window) for row in y[:,:n_train]]
    subspace = namespace["subspace"](np.hstack(matrices))
    recurrence = namespace["recurrence"](subspace)
    prediction = np.array([namespace["extend"](namespace["smooth"](h,subspace),
        recurrence,y.shape[1]-n_train) for h in matrices])[:,n_validation:]
    return {"future_prediction":prediction,
        "future_observed_nrmse":nrmse(prediction,y[:,n_train+n_validation:]),
        "selected_ssa_rank":subspace.shape[1],"lag_window":window,
        "method":"notebook horizontal MSSA recurrence, equal lag window"}


def main():
    started = time.perf_counter()
    with np.load(SOURCE, allow_pickle=False) as record:
        # No access to harcon, constituents, station IDs, or catalogue phases.
        y = record["eta"].astype(float)
        s = record["distance_km"][:, None].astype(float)
        t = record["t"].astype(float)
    if not np.isfinite(y).all() or not np.allclose(np.diff(t), np.diff(t).mean()):
        raise ValueError("this assay requires a complete uniform measured record")
    n_train, n_validation = int(.6*len(t)), int(.2*len(t))
    future_start = n_train+n_validation
    settings = Settings(n_train=n_train, n_validation=n_validation,
        n_future=len(t)-future_start, lag_window=128, max_steps=6,
        k_candidates=(16, 32), frontend_mode="dynamic",
        dynamic_ranks=(4, 8, 16), phase_metric_mode="field",
        temporal_gate_mode="iid_reference", joint_frequency_mode="harmonic_nuisance",
        conditional_model_mode="fourier_sieve", enforce_validation_improvement=True)
    spatial = NodalSpatialModel.from_positions(s)
    fitted = fit_iterative_ssa(y[:, :n_train], y[:, n_train:future_start],
        settings, sampling_model=spatial,
        noise_estimate=noise_from_training_high_band(y[:, :n_train]))
    model = fitted["clock_models"][-1]
    future_indices = np.arange(future_start, len(t))
    prediction = predict_field(model, future_indices, s, sampling_model=spatial)
    future_mass = spatial.coordinates(y[:, future_start:])
    train_phases, future_phases, readouts = [], [], []
    frequencies = np.asarray(fitted["acquired_frequencies"])
    for j, frontend in enumerate(fitted["frozen_frontends"]):
        residual = future_mass-fitted["clock_models"][j].predict(future_indices)
        applied = apply_ssa_character(frontend, residual)
        z = applied["normalized_character"]
        train_phase = np.asarray(frontend["phase"])
        origin = np.angle(np.mean(np.exp(1j*(train_phase -
            frequencies[j]*np.arange(len(train_phase))))))
        phase = np.angle(np.exp(1j*(applied["phase"]-origin)))
        train_phases.append(np.angle(np.exp(1j*(train_phase-origin))))
        future_phases.append(phase)
        readouts.append({"character": j, "frozen_filter": True,
            "future_phase_step_coherence": float(abs(np.mean(z[1:]*np.conj(z[:-1])))),
            "future_radius_cv": float(np.std(applied["radii"])/np.mean(applied["radii"])),
            "future_phase_mae_to_inferred_clock_rad": float(np.mean(abs(
                np.angle(np.exp(1j*(phase-frequencies[j]*future_indices[:len(phase)]))))))})
    # Both baselines have data-only model selection and untouched future.
    joint = fit_joint_baseline(y, t)
    classical = horizontal_mssa_equal_window(y, t, settings.lag_window)
    arrays = OUT / "real_phase_factors_coordinates.npz"
    np.savez_compressed(arrays, train_phase=np.asarray(train_phases),
        future_phase=np.asarray(future_phases), future_t=t[future_start:],
        frequencies_rad_per_sample=frequencies,
        future_prediction=prediction, future_observed=y[:, future_start:],
        baseline_future_prediction=joint["future_prediction"],
        original_mssa_future_prediction=classical["future_prediction"])
    payload = {"source": str(SOURCE),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "data": "cached measured NOAA river-level field",
        "fit_inputs": ["eta", "distance_km", "t"],
        "catalogue_metadata_used": False, "settings": asdict(settings),
        "record_shape": list(y.shape), "split": [n_train,n_validation,len(t)-future_start],
        "inference": fitted["audit"], "frozen_readouts": readouts,
        "frequencies_cycles_per_hour": (frequencies/(2*np.pi*np.diff(t).mean())).tolist(),
        "future_observed_nrmse": nrmse(prediction,y[:,future_start:]),
        "joint_baseline_future_observed_nrmse": joint["future_observed_nrmse"],
        "horizontal_mssa_equal_window_future_observed_nrmse": classical["future_observed_nrmse"],
        "horizontal_mssa_equal_window_rank":classical["selected_ssa_rank"],
        "selected_joint_baseline_lines": joint["selected_oscillation_line_count"],
        "coordinates": str(arrays),
        "interpretation": "learned periodic factor; residual is not certified noise or a complete torus",
        "elapsed_seconds": time.perf_counter()-started}
    target = OUT / "real_phase_factors.json"
    target.write_text(json.dumps(serializable(payload),ensure_ascii=False,indent=2,
                                allow_nan=False)+"\n")
    print(json.dumps({k:payload[k] for k in ("future_observed_nrmse",
        "joint_baseline_future_observed_nrmse","horizontal_mssa_equal_window_future_observed_nrmse",
        "frequencies_cycles_per_hour","elapsed_seconds")},indent=2),flush=True)


if __name__ == "__main__":
    main()
