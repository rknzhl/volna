"""Final fixed phase-factor pipeline on measured NOAA river levels.

Reads eta, distance_km and t only; no constituent catalogue or known dynamics.
The residual need not belong to a finite-dimensional quasiperiodic model.
"""

from project_paths import RESULTS_DIR

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from final_phase_factor_assay import fixed_settings, fingerprints
from iterative_ssa import Settings, fit_iterative_ssa, predict_field
from nodal_spatial import NodalSpatialModel
from run_actual_wave_iterative import noise_from_training_high_band, serializable
from run_actual_wave_mssa_baseline import fit_joint_baseline
from run_actual_wave_revised import recurrent_mssa
from ssa_character import apply_field_character, aligned_character_rows


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[1] / "data/real/tides.npz"
OUT = RESULTS_DIR
TARGET = OUT / "real_phase_factors_final.json"


def nrmse(prediction, reference):
    return float(np.linalg.norm(prediction-reference)/np.linalg.norm(reference))


def main():
    if TARGET.exists():
        raise FileExistsError("preserve the existing final assay")
    started = time.perf_counter()
    hashes = fingerprints()
    hashes[Path(__file__).name] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with np.load(SOURCE, allow_pickle=False) as archive:
        y = archive["eta"].astype(float)
        s = archive["distance_km"].astype(float)[:, None]
        t = archive["t"].astype(float)
    if not np.isfinite(y).all() or not np.allclose(np.diff(t),np.diff(t).mean()):
        raise ValueError("complete uniformly sampled measured field required")
    n_train, n_validation = int(.6*len(t)), int(.2*len(t))
    start = n_train+n_validation
    settings = fixed_settings(Settings(n_train=n_train,n_validation=n_validation,
        n_future=len(t)-start,lag_window=128,k_candidates=(16,32),
        frontend_mode="dynamic",dynamic_ranks=(4,8,16),phase_metric_mode="field",
        temporal_gate_mode="iid_reference",joint_frequency_mode="harmonic_nuisance"))
    spatial = NodalSpatialModel.from_positions(s)
    noise = noise_from_training_high_band(y[:,:n_train])
    fitted = fit_iterative_ssa(y[:,:n_train],y[:,n_train:start],settings,
        sampling_model=spatial,noise_estimate=noise)
    future_indices = np.arange(start,len(t))
    prediction = predict_field(fitted["clock_models"][-1],future_indices,s,
        sampling_model=spatial)
    future_mass = spatial.coordinates(y[:,start:])
    frequencies = np.asarray(fitted["acquired_frequencies"])
    train_phases, future_phases, readouts = [], [], []
    for j, frontend in enumerate(fitted["frozen_frontends"]):
        applied = apply_field_character(frontend,future_mass,
            fitted["clock_models"][j].predict(future_indices))
        train_phase = np.asarray(frontend["phase"])
        # A training-only origin; never calibrate on future observations.
        origin = np.angle(np.mean(np.exp(1j*(train_phase-
            frequencies[j]*np.arange(len(train_phase))))))
        phase = np.angle(np.exp(1j*(applied["phase"]-origin)))
        train_phases.append(np.angle(np.exp(1j*(train_phase-origin))))
        future_phases.append(phase)
        z = applied["normalized_character"]
        readouts.append({"character":j,"frozen_filter":True,
            "window":frontend["window"],
            "readout_source":frontend.get("readout_source","residual"),
            "future_phase_step_coherence":float(abs(np.mean(z[1:]*np.conj(z[:-1])))),
            "future_radius_cv":float(np.std(applied["radii"])/np.mean(applied["radii"])),
            "future_phase_mae_to_inferred_clock_rad":float(np.mean(abs(np.angle(
                np.exp(1j*(phase-frequencies[j]*future_indices[:len(phase)]))))))})
    train_phases = aligned_character_rows(train_phases)
    future_phases = aligned_character_rows(future_phases)
    joint = fit_joint_baseline(y,t)
    max_window = fitted["audit"]["maximum_phase_readout_window"]
    mssa = {window:recurrent_mssa(y[:,:n_train],y[:,n_train:start],len(t)-start,window)
            for window in dict.fromkeys((128,max_window))}
    arrays = OUT / "real_phase_factors_final_coordinates.npz"
    np.savez_compressed(arrays,train_phase=train_phases,future_phase=future_phases,
        future_phase_t=t[start:start+future_phases.shape[1]],future_t=t[start:],
        frequencies_rad_per_sample=frequencies,future_prediction=prediction,
        future_observed=y[:,start:],baseline_future_prediction=joint["future_prediction"],
        **{"mssa_window"+str(window)+"_prediction":row["prediction"] for window,row in mssa.items()})
    current = fingerprints()
    current[Path(__file__).name] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    payload = {"status":"finished","source":str(SOURCE),
        "source_sha256":hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "source_fingerprints":hashes,"source_fingerprints_unchanged":current==hashes,
        "data":"cached measured NOAA river-level field","fit_inputs":["eta","distance_km","t"],
        "catalogue_metadata_used":False,"settings":asdict(settings),"record_shape":list(y.shape),
        "split":[n_train,n_validation,len(t)-start],"noise_estimate_train_only":noise,
        "inference":fitted["audit"],"frozen_readouts":readouts,
        "frequencies_cycles_per_hour":(frequencies/(2*np.pi*np.diff(t).mean())).tolist(),
        "future_observed_nrmse":nrmse(prediction,y[:,start:]),
        "joint_baseline_future_observed_nrmse":joint["future_observed_nrmse"],
        "mssa":{str(window):dict({key:value for key,value in row.items() if key!="prediction"},
            future_observed_nrmse=nrmse(row["prediction"],y[:,start:])) for window,row in mssa.items()},
        "selected_joint_baseline_lines":joint["selected_oscillation_line_count"],
        "coordinates":str(arrays),"coordinates_sha256":hashlib.sha256(arrays.read_bytes()).hexdigest(),
        "interpretation":"learned phase factors; residual is not certified noise or a complete torus",
        "elapsed_seconds":time.perf_counter()-started}
    TARGET.write_text(json.dumps(serializable(payload),ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    print(json.dumps({key:payload[key] for key in ("future_observed_nrmse",
        "joint_baseline_future_observed_nrmse","frequencies_cycles_per_hour",
        "source_fingerprints_unchanged","elapsed_seconds")},indent=2),flush=True)


if __name__ == "__main__":
    main()
