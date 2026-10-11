"""Observation-only integration check of the optional observer fallback.

Keep the previous finite dictionary so its observed near-relation invokes
the fallback.  This is a declared component verification, not another final
method configuration.  No generator phases/frequencies are opened or scored.
"""

from project_paths import RESULTS_DIR

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np

from final_phase_factor_assay import fixed_settings
from iterative_ssa import Settings, fit_iterative_ssa, _observed_acquired_characters, compact_frontend
from nodal_spatial import NodalSpatialModel
from run_actual_wave_revised import fixture_inputs
import run_actual_wave_iterative as wave_driver
from ssa_character import apply_field_character, aligned_character_rows


HERE = Path(__file__).resolve().parent


def main():
    hashes = wave_driver.fingerprints()
    for name in ("verify_integrated_observer_fallback.py", "final_phase_factor_assay.py", "run_actual_wave_revised.py"):
        hashes[name] = hashlib.sha256((HERE/name).read_bytes()).hexdigest()
    y, positions, times, _ = fixture_inputs("r2_infinite_mixed_products")
    settings = replace(fixed_settings(Settings(n_train=3200, n_validation=1600, n_future=1600,
        lag_window=128, frontend_mode="dynamic", dynamic_ranks=(4,8,16),
        joint_frequency_mode="harmonic_nuisance", phase_metric_mode="field", temporal_gate_mode="iid_reference")),
        fourier_sieve_mode="legacy")
    spatial = NodalSpatialModel.from_positions(positions)
    noise = wave_driver.noise_from_training_high_band(y[:, :3200])
    fitted = fit_iterative_ssa(y[:, :3200], y[:, 3200:4800], settings, sampling_model=spatial, noise_estimate=noise)
    mass = spatial.to_mass_coordinates(y)
    actual_train = aligned_character_rows([apply_field_character(front, mass[:,:3200],
        fitted["clock_models"][index].predict(times[:3200]))["normalized_character"]
        for index,front in enumerate(fitted["frozen_frontends"])])
    stored_train = aligned_character_rows([front["normalized_character"] for front in fitted["frozen_frontends"]])
    mismatch = float(np.max(abs(np.angle(actual_train*np.conj(stored_train)))))
    future = aligned_character_rows([apply_field_character(front, mass[:,4800:],
        fitted["clock_models"][index].predict(times[4800:]))["normalized_character"]
        for index,front in enumerate(fitted["frozen_frontends"])])
    refinements = [{"step":s["step"], "status":s["status"],
        "validation_gain_positive":s.get("validation_gain_positive"), "refinement":s["phase_readout_refinement_attempt"]}
        for s in fitted["audit"]["stages"] if "phase_readout_refinement_attempt" in s]
    payload={"status":"finished", "input_scope":"observations only; no generator phases/frequencies opened",
        "test_scope":"legacy-dictionary component verification only; final assay uses expanded dictionary",
        "source_fingerprints":hashes, "source_fingerprints_current":all(hashlib.sha256((HERE/k).read_bytes()).hexdigest()==v for k,v in hashes.items()),
        "final_status":fitted["audit"]["status"], "settings":settings.__dict__,
        "refinement_stages":refinements, "returned_frontends":[compact_frontend(front) for front in fitted["frozen_frontends"]],
        "returned_train_readout_max_phase_difference_rad":mismatch,
        "returned_source_and_windows":[(front.get("readout_source","residual"),front["window"]) for front in fitted["frozen_frontends"]],
        "final_relation_check_of_returned_readouts":fitted["audit"]["final_observed_relation_check_of_returned_filters"],
        "field_validation_mse_final":float(np.mean(np.sum((mass[:,3200:4800]-fitted["clock_models"][-1].predict(times[3200:4800]))**2,axis=0))),
        "future_used_for_selection":False}
    wave_driver.save(payload,RESULTS_DIR / 'integrated_observer_fallback_verification.json')
    np.savez_compressed(RESULTS_DIR / 'integrated_observer_fallback_verification_coordinates.npz', train=actual_train,
                       future=future, learned_frequencies=fitted["acquired_frequencies"])
    print(json.dumps({"status":payload["final_status"],"source_windows":payload["returned_source_and_windows"],
        "max_returned_train_phase_difference":mismatch,"refinement_count":len(refinements),
        "refinements":[{"step":r["step"],"status":r["status"],"supported":r["refinement"].get("status"),
                        "window":r["refinement"].get("selected_window"),"field_gain_positive":r["validation_gain_positive"]} for r in refinements]}),flush=True)


if __name__=="__main__":
    main()
