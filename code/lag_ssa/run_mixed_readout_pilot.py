"""Training-selected convolution phase-readout pilot for the mixed-field flag.

The existing assay supplies only observation-selected frequencies and a
proposed integer relation. Profiles, Hann window and phase origins use train
only. Validation keeps the existing .95 gate. Future is scoring only.
No true phases, frequencies or mode labels are opened by this script.
"""

from __future__ import annotations

from project_paths import RESULTS_DIR

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np

import run_actual_wave_iterative as wave_driver
from iterative_ssa import Settings, fit_iterative_ssa, character_frontend, _observed_acquired_characters
from nodal_spatial import NodalSpatialModel
from phase_factor_regression import observed_relation_check
from phase_readout_refinement import fit_field_character_observer, observer_quality
from run_actual_wave_revised import fixture_inputs
from ssa_character import apply_ssa_character


HERE = Path(__file__).resolve().parent
OUTPUT = RESULTS_DIR / 'mixed_readout_pilot.json'


def relation_rows(readouts):
    count = min(len(row["normalized_character"]) for row in readouts)
    return np.asarray([row["normalized_character"][:count] for row in readouts])


def main():
    started = time.perf_counter()
    source_names = ("phase_readout_refinement.py", "run_mixed_readout_pilot.py", "iterative_ssa.py", "phase_factor_regression.py")
    initial_hashes = {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in source_names}
    y, positions, times, _ = fixture_inputs("r2_infinite_mixed_products")
    settings = Settings(n_train=3200, n_validation=1600, n_future=1600,
                        lag_window=128, frontend_mode="dynamic", dynamic_ranks=(4, 8, 16),
                        joint_frequency_mode="harmonic_nuisance", phase_metric_mode="field",
                        temporal_gate_mode="iid_reference", k_candidates=(16, 32),
                        conditional_model_mode="fourier_sieve", enforce_validation_improvement=True,
                        phase_relation_mode="observed")
    spatial = NodalSpatialModel.from_positions(positions)
    noise = wave_driver.noise_from_training_high_band(y[:, :3200])
    print("fit mixed observed-relation assay", flush=True)
    fitted = fit_iterative_ssa(y[:, :3200], y[:, 3200:4800], settings,
                               sampling_model=spatial, noise_estimate=noise)
    audit = fitted["audit"]
    unresolved = audit["stages"][-1]
    if unresolved["status"] != "unresolved_near_relation":
        raise RuntimeError("pilot expected the observation-supported near-relation flag")
    candidate_frequency = float(unresolved["candidate"]["frequency"])
    proposed_label = np.asarray(unresolved["old_factor_phase_polynomial_association"]["label"], dtype=int)
    lattice = audit["final_lattice"]
    products = np.asarray(lattice["products"], dtype=int)
    acquired_relation = proposed_label @ products
    relation = np.r_[1, -acquired_relation][None, :]
    frequencies = np.r_[candidate_frequency, fitted["acquired_frequencies"]]
    mass = spatial.to_mass_coordinates(spatial.recover(y))
    train_times = times[:3200]
    validation_times = times[3200:4800]
    future_times = times[4800:]
    residual_train = mass[:, :3200] - fitted["clock_models"][-1].predict(train_times)
    residual_validation = mass[:, 3200:4800] - fitted["clock_models"][-1].predict(validation_times)
    residual_future = mass[:, 4800:] - fitted["clock_models"][-1].predict(future_times)
    sources_train = [residual_train] + [mass[:, :3200]] * len(fitted["acquired_frequencies"])
    sources_validation = [residual_validation] + [mass[:, 3200:4800]] * len(fitted["acquired_frequencies"])
    sources_future = [residual_future] + [mass[:, 4800:]] * len(fitted["acquired_frequencies"])
    covariance = noise["sigma"] ** 2 * np.einsum("ij,kj->ik", spatial.mass_sqrt,
                                                spatial.mass_sqrt, optimize=False)

    # A disjoint inner training suffix selects L; it never opens validation.
    split = 2304
    grid = []
    for window in (128, 256, 512):
        observers = [fit_field_character_observer(source[:, :split], frequency, window)
                     for source, frequency in zip(sources_train, frequencies)]
        inner_train = relation_rows([observer.apply(source[:, :split])
                                     for source, observer in zip(sources_train, observers)])
        inner_holdout = relation_rows([observer.apply(source[:, split:])
                                       for source, observer in zip(sources_train, observers)])
        check = observed_relation_check(inner_train, inner_holdout, relation, coherence_threshold=.95)
        item = check["relations"][0]
        score = min(item["train_coherence"], item["validation_coherence"],
                    item["validation_frozen_origin_agreement"])
        grid.append({"window": window, "minimum_inner_relation_agreement": score,
                     "train_inner_relation_check": check,
                     "inner_holdout_quality": [observer_quality(observer.apply(source[:, split:]), frequency,
                                                              covariance, observer)
                                               for source, frequency, observer in zip(sources_train, frequencies, observers)]})
    selected = max(grid, key=lambda item: (item["minimum_inner_relation_agreement"], -item["window"]))
    window = selected["window"]
    observers = [fit_field_character_observer(source, frequency, window)
                 for source, frequency in zip(sources_train, frequencies)]
    train_readouts = [observer.apply(source) for source, observer in zip(sources_train, observers)]
    validation_readouts = [observer.apply(source) for source, observer in zip(sources_validation, observers)]
    future_readouts = [observer.apply(source) for source, observer in zip(sources_future, observers)]
    convolution_train = relation_rows(train_readouts)
    convolution_validation = relation_rows(validation_readouts)
    convolution_future = relation_rows(future_readouts)
    validation_check = observed_relation_check(convolution_train, convolution_validation, relation, .95)
    future_check = observed_relation_check(convolution_train, convolution_future, relation, .95)

    # Reproduce the unchanged frozen SSA candidate for an independent comparison.
    candidate_frontend = character_frontend(residual_train, noise["sigma"], settings, spatial)
    ssa_train, ssa_validation = _observed_acquired_characters(
        fitted["frozen_frontends"], fitted["clock_models"], mass[:, 3200:4800], validation_times)
    _, ssa_future = _observed_acquired_characters(
        fitted["frozen_frontends"], fitted["clock_models"], mass[:, 4800:], future_times)
    ssa_train = np.vstack((candidate_frontend["normalized_character"], ssa_train))
    ssa_validation = np.vstack((apply_ssa_character(candidate_frontend, residual_validation)["normalized_character"], ssa_validation))
    ssa_future = np.vstack((apply_ssa_character(candidate_frontend, residual_future)["normalized_character"], ssa_future))
    hashes = {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in source_names}
    payload = {"case": "r2_infinite_mixed_products", "status": "finished",
               "settings": asdict(settings), "source_fingerprints": hashes,
               "source_fingerprints_unchanged": hashes == initial_hashes,
               "input_scope": "Y,s,t; learned frequencies and relation from observed assay; no evaluation truth read",
               "learned_frequencies_rad_per_sample": frequencies.tolist(),
               "proposed_relation_in_acquired_characters": relation.tolist(),
               "assay_status_before_refinement": audit["status"],
               "window_selection": "maximum minimum train/inner-holdout phase-product agreement; tie chooses shorter window",
               "inner_profile_training_samples": split,
               "frequencies_estimated_from_full_training_record": True,
               "inner_residual_model_fitted_on_full_training_record": True,
               "independent_validation_or_future_used_for_window_selection": False,
               "window_selection_grid": grid, "selected_window": window,
               "frozen_observers": [observer.metadata() for observer in observers],
               "validation_convolution_relation_check": validation_check,
               "future_convolution_relation_check_scoring_only": future_check,
               "validation_convolution_quality": [observer_quality(readout, frequency, covariance, observer)
                                                   for readout, frequency, observer in zip(validation_readouts, frequencies, observers)],
               "future_convolution_quality_scoring_only": [observer_quality(readout, frequency, covariance, observer)
                                                           for readout, frequency, observer in zip(future_readouts, frequencies, observers)],
               "existing_ssa_validation_relation_check": observed_relation_check(ssa_train, ssa_validation, relation, .95),
               "existing_ssa_future_relation_check_scoring_only": observed_relation_check(ssa_train, ssa_future, relation, .95),
               "decoder_unchanged_by_this_pilot": True,
               "remaining_scope": "one proposed relation; no discovery completeness, spatial interpolation or novelty claim",
               "elapsed_seconds": time.perf_counter() - started}
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    np.savez_compressed(RESULTS_DIR / 'mixed_readout_pilot_coordinates.npz',
                        convolution_train=convolution_train, convolution_validation=convolution_validation,
                        convolution_future=convolution_future, ssa_train=ssa_train,
                        ssa_validation=ssa_validation, ssa_future=ssa_future,
                        relation=relation, learned_frequencies=frequencies)
    print(json.dumps({"selected_window": window, "validation": validation_check,
                      "future": future_check, "elapsed_seconds": payload["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
