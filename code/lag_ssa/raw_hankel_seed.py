"""Training-fitted initialization using exactly the observed Hankel window.

Strong independent characters select a finite set of covering lifts. Their
future angles are raw-field linear readouts. Absolute future time and model
clock subtraction are not accepted by this API.
"""

import numpy as np

from phase_readout_refinement import fit_field_character_observer


def initialize_from_raw_strong_characters(mass_training, mass_observed, projector):
    rows = np.asarray(projector.multistart_metadata["strong_independent_character_rows"], dtype=int)
    if rows.shape != (projector.rank, projector.rank):
        raise ValueError("one independent learned character per phase is required")
    frequencies = np.einsum("ij,j->i", rows, projector.frequencies, optimize=False)
    window = projector.settings.window
    future_phases, training_phases, origins, metadata = [], [], [], []
    for frequency in frequencies:
        observer = fit_field_character_observer(mass_training, float(frequency), window)
        train_phase = observer.apply(mass_training)["phase"]
        origin = np.angle(np.mean(np.exp(1j*(train_phase-frequency*np.arange(len(train_phase))))))
        training_phases.append(np.angle(np.exp(1j*(train_phase-origin))))
        future_phases.append(np.angle(np.exp(1j*(observer.apply(mass_observed)["phase"]-origin))))
        origins.append(origin)
        metadata.append(observer.metadata())
    seed = np.linalg.solve(rows.astype(float), np.asarray(future_phases))
    train_seed = np.linalg.solve(rows.astype(float), np.asarray(training_phases))
    return {"phase": seed, "training_phase": train_seed,
        "metadata": {"strong_character_labels": rows.tolist(),
            "learned_character_frequencies": frequencies.tolist(), "seed_window": window,
            "training_origins": origins, "raw_observers": metadata,
            "future_absolute_time_used": False, "future_prefix_clock_subtraction_used": False,
            "seed_and_objective_use_identical_observation_window": True,
            "cover_lifts": projector.multistart_metadata}}
