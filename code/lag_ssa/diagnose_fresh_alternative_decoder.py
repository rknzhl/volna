"""Read-only, post-freeze decomposition of the repaired fresh-case decoder.

No fitting, frequency selection, support selection or output mutation occurs.
True phases/catalogue are used only to evaluate an already frozen model and
identify the source of its remaining reconstruction error.
"""

from project_paths import RESULTS_DIR

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from sympy import Matrix


HERE = Path(__file__).resolve().parent
SOURCE = RESULTS_DIR / 'fresh_alternative_phase_pipeline.json'
OUTPUT = RESULTS_DIR / 'fresh_alternative_decoder_diagnostics.json'
TARGET = "fresh_r3_nofund_cover3"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_archive(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def canonical(mode, coefficient):
    sign = next((int(value) for value in mode if value), 0)
    if sign == 0:
        raise ValueError("nonconstant signed dictionary required")
    return (tuple(int(value) for value in mode), coefficient) if sign > 0 else (
        tuple(-int(value) for value in mode), np.conj(coefficient))


def coefficient_dictionary(modes, coefficients):
    result = {}
    for mode, coefficient in zip(modes, coefficients):
        key, value = canonical(mode, coefficient)
        result[key] = result.get(key, 0.) + value
    return result


def field(labels, coefficients, phase):
    count = len(labels)
    signed = (coefficients[1:count + 1] - 1j * coefficients[count + 1:]) / 2
    angles = np.einsum("ij,jt->it", labels, phase, optimize=False)
    return coefficients[0, :, None] + 2 * np.real(np.einsum(
        "mq,mt->qt", signed, np.exp(1j * angles), optimize=False))


def score(prediction, clean, mass):
    error = prediction - clean
    return dict(sensor_nrmse=float(np.sqrt(np.sum(error ** 2) / np.sum(clean ** 2))),
        mass_nrmse=float(np.sqrt(np.einsum("it,ij,jt->", error, mass, error, optimize=False) /
                                  np.einsum("it,ij,jt->", clean, mass, clean, optimize=False))))


def haar_decomposition(labels, coefficients, physical_basis, origin, truth):
    count = len(labels)
    complex_coefficients = (coefficients[1:count + 1] - 1j * coefficients[count + 1:]) / 2
    physical_modes = np.einsum("ij,jk->ik", labels, physical_basis, optimize=False)
    gauge = np.exp(1j * np.einsum("ij,j->i", labels, origin, optimize=False))
    learned = coefficient_dictionary(physical_modes, complex_coefficients * gauge[:, None])
    norm2 = lambda value: float(np.sum(abs(value) ** 2))
    signal = 2 * sum(norm2(value) for value in truth.values())
    missing = 2 * sum(norm2(value) for key, value in truth.items() if key not in learned)
    extra = 2 * sum(norm2(value) for key, value in learned.items() if key not in truth)
    bias = 2 * sum(norm2(learned[key] - value) for key, value in truth.items() if key in learned)
    mean = norm2(coefficients[0])
    total = missing + extra + bias + mean
    fractions = {key: value / signal for key, value in
                 (("missing_true_characters", missing), ("extra_nontrue_characters", extra),
                  ("included_character_coefficient_bias", bias), ("mean", mean))}
    return dict(retained_character_count=count, true_character_count=len(truth),
        represented_true_character_count=sum(key in learned for key in truth),
        extra_character_count=sum(key not in truth for key in learned),
        true_field_haar_mass_energy=signal, squared_error_over_signal=fractions,
        fraction_of_total_squared_error={key: value * signal / total for key, value in fractions.items()},
        haar_mass_nrmse=float(np.sqrt(total / signal))), learned


def main():
    if OUTPUT.exists():
        raise FileExistsError("preserve completed read-only diagnostic")
    payload = json.loads(SOURCE.read_text())
    if payload["status"] != "finished" or not payload["all_six_attempts_and_outputs_frozen_before_truth_open"]:
        raise ValueError("requires completed immutable replay")
    source_hash = digest(SOURCE)
    row = next(row for row in payload["rows"] if row["run"] == TARGET)
    entry = next(entry for entry in payload["manifest"] if entry["run"] == TARGET)
    paths = {"public_input": (entry["input"], entry["input_sha256"]),
             "evaluation": (entry["evaluation"], entry["evaluation_sha256"]),
             "discovery_model": (row["discovery"]["source_model"], row["discovery"]["source_model_sha256"]),
             "dictionary_model": (row["dictionary"]["frozen_parameters"], row["dictionary"]["frozen_parameters_sha256"]),
             "dictionary_output": (row["dictionary"]["frozen_output"], row["dictionary"]["frozen_output_sha256"])}
    initial_hashes = {key: digest(path) for key, (path, expected) in paths.items()}
    if any(initial_hashes[key] != expected for key, (_, expected) in paths.items()):
        raise ValueError("frozen source archive changed before diagnostic")
    parameters = load_archive(row["dictionary"]["frozen_parameters"])
    arrays = load_archive(row["dictionary"]["frozen_output"])
    evaluation = load_archive(entry["evaluation"])
    basis = np.asarray(row["module_truth_only"]["physical_basis"], dtype=int)
    exact_basis = Matrix(basis.tolist())
    if abs(exact_basis.det()) != 1:
        raise ValueError("this target diagnostic requires a complete primitive square basis")
    inverse = np.asarray(exact_basis.inv().tolist(), dtype=int)
    train_count = entry["n_train"]
    cutoff = train_count + entry["n_validation"]
    true_train = np.einsum("ij,jt->it", basis, evaluation["theta"][:, :train_count], optimize=False)
    origin = np.angle(np.mean(np.exp(1j * (
        parameters["frequencies"][:, None] * np.arange(train_count) - true_train)), axis=1))
    oracle_phase = np.einsum("ij,jt->it", basis, evaluation["theta"][:, cutoff:], optimize=False) + origin[:, None]
    clean = evaluation["clean"][:, cutoff:]
    mass_to_sensor = parameters["mass_to_sensor"]
    profiles_mass = np.einsum("ij,mj->mi", parameters["spatial_mass_sqrt"],
                              evaluation["catalog_coefficients"], optimize=False)
    truth = coefficient_dictionary(evaluation["catalog_modes"], profiles_mass)
    catalog_clean = 2 * np.real(np.einsum("mq,mt->qt", evaluation["catalog_coefficients"],
        np.exp(1j * np.einsum("ij,jt->it", evaluation["catalog_modes"], evaluation["theta"], optimize=False)),
        optimize=False))
    variants = {}
    for name, labels, coefficients, observed_key in (
            ("pruned", parameters["labels"], parameters["coefficients_mass"], "prediction_endpoint"),
            ("unpruned", parameters["0__labels"], parameters["0__coefficients_mass"], "old_prediction_endpoint")):
        prediction_oracle = np.einsum("ij,jt->it", mass_to_sensor,
            field(labels, coefficients, oracle_phase), optimize=False)
        decomposition, learned = haar_decomposition(labels, coefficients, basis, origin, truth)
        variants[name] = dict(observed_phase_score=score(arrays[observed_key], clean, parameters["physical_mass"]),
            oracle_phase_score_evaluation_only=score(prediction_oracle, clean, parameters["physical_mass"]),
            haar_coefficient_error_decomposition=decomposition)
    original_labels, retained_labels = parameters["0__labels"], parameters["labels"]
    missing = []
    for mode in evaluation["catalog_modes"]:
        label = np.einsum("i,ij->j", mode, inverse, optimize=False)
        in_original = bool(np.any(np.all(original_labels == label, axis=1) | np.all(original_labels == -label, axis=1)))
        in_retained = bool(np.any(np.all(retained_labels == label, axis=1) | np.all(retained_labels == -label, axis=1)))
        if not in_retained:
            key, _ = canonical(mode, np.zeros(len(mass_to_sensor)))
            missing.append(dict(physical_character=mode.tolist(), final_basis_character=label.tolist(),
                final_basis_l1=int(np.sum(abs(label))), present_before_protected_pruning=in_original,
                removed_by_protected_pruning=bool(in_original and not in_retained),
                fraction_of_true_haar_mass_energy=float(2 * np.sum(abs(truth[key]) ** 2) /
                    variants["pruned"]["haar_coefficient_error_decomposition"]["true_field_haar_mass_energy"])))
    true_basis_frequency = np.einsum("ij,j->i", basis, evaluation["alpha"], optimize=False)
    true_acquired_frequency = np.einsum("ij,j->i", parameters["acquired_labels"], true_basis_frequency, optimize=False)
    acquired = np.asarray(row["discovery"]["inference"]["acquired_frequencies"])
    true_catalog_frequency = np.einsum("ij,j->i", evaluation["catalog_modes"], evaluation["alpha"], optimize=False)
    extras = []
    count = len(retained_labels)
    retained_complex = (parameters["coefficients_mass"][1:count + 1] -
                        1j * parameters["coefficients_mass"][count + 1:]) / 2
    for label, profile in zip(retained_labels, retained_complex):
        mode = np.einsum("i,ij->j", label, basis, optimize=False)
        key, _ = canonical(mode, profile)
        if key in truth:
            continue
        frequency = float(np.einsum("i,i->", label, parameters["frequencies"], optimize=False))
        gaps = np.angle(np.exp(1j * (frequency - np.r_[true_catalog_frequency, -true_catalog_frequency])))
        nearest = int(np.argmin(abs(gaps)))
        nearest_mode = evaluation["catalog_modes"][nearest % len(true_catalog_frequency)] * (
            1 if nearest < len(true_catalog_frequency) else -1)
        extras.append(dict(final_basis_character=label.tolist(), physical_character=mode.tolist(),
            nearest_true_temporal_character=nearest_mode.tolist(),
            nearest_true_frequency_gap_rad_per_sample=float(gaps[nearest]),
            coefficient_pair_mass_energy=float(4 * np.sum(abs(profile) ** 2)),
            same_physical_character_as_nearest=False))
    result = dict(status="finished", created_at=datetime.now(timezone.utc).isoformat(),
        run=TARGET, source=str(SOURCE), source_sha256=source_hash,
        diagnostic_source_sha256=digest(__file__), frozen_archive_hashes=initial_hashes,
        public_only_inference_already_completed=True, no_model_refitted=True,
        no_frequency_label_coefficient_or_phase_output_changed=True, no_new_candidate_selection=True,
        oracle_phase_used_for_evaluation_only=True, oracle_phase_is_method_output=False,
        truth_gauge_fit_on_train_only=True, future_truth_used_to_fit_gauge=False,
        exact_physical_basis=basis.tolist(), basis_determinant=int(exact_basis.det()),
        train_origin_gauge_rad=origin.tolist(), variants=variants,
        missing_true_characters=missing, retained_nontrue_characters=extras,
        rotation_diagnostics=dict(units="radians per sample", acquired_frequencies=acquired.tolist(),
            acquired_frequency_errors_wrapped=(np.angle(np.exp(1j * (acquired - true_acquired_frequency)))).tolist(),
            decoder_basis_frequencies=parameters["frequencies"].tolist(),
            decoder_basis_frequency_errors_wrapped=(np.angle(np.exp(1j * (
                parameters["frequencies"] - true_basis_frequency)))).tolist(),
            acquired_and_final_projected_rotation_are_distinct=True,
            strong_first_acquired_final_model_error_rad_per_sample=float(np.angle(np.exp(1j * (
                np.einsum("i,i->", parameters["acquired_labels"][0], parameters["frequencies"], optimize=False) -
                true_acquired_frequency[0]))))),
        finite_catalog_reconstruction_max_absolute_difference=float(np.max(abs(catalog_clean - evaluation["clean"]))),
        phase_scores_by_case=[dict(run=item["run"], phase=item["phase_truth_only"],
            full_observable_state_faithful=item["full_physical_module_and_group_correct_truth_only"])
            for item in payload["rows"]],
        replay_summary=payload["summary"],
        interpretation="The target's oracle-phase composition is almost as inaccurate as its observed-phase composition. Three true characters were absent before pruning; three retained nontrue characters also contribute error. The remaining field failure is predominantly frozen-decoder support/coefficient error; this does not identify the causal share of frequency bias versus earlier dictionary selection.",
        haar_metric_scope="uniform physical torus mass norm; finite future trajectory includes cross terms and uses a different normalization",
        missing_dictionary_does_not_mean_missing_phase_module=True,
        no_global_relabeling_fix_evaluated=True,
        previously_examined_posthoc_replay=True, independent_external_validation=False)
    result["all_frozen_archives_and_source_unchanged"] = bool(digest(SOURCE) == source_hash and
        all(digest(path) == initial_hashes[key] for key, (path, _) in paths.items()))
    result["all_frozen_inference_sources_unchanged"] = all(digest(HERE / name) == expected
        for name, expected in payload["source_fingerprints"].items())
    OUTPUT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(output=str(OUTPUT), variants=variants, missing_true_characters=missing,
        rotation_diagnostics=result["rotation_diagnostics"],
        archives_unchanged=result["all_frozen_archives_and_source_unchanged"],
        sources_unchanged=result["all_frozen_inference_sources_unchanged"])))


if __name__ == "__main__":
    main()
