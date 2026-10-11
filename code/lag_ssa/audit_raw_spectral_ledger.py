"""Independent post-freeze audit/scoring of the raw observed-line ledger.

Run only after the author/root explicitly confirms all six inference outputs
are frozen. No evaluator is opened before all six deterministic forecasts
have also been saved here. No fitting, selection or parameter mutation occurs.
"""

from project_paths import RESULTS_DIR

from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sympy import Matrix
from sympy.matrices.normalforms import hermite_normal_form


HERE = Path(__file__).resolve().parent
SOURCE = RESULTS_DIR / 'raw_spectral_ledger.json'
EVALUATION_MANIFEST = RESULTS_DIR / 'fresh_phase_pipeline.json'
ROOT = RESULTS_DIR / 'raw_spectral_ledger_audit'
OUTPUT = ROOT.with_suffix(".json")
MATCHING_TOLERANCE_TRAIN_BINS = .25


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_archive(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def row_hnf(rows, dimension):
    if not rows:
        return []
    result = hermite_normal_form(Matrix(rows).T).T
    return [[int(value) for value in row] for row in result.tolist()]


def predict_complex(mean, profiles, frequencies, center, times):
    return mean[:, None] + 2 * np.real(np.einsum("mq,mt->qt", profiles,
        np.exp(1j * frequencies[:, None] * (times[None, :] - center)), optimize=False))


def same_real_character(first, second):
    return bool(np.array_equal(first, second) or np.array_equal(first, -np.asarray(second)))


def field_score(prediction, clean, mass):
    error = prediction - clean
    return dict(sensor_nrmse=float(np.sqrt(np.sum(error ** 2) / np.sum(clean ** 2))),
        sensor_rmse=float(np.sqrt(np.mean(error ** 2))),
        nodal_mass_nrmse=float(np.sqrt(np.einsum("it,ij,jt->", error, mass, error, optimize=False) /
            np.einsum("it,ij,jt->", clean, mass, clean, optimize=False))))


def audit_parameters(row, entry):
    if digest(entry["input"]) != entry["input_sha256"]:
        raise ValueError("public observations changed")
    if digest(row["frozen_parameters"]) != row["frozen_parameters_sha256"]:
        raise ValueError("frozen ledger parameters changed")
    parameters = load_archive(row["frozen_parameters"])
    with np.load(entry["input"], allow_pickle=False) as public:
        if set(public.files) != {"Y", "s", "t"}:
            raise ValueError("exact public archive schema required")
        times = public["t"].copy()
    count = len(parameters["frequencies"])
    coefficient = parameters["coefficients_sensor_center"]
    if coefficient.shape[0] != 1 + 2 * count:
        raise ValueError("mean/all-cos/all-sin coefficient order required")
    profile = (coefficient[1:count + 1] - 1j * coefficient[count + 1:]) / 2
    center = float(parameters["time_center"])
    train, cutoff = entry["n_train"], entry["n_train"] + entry["n_validation"]
    if center != float(np.mean(times[:train])):
        raise ValueError("TRAIN center changed")
    predicted_train = predict_complex(coefficient[0], profile, parameters["frequencies"], center, times[:train])
    predicted_validation = predict_complex(coefficient[0], profile, parameters["frequencies"], center, times[train:cutoff])
    future_prediction = predict_complex(coefficient[0], profile, parameters["frequencies"], center, times[cutoff:])
    frozen_path = ROOT / (entry["run"] + ".npz")
    np.savez_compressed(frozen_path, future_times=times[cutoff:], predicted_future=future_prediction,
        frequencies=parameters["frequencies"], profiles_sensor_center=profile,
        mean_sensor=coefficient[0], time_center=np.asarray(center))
    checks = dict(run=entry["run"], status="forecast_frozen_before_evaluation", future_output=str(frozen_path),
        future_output_sha256=digest(frozen_path), source_parameter_sha256=digest(row["frozen_parameters"]),
        input_sha256=digest(entry["input"]), line_count=count,
        independent_complex_train_max_difference=float(np.max(abs(predicted_train - parameters["predicted_train"]))),
        independent_complex_validation_max_difference=float(np.max(abs(predicted_validation - parameters["predicted_validation"]))),
        center_profile_max_difference=float(np.max(abs(profile - parameters["complex_profiles_sensor_center"]))) if count else 0.,
        origin_profile_max_difference=float(np.max(abs(profile * np.exp(-1j * parameters["frequencies"] * center)[:, None] -
            parameters["complex_profiles_sensor_origin"]))) if count else 0.,
        no_absolute_future_observations_in_prediction=True, prediction_scope="clock forecast from independent frozen free-line frequencies")
    return checks, parameters


def score_lines(checks, parameters, entry, evaluation_entry):
    if evaluation_entry["input_sha256"] != entry["input_sha256"]:
        raise ValueError("independent evaluator does not correspond to original public input")
    if digest(evaluation_entry["evaluation"]) != evaluation_entry["evaluation_sha256"]:
        raise ValueError("independent evaluation archive changed")
    evaluation = load_archive(evaluation_entry["evaluation"])
    prediction = load_archive(checks["future_output"])
    cutoff = entry["n_train"] + entry["n_validation"]
    if not np.array_equal(prediction["future_times"], np.arange(cutoff, cutoff + entry["n_future"])):
        raise ValueError("future targets differ from predeclared block")
    frequencies = parameters["frequencies"]
    true_labels, true_coefficients = evaluation["catalog_modes"], evaluation["catalog_coefficients"]
    true_frequencies = np.angle(np.exp(1j * np.einsum("ij,j->i", true_labels, evaluation["alpha"], optimize=False)))
    mass, center = parameters["physical_mass"], float(parameters["time_center"])
    true_energy = 2 * np.real(np.einsum("mi,ij,mj->m", np.conj(true_coefficients), mass,
        true_coefficients, optimize=False))
    tolerance = MATCHING_TOLERANCE_TRAIN_BINS * 2 * np.pi / entry["n_train"]
    profiles = parameters["complex_profiles_sensor_center"]
    matched_rows, matched_support, matches = [], set(), []
    oracle_frequency = frequencies.copy()
    for line, (frequency, profile) in enumerate(zip(frequencies, profiles)):
        near = []
        for index, rotation in enumerate(true_frequencies):
            for sign in (1, -1):
                error = float(np.angle(np.exp(1j * (frequency - sign * rotation))))
                if abs(error) <= tolerance:
                    near.append(dict(catalog_index=index, conjugacy_sign=sign,
                        physical_character=(sign * true_labels[index]).tolist(),
                        frequency_error_rad_per_sample=error, catalog_haar_mass_energy=float(true_energy[index])))
        unique = {tuple(match["physical_character"]) for match in near}
        information = dict(line_index=line, learned_frequency=float(frequency), matches=near,
            status="unique" if len(unique) == 1 else "ambiguous" if unique else "unmatched")
        if len(unique) == 1:
            match = near[0]
            index, sign = match["catalog_index"], match["conjugacy_sign"]
            matched_rows.append(list(next(iter(unique)))); matched_support.add(index)
            wrapped_rotation = sign * true_frequencies[index]
            oracle_frequency[line] = wrapped_rotation
            # Center is half-integer. The true temporal frequency must be
            # wrapped BEFORE center rotation, otherwise sampled aliases give
            # a spurious center-gauge phase offset.
            coefficient = true_coefficients[index] if sign > 0 else np.conj(true_coefficients[index])
            origin = float(np.einsum("i,i->", sign * true_labels[index], evaluation["theta"][:, 0], optimize=False))
            reference = coefficient * np.exp(1j * (origin + wrapped_rotation * center))
            difference = profile - reference
            information["profile_center_gauge_mass_relative_error"] = float(np.sqrt(
                np.real(np.einsum("i,ij,j->", np.conj(difference), mass, difference, optimize=False)) /
                np.real(np.einsum("i,ij,j->", np.conj(reference), mass, reference, optimize=False))))
            information["profile_reference_gauge"] = "true TRAIN-center sampled wrapped rotation plus physical initial phase; no future origin fit"
        matches.append(information)
    physical_rank = int(evaluation["rank"])
    recovered = row_hnf(matched_rows, physical_rank)
    visible = row_hnf(evaluation["group_modes"].tolist(), physical_rank)
    module_missing = [row_hnf(matched_rows + [mode.tolist()], physical_rank) != recovered for mode in true_labels]
    absent_support = [index not in matched_support for index in range(len(true_labels))]
    conditional = predict_complex(parameters["coefficients_sensor_center"][0], profiles, oracle_frequency,
        center, prediction["future_times"])
    clean = evaluation["clean"][:, cutoff:]
    checks.update(status="finished", evaluation_sha256=digest(evaluation_entry["evaluation"]),
        future_clock_field_score=field_score(prediction["predicted_future"], clean, mass),
        conditional_true_frequency_substitution_score_evaluation_only=field_score(conditional, clean, mass),
        conditional_substitution_scope="replace only uniquely catalog-matched free-line rotations; unmatched/ambiguous lines retain learned rotations; profiles/mean fixed",
        conditional_frequency_substitution_is_method_output=False,
        no_future_truth_fitted_origin_or_profile=True, line_matching=matches,
        matching_tolerance_rad_per_sample=tolerance, matching_tolerance_train_bins=MATCHING_TOLERANCE_TRAIN_BINS,
        unique_line_count=sum(match["status"] == "unique" for match in matches),
        unmatched_line_count=sum(match["status"] == "unmatched" for match in matches),
        ambiguous_line_count=sum(match["status"] == "ambiguous" for match in matches),
        uniquely_matched_character_module_HNF_truth_only=recovered,
        full_visible_character_module_HNF_truth_only=visible,
        full_visible_module_covered_by_unique_line_matches_truth_only=recovered == visible,
        rank_of_uniquely_matched_module_truth_only=len(recovered), physical_rank_truth_only=physical_rank,
        phase_rank_inferred_by_method=False, spectral_line_count_is_phase_rank=False,
        missing_catalog_line_haar_mass_energy_fraction=float(np.sum(true_energy[absent_support]) / np.sum(true_energy)),
        missing_character_module_haar_mass_energy_fraction=float(np.sum(true_energy[module_missing]) / np.sum(true_energy)),
        bounded_catalog_scope="all finite modes or first64 Poisson harmonics; no unseen infinite-tail certificate",
        catalog_given_to_inference=False, observed_phase_encoding_performed=False,
        nonlinear_window_MSSA_comparison_performed=False)
    if entry["run"] == "fresh_r2_primitive_cover6":
        target = np.asarray([1, 1]); false = np.asarray([39, 0])
        target_frequency = float(np.einsum("i,i->", target, evaluation["alpha"], optimize=False))
        false_frequency = float(np.angle(np.exp(1j * np.einsum("i,i->", false, evaluation["alpha"], optimize=False))))
        checks["first_case_requested_character_check"] = dict(
            true_1_1_uniquely_retained=any(match["status"] == "unique" and any(
                same_real_character(item["physical_character"], target) for item in match["matches"]) for match in matches),
            true_1_1_nearest_learned_frequency_gap=float(np.min(abs(np.angle(np.exp(1j * (
                frequencies[:, None] - np.asarray([target_frequency, -target_frequency])[None, :])))))) if len(frequencies) else None,
            false_39_0_nearest_learned_frequency_gap=float(np.min(abs(np.angle(np.exp(1j * (
                frequencies[:, None] - np.asarray([false_frequency, -false_frequency])[None, :])))))) if len(frequencies) else None,
            false_39_0_is_present_in_true_field_catalog=any(same_real_character(mode, false) for mode in true_labels),
            permanent_integer_labels_assigned_by_ledger=False)
    if entry["run"] == "fresh_r3_infinite_weak":
        checks["third_independent_phase_requested_character_check"] = dict(
            true_e3_uniquely_retained=any(match["status"] == "unique" and any(
                same_real_character(item["physical_character"], [0, 0, 1]) for item in match["matches"]) for match in matches),
            full_rank_three_unique_line_module_truth_only=len(recovered) == 3,
            independence_or_true_rank_certified_by_method=False)
    return checks


def main(confirm_all_frozen=False):
    if not confirm_all_frozen:
        raise RuntimeError("wait for author/root all-six-output freeze confirmation before running")
    if OUTPUT.exists() or ROOT.exists():
        raise FileExistsError("preserve completed ledger audit")
    payload = json.loads(SOURCE.read_text())
    if (payload["status"] != "finished_inference_only" or len(payload["rows"]) != 6 or
            not payload["all_six_attempts_and_outputs_frozen_before_truth"] or payload["evaluation_files_opened"]):
        raise ValueError("requires all six frozen public-only inference attempts")
    source_hash = digest(SOURCE)
    if any(digest(HERE / name) != expected for name, expected in payload["source_fingerprints"].items()):
        raise ValueError("ledger inference source changed")
    ROOT.mkdir(exist_ok=False)
    checks, frozen = [], []
    for row, entry in zip(payload["rows"], payload["manifest"]):
        if row["run"] != entry["run"]:
            raise ValueError("manifest ordering changed")
        if row["run_status"] == "fitted_and_frozen":
            audit, parameters = audit_parameters(row, entry)
            checks.append(audit); frozen.append((audit, parameters, entry))
        else:
            checks.append(dict(run=row["run"], status=row["run_status"], exception=row.get("exception")))
    result = dict(status="all_forecasts_frozen_before_evaluation", created_at=datetime.now(timezone.utc).isoformat(),
        source=str(SOURCE), source_sha256=source_hash, audit_source_sha256=digest(__file__), rows=checks,
        all_six_inference_attempts_frozen_before_truth=True, all_deterministic_forecasts_saved_before_evaluation=True,
        evaluation_files_opened=False, explicit_author_root_freeze_confirmation_received=True,
        no_model_refitted=True, no_threshold_or_candidate_chosen_by_truth=True,
        prior_cases_previously_examined=True, independent_external_validation=False,
        scoring_scope="raw sensor clock forecast; not nonlinear observed-window phase reconstruction",
        rank_or_latent_group_inference_added=False)
    OUTPUT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    evaluation_manifest = json.loads(EVALUATION_MANIFEST.read_text())
    evaluation_by_run = {entry["run"]: entry for entry in evaluation_manifest["manifest"]}
    result["evaluation_manifest_sha256"] = digest(EVALUATION_MANIFEST)
    for audit, parameters, entry in frozen:
        score_lines(audit, parameters, entry, evaluation_by_run[entry["run"]])
    result.update(status="finished", evaluation_files_opened=True,
        inference_sources_unchanged=all(digest(HERE / name) == expected for name, expected in payload["source_fingerprints"].items()),
        inference_result_unchanged=digest(SOURCE) == source_hash,
        all_parameter_hashes_unchanged=all(digest(row["frozen_parameters"]) == row["frozen_parameters_sha256"]
            for row in payload["rows"] if row["run_status"] == "fitted_and_frozen"),
        all_forecast_hashes_unchanged=all(digest(row["future_output"]) == row["future_output_sha256"]
            for row in checks if row["status"] == "finished"),
        all_public_inputs_unchanged=all(digest(entry["input"]) == entry["input_sha256"] for entry in payload["manifest"]))
    result["summary"] = dict(finished_rows=len(frozen), requested_rows=6,
        complete_visible_module_from_unique_matches_truth_only=sum(row["full_visible_module_covered_by_unique_line_matches_truth_only"]
            for row in checks if row["status"] == "finished"),
        method_inferred_r=False, no_observed_window_MSSA_comparison=True)
    OUTPUT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(output=str(OUTPUT), summary=result["summary"],
        rows=[dict(run=row["run"], lines=row.get("line_count"), score=row.get("future_clock_field_score"),
                   unique=row.get("unique_line_count"), ambiguous=row.get("ambiguous_line_count"), unmatched=row.get("unmatched_line_count"))
              for row in checks])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm-all-six-frozen", action="store_true")
    main(parser.parse_args().confirm_all_six_frozen)
