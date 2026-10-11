"""Corroborate saved SSA candidates in raw validation with frozen nuisances.

This is a retrospective diagnostic, not a core algorithm change. Frequencies,
harmonic labels, and BIC dictionary choices come from training-stage audits.
Only sinusoid amplitudes are fitted in validation. The central chi-square
reference assumes the old finite dictionary is the correct conditional mean;
infinite/mixed tails and frequency error can invalidate that assumption.
Physical mode labels enter only after all raw-data diagnostics for a row.
"""

from __future__ import annotations

from project_paths import RESULTS_DIR

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.linalg import lstsq, svdvals
from scipy.stats import chi2

from iterative_fields import AREA, _lattice_signature, generate_field, score_frequency


OUT = RESULTS_DIR
SOURCE = OUT / "iterative_ssa_residual_v4_robustness.json"
DESTINATION = OUT / "raw_character_corroboration.json"
P_GATE = 1e-8
NUMERICAL_RELATIVE_TOLERANCE = 1e-10
CATALOG_HARMONIC_ORDER = 12


def _serialize(value):
    if isinstance(value, np.ndarray):
        return _serialize(value.tolist())
    if isinstance(value, np.generic):
        return _serialize(value.item())
    if isinstance(value, dict):
        return {str(key): _serialize(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_serialize(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _save(payload: dict) -> None:
    payload["last_update_utc"] = datetime.now(timezone.utc).isoformat()
    temporary = DESTINATION.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(_serialize(payload), indent=2,
                                    allow_nan=False) + "\n")
    temporary.replace(DESTINATION)


def regenerate_observation(row: dict) -> tuple[np.ndarray, dict]:
    """Reproduce runner's one full-record RNG draw before slicing validation."""
    settings = row["settings"]
    n_train, n_validation, n_future = (settings[key] for key in
                                      ("n_train", "n_validation", "n_future"))
    generated = generate_field(row["case"], np.arange(n_train+n_validation+n_future),
                               n_train=n_train)
    clean = generated["clean"]
    scale = float(np.sqrt(np.mean(np.sum(AREA[:, None]*clean[:, :n_train]**2,
                                         axis=0))))
    applied_sigma = row["relative_sensor_noise"]*scale
    observed = clean+applied_sigma*np.random.default_rng(row["seed"]).standard_normal(clean.shape)
    validation = observed[:, n_train:n_train+n_validation]
    return validation, {"generator_noise_sigma": applied_sigma,
                        "validation_sha256": hashlib.sha256(validation.tobytes()).hexdigest()}


def frozen_prior_dictionary(previous: dict | None, n_train: int) -> dict:
    if previous is None:
        return {"frequencies": [], "line_labels": [], "selected_harmonic_order": 0,
                "training_bic": None, "times_center": (n_train-1)/2,
                "source_stage_step": None, "dictionary": "constant only"}
    joint = previous["joint_frequency_fit"]
    selected = joint.get("nuisance_audit", {})
    frequencies = np.asarray(joint["frequencies"], dtype=float)
    labels = selected.get("line_labels")
    if labels is None:
        raise ValueError("saved harmonic nuisance line_labels are missing")
    labels = np.asarray(labels, dtype=int)
    if labels.ndim != 2 or labels.shape[1] != len(frequencies):
        raise ValueError("saved prior frequencies and nuisance labels disagree")
    return {"frequencies": frequencies.tolist(), "line_labels": labels.tolist(),
            "selected_harmonic_order": selected.get("harmonic_order"),
            "training_bic": selected.get("bic"),
            "times_center": selected.get("times_center", (n_train-1)/2),
            "source_stage_step": previous["step"],
            "dictionary": "training BIC selected acquired lines and their tied individual harmonics",
            "training_dictionary_audit": selected.get("nuisance_audit", {}),
            "frequencies_source": "prior accepted stage joint_frequency_fit.frequencies"}


def raw_validation_diagnostic(validation: np.ndarray, absolute_times: np.ndarray,
                              candidate: float, prior: dict,
                              noise_sigma_squared: float) -> dict:
    """Frozen-frequency nested linear fits; no dictionary selection on val."""
    y = np.asarray(validation, dtype=float).T
    times = np.asarray(absolute_times, dtype=float)-prior["times_center"]
    old_frequencies = np.asarray(prior["frequencies"], dtype=float)
    if len(old_frequencies):
        labels = np.asarray(prior["line_labels"], dtype=int)
        lines = np.einsum("ij,j->i", labels, old_frequencies, optimize=False)
        angles = times[:, None]*lines[None, :]
        old = np.column_stack((np.ones(len(times)), np.cos(angles), np.sin(angles)))
    else:
        lines = np.empty(0)
        old = np.ones((len(times), 1))
    candidate_design = np.column_stack((np.cos(candidate*times), np.sin(candidate*times)))
    fitted, _, old_rank, _ = lstsq(old, np.column_stack((y, candidate_design)),
                                  cond=NUMERICAL_RELATIVE_TOLERANCE,
                                  lapack_driver="gelsy")
    q = y.shape[1]
    residual = y-np.einsum("ij,jk->ik", old, fitted[:, :q], optimize=False)
    innovation = candidate_design-np.einsum("ij,jk->ik", old, fitted[:, q:], optimize=False)
    old_singular = svdvals(old)
    new_singular = svdvals(innovation)
    innovation_threshold = NUMERICAL_RELATIVE_TOLERANCE*max(1., float(svdvals(candidate_design)[0]))
    increment_rank = int(np.sum(new_singular > innovation_threshold))
    correction, _, _, _ = lstsq(innovation, residual,
                                cond=NUMERICAL_RELATIVE_TOLERANCE,
                                lapack_driver="gelsy")
    alternative_residual = residual-np.einsum("ij,jk->ik", innovation, correction, optimize=False)
    old_rss = float(np.sum(residual**2))
    new_rss = float(np.sum(alternative_residual**2))
    improvement = max(0., old_rss-new_rss)
    statistic = improvement/noise_sigma_squared
    nominal_df = 2*q
    p_reference = float(chi2.sf(statistic, nominal_df))
    full_rank = old_rank == old.shape[1] and increment_rank == 2
    accepted = bool(full_rank and p_reference < P_GATE)
    fraction = float(np.sum(innovation**2)/max(np.sum(candidate_design**2), 1e-30))
    old_dof = (len(times)-old_rank)*q
    return {"accepted_by_raw_reference": accepted,
            "candidate_frequency_before_current_joint_fit": candidate,
            "validation_rss_old": old_rss, "validation_rss_with_candidate": new_rss,
            "validation_rss_improvement": improvement,
            "statistic_using_train_spatial_orthogonal_noise": statistic,
            "chi_square_reference_df": nominal_df,
            "chi_square_reference_p": p_reference, "p_gate": P_GATE,
            "old_design_columns": old.shape[1], "old_design_rank": int(old_rank),
            "old_design_condition": float(old_singular[0]/max(old_singular[-1], 1e-30)),
            "candidate_innovation_rank": increment_rank,
            "candidate_innovation_singular_values": new_singular.tolist(),
            "candidate_energy_fraction_outside_old_dictionary": fraction,
            "candidate_collinear_or_old_design_deficient": not full_rank,
            "old_line_frequencies_from_train_dictionary": lines.tolist(),
            "old_residual_variance_over_noise_reference": old_rss/max(old_dof, 1)/noise_sigma_squared,
            "status": "diagnostic_only_finite_dictionary_misspecification_and_frequency_error_not_calibrated",
            "null_mean_correctness_certified": False,
            "infinite_or_mixed_tail_bias_bounded": False,
            "amplitudes_fitted_in_validation_only": True,
            "candidate_or_dictionary_selected_in_validation": False}


def score_prefix(row: dict, accepted: list[dict], prefix_length: int) -> dict:
    """Truth-only score of the original saved prefix, with no refitting."""
    if not prefix_length:
        return {"prefix_length": 0, "candidate_rank": 0,
                "complete_observable_module_rank_relation_truth_only": False,
                "reason": "no accepted character retained"}
    stage = accepted[prefix_length-1]
    frequencies = stage["joint_frequency_fit"]["frequencies"]
    scores = [score_frequency(row["case"], f, 2*np.pi/row["settings"]["n_train"],
                              CATALOG_HARMONIC_ORDER, row["settings"]["n_train"])
              for f in frequencies]
    distinct = [{tuple(match["mode"]) for match in score["matches"]} for score in scores]
    unique = all(len(modes) == 1 for modes in distinct)
    lattice = stage.get("lattice")
    result = {"prefix_length": prefix_length,
              "candidate_rank": lattice.get("rank") if lattice else None,
              "saved_prefix_frequencies": frequencies,
              "labels_truth_only": scores,
              "unique_physical_labels_truth_only": unique,
              "complete_observable_module_rank_relation_truth_only": False}
    if not unique or not lattice or lattice.get("rank") is None:
        return result
    labels = [next(iter(modes)) for modes in distinct]
    signature = _lattice_signature(labels, row["physical_phase_rank_truth_only"])
    truth = row["observable_lattice_truth_only"]
    module_complete = signature["rank"] == truth["rank"] and signature["basis"] == truth["basis"]
    integer = np.asarray(labels, dtype=int)
    physical_basis = np.asarray(lattice["products"], dtype=int)@integer
    relation = np.array_equal(np.asarray(lattice["peak_labels"], dtype=int)@physical_basis, integer)
    result.update({"acquired_module_truth_only": signature,
                   "module_complete_truth_only": module_complete,
                   "physical_relation_valid_truth_only": relation,
                   "physical_basis_labels_truth_only": physical_basis.tolist(),
                   "complete_observable_module_rank_relation_truth_only": bool(
                       module_complete and relation and lattice["rank"] == truth["rank"])})
    return result


def evaluate(row: dict, source_row_index: int) -> dict:
    validation, generation = regenerate_observation(row)
    settings = row["settings"]
    times = np.arange(settings["n_train"], settings["n_train"]+settings["n_validation"])
    sigma_squared = row["inference"]["noise_estimate"]["sigma_squared"]
    accepted = [stage for stage in row["inference"]["stages"]
                if stage["status"].startswith("accepted_character")]
    tests = []
    previous = None
    for index, stage in enumerate(accepted):
        prior = frozen_prior_dictionary(previous, settings["n_train"])
        candidate = float(stage["candidate"]["frequency"])
        test = raw_validation_diagnostic(validation, times, candidate, prior, sigma_squared)
        test.update({"accepted_character_index": index+1, "source_stage_step": stage["step"],
                     "source_row_index": source_row_index,
                     "source_stage_status": stage["status"],
                     "candidate_training_audit": stage["candidate"],
                     "prior_training_dictionary_snapshot": prior,
                     "original_old_model_peak_frequencies": stage["old_model"].get("model_peak_frequencies"),
                     "candidate_frequency_after_current_joint_fit": stage["joint_frequency_fit"]["frequencies"][-1]})
        tests.append(test)
        previous = stage
    # Truth scoring begins only after every saved raw-data test is complete.
    assert abs(generation["generator_noise_sigma"]-row["applied_noise_sigma_truth_only"]) < 1e-12
    for test in tests:
        scoring = score_frequency(row["case"], test["candidate_frequency_before_current_joint_fit"],
                                   2*np.pi/settings["n_train"], CATALOG_HARMONIC_ORDER,
                                   settings["n_train"])
        fitted_scoring = score_frequency(row["case"], test["candidate_frequency_after_current_joint_fit"],
                                          2*np.pi/settings["n_train"], CATALOG_HARMONIC_ORDER,
                                          settings["n_train"])
        test["candidate_frequency_truth_score_only"] = scoring
        test["original_joint_fitted_frequency_truth_score_only"] = fitted_scoring
        test["original_joint_fitted_line_unmatched_in_bounded_catalog_truth_only"] = not fitted_scoring["matches"]
        test["candidate_unique_physical_label_truth_only"] = len({tuple(m["mode"]) for m in scoring["matches"]}) == 1
    first_rejected = next((i for i, test in enumerate(tests) if not test["accepted_by_raw_reference"]), len(tests))
    original_correct = bool(row["acquired_module_complete_truth_only"]
                             and row["r_candidate_matches_observable_truth_only"]
                             and row.get("inferred_integer_relation_matches_physical_labels_truth_only"))
    return {"case": row["case"], "noise": row["relative_sensor_noise"], "seed": row["seed"],
            "source_row_index": source_row_index, "accepted_stage_count": len(tests),
            "validation_regeneration": generation, "raw_tests": tests,
            "original_module_rank_relation_correct_truth_only": original_correct,
            "retained_prefix_length_before_first_rejection": first_rejected,
            "prefix_truth_scoring": score_prefix(row, accepted, first_rejected),
            "all_tests_follow_original_histories_not_counterfactual_refits": True}


def summarize(rows: list[dict]) -> dict:
    tests = [test for row in rows for test in row.get("raw_tests", [])]
    unmatched = [t for t in tests if t["original_joint_fitted_line_unmatched_in_bounded_catalog_truth_only"]]
    matched = [t for t in tests if not t["original_joint_fitted_line_unmatched_in_bounded_catalog_truth_only"]]
    return {"rows": len(rows), "accepted_stages_tested": len(tests),
            "raw_reference_accepted_stages": sum(t["accepted_by_raw_reference"] for t in tests),
            "raw_reference_rejected_stages": sum(not t["accepted_by_raw_reference"] for t in tests),
            "original_joint_fitted_unmatched_lines": len(unmatched),
            "original_joint_fitted_unmatched_lines_rejected": sum(not t["accepted_by_raw_reference"] for t in unmatched),
            "original_joint_fitted_catalog_matched_lines": len(matched),
            "original_joint_fitted_catalog_matched_lines_retained": sum(t["accepted_by_raw_reference"] for t in matched),
            "collinear_or_old_design_deficient_stages": sum(t["candidate_collinear_or_old_design_deficient"] for t in tests),
            "original_correct_rows": sum(r["original_module_rank_relation_correct_truth_only"] for r in rows),
            "correct_original_prefix_after_first_rejection_rows": sum(r["prefix_truth_scoring"]["complete_observable_module_rank_relation_truth_only"] for r in rows),
            "repaired_incorrect_rows": sum(not r["original_module_rank_relation_correct_truth_only"] and r["prefix_truth_scoring"]["complete_observable_module_rank_relation_truth_only"] for r in rows),
            "damaged_original_correct_rows": sum(r["original_module_rank_relation_correct_truth_only"] and not r["prefix_truth_scoring"]["complete_observable_module_rank_relation_truth_only"] for r in rows)}


def main() -> None:
    started = time.monotonic()
    source = json.loads(SOURCE.read_text())
    payload = {"description": __doc__, "assay_complete": False,
               "source": str(SOURCE), "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
               "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "p_gate": P_GATE, "nominal_chi_square_df": "2 * raw sensor count",
               "rows_expected": len(source["rows"]), "rows": [],
               "reference_calibrated_for_infinite_or_learned_residual_model": False,
               "candidate_frequencies_from_train_before_current_joint_fit": True,
               "old_dictionary_from_prior_train_stage_BIC_audit": True,
               "validation_amplitude_fits_allowed_no_frequency_or_order_reselection": True,
               "prefix_evaluation_is_retrospective_not_full_pipeline_rerun": True}
    _save(payload)
    for index, row in enumerate(source["rows"]):
        result = evaluate(row, index)
        payload["rows"].append(result)
        _save(payload)
        if (index+1)%10 == 0:
            print(f"SAVED {index+1}/{len(source['rows'])} rows", flush=True)
    payload["assay_complete"] = True
    payload["summary_overall"] = summarize(payload["rows"])
    payload["summary_by_noise"] = [{"noise": noise, **summarize([r for r in payload["rows"] if r["noise"] == noise])}
                                    for noise in source["noise_levels"]]
    payload["rejected_stage_records"] = [{"case": r["case"], "noise": r["noise"], "seed": r["seed"],
        "accepted_character_index": t["accepted_character_index"],
        "candidate_frequency": t["candidate_frequency_before_current_joint_fit"],
        "p_reference": t["chi_square_reference_p"],
        "original_fitted_line_unmatched_truth_only": t["original_joint_fitted_line_unmatched_in_bounded_catalog_truth_only"]}
        for r in payload["rows"] for t in r["raw_tests"] if not t["accepted_by_raw_reference"]]
    payload["unmatched_extra_stage_records"] = [{"case": r["case"], "noise": r["noise"], "seed": r["seed"],
        "accepted_character_index": t["accepted_character_index"],
        "candidate_frequency": t["candidate_frequency_before_current_joint_fit"],
        "joint_fitted_frequency": t["candidate_frequency_after_current_joint_fit"],
        "accepted_by_raw_reference": t["accepted_by_raw_reference"],
        "p_reference": t["chi_square_reference_p"],
        "old_residual_variance_over_noise_reference": t["old_residual_variance_over_noise_reference"]}
        for r in payload["rows"] for t in r["raw_tests"]
        if t["original_joint_fitted_line_unmatched_in_bounded_catalog_truth_only"]]
    payload["elapsed_seconds"] = time.monotonic()-started
    _save(payload)
    print(json.dumps(payload["summary_overall"], indent=2), flush=True)
    print(str(DESTINATION), flush=True)


if __name__ == "__main__":
    main()
