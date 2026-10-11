"""Independent post-freeze polynomial/readout audit; no fitting or selection.

The benchmark has been inspected before.  The observed-only audit of ALL six
archives is saved before opening the existing evaluator or its physical line
labels.  Clean-data calculations below are diagnostics, never method outputs.
"""

from project_paths import RESULTS_DIR

from datetime import datetime, timezone
from functools import lru_cache
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.linalg import lstsq, pinv


HERE = Path(__file__).resolve().parent
SOURCE = RESULTS_DIR / 'wick_character_assay.json'
PROTOCOL = RESULTS_DIR / 'wick_character_assay_protocol.json'
GROUP_AUDIT = RESULTS_DIR / 'ledger_group_selection_audit.json'
EVALUATION_MANIFEST = RESULTS_DIR / 'fresh_phase_pipeline.json'
OUTPUT = RESULTS_DIR / 'wick_character_assay_audit.json'
ROOT = RESULTS_DIR / 'wick_character_assay_audit'
READOUT_RCOND = 1e-10
NUMERICAL_AUDIT_RELATIVE_TOLERANCE = 1e-7


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_archive(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def save(payload):
    temporary = OUTPUT.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(OUTPUT)


def max_difference(first, second):
    return float(np.max(abs(np.asarray(first) - np.asarray(second)), initial=0.))


def complex_pair(value):
    value = complex(value)
    return [value.real, value.imag]


def close_scaled(first, second):
    scale = max(1., float(np.max(abs(np.asarray(second)), initial=0.)))
    return max_difference(first, second) <= NUMERICAL_AUDIT_RELATIVE_TOLERANCE * scale


def origin_bank(parameters, length):
    """Real cos/sin expansion, independent of the runner's complex assembly."""
    nu = parameters["frequencies"]
    count = len(nu)
    coefficient = parameters["coefficients_sensor_center"]
    center = float(parameters["time_center"])
    cosine, sine = np.cos(nu * center)[:, None], np.sin(nu * center)[:, None]
    profile_cos = coefficient[1:1 + count] * cosine - coefficient[1 + count:] * sine
    profile_sin = coefficient[1:1 + count] * sine + coefficient[1 + count:] * cosine
    root = parameters["spatial_mass_sqrt"]
    mass_cos = np.einsum("qi,ji->jq", root, profile_cos, optimize=False)
    mass_sin = np.einsum("qi,ji->jq", root, profile_sin, optimize=False)
    angles = np.arange(length)[:, None] * nu[None, :]
    cosine, sine = np.cos(angles), np.sin(angles)
    bank_cos = (np.einsum("lj,jq->lqj", cosine, mass_cos, optimize=False)
                + np.einsum("lj,jq->lqj", sine, mass_sin, optimize=False))
    bank_sin = (-np.einsum("lj,jq->lqj", sine, mass_cos, optimize=False)
                + np.einsum("lj,jq->lqj", cosine, mass_sin, optimize=False))
    bank = np.concatenate((bank_cos, bank_sin), axis=2).reshape(length * root.shape[0], 2 * count) / np.sqrt(length)
    independent_origin = (profile_cos - 1j * profile_sin) / 2
    return bank, max_difference(independent_origin, parameters["complex_profiles_sensor_origin"])


def observed_windows(raw, mean, length):
    centered = (raw - mean[:, None]).T
    return np.lib.stride_tricks.sliding_window_view(centered, length, axis=0).transpose(0, 2, 1)


def independent_readout(parameters, raw, sigma, length):
    bank, origin_error = origin_bank(parameters, length)
    root = parameters["spatial_mass_sqrt"]
    windows = observed_windows(raw, parameters["coefficients_sensor_center"][0], length)
    mass_windows = np.einsum("qi,nli->nlq", root, windows, optimize=False) / np.sqrt(length)
    h = mass_windows.reshape(len(windows), -1)
    # Gelsd independently solves every measured window; no frozen phases enter.
    solution, _, rank, singular = lstsq(bank, h.T, cond=READOUT_RCOND, lapack_driver="gelsd")
    if rank != bank.shape[1]:
        raise np.linalg.LinAlgError("independent full-bank inverse unresolved; no truncation allowed")
    inverse, inverse_rank = pinv(bank, rtol=READOUT_RCOND, return_rank=True)
    if inverse_rank != rank:
        raise np.linalg.LinAlgError("independent inverse rank mismatch")
    raw_readout = np.einsum("als,si->ali", inverse.reshape(rank, length, root.shape[0]), root, optimize=False) / np.sqrt(length)
    # Full mass-window covariance, not an orthogonal-frequency approximation.
    blocks = inverse.reshape(rank, length, root.shape[0])
    covariance = sigma ** 2 / length * np.einsum("ali,ij,blj->ab", blocks,
        parameters["physical_mass"], blocks, optimize=False)
    covariance_raw = sigma ** 2 * np.einsum("ali,bli->ab", raw_readout, raw_readout, optimize=False)
    identity = np.einsum("ap,pb->ab", inverse, bank, optimize=False)
    return solution.T, covariance, raw_readout, dict(
        rank=int(rank), columns=bank.shape[1], singular_values=singular.tolist(),
        center_to_origin_profile_max_difference=origin_error,
        inverse_identity_max_difference=max_difference(identity, np.eye(rank)),
        mass_vs_raw_covariance_max_difference=max_difference(covariance, covariance_raw),
        bank_row_layout="lag-major,sensor-minor", bank_column_layout="allcos,allsin",
        mean_is_frozen_TRAIN_sensor_intercept=True, known_clock_used_for_readout=False)


@lru_cache(maxsize=None)
def partial_internal_matchings(positions):
    """Explicit pair/single partitions; no subset-value Wick recurrence."""
    if not positions:
        return (((), ()),)
    first, remaining = positions[0], positions[1:]
    result = [((first,) + singles, pairs) for singles, pairs in partial_internal_matchings(remaining)]
    for index, second in enumerate(remaining):
        rest = remaining[:index] + remaining[index + 1:]
        result.extend((singles, ((first, second),) + pairs)
                      for singles, pairs in partial_internal_matchings(rest))
    return tuple(result)


def explicit_products(beta, covariance, relation):
    count = beta.shape[1] // 2
    factors = [(j, 1 if power > 0 else -1) for j, power in enumerate(relation) for _ in range(abs(int(power)))]
    degree = len(factors)
    if not 0 < degree <= 8:
        raise ValueError("fixed nonzero degree <=8 required")
    values = np.column_stack([beta[:, j] + 1j * sign * beta[:, count + j] for j, sign in factors])
    contractions = np.empty((degree, degree), dtype=complex)
    for first, (j, sign_first) in enumerate(factors):
        for second, (k, sign_second) in enumerate(factors):
            contractions[first, second] = (covariance[j, k] - sign_first * sign_second * covariance[count + j, count + k]
                + 1j * (sign_second * covariance[j, count + k] + sign_first * covariance[count + j, k]))
    corrected = np.zeros(len(beta), dtype=complex)
    matchings = partial_internal_matchings(tuple(range(degree)))
    for singles, pairs in matchings:
        term = np.prod(values[:, singles], axis=1) if singles else np.ones(len(beta), dtype=complex)
        for first, second in pairs:
            term = term * (-contractions[first, second])
        corrected += term
    return np.prod(values, axis=1), corrected, len(matchings)


def polynomial_audit(entry, row, parameters, saved, beta, covariance, length):
    raw_products, corrected_products, relation_checks = [], [], []
    if len(entry["relations"]) != len(row["relations"]) or len(row["relations"]) != len(saved["relation_rows"]):
        raise ValueError("fixed hypothesis count changed")
    count_windows = len(beta)
    overlap = min(length - 1, count_windows - 1)
    averaging_factor = (count_windows * (1 + 2 * overlap) - overlap * (overlap + 1)) / count_windows ** 2
    for index, (proposal, reported) in enumerate(zip(entry["relations"], row["relations"])):
        relation = proposal["row"]
        if (relation != reported["row"] or relation != saved["relation_rows"][index].tolist()
                or reported["relation_accepted"] or reported["relation_rejected"]):
            raise ValueError("frozen relation or nonselection scope changed")
        raw, corrected, matchings = explicit_products(beta, covariance, relation)
        raw_products.append(raw); corrected_products.append(corrected)
        clock = np.exp(1j * np.einsum("nj,j->n", saved["starts"][:, None] * parameters["frequencies"], relation, optimize=False))
        mean_upper = reported["pointwise_variance_upper_bound"] * averaging_factor
        relation_checks.append(dict(proposal_index=proposal["proposal_index"], row=relation, degree=proposal["degree"],
            enumerated_internal_matchings=matchings,
            raw_polynomial_max_difference=max_difference(raw, saved["raw_products"][index]),
            corrected_polynomial_max_difference=max_difference(corrected, saved["corrected_products"][index]),
            corrected_polynomial_check_passed=close_scaled(corrected, saved["corrected_products"][index]),
            reported_corrected_mean_max_difference=max_difference(np.asarray(complex_pair(np.mean(corrected))), reported["corrected_mean"]),
            free_clock_reference_modulus_difference=abs(abs(np.mean(clock)) - reported["free_clock_mean_modulus"]),
            overlap_variance_formula_difference=abs(mean_upper - reported["overlapping_mean_variance_upper_bound"]),
            overlap_factor=averaging_factor))
    return np.asarray(raw_products), np.asarray(corrected_products), relation_checks


def observed_only(entry, row, length):
    for path, expected in ((entry["input"], entry["input_sha256"]),
            (entry["parameters"], entry["parameters_sha256"]), (row["archive"], row["archive_sha256"])):
        if digest(path) != expected:
            raise ValueError("immutable source/input/assay archive changed")
    parameters, saved = load_archive(entry["parameters"]), load_archive(row["archive"])
    public = load_archive(entry["input"])
    if set(public) != {"Y", "s", "t"}:
        raise ValueError("only public Y,s,t input schema required")
    first = entry["n_train"] + entry["n_validation"]
    last = first + entry["n_future"]
    starts = np.arange(first, last - length + 1)
    if (not np.array_equal(public["t"][first:last], np.arange(first, last))
            or not np.array_equal(saved["starts"], starts)
            or not np.array_equal(public["s"], parameters["sensor_positions"])):
        raise ValueError("future interval/positions/lag starts changed")
    beta, covariance, readout, bank = independent_readout(parameters, public["Y"][:, first:last], entry["sigma_estimate"], length)
    raw, corrected, relations = polynomial_audit(entry, row, parameters, saved, beta, covariance, length)
    path = ROOT / (entry["run"] + ".npz")
    np.savez_compressed(path, beta_independent=beta, covariance_independent=covariance,
        readout_raw_independent=readout, raw_products_independent=raw,
        corrected_products_independent=corrected, starts=starts, relation_rows=saved["relation_rows"])
    bank.update(beta_lstsq_max_difference=max_difference(beta, saved["beta"]),
        covariance_max_difference=max_difference(covariance, saved["covariance"]),
        readout_raw_max_difference=max_difference(readout, saved["readout_raw"]),
        beta_lstsq_check_passed=close_scaled(beta, saved["beta"]),
        covariance_check_passed=close_scaled(covariance, saved["covariance"]))
    return dict(run=entry["run"], status="observed_only_numerical_audit_frozen", bank=bank,
        relations=relations, numerical_archive=str(path), numerical_archive_sha256=digest(path),
        future_observation_interval=[first, last], starts=int(len(starts)),
        original_assay_archive_sha256=row["archive_sha256"], public_input_sha256=entry["input_sha256"],
        frozen_parameters_sha256=entry["parameters_sha256"]), (parameters, saved, beta, covariance, readout)


def postfreeze_diagnosis(check, entry, row, bundle, physical_row, evaluation_entry, length):
    if evaluation_entry["input_sha256"] != entry["input_sha256"]:
        raise ValueError("evaluation/public archive identity mismatch")
    if digest(evaluation_entry["evaluation"]) != evaluation_entry["evaluation_sha256"]:
        raise ValueError("independent evaluator changed")
    parameters, saved, beta, covariance, readout = bundle
    truth = load_archive(evaluation_entry["evaluation"])
    physical = physical_row["physical_basis_P_truth_only"]
    count = len(parameters["frequencies"])
    if len(physical) != count:
        raise ValueError("old postfreeze physical line order mismatch")
    old_frequencies = np.asarray([item["frequency"] for item in physical_row["raw_catalog_matching"]])
    if not np.array_equal(old_frequencies, parameters["frequencies"]):
        raise ValueError("physical-label audit does not match the unchanged free-bank lines")
    first = entry["n_train"] + entry["n_validation"]
    last = first + entry["n_future"]
    clean_windows = observed_windows(truth["clean"][:, first:last], parameters["coefficients_sensor_center"][0], length)
    clean_beta = np.einsum("ali,nli->na", readout, clean_windows, optimize=False)
    true_sigma = float(truth["sigma"])
    covariance_true = covariance * (true_sigma / entry["sigma_estimate"]) ** 2
    clean_z = clean_beta[:, :count] + 1j * clean_beta[:, count:]
    true_rotations = np.einsum("ij,j->i", np.asarray(physical, dtype=float), truth["alpha"], optimize=False)
    true_unit_z = np.exp(1j * saved["starts"][:, None] * true_rotations[None, :])
    relations = []
    clean_products, true_noise_corrected, plug_in_expected = [], [], []
    for index, (proposal, reported) in enumerate(zip(entry["relations"], row["relations"])):
        relation = proposal["row"]
        physical_label = [sum(int(relation[j]) * int(physical[j][k]) for j in range(count))
                          for k in range(len(physical[0]))]
        genuine = all(value == 0 for value in physical_label)
        clean_product, _, _ = explicit_products(clean_beta, np.zeros_like(covariance), relation)
        _, actual_known_noise_corrected, _ = explicit_products(beta, covariance_true, relation)
        # E[H_Sigmahat(mu+eta_true)]=H_(Sigmahat-Sigmatrue)(mu).
        _, expected_plugin, _ = explicit_products(clean_beta, covariance - covariance_true, relation)
        mean_clean = complex(np.mean(clean_product))
        mean_raw = complex(np.mean(saved["raw_products"][index]))
        mean_corrected = complex(np.mean(saved["corrected_products"][index]))
        mean_true_corrected = complex(np.mean(actual_known_noise_corrected))
        mean_plugin_expected = complex(np.mean(expected_plugin))
        ideal_true = np.exp(1j * saved["starts"] * float(np.einsum("i,i->", physical_label, truth["alpha"], optimize=False)))
        ideal_clock = np.exp(1j * saved["starts"] * float(np.einsum("i,i->", relation, parameters["frequencies"], optimize=False)))
        normalized_clean_product = np.ones(len(clean_beta), dtype=complex)
        for coordinate, power in enumerate(relation):
            if power:
                if np.any(abs(clean_z[:, coordinate]) == 0):
                    normalized_clean_product = None
                    break
                phase_coordinate = clean_z[:, coordinate] / abs(clean_z[:, coordinate])
                normalized_clean_product *= phase_coordinate ** int(power)
        relations.append(dict(proposal_index=proposal["proposal_index"], row=relation, degree=proposal["degree"],
            physically_valid_relation_truth_only=genuine, physical_resultant_label_truth_only=physical_label,
            clean_linear_readout_polynomial_mean=complex_pair(mean_clean),
            ideal_true_origin_unit_character_mean=complex_pair(np.mean(ideal_true)),
            ideal_true_character_finite_record_coherence=abs(complex(np.mean(ideal_true))),
            ideal_free_clock_unit_character_mean=complex_pair(np.mean(ideal_clock)),
            clean_normalized_phase_coherence_diagnostic_only=(abs(complex(np.mean(normalized_clean_product)))
                if normalized_clean_product is not None else None),
            clean_phase_normalization_used_for_polynomial_or_selection=False,
            clean_mean_modulus=abs(mean_clean), corrected_mean_modulus=abs(mean_corrected),
            raw_mean_error_against_clean_readout=abs(mean_raw - mean_clean),
            corrected_mean_error_against_clean_readout=abs(mean_corrected - mean_clean),
            known_true_noise_corrected_mean_error_against_clean_readout=abs(mean_true_corrected - mean_clean),
            correction_mean_magnitude=abs(mean_corrected - mean_raw),
            clean_polynomial_mean_model_bias_vs_ideal_clock=abs(mean_clean - np.mean(ideal_clock)),
            clean_polynomial_mean_model_bias_vs_ideal_true_character=abs(mean_clean - np.mean(ideal_true)),
            clean_polynomial_rms_model_bias_vs_ideal_true_character=float(np.sqrt(np.mean(abs(clean_product - ideal_true) ** 2))),
            plugin_covariance_expectation_mean=complex_pair(mean_plugin_expected),
            plugin_covariance_expectation_bias=abs(mean_plugin_expected - mean_clean),
            corrected_mean_error_against_plugin_expectation=abs(mean_corrected - mean_plugin_expected),
            conditional_reported_mean_std_reference=reported["conditional_mean_std_upper_bound"],
            corrected_departure_q1_unclipped=2 - 2 * mean_corrected.real,
            clean_departure_q1_unclipped=2 - 2 * mean_clean.real,
            ideal_true_unit_departure_q1=2 - 2 * float(np.real(np.mean(ideal_true))),
            correction_reduced_realized_clean_mean_error=abs(mean_corrected - mean_clean) < abs(mean_raw - mean_clean),
            no_relation_accepted_or_rejected=True))
        clean_products.append(clean_product); true_noise_corrected.append(actual_known_noise_corrected)
        plug_in_expected.append(expected_plugin)
    evaluation_path = ROOT / (entry["run"] + "__postfreeze_diagnosis.npz")
    np.savez_compressed(evaluation_path, clean_beta=clean_beta, true_unit_z_origin=true_unit_z,
        covariance_true_noise=covariance_true, clean_products=np.asarray(clean_products),
        corrected_products_known_true_noise=np.asarray(true_noise_corrected),
        plugin_corrected_expectation_given_clean=np.asarray(plug_in_expected))
    check.update(status="finished", physical_basis_P_truth_only=physical,
        evaluation_sha256=evaluation_entry["evaluation_sha256"],
        evaluation_diagnostic_archive=str(evaluation_path), evaluation_diagnostic_archive_sha256=digest(evaluation_path),
        sigma_estimated=entry["sigma_estimate"], sigma_true_evaluation_only=true_sigma,
        estimated_noise_variance_over_true_noise_variance=(entry["sigma_estimate"] / true_sigma) ** 2,
        clean_coordinate_modulus_mean=np.mean(abs(clean_z), axis=0).tolist(),
        clean_coordinate_modulus_rms_departure_from_one=np.sqrt(np.mean((abs(clean_z) - 1) ** 2, axis=0)).tolist(),
        clean_coordinate_modulus_max_departure_from_one=np.max(abs(abs(clean_z) - 1), axis=0).tolist(),
        clean_coordinate_unit_circle_condition_exactly_valid=False,
        observed_readout_noise_empirical_squared_norm_mean=float(np.mean(np.sum((beta - clean_beta) ** 2, axis=1))),
        true_noise_readout_covariance_trace=float(np.trace(covariance_true)),
        relations_postfreeze=relations,
        interpretation="Wick corrects additive known Gaussian noise contractions of the ACTUAL linear readout polynomial; amplitude/tail/profile/rotation errors survive",
        physical_labels_used_only_after_all_observed_audits_frozen=True,
        ideal_origin_gauge="matched physical sampled rotations times absolute window START; initial physical phase is carried by frozen profile, no heldout gauge fitted")


def aggregate(rows):
    relations = [item for row in rows for item in row["relations_postfreeze"]]
    numerical = [item for row in rows for item in row["relations"]]
    result = dict(rows=len(rows), fixed_hypotheses=len(relations),
        all_polynomial_checks_passed=all(item["corrected_polynomial_check_passed"] for item in numerical),
        all_bank_readout_checks_passed=all(row["bank"]["beta_lstsq_check_passed"] and row["bank"]["covariance_check_passed"] for row in rows),
        maximum_corrected_polynomial_difference=max(item["corrected_polynomial_max_difference"] for item in numerical),
        maximum_beta_lstsq_difference=max(row["bank"]["beta_lstsq_max_difference"] for row in rows),
        maximum_covariance_difference=max(row["bank"]["covariance_max_difference"] for row in rows),
        maximum_overlap_variance_formula_difference=max(item["overlap_variance_formula_difference"] for item in numerical))
    for name, subset in (("all", relations), ("genuine", [r for r in relations if r["physically_valid_relation_truth_only"]]),
                         ("false", [r for r in relations if not r["physically_valid_relation_truth_only"]])):
        result[name] = dict(count=len(subset),
            corrected_modulus_above_one=sum(r["corrected_mean_modulus"] > 1 for r in subset),
            correction_reduced_realized_clean_mean_error=sum(r["correction_reduced_realized_clean_mean_error"] for r in subset),
            mean_correction_magnitude=float(np.mean([r["correction_mean_magnitude"] for r in subset])) if subset else None,
            maximum_correction_magnitude=max((r["correction_mean_magnitude"] for r in subset), default=None),
            median_raw_clean_mean_error=float(np.median([r["raw_mean_error_against_clean_readout"] for r in subset])) if subset else None,
            median_corrected_clean_mean_error=float(np.median([r["corrected_mean_error_against_clean_readout"] for r in subset])) if subset else None,
            median_clean_model_bias_vs_ideal_true_character=float(np.median([r["clean_polynomial_mean_model_bias_vs_ideal_true_character"] for r in subset])) if subset else None)
    return result


def main(confirm_all_frozen=False):
    if not confirm_all_frozen:
        raise RuntimeError("explicit root all-six terminal freeze gate required")
    if OUTPUT.exists() or ROOT.exists():
        raise FileExistsError("preserve any previous completed audit")
    payload, prepared = json.loads(SOURCE.read_text()), json.loads(PROTOCOL.read_text())
    if (payload["status"] != "finished_measured_window_assay" or len(payload["rows"]) != 6
            or not payload["all_six_outputs_frozen_before_new_evaluation"] or payload["evaluation_files_opened"]
            or not payload["source_input_hashes_unchanged"] or payload["relation_count"] != 122):
        raise ValueError("all six fixed public-only assay outputs must be frozen")
    source_hash, protocol_hash, audit_hash = digest(SOURCE), digest(PROTOCOL), digest(__file__)
    if payload["protocol_sha256"] != protocol_hash:
        raise ValueError("prepared protocol changed")
    for key in ("source_sha256", "settings", "source_fingerprints", "manifest", "relation_count"):
        if payload[key] != prepared[key]:
            raise ValueError("assay declaration changed")
    if digest(payload["source"]) != payload["source_sha256"]:
        raise ValueError("original group result changed")
    if any(digest(HERE / name) != expected for name, expected in payload["source_fingerprints"].items()):
        raise ValueError("frozen assay sources changed")
    ROOT.mkdir()
    rows, bundles = [], []
    length = payload["settings"]["window"]
    for entry, row in zip(payload["manifest"], payload["rows"]):
        if entry["run"] != row["run"]:
            raise ValueError("manifest order changed")
        check, bundle = observed_only(entry, row, length)
        rows.append(check); bundles.append(bundle)
    result = dict(status="all_six_observed_only_audits_frozen_before_evaluation", created_at=datetime.now(timezone.utc).isoformat(),
        source=str(SOURCE), source_sha256=source_hash, protocol_sha256=protocol_hash,
        audit_source_sha256=audit_hash, rows=rows, evaluation_files_opened=False,
        explicit_root_all_six_terminal_gate_received=True,
        all_six_observed_only_numerical_outputs_frozen_before_evaluation=True,
        no_fitting_no_selection_no_parameter_changes=True,
        previously_examined_benchmark_not_fresh_external_validation=True,
        confidence_interval_or_exact_group_certificate_claimed=False,
        numerical_audit_relative_tolerance=NUMERICAL_AUDIT_RELATIVE_TOLERANCE)
    save(result)
    physical_audit = json.loads(GROUP_AUDIT.read_text())
    evaluation_manifest = json.loads(EVALUATION_MANIFEST.read_text())
    if physical_audit["source_sha256"] != payload["source_sha256"]:
        raise ValueError("physical-label audit was made for a different frozen group source")
    physical_by_run = {row["run"]: row for row in physical_audit["rows"]}
    evaluation_by_run = {row["run"]: row for row in evaluation_manifest["manifest"]}
    result.update(group_physical_labels_audit_sha256=digest(GROUP_AUDIT),
        evaluation_manifest_sha256=digest(EVALUATION_MANIFEST), evaluation_files_opened=True)
    for check, entry, row, bundle in zip(rows, payload["manifest"], payload["rows"], bundles):
        postfreeze_diagnosis(check, entry, row, bundle, physical_by_run[entry["run"]], evaluation_by_run[entry["run"]], length)
    preserved = dict(assay_result_unchanged=digest(SOURCE) == source_hash,
        protocol_unchanged=digest(PROTOCOL) == protocol_hash, audit_source_unchanged=digest(__file__) == audit_hash,
        old_group_result_unchanged=digest(payload["source"]) == payload["source_sha256"],
        all_assay_sources_unchanged=all(digest(HERE / name) == expected for name, expected in payload["source_fingerprints"].items()),
        all_public_inputs_unchanged=all(digest(entry["input"]) == entry["input_sha256"] for entry in payload["manifest"]),
        all_parameters_unchanged=all(digest(entry["parameters"]) == entry["parameters_sha256"] for entry in payload["manifest"]),
        all_original_archives_unchanged=all(digest(row["archive"]) == row["archive_sha256"] for row in payload["rows"]),
        all_numerical_audit_archives_unchanged=all(digest(row["numerical_archive"]) == row["numerical_archive_sha256"] for row in rows),
        physical_audit_unchanged=digest(GROUP_AUDIT) == result["group_physical_labels_audit_sha256"],
        evaluation_manifest_unchanged=digest(EVALUATION_MANIFEST) == result["evaluation_manifest_sha256"],
        all_evaluation_archives_unchanged=all(digest(evaluation_by_run[row["run"]]["evaluation"]) == row["evaluation_sha256"] for row in rows))
    if not all(preserved.values()):
        raise ValueError("immutable artifact changed during independent audit")
    result.update(status="finished", preserved_hash_checks=preserved, summary=aggregate(rows))
    save(result)
    print(json.dumps(dict(output=str(OUTPUT), summary=result["summary"])), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm-all-six-frozen", action="store_true")
    main(parser.parse_args().confirm_all_six_frozen)
