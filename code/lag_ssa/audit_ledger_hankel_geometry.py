"""Independent, read-only artifact audit of the frozen 59 geometry rows.

No sampler, lattice proposal generator, model fitter or evaluator is called.
Only reported witness points/directions are recomputed, in the ORIGINAL mass
lag bank with independent SciPy SVD/null-space routines and explicit einsum.
"""

from project_paths import RESULTS_DIR

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.linalg import null_space, svd
from sympy import Matrix, Rational


HERE = Path(__file__).resolve().parent
SOURCE = RESULTS_DIR / 'ledger_hankel_geometry.json'
OUTPUT = RESULTS_DIR / 'ledger_hankel_geometry_audit.json'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_parameters(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def exact_matrix(rows, count, values):
    return Matrix(rows, count, [int(value) for row in values for value in row])


def exact_model_checks(parameters, presentation, count):
    rank = presentation["relation_rank"]
    relation = exact_matrix(rank, count, presentation["relation_rows_HNF"])
    left = exact_matrix(rank, rank, presentation["U"])
    right = exact_matrix(count, count, presentation["V"])
    inverse = exact_matrix(count, count, presentation["V_inverse"])
    diagonal = Matrix.zeros(rank, count)
    for index, value in enumerate(presentation["smith_diagonal"]):
        diagonal[index, index] = int(value)
    checks = dict(exact_Smith_identity=left * relation * right == diagonal,
        exact_unimodular_transforms=abs(int(left.det())) == 1 and abs(int(right.det())) == 1,
        exact_inverse=right * inverse == Matrix.eye(count),
        exact_relation_rank=relation.rank() == rank,
        continuous_dimension_consistent=presentation["continuous_dimension"] == count - rank)
    for key, expected in (("relation_rows_HNF_integer_text", relation), ("U_integer_text", left),
                          ("V_integer_text", right), ("V_inverse_integer_text", inverse)):
        values = parameters[key]
        checks[key + "_matches"] = bool(values.shape == expected.shape and
            [int(value) for value in values.flat] == [int(value) for value in expected])
    return relation, checks


def exact_lattice_checks(relation, information):
    count, rank = relation.cols, relation.rows
    points = information.get("points", [])
    if not points:
        return dict(point_count=0, all_exact_checks_passed=True, points=[])
    if rank in (0, count):
        raise ValueError("nontrivial connected normal lattice points in an unsupported rank")
    normal = relation.T * (relation * relation.T).inv() * relation
    tangent = Matrix.eye(count) - normal
    answers, seen = [], set()
    for point in points:
        preimage = Matrix([int(value) for value in point["integer_preimage"]])
        projected = Matrix([Rational(value) for value in point["projected_normal_turns"]])
        wrapped = Matrix([Rational(value) for value in point["wrapped_phase_turns"]])
        winding = relation * wrapped
        key = tuple(wrapped)
        checks = dict(normal_preimage_matches=normal * preimage == projected,
            wrapped_plus_normal_is_integer=all(value.is_Integer for value in wrapped + projected),
            relation_times_wrapped_is_integer=all(value.is_Integer for value in winding),
            exact_windings_match=[int(value) for value in winding] == point["exact_membership_windings"],
            identity_component_lift=relation * tangent * preimage == Matrix.zeros(rank, 1),
            connected_lift_differs_by_integer=all(value.is_Integer for value in tangent * preimage - wrapped),
            normal_length_matches=(projected.T * projected)[0] == Rational(point["unwrapped_normal_length_in_turns_squared"]),
            wrapped_nonidentity=any(wrapped), wrapped_distinct=key not in seen,
            exact_half_open_wrap=all(-Rational(1, 2) <= value < Rational(1, 2) for value in wrapped))
        seen.add(key)
        answers.append(dict(basis_index=point["basis_index"], **checks))
    return dict(point_count=len(points), points=answers,
        all_exact_checks_passed=all(all(value for key, value in point.items() if key != "basis_index") for point in answers),
        pair_count_matches=information.get("pair_count") == information.get("base_points") * len(points))


def original_bank(parameters, settings):
    frequencies = np.asarray(parameters["frequencies"], dtype=float)
    count = len(frequencies)
    coefficient = parameters["coefficients_sensor_center"]
    center = float(parameters["time_center"])
    profile_center = (coefficient[1:count + 1] - 1j * coefficient[count + 1:]) / 2
    profile_origin = profile_center * np.exp(-1j * frequencies * center)[:, None]
    mass_root, mass = parameters["spatial_mass_sqrt"], parameters["physical_mass"]
    profile_mass = np.einsum("ij,mj->mi", mass_root, profile_origin, optimize=False)
    lag_rotation = np.exp(1j * np.arange(settings["window"])[:, None] * frequencies[None, :])
    complex_columns = (2 / np.sqrt(settings["window"])) * np.einsum(
        "lm,mq->lqm", lag_rotation, profile_mass, optimize=False)
    bank = np.concatenate((complex_columns.real, -complex_columns.imag), axis=2).reshape(
        settings["window"] * len(mass), 2 * count)
    return bank, profile_origin, profile_mass, mass_root, mass


def field_tangent(bank, domain_tangent, phase):
    dimension = domain_tangent.shape[1]
    if not dimension:
        return np.empty((len(bank), 0)), np.empty(0)
    derivative = np.vstack((-np.sin(phase)[:, None] * domain_tangent,
                             np.cos(phase)[:, None] * domain_tangent))
    jacobian = np.einsum("ij,jk->ik", bank, derivative, optimize=False)
    basis, singular, _ = svd(jacobian, full_matrices=False, lapack_driver="gesvd")
    return basis, singular


def remove_tangent(vector, tangent):
    coordinate = np.einsum("ij,i->j", tangent, vector, optimize=False)
    return vector - np.einsum("ij,j->i", tangent, coordinate, optimize=False)


def audit_witness(bank, domain_tangent, witness, settings):
    if witness is None:
        return None
    first = np.asarray(witness["phase_first"])
    tangent, tangent_singular = field_tangent(bank, domain_tangent, first)
    valid = bool(not len(tangent_singular) or tangent_singular[-1] >
                 settings["tangent_rank_rcond"] * tangent_singular[0])
    if "phase_second" in witness:
        second = np.asarray(witness["phase_second"])
        delta = np.r_[np.cos(second) - np.cos(first), np.sin(second) - np.sin(first)]
        chord = np.einsum("ij,j->i", bank, delta, optimize=False)
        normal = remove_tangent(chord, tangent)
        square, normal_length = float(np.sum(chord ** 2)), float(np.linalg.norm(normal))
        upper = square / (2 * normal_length)
        answer = dict(kind="reported original-window chord only", recomputed_upper=upper,
            chord_distance=float(np.sqrt(square)), normal_distance=normal_length,
            chord_distance_difference=float(np.sqrt(square) - witness["chord_distance"]),
            normal_distance_difference=float(normal_length - witness["normal_distance"]))
    elif witness["source"] == "analytic directional normal-curvature limit":
        direction = np.asarray(witness["phase_direction"])
        first_derivative = np.einsum("ij,j->i", bank,
            np.r_[-np.sin(first) * direction, np.cos(first) * direction], optimize=False)
        second_derivative = np.einsum("ij,j->i", bank,
            np.r_[-np.cos(first) * direction ** 2, -np.sin(first) * direction ** 2], optimize=False)
        normal = remove_tangent(second_derivative, tangent)
        upper = float(np.sum(first_derivative ** 2) / np.linalg.norm(normal))
        answer = dict(kind="reported original-window analytic direction only", recomputed_upper=upper)
    else:
        raise ValueError("unsupported saved witness kind")
    answer.update(tangent_rank_valid_at_declared_tolerance=valid,
        upper_absolute_difference=abs(upper - witness["upper"]),
        upper_relative_difference=abs(upper - witness["upper"]) / max(abs(witness["upper"]), 1e-30),
        no_new_points_directions_or_search=True)
    return answer


def audit_row(row, model, settings):
    parameters = load_parameters(model["parameters"])
    original, origin, mass_profiles, mass_root, mass = original_bank(parameters, settings)
    count = len(parameters["frequencies"])
    relation, exact = exact_model_checks(parameters, model["presentation"], count)
    q, singular, vt = svd(original, full_matrices=False, lapack_driver="gesvd")
    thin = singular[:, None] * vt
    reconstruction = np.einsum("ij,jk->ik", q, thin, optimize=False)
    gram = np.einsum("ij,ik->jk", original, original, optimize=False)
    thin_gram = np.einsum("ij,ik->jk", thin, thin, optimize=False)
    numerical_rank = int(np.sum(singular > settings["bank_rank_rcond"] * singular[0]))
    q_blocks = q.reshape(settings["window"], len(mass), len(singular))
    weighted = np.einsum("ij,ljk->lik", mass, q_blocks, optimize=False)
    sigma = float(model["noise_estimate_train_only"]["sigma"])
    covariance = sigma ** 2 / settings["window"] * np.einsum("lai,laj->ij", q_blocks, weighted, optimize=False)
    covariance = (covariance + covariance.T) / 2
    adjoint_noise_root = np.einsum("ba,lbk->lak", mass_root, q_blocks, optimize=False)
    factor_covariance = sigma ** 2 / settings["window"] * np.einsum(
        "lai,laj->ij", adjoint_noise_root, adjoint_noise_root, optimize=False)
    eigenvalues = np.linalg.eigvalsh(covariance)
    radius = float(np.sqrt(np.trace(covariance)) + np.sqrt(
        2 * eigenvalues[-1] * np.log(1 / settings["one_window_alpha"])))
    if relation.rows:
        floating_relation = np.asarray(relation.tolist(), dtype=float)
        domain_tangent = null_space(floating_relation, rcond=settings["relation_rank_rcond"])
        _, relation_singular, _ = svd(floating_relation, full_matrices=False, lapack_driver="gesvd")
        resolved_rank = int(np.sum(relation_singular > settings["relation_rank_rcond"] * relation_singular[0]))
        tangent_residual = float(np.linalg.norm(np.einsum(
            "ij,jk->ik", floating_relation, domain_tangent, optimize=False)))
    else:
        domain_tangent = np.eye(count)
        resolved_rank, tangent_residual = 0, 0.
    reported = row["diagnostic"]
    stored_bank = reported["bank"]
    witness = audit_witness(original, domain_tangent, reported["best_witness"], settings)
    upper = witness["recomputed_upper"] if witness else None
    signed = np.r_[parameters["frequencies"], -parameters["frequencies"]]
    gaps = abs(np.angle(np.exp(1j * (signed[:, None] - signed[None, :]))))
    np.fill_diagonal(gaps, np.inf)
    theoretical = bool(settings["window"] >= 2 * count and np.all(np.linalg.norm(mass_profiles, axis=1) > 0)
        and np.min(gaps) > 1e-12 and np.all(abs(np.angle(np.exp(1j * signed))) > 1e-12))
    resolved = bool(numerical_rank == 2 * count and resolved_rank == relation.rows)
    ratio = upper / (radius + settings["model_tail_bound"]) if upper is not None else None
    rejection = bool(resolved and theoretical and upper is not None and upper <= radius + settings["model_tail_bound"])
    status = ("conditional_model_only_rejection_witness" if rejection else "bounded_search_no_rejection_found"
              if resolved and theoretical else "unresolved_numerical_embedding_witness")
    flags = dict(rejection_matches=rejection == reported["model_only_noise_scale_rejection_numerical"],
        geometry_status_matches=status == reported["geometry_status"],
        numerical_bank_rank_matches=numerical_rank == stored_bank["bank_numerical_rank"],
        relation_rank_matches=resolved_rank == stored_bank["relation_numerical_rank"],
        all_columns_kept=stored_bank["all_columns_kept_in_thin_bank"] and stored_bank["no_svd_truncation_used_for_distances"],
        no_positive_acceptance_certificate=reported["no_positive_acceptance_certificate"],
        no_global_minimum_or_lower_bound=not reported["reach_lower_bound_or_global_minimum_established"],
        no_validated_floating_certificate=not reported["validated_floating_point_certificate"],
        optimistic_tail_scope=stored_bank["model_tail_bound_is_optimistic_assumption"] and not stored_bank["true_domain_or_true_field_bound_available"],
        no_calibrated_estimated_sigma=not stored_bank["estimated_sigma_calibrated"],
        no_evaluator_or_retune=reported["no_evaluation_or_catalogue_used"] and reported["no_model_refit_or_selection_retuning"])
    return dict(run=row["run"], model_index=row["model_index"], selected=row["selected"], status="audited",
        parameters_sha256=digest(model["parameters"]), exact_model_checks=exact,
        exact_lattice_proposals=exact_lattice_checks(relation, reported["methods"]["exact_projected_lattice"]),
        profile_origin_max_difference=float(np.max(abs(origin - parameters["complex_profiles_sensor_origin"]))),
        numerical_bank_rank=numerical_rank, real_bank_column_count=2 * count,
        independent_singular_values_max_difference=float(np.max(abs(singular - np.asarray(stored_bank["bank_singular_values"])))),
        original_bank_reconstruction_relative_defect=float(np.linalg.norm(original - reconstruction) / np.linalg.norm(original)),
        original_full_Gram_relative_defect=float(np.linalg.norm(gram - thin_gram) / np.linalg.norm(gram)),
        covariance_direct_vs_root_factor_max_difference=float(np.max(abs(covariance - factor_covariance))),
        covariance_trace_absolute_difference=abs(float(np.trace(covariance)) - stored_bank["covariance_trace"]),
        covariance_max_eigenvalue_absolute_difference=abs(float(eigenvalues[-1]) - stored_bank["covariance_maximum_eigenvalue"]),
        noise_radius_absolute_difference=abs(radius - stored_bank["projected_noise_radius"]),
        independent_noise_radius=radius, independent_relation_null_residual=tangent_residual,
        witness=witness, ratio_absolute_difference=abs(ratio - reported["reach_upper_to_projected_noise_ratio"]) if ratio is not None else None,
        independent_reach_upper_to_noise_ratio=ratio, independent_conditional_rejection=rejection,
        consistent_conditional_flags=flags)


def main():
    if OUTPUT.exists():
        raise FileExistsError("preserve completed geometry audit")
    started = time.perf_counter()
    audit_source_hash = digest(__file__)
    payload = json.loads(SOURCE.read_text())
    if payload["status"] != "finished_read_only_geometry" or len(payload["models"]) != 59:
        raise ValueError("requires the completed immutable 59-model diagnostic")
    source_hash = digest(SOURCE)
    protocol = json.loads(Path(payload["protocol_file"]).read_text())
    group_source = json.loads(Path(payload["source"]).read_text())
    provenance = dict(geometry_protocol_hash=digest(payload["protocol_file"]) == payload["protocol_sha256"],
        group_source_hash=digest(payload["source"]) == payload["source_sha256"],
        geometry_sources=all(digest(HERE / name) == expected for name, expected in payload["source_fingerprints"].items()),
        frozen_inference_sources=all(digest(HERE / name) == expected for name, expected in payload["frozen_inference_source_fingerprints"].items()),
        protocol_models_match=protocol["models"] == payload["models"],
        protocol_settings_match=protocol["settings"] == payload["settings"],
        group_protocol_hash=digest(group_source["protocol_file"]) == group_source["protocol_sha256"],
        group_raw_ledger_hash=digest(group_source["raw_ledger_source"]) == group_source["raw_ledger_source_sha256"],
        model_hashes=all(digest(model["parameters"]) == model["parameters_sha256"] for model in payload["models"]),
        public_input_hashes_only=all(digest(entry["input"]) == entry["input_sha256"] for entry in group_source["manifest"]),
        raw_ledger_parameter_hashes=all(digest(entry["ledger_parameters"]) == entry["ledger_parameters_sha256"] for entry in group_source["manifest"]),
        original_no_evaluation_flags=not payload["evaluation_files_opened"] and not group_source["evaluation_files_opened"])
    if not all(provenance.values()):
        raise ValueError("immutable provenance check failed")
    rows = []
    for index, (row, model) in enumerate(zip(payload["rows"], payload["models"])):
        if row["run"] != model["run"] or row["model_index"] != model["model_index"]:
            raise ValueError("frozen model/witness ordering mismatch")
        try:
            answer = audit_row(row, model, payload["settings"])
        except Exception as error:
            answer = dict(run=row["run"], model_index=row["model_index"], selected=row["selected"],
                status="audit_exception", exception=f"{type(error).__name__}: {error}")
        rows.append(answer)
        if (index + 1) % 15 == 0:
            print(json.dumps(dict(audited=index + 1, total=len(payload["models"]),
                exceptions=sum(item["status"] != "audited" for item in rows))), flush=True)
    finished = [row for row in rows if row["status"] == "audited"]
    differences = lambda key: max((row[key] for row in finished if row.get(key) is not None), default=0.)
    witness_difference = lambda key: max((row["witness"][key] for row in finished if row.get("witness")), default=0.)
    summary = dict(models_total=len(rows), audited=len(finished), exceptions=len(rows) - len(finished),
        all_exact_model_checks=all(all(row["exact_model_checks"].values()) for row in finished),
        exact_lattice_point_count=sum(row["exact_lattice_proposals"]["point_count"] for row in finished),
        all_exact_lattice_checks=all(row["exact_lattice_proposals"]["all_exact_checks_passed"] and
            row["exact_lattice_proposals"].get("pair_count_matches", True) for row in finished),
        all_conditional_flags_match=all(all(row["consistent_conditional_flags"].values()) for row in finished),
        all_real_banks_full_numerical_column_rank=all(row["numerical_bank_rank"] == row["real_bank_column_count"] for row in finished),
        conditional_rejections=sum(row["independent_conditional_rejection"] for row in finished),
        selected_rejections=sum(row["selected"] and row["independent_conditional_rejection"] for row in finished),
        maximum_profile_origin_difference=differences("profile_origin_max_difference"),
        maximum_singular_values_difference=differences("independent_singular_values_max_difference"),
        maximum_reconstruction_relative_defect=differences("original_bank_reconstruction_relative_defect"),
        maximum_full_Gram_relative_defect=differences("original_full_Gram_relative_defect"),
        maximum_noise_radius_difference=differences("noise_radius_absolute_difference"),
        maximum_covariance_trace_difference=differences("covariance_trace_absolute_difference"),
        maximum_covariance_eigenvalue_difference=differences("covariance_max_eigenvalue_absolute_difference"),
        maximum_covariance_direct_vs_factor_difference=differences("covariance_direct_vs_root_factor_max_difference"),
        maximum_witness_upper_absolute_difference=witness_difference("upper_absolute_difference"),
        maximum_witness_upper_relative_difference=witness_difference("upper_relative_difference"),
        maximum_ratio_difference=differences("ratio_absolute_difference"),
        selected_models=[dict(run=row["run"], model_index=row["model_index"],
            reach_upper_to_noise_ratio=row["independent_reach_upper_to_noise_ratio"],
            conditional_rejection=row["independent_conditional_rejection"]) for row in finished if row["selected"]],
        original_summary_matches=payload["summary"]["conditional_model_only_rejections"] ==
            sum(row["independent_conditional_rejection"] for row in finished) and
            payload["summary"]["selected_model_only_rejections"] ==
            sum(row["selected"] and row["independent_conditional_rejection"] for row in finished))
    unchanged = bool(digest(SOURCE) == source_hash and digest(__file__) == audit_source_hash and
        all(digest(HERE / name) == expected for name, expected in payload["source_fingerprints"].items()) and
        all(digest(model["parameters"]) == model["parameters_sha256"] for model in payload["models"]))
    result = dict(status="finished_read_only_audit", source=str(SOURCE), source_sha256=source_hash,
        audit_source_sha256=audit_source_hash, source_and_all_models_unchanged=unchanged,
        provenance=provenance, rows=rows, summary=summary, total_elapsed_seconds=time.perf_counter() - started,
        no_evaluation_catalogue_or_true_phase_loaded=True, no_public_observation_arrays_loaded=True,
        no_geometry_search_or_sampling_or_proposals_generated=True, no_model_fitting_or_selection=True,
        original_W_used_for_saved_witness_checks=True, floating_matmul_used=False,
        independent_linear_algebra="SciPy gesvd and null_space; floating products via unoptimized einsum",
        numerical_agreement_is_not_interval_certificate=True, positive_acceptance_certificates=0)
    OUTPUT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(output=str(OUTPUT), unchanged=unchanged, summary=summary)), flush=True)


if __name__ == "__main__":
    main()
