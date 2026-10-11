"""Independent post-freeze ALL-model algebra/numerical/forecast audit.

No fit, proposal, threshold, parameter or frozen result is changed. Evaluator
archives open only after the completed group-stage freeze/hash checks. Exact
physical labels are posthoc and never returned to the inference implementation.
"""

from project_paths import RESULTS_DIR

from datetime import datetime, timezone
from math import gcd, lcm
import hashlib
import json
from pathlib import Path

import numpy as np
from sympy import Integer, Matrix

from fresh_group_image_audit import row_hnf, integer_kernel_rows


HERE = Path(__file__).resolve().parent
SOURCE = RESULTS_DIR / 'ledger_group_selection.json'
LEDGER = RESULTS_DIR / 'raw_spectral_ledger.json'
EVALUATION_ROOT = RESULTS_DIR / 'fresh_phase_pipeline/evaluation'
OUTPUT = RESULTS_DIR / 'ledger_group_selection_audit.json'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def symbolic_integer(value):
    if isinstance(value, Integer):
        return int(value)
    raise TypeError("unsupported audit result type: "+type(value).__name__)


def matrix_array(array):
    values = np.asarray(array)
    return Matrix(values.shape[0], values.shape[1], [int(value) for value in values.ravel()])


def metrics(matrix):
    rows = matrix.tolist()
    return dict(shape=[matrix.rows, matrix.cols], maximum_absolute_entry=max((abs(int(v)) for v in matrix), default=0),
        maximum_row_l1=max((sum(abs(int(v)) for v in row) for row in rows), default=0),
        total_l1=sum(abs(int(v)) for v in matrix))


def matching(frequencies, alpha, catalogue, tolerance):
    rows, records = [], []
    for frequency in frequencies:
        matches = {}
        for mode in catalogue:
            wrapped = float(np.angle(np.exp(1j*float(mode@alpha))))
            if abs(wrapped) <= 1e-12 or abs(abs(wrapped)-np.pi) <= 1e-12:
                continue
            label = tuple(int(v) for v in (mode if wrapped > 0 else -mode))
            gap = abs(float(frequency)-abs(wrapped))
            if gap <= tolerance:
                matches[label] = dict(physical_label=list(label), frequency_gap=gap,
                                      true_sampled_frequency=abs(wrapped))
        found = list(matches.values())
        records.append(dict(frequency=float(frequency), matches=found, unique=len(found) == 1))
        if len(found) == 1:
            rows.append(found[0]["physical_label"])
    return records, Matrix(rows) if len(rows) == len(frequencies) else None


def forecast(coefficient, nu, center, times):
    count = len(nu)
    angle = nu[:, None]*(np.asarray(times)[None, :]-center)
    return (coefficient[0, :, None]
        +np.einsum("jq,jt->qt", coefficient[1:1+count], np.cos(angle), optimize=False)
        +np.einsum("jq,jt->qt", coefficient[1+count:], np.sin(angle), optimize=False))


def score_error(prediction, clean, mass):
    residual = prediction-clean
    numerator, denominator = float(np.sum(residual**2)), float(np.sum(clean**2))
    mass_numerator = float(np.einsum("qt,qr,rt->", residual, mass, residual, optimize=False))
    mass_denominator = float(np.einsum("qt,qr,rt->", clean, mass, clean, optimize=False))
    return dict(sensor_nrmse=float(np.sqrt(numerator/denominator)),
                mass_nrmse=float(np.sqrt(mass_numerator/mass_denominator)))


def main():
    if OUTPUT.exists():
        raise FileExistsError("preserve prior group audit")
    source, ledger = json.loads(SOURCE.read_text()), json.loads(LEDGER.read_text())
    if (source["status"] != "finished_inference_only"
            or not source["all_six_attempts_and_outputs_frozen_before_truth"]
            or source["evaluation_files_opened"] or not source["source_and_input_hashes_unchanged"]):
        raise ValueError("explicit completed all6 inference freeze required")
    initial_sources = {name:digest(HERE/name) for name in source["source_fingerprints"]}
    if initial_sources != source["source_fingerprints"]:
        raise ValueError("group source fingerprint changed before scoring")
    source_hash, ledger_hash = digest(SOURCE), digest(LEDGER)
    ledger_rows = {row["run"]:row for row in ledger["rows"]}
    for entry in source["manifest"]:
        if digest(entry["input"]) != entry["input_sha256"] or digest(entry["ledger_parameters"]) != entry["ledger_parameters_sha256"]:
            raise ValueError("frozen public/ledger input changed")
    for row in source["rows"]:
        if row["run_status"] == "fitted_and_frozen":
            for model in row["model_files"]:
                if digest(model["parameters"]) != model["parameters_sha256"]:
                    raise ValueError("frozen group model changed before any truth archive")
    rows, evaluation_hashes = [], {}
    for row, entry in zip(source["rows"], source["manifest"]):
        if row["run_status"] != "fitted_and_frozen":
            rows.append(dict(run=row["run"], inference_status=row["run_status"]))
            continue
        with np.load(entry["ledger_parameters"], allow_pickle=False) as archive:
            raw = {key:archive[key] for key in archive.files}
        evaluation = EVALUATION_ROOT/(row["run"]+".npz")
        evaluation_hashes[str(evaluation)] = digest(evaluation)
        with np.load(evaluation, allow_pickle=False) as archive:
            truth = {key:archive[key] for key in ("clean", "alpha", "catalog_modes", "group_modes", "sigma")}
        tolerance = .25*2*np.pi/entry["n_train"]
        matches, physical = matching(raw["frequencies"], truth["alpha"], truth["catalog_modes"], tolerance)
        if physical is None:
            raise ArithmeticError("postfreeze baseline line identities not uniquely matched")
        kernel = integer_kernel_rows(physical)
        visible = row_hnf(truth["group_modes"].tolist(), physical.cols)
        recovered = row_hnf(physical.tolist(), physical.cols)
        full_module = recovered == visible
        covariance = raw["conditional_frequency_covariance"]
        covariance_inverse = np.linalg.inv(covariance)
        true_nu = np.asarray([record["matches"][0]["true_sampled_frequency"] for record in matches])
        base_error = raw["frequencies"]-true_nu
        cutoff = entry["n_train"]+entry["n_validation"]
        future_times = np.arange(cutoff, cutoff+entry["n_future"])
        clean_future = truth["clean"][:, cutoff:cutoff+entry["n_future"]]
        true_eligible, true_all, proposal_records = [], [], []
        first_false_eligible = None
        for index, proposal in enumerate(row["inference"]["proposals"]["all_proposals"]):
            a = Matrix(1, physical.rows, proposal["row"])
            valid = a*physical == Matrix.zeros(1, physical.cols)
            if valid:
                true_all.append(proposal["row"])
                if proposal["eligible_for_nested_refit"]:
                    true_eligible.append(proposal["row"])
            actual_residue = float(np.angle(np.exp(1j*float(np.asarray(proposal["row"])@true_nu))))
            deviation = proposal["conditional_standard_deviation"]
            proposal_records.append(dict(index=index, row=proposal["row"], physically_valid_truth_only=bool(valid),
                eligible=proposal["eligible_for_nested_refit"], conditional_z=proposal["conditional_z"],
                true_frequency_residue=actual_residue,
                conditional_sd_over_true_nonrelation_gap=(deviation/abs(actual_residue)
                    if deviation is not None and abs(actual_residue) > 1e-12 else None)))
        eligible_order = row["inference"]["proposals"]["eligible"]
        for index, proposal in enumerate(eligible_order):
            if Matrix(1, physical.rows, proposal["row"])*physical != Matrix.zeros(1, physical.cols):
                first_false_eligible = dict(index=index, row=proposal["row"], conditional_z=proposal["conditional_z"],
                    provenance=proposal["provenance"])
                break
        plan_records = []
        for plan in row["inference"]["lattice_plans"]["plans"]:
            relation = row_hnf(plan["relation_rows_HNF"], physical.rows)
            plan_records.append(dict(plan_index=plan["model_index"], relation_rank=relation.rows,
                every_relation_physically_valid_truth_only=relation*physical == Matrix.zeros(relation.rows, physical.cols),
                equals_true_observed_image_kernel_truth_only=relation == kernel,
                relation_complexity=metrics(relation)))
        models = []
        for metadata in row["model_files"]:
            with np.load(metadata["parameters"], allow_pickle=False) as archive:
                parameters = {key:archive[key] for key in archive.files}
            nu, coefficient = parameters["frequencies"], parameters["coefficients_sensor_center"]
            center = float(parameters["time_center"])
            train_angles = (np.arange(entry["n_train"])-center)[:, None]*nu[None, :]
            design = np.column_stack((np.ones(entry["n_train"]), np.cos(train_angles), np.sin(train_angles)))
            independent_gram = np.einsum("ni,nj->ij", design, design, optimize=False)
            difference = abs(parameters["temporal_gram"]-independent_gram)
            origin = .5*(coefficient[1:1+len(nu)]-1j*coefficient[1+len(nu):])*np.exp(-1j*nu*center)[:, None]
            origin_defect = float(np.max(abs(origin-parameters["complex_profiles_sensor_origin"])))
            relation = matrix_array(parameters["relation_rows_HNF_integer_text"])
            u, v = matrix_array(parameters["U_integer_text"]), matrix_array(parameters["V_integer_text"])
            inverse = matrix_array(parameters["V_inverse_integer_text"])
            diagonal = [int(value) for value in parameters["smith_diagonal_integer_text"]]
            d_matrix = Matrix.zeros(relation.rows, len(nu))
            for i,value in enumerate(diagonal):d_matrix[i, i] = value
            exact_smith = (u*relation*v == d_matrix and v*inverse == Matrix.eye(len(nu))
                           and abs(int(v.det())) == 1 and abs(int(u.det())) == 1)
            image_correct = relation == kernel
            physically_valid = relation*physical == Matrix.zeros(relation.rows, physical.cols)
            predicted = forecast(coefficient, nu, center, future_times)
            prediction_origin = coefficient[0, :, None]+2*np.real(np.einsum("jq,jt->qt", origin,
                                       np.exp(1j*nu[:, None]*future_times[None, :]), optimize=False))
            free_weights = v[:, relation.rows:]
            finite_indices = [i for i,order in enumerate(diagonal) if order > 1]
            finite_weights = Matrix(len(nu), len(finite_indices),
                [int(v[j,i]) % diagonal[i] for j in range(len(nu)) for i in finite_indices])
            centered_finite = Matrix(len(nu), len(finite_indices),
                [min(int(v[j,i]) % diagonal[i], diagonal[i]-(int(v[j,i]) % diagonal[i]))
                 for j in range(len(nu)) for i in finite_indices])
            shift = nu-raw["frequencies"]
            frequency_truth_error = nu-true_nu
            models.append(dict(model_index=metadata["model_index"], model_kind=metadata["model_kind"],
                selected=metadata["model_index"] == row["selected_model_index"],
                relation_rank=relation.rows, continuous_dimension=len(nu)-relation.rows,
                finite_orders=[diagonal[i] for i in finite_indices], component_count=metadata["presentation"]["component_count"],
                observer_component_budget_available=metadata["presentation"]["observer_component_budget_available"],
                finite_rotation_orbit_order=metadata["presentation"]["finite_rotation_orbit_order"],
                every_relation_physically_valid_truth_only=bool(physically_valid),
                equals_true_observed_image_kernel_truth_only=bool(image_correct),
                full_visible_module_and_model_image_correct_truth_only=bool(full_module and image_correct),
                exact_smith_decomposition_verified=bool(exact_smith), relation_complexity=metrics(relation),
                free_character_weight_complexity=metrics(free_weights),
                finite_character_weight_complexity=metrics(finite_weights),
                centered_finite_character_weight_complexity=metrics(centered_finite),
                free_weight_norm_is_current_Smith_gauge=True,
                one_free_angle_weight_magnitudes_invariant_up_to_sign=len(nu)-relation.rows == 1,
                conditional_frequency_shift_mahalanobis=float(np.einsum("i,ij,j->", shift, covariance_inverse, shift, optimize=False)),
                true_frequency_error_in_baseline_covariance_mahalanobis=float(np.einsum("i,ij,j->", frequency_truth_error,
                    covariance_inverse, frequency_truth_error, optimize=False)),
                maximum_true_frequency_error_over_baseline_se=float(np.max(abs(frequency_truth_error)/np.sqrt(np.diag(covariance)))),
                forecast=score_error(predicted, clean_future, raw["physical_mass"]),
                independent_gram_max_absolute_difference=float(np.max(difference)),
                independent_gram_max_relative_scale_difference=float(np.max(difference)/max(1.,float(np.max(abs(independent_gram))))),
                independent_gram_check_passed=bool(np.allclose(parameters["temporal_gram"], independent_gram, atol=1e-9, rtol=1e-12)),
                origin_profile_formula_max_difference=origin_defect,
                centered_vs_origin_forecast_max_difference=float(np.max(abs(predicted-prediction_origin)))))
        selected = next(model for model in models if model["selected"])
        true_all_basis = row_hnf(true_all, physical.rows)
        true_eligible_basis = row_hnf(true_eligible, physical.rows)
        rows.append(dict(run=row["run"], physical_basis_P_truth_only=physical.tolist(),
            true_image_kernel_HNF_truth_only=kernel.tolist(), visible_module_HNF_truth_only=visible.tolist(),
            recovered_observed_module_HNF_truth_only=recovered.tolist(), complete_visible_module_truth_only=bool(full_module),
            raw_catalog_matching=matches, catalogue_matching_tolerance=tolerance,
            all_true_proposals_span_complete_kernel_truth_only=true_all_basis == kernel,
            eligible_true_proposals_span_complete_kernel_truth_only=true_eligible_basis == kernel,
            true_eligible_relation_count=len(true_eligible), first_false_eligible_relation_truth_only=first_false_eligible,
            any_nested_plan_equals_true_kernel_truth_only=any(plan["equals_true_observed_image_kernel_truth_only"] for plan in plan_records),
            any_stored_model_equals_true_kernel_truth_only=any(model["equals_true_observed_image_kernel_truth_only"] for model in models),
            supplied_noise_variance_over_true_noise_variance_truth_only=(entry["noise_estimate_train_only"]["sigma"]**2/float(truth["sigma"])**2),
            baseline_true_frequency_error_mahalanobis=float(np.einsum("i,ij,j->", base_error,covariance_inverse,base_error,optimize=False)),
            relation_proposals_posthoc=proposal_records, nested_path_posthoc=plan_records,
            all_models=models, selected_model=selected,
            forecast_scope="clock forecast from frozen TRAIN line coefficients; no observed-window phase encoding or MSSA comparison"))
    if digest(SOURCE) != source_hash or digest(LEDGER) != ledger_hash:
        raise ValueError("immutable inference results changed during audit")
    if {name:digest(HERE/name) for name in source["source_fingerprints"]} != initial_sources:
        raise ValueError("immutable group sources changed during audit")
    payload = dict(status="finished", created_at=datetime.now(timezone.utc).isoformat(), source=str(SOURCE),
        source_sha256=source_hash, raw_ledger_sha256=ledger_hash, audit_source_sha256=digest(Path(__file__)),
        evaluation_hashes=evaluation_hashes, all_six_group_outputs_frozen_before_truth=True,
        no_source_parameter_threshold_or_fit_changed=True, rows=rows,
        summary=dict(fitted_runs=len(rows), exact_observed_image_selected=sum(r["selected_model"]["equals_true_observed_image_kernel_truth_only"] for r in rows),
            complete_visible_module=sum(r["complete_visible_module_truth_only"] for r in rows),
            complete_module_and_image_selected=sum(r["selected_model"]["full_visible_module_and_model_image_correct_truth_only"] for r in rows),
            true_kernel_in_any_nested_plan=sum(r["any_nested_plan_equals_true_kernel_truth_only"] for r in rows),
            eligible_true_rows_span_kernel=sum(r["eligible_true_proposals_span_complete_kernel_truth_only"] for r in rows),
            all_model_grams_passed=all(m["independent_gram_check_passed"] for r in rows for m in r["all_models"]),
            max_independent_gram_difference=max(m["independent_gram_max_absolute_difference"] for r in rows for m in r["all_models"])))
    OUTPUT.write_text(json.dumps(payload, indent=2, allow_nan=False, default=symbolic_integer)+"\n")
    print(json.dumps(payload["summary"]), flush=True)
    for row in rows:
        selected = row["selected_model"]
        print(json.dumps(dict(run=row["run"], dimension=selected["continuous_dimension"], finite=selected["finite_orders"],
            image_correct=selected["equals_true_observed_image_kernel_truth_only"], kernel_available=row["eligible_true_proposals_span_complete_kernel_truth_only"],
            kernel_in_path=row["any_nested_plan_equals_true_kernel_truth_only"], weights=selected["free_character_weight_complexity"],
            forecast=selected["forecast"])), flush=True)


if __name__ == "__main__":main()
