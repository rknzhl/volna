"""Independent post-freeze audit of ALL bounded-beam RAW group refits.

No fitting, proposal, ranking rule or frozen artifact is changed. Previously
frozen physical labels are opened only after all six source/model hash checks.
Exact Z-lattice image equality and complete visible character module are tested
separately. Forecast scores use frozen clock profiles, not window encodings.
"""

from project_paths import RESULTS_DIR

from datetime import datetime, timezone
from math import gcd, isqrt, lcm
import hashlib
import json
from pathlib import Path

import numpy as np
from sympy import Integer, Matrix
from sympy.matrices.normalforms import hermite_normal_form


HERE = Path(__file__).resolve().parent
SOURCE = RESULTS_DIR / 'ledger_beam_group_selection.json'
TRUTH_AUDIT = RESULTS_DIR / 'ledger_group_selection_audit.json'
EVALUATION_ROOT = RESULTS_DIR / 'fresh_phase_pipeline/evaluation'
OUTPUT = RESULTS_DIR / 'ledger_beam_group_selection_audit.json'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def exact_json(value):
    if isinstance(value, Integer):
        return int(value)
    raise TypeError("unsupported audit value: "+type(value).__name__)


def matrix(rows, dimension):
    return Matrix(len(rows), dimension, [int(x) for row in rows for x in row])


def row_hnf(rows, dimension):
    return hermite_normal_form(matrix(rows, dimension).T).T


def array_matrix(array):
    values = np.asarray(array)
    return Matrix(*values.shape, [int(x) for x in values.ravel()])


def forecast(coefficient, nu, center, times):
    j = len(nu)
    angles = nu[:, None]*(np.asarray(times)[None, :]-center)
    return (coefficient[0, :, None]
            +np.einsum("jq,jt->qt", coefficient[1:1+j], np.cos(angles), optimize=False)
            +np.einsum("jq,jt->qt", coefficient[1+j:], np.sin(angles), optimize=False))


def score(prediction, clean, mass):
    residual = prediction-clean
    return dict(sensor_nrmse=float(np.sqrt(np.sum(residual**2)/np.sum(clean**2))),
        mass_nrmse=float(np.sqrt(np.einsum("qt,qr,rt->", residual, mass, residual, optimize=False)
                                /np.einsum("qt,qr,rt->", clean, mass, clean, optimize=False))))


def relation_assessment(relation, physical, kernel):
    residual = relation*physical
    valid = residual == Matrix.zeros(relation.rows, physical.cols)
    equal = relation == kernel
    classification = ("exact_observed_image_annihilator" if equal else
        "missing_coordinate_relations_larger_presented_group" if valid else
        "false_physical_constraints")
    return dict(equals_true_observed_image_kernel_truth_only=bool(equal),
        every_relation_physically_valid_truth_only=bool(valid),
        relation_times_physical_P_truth_only=residual.tolist(),
        physically_false_HNF_row_count_truth_only=sum(any(value for value in residual.row(i))
                                                    for i in range(residual.rows)),
        classification_truth_only=classification)


def checked_inputs():
    """Do every six-case freeze check before opening any truth audit/archive."""
    source = json.loads(SOURCE.read_text())
    if (source["status"] != "finished_inference_only"
            or len(source["rows"]) != 6 or len(source["manifest"]) != 6
            or not source["all_six_attempts_and_outputs_frozen_before_truth"]
            or not source["source_and_input_hashes_unchanged"]
            or source["evaluation_files_opened"] or source["future_values_given_to_fit"]
            or any(row["run_status"] != "fitted_and_frozen" for row in source["rows"])):
        raise ValueError("completed successful all-six inference freeze is required")
    protected = {str(SOURCE):digest(SOURCE), str(source["protocol_file"]):source["protocol_sha256"],
        source["source_group"]:source["source_group_sha256"],
        source["source_alternatives"]:source["source_alternatives_sha256"]}
    protected.update({str(HERE/name):expected for name,expected in source["source_fingerprints"].items()})
    protected.update({model["parameters"]:model["parameters_sha256"]
                      for model in source["protected_original_group_models"]})
    for entry in source["manifest"]:
        for key in ("input", "ledger_parameters", "alternative_states"):
            protected[entry[key]] = entry[key+"_sha256"]
    for row in source["rows"]:
        if row["evaluation_file_opened"] or row["true_rank_or_physical_labels_loaded"]:
            raise ValueError("truth was declared loaded by inference")
        protected.update({model["parameters"]:model["parameters_sha256"] for model in row["model_files"]})
    for path,expected in protected.items():
        if digest(path) != expected:
            raise ValueError("changed frozen artifact before truth: "+path)
    return source, protected


def main():
    if OUTPUT.exists():
        raise FileExistsError("preserve previous beam-group audit")
    source, protected = checked_inputs()
    # All six inference outputs and all old59 archives have now been checked.
    truth_audit_hash = digest(TRUTH_AUDIT)
    truth_audit = json.loads(TRUTH_AUDIT.read_text())
    if (truth_audit["status"] != "finished"
            or truth_audit["source_sha256"] != source["source_group_sha256"]
            or not truth_audit["all_six_group_outputs_frozen_before_truth"]):
        raise ValueError("previously frozen truth-only audit refers to another group source")
    truths = {row["run"]:row for row in truth_audit["rows"]}
    rows, evaluation_hashes = [], {}
    for row, entry in zip(source["rows"], source["manifest"]):
        if row["run"] != entry["run"]:
            raise ValueError("row/manifest order mismatch")
        known = truths[row["run"]]
        physical = Matrix(known["physical_basis_P_truth_only"])
        kernel = matrix(known["true_image_kernel_HNF_truth_only"], physical.rows)
        if row_hnf(kernel.tolist(), physical.rows) != kernel or kernel*physical != Matrix.zeros(kernel.rows, physical.cols):
            raise ArithmeticError("cached exact truth-only kernel is inconsistent")
        full_visible = bool(known["complete_visible_module_truth_only"])
        with np.load(entry["input"], allow_pickle=False) as public:
            raw_validation = public["Y"][:, entry["n_train"]:entry["n_train"]+entry["n_validation"]]
            validation_times = public["t"][entry["n_train"]:entry["n_train"]+entry["n_validation"]]
        with np.load(entry["ledger_parameters"], allow_pickle=False) as archive:
            mass = archive["physical_mass"]
        evaluation = EVALUATION_ROOT/(row["run"]+".npz")
        evaluation_hashes[str(evaluation)] = digest(evaluation)
        if evaluation_hashes[str(evaluation)] != truth_audit["evaluation_hashes"][str(evaluation)]:
            raise ValueError("evaluation archive changed since previous frozen audit")
        cutoff = entry["n_train"]+entry["n_validation"]
        future_times = np.arange(cutoff, cutoff+entry["n_future"])
        with np.load(evaluation, allow_pickle=False) as archive:
            clean_future = archive["clean"][:, cutoff:cutoff+entry["n_future"]]
        states = entry["final_sieve_states"]
        by_id = {state["state_id"]:state for state in states}
        with Path(entry["alternative_states"]).open() as handle:
            archive_states = json.load(handle)
        original_states = {state["state_id"]:state for state in archive_states["all_registered_states"]}
        expected_states = [original_states[key] for key in archive_states["final_beam_state_ids"]
                           if original_states[key]["representation_sieve_passed"]]
        if states != expected_states:
            raise ArithmeticError("candidate set differs from full frozen final beam+sieve")
        candidate_attempts = row["inference"]["all_model_attempts"][2:]
        if ([attempt["candidate_id"] for attempt in candidate_attempts] != list(by_id)
                or row["inference"]["candidate_state_ids"] != list(by_id)
                or not row["inference"]["all_frozen_final_sieve_states_attempted"]):
            raise ArithmeticError("not every frozen eligible state was attempted in order")
        candidate_records = []
        for attempt in candidate_attempts:
            state = by_id[attempt["candidate_id"]]
            if (attempt["frozen_final_state"] != state or attempt["relation_rows_HNF"] != state["relation_rows_HNF"]
                    or attempt["integer_windings"] != state["integer_windings"]):
                raise ArithmeticError("candidate exact algebra changed before fitting")
            r = matrix(state["relation_rows_HNF"], physical.rows)
            candidate_records.append(dict(state_id=state["state_id"], status=attempt["status"],
                stored_model_index=attempt.get("stored_model_index"), free_dimension=state["free_dimension"],
                validation_raw_sensor_mse=attempt.get("validation_raw_sensor_mse"),
                **relation_assessment(r, physical, kernel)))
        models = []
        for metadata in row["model_files"]:
            with np.load(metadata["parameters"], allow_pickle=False) as archive:
                params = {key:archive[key] for key in archive.files}
            for name,array in params.items():
                if array.dtype.kind not in "US" and not np.isfinite(array).all():
                    raise FloatingPointError("nonfinite stored model "+name)
            nu, coefficient, center = params["frequencies"], params["coefficients_sensor_center"], float(params["time_center"])
            r = array_matrix(params["relation_rows_HNF_integer_text"])
            u, v = array_matrix(params["U_integer_text"]), array_matrix(params["V_integer_text"])
            inverse = array_matrix(params["V_inverse_integer_text"])
            diagonal = [int(value) for value in params["smith_diagonal_integer_text"]]
            d = Matrix.zeros(r.rows, physical.rows)
            for i,value in enumerate(diagonal):
                d[i,i] = value
            if (row_hnf(r.tolist(), physical.rows) != r or r.rows != len(diagonal)
                    or u*r*v != d or v*inverse != Matrix.eye(physical.rows)
                    or abs(int(u.det())) != 1 or abs(int(v.det())) != 1
                    or r.tolist() != metadata["presentation"]["relation_rows_HNF"]
                    or v.tolist() != metadata["presentation"]["V"]):
                raise ArithmeticError("stored exact HNF/reduced Smith gauge differs")
            free = v[:, r.rows:]
            finite_indices = [i for i,value in enumerate(diagonal) if value > 1]
            finite = [[int(v[j,i]) % diagonal[i] for i in finite_indices] for j in range(physical.rows)]
            if (free != array_matrix(params["all_free_character_weights_integer_text"])
                    or matrix(finite,len(finite_indices)) != array_matrix(params["all_finite_character_weights_integer_text"])):
                raise ArithmeticError("stored free/finite characters differ from exact gauge")
            components = 1
            for i in finite_indices:
                components *= diagonal[i]
            state_id = metadata.get("beam_state_id")
            equality_defect, orbit_order, complexity = None, 1, None
            if state_id is not None:
                state = by_id[state_id]
                if (r.tolist() != state["relation_rows_HNF"] or v.tolist() != state["V_reduced"]
                        or inverse.tolist() != state["V_inverse_reduced"]):
                    raise ArithmeticError("stored model changed frozen candidate presentation")
                winding = Matrix(state["integer_windings"]) if r.rows else Matrix.zeros(0,1)
                advance = u*winding
                for i in finite_indices:
                    orbit_order = lcm(orbit_order,diagonal[i]//gcd(diagonal[i],int(advance[i])))
                numeric_r = np.asarray(r.tolist(),float).reshape(r.shape)
                equality_defect = float(np.max(abs(np.einsum("ij,j->i",numeric_r,nu,optimize=False)
                    -2*np.pi*np.asarray(state["integer_windings"])))) if r.rows else 0.0
                centered = [[value if 2*value <= diagonal[i] else value-diagonal[i]
                             for value,i in zip(values,finite_indices)] for values in finite]
                complexity = max((sum(abs(int(x)) for x in free.row(j))+sum(abs(x) for x in centered[j])
                                  for j in range(physical.rows)),default=0)
                if (complexity != state["whole_representation_complexity"] or complexity > isqrt(entry["n_train"])
                        or components > entry["n_train"] or orbit_order != components):
                    raise ArithmeticError("stored model violates frozen computational eligibility")
            angle = (np.arange(entry["n_train"])-center)[:, None]*nu[None,:]
            design = np.column_stack((np.ones(entry["n_train"]),np.cos(angle),np.sin(angle)))
            gram = np.einsum("ni,nj->ij",design,design,optimize=False)
            gram_difference = float(np.max(abs(gram-params["temporal_gram"])))
            if not np.allclose(gram,params["temporal_gram"],atol=1e-9,rtol=1e-12):
                raise ArithmeticError("stored temporal Gram fails independent reconstruction")
            origin = .5*(coefficient[1:1+len(nu)]-1j*coefficient[1+len(nu):])*np.exp(-1j*nu*center)[:,None]
            origin_difference = float(np.max(abs(origin-params["complex_profiles_sensor_origin"])))
            val_prediction = forecast(coefficient,nu,center,validation_times)
            val_mse = float(np.mean((raw_validation-val_prediction)**2))
            val_difference = abs(val_mse-metadata["validation_raw_sensor_mse"])
            if val_difference > 1e-12*max(1.,val_mse) or origin_difference > 1e-10:
                raise ArithmeticError("stored profile gauge or RAW VAL score differs")
            future_prediction = forecast(coefficient,nu,center,future_times)
            models.append(dict(model_index=metadata["model_index"], model_kind=metadata["model_kind"],
                beam_state_id=state_id, selected=metadata["model_index"] == row["selected_model_index"],
                relation_rows_HNF=r.tolist(), continuous_dimension=physical.rows-r.rows,
                finite_orders=[diagonal[i] for i in finite_indices], component_count=components,
                finite_rotation_orbit_order=orbit_order, frozen_gauge_complexity=complexity,
                numeric_equality_max_defect=equality_defect,
                exact_reduced_Smith_and_all_characters_verified=True,
                independent_gram_max_difference=gram_difference,
                origin_profile_max_difference=origin_difference,
                raw_validation_mse=metadata["validation_raw_sensor_mse"],
                independently_recomputed_validation_mse=val_mse, validation_score_max_difference=val_difference,
                forecast=score(future_prediction,clean_future,mass),
                full_visible_module_and_image_correct_truth_only=bool(full_visible and r == kernel),
                **relation_assessment(r,physical,kernel)))
        # Replay exactly the inference comparison, rather than applying a new tolerance.
        replay_best = 0
        for index,model in enumerate(models):
            if model["raw_validation_mse"] < models[replay_best]["raw_validation_mse"]:
                replay_best = index
        if replay_best != row["selected_model_index"] or [m["model_index"] for m in models] != list(range(len(models))):
            raise ArithmeticError("selection is not strict RAW VAL minimum with earlier ties")
        selected = models[replay_best]
        correct_candidates = [record for record in candidate_records if record["equals_true_observed_image_kernel_truth_only"]]
        correct_models = [m for m in models if m["equals_true_observed_image_kernel_truth_only"]]
        old = known["selected_model"]
        rows.append(dict(run=row["run"], candidate_count=len(states), all_candidates_attempted_in_frozen_order=True,
            stored_model_count=len(models), physical_P_truth_only=physical.tolist(),
            true_image_kernel_HNF_truth_only=kernel.tolist(),
            complete_visible_module_truth_only=full_visible,
            correct_kernel_final_sieve_candidate_available_truth_only=bool(correct_candidates),
            correct_kernel_candidate_fit_admissible_truth_only=any(record["status"] == "admissible_tied_model" for record in correct_candidates),
            correct_kernel_stored_model_available_truth_only=bool(correct_models),
            correct_kernel_candidate_records_truth_only=correct_candidates,
            all_candidate_assessments_truth_only=candidate_records, all_stored_models=models,
            selected_model=selected, strict_raw_validation_selection_replayed=True,
            old_greedy_selected=dict(continuous_dimension=old["continuous_dimension"],finite_orders=old["finite_orders"],
                classification_truth_only=("exact_observed_image_annihilator" if old["equals_true_observed_image_kernel_truth_only"] else
                    "missing_coordinate_relations_larger_presented_group" if old["every_relation_physically_valid_truth_only"] else
                    "false_physical_constraints"),
                equals_true_observed_image_kernel_truth_only=old["equals_true_observed_image_kernel_truth_only"],
                full_visible_module_and_image_correct_truth_only=old["full_visible_module_and_model_image_correct_truth_only"],
                forecast=old["forecast"]),
            old_and_new_selected_exact_R_equal=matrix(old_relation_rows(known),physical.rows)
                ==matrix(selected["relation_rows_HNF"],physical.rows),
            clock_forecast_scope="frozen TRAIN line profiles; not observed-window inverse or MSSA comparison"))
    for path,expected in {**protected, str(TRUTH_AUDIT):truth_audit_hash, **evaluation_hashes}.items():
        if digest(path) != expected:
            raise ValueError("immutable artifact changed during audit: "+path)
    summary = dict(runs=len(rows),candidate_attempts=sum(r["candidate_count"] for r in rows),
        stored_models=sum(r["stored_model_count"] for r in rows),
        correct_kernel_candidate_available=sum(r["correct_kernel_final_sieve_candidate_available_truth_only"] for r in rows),
        correct_kernel_fit_admissible=sum(r["correct_kernel_candidate_fit_admissible_truth_only"] for r in rows),
        exact_observed_image_selected=sum(r["selected_model"]["equals_true_observed_image_kernel_truth_only"] for r in rows),
        complete_visible_module=sum(r["complete_visible_module_truth_only"] for r in rows),
        complete_module_and_image_selected=sum(r["selected_model"]["full_visible_module_and_image_correct_truth_only"] for r in rows),
        selected_false_physical_constraints=sum(r["selected_model"]["classification_truth_only"] == "false_physical_constraints" for r in rows),
        selected_missing_relations_overdimension=sum(r["selected_model"]["classification_truth_only"] == "missing_coordinate_relations_larger_presented_group" for r in rows),
        old_exact_observed_image_selected=sum(r["old_greedy_selected"]["equals_true_observed_image_kernel_truth_only"] for r in rows),
        old_complete_module_and_image_selected=sum(r["old_greedy_selected"]["full_visible_module_and_image_correct_truth_only"] for r in rows),
        all_strict_validation_choices_replayed=True,all_frozen_gauges_and_all_character_weights_verified=True,
        all_model_grams_verified=True,max_independent_gram_difference=max(m["independent_gram_max_difference"] for r in rows for m in r["all_stored_models"]),
        protected_original_model_count=len(source["protected_original_group_models"]))
    payload = dict(status="finished",created_at=datetime.now(timezone.utc).isoformat(),
        source=str(SOURCE),source_sha256=protected[str(SOURCE)],truth_audit=str(TRUTH_AUDIT),truth_audit_sha256=truth_audit_hash,
        audit_source_sha256=digest(Path(__file__)),evaluation_hashes=evaluation_hashes,
        explicit_all_six_source_output_freeze_checked_before_truth=True,
        no_fit_candidate_threshold_beam_validation_or_frozen_artifact_changed=True,
        exact_unsaturated_Z_lattice_equality_not_Q_span=True,prior_benchmarks_not_independent_external_test=True,
        observed_window_phase_or_MSSA_comparison_performed=False,rows=rows,summary=summary)
    OUTPUT.write_text(json.dumps(payload,indent=2,allow_nan=False,default=exact_json)+"\n")
    print(json.dumps(summary),flush=True)
    for row in rows:
        selected=row["selected_model"]
        print(json.dumps(dict(run=row["run"],dimension=selected["continuous_dimension"],
            classification=selected["classification_truth_only"],full_state=selected["full_visible_module_and_image_correct_truth_only"],
            correct_candidate=row["correct_kernel_final_sieve_candidate_available_truth_only"],
            correct_fit=row["correct_kernel_candidate_fit_admissible_truth_only"],
            forecast=selected["forecast"],old_forecast=row["old_greedy_selected"]["forecast"])),flush=True)


def old_relation_rows(known):
    """Read old selected exact R from its already-frozen inference metadata."""
    original=json.loads(Path(json.loads(SOURCE.read_text())["source_group"]).read_text())
    row=next(row for row in original["rows"] if row["run"] == known["run"])
    return row["model_files"][row["selected_model_index"]]["presentation"]["relation_rows_HNF"]


if __name__ == "__main__":
    main()
