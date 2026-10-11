"""Posthoc exact group-image supplement; no frozen inference files are edited.

P maps physical phases to the saved acquired-factor chart. For the intended
nonresonant physical rotation, the true phase image has annihilator ker_Z(P.T).
Equality of relation lattices tests the group image; equality of row_Z(P) with
the full visible physical character module separately tests faithfulness.
All physical labels here are evaluation-only, after the completed fresh freeze.
"""

from project_paths import RESULTS_DIR

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from sympy import Matrix, ZZ
from sympy.matrices.normalforms import hermite_normal_form, smith_normal_form

from group_witness_conditioning import conditioning


HERE = Path(__file__).resolve().parent
SOURCE = RESULTS_DIR / 'fresh_phase_pipeline.json'
OUTPUT = RESULTS_DIR / 'fresh_group_image_audit.json'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def matrix(rows, dimension):
    return Matrix(len(rows), dimension, [int(x) for row in rows for x in row])


def row_hnf(rows, dimension):
    return hermite_normal_form(matrix(rows, dimension).T).T


def egcd(a, b):
    old_r, r, old_x, x, old_y, y = a, b, 1, 0, 0, 1
    while r:
        quotient = old_r // r
        old_r, r = r, old_r-quotient*r
        old_x, x = x, old_x-quotient*x
        old_y, y = y, old_y-quotient*y
    if old_r < 0:
        old_r, old_x, old_y = -old_r, -old_x, -old_y
    return old_r, old_x, old_y


def integer_kernel_rows(p):
    """Full integer kernel, by unimodular intersections, not rational rounding."""
    columns = Matrix.eye(p.rows)
    for constraint in p.T.tolist():
        values = Matrix(1, p.rows, constraint) * columns
        if not any(values):
            continue
        transform = Matrix.eye(columns.cols)
        pivot = next(j for j in range(columns.cols) if values[j])
        if pivot:
            transform.col_swap(0, pivot)
            values = values * transform
        for j in range(1, columns.cols):
            a, b = int(values[0]), int(values[j])
            if not b:
                continue
            g, x, y = egcd(a, b)
            operation = Matrix.eye(columns.cols)
            operation[0, 0], operation[j, 0] = x, y
            operation[0, j], operation[j, j] = -b // g, a // g
            transform = transform * operation
            values = values * operation
        columns = columns * transform[:, 1:]
    kernel = hermite_normal_form(columns).T
    if kernel * p != Matrix.zeros(kernel.rows, p.cols):
        raise ArithmeticError("integer kernel does not annihilate P exactly")
    if kernel.rows != p.rows-p.rank():
        raise ArithmeticError("integer kernel has incorrect rank")
    return kernel


def presentation(rows, dimension):
    hnf = row_hnf(rows, dimension)
    snf = smith_normal_form(hnf, domain=ZZ)
    diagonal = [abs(int(snf[i, i])) for i in range(min(snf.shape)) if snf[i, i]]
    return dict(row_basis_HNF=hnf.tolist(), rank=hnf.rows,
        smith_nonzero_diagonal=diagonal,
        finite_invariant_factors=[item for item in diagonal if item > 1],
        connected_torus_dimension=dimension-hnf.rows,
        relation_rows_saturated_by_audit=False)


def relation_assessment(rows, p, kernel):
    hnf = row_hnf(rows, p.rows)
    residual = hnf*p
    exact_valid = residual == Matrix.zeros(hnf.rows, p.cols)
    equal = hnf == kernel
    if equal:
        classification = "exact_image_annihilator"
    elif exact_valid:
        classification = "missing_redundant_coordinate_relations_larger_latent_group"
    else:
        classification = "false_physical_constraint_not_coordinate_redundancy"
    return dict(presentation=presentation(rows, p.rows),
        exact_relation_times_P=residual.tolist(),
        every_relation_valid_on_physical_image=exact_valid,
        equals_full_image_annihilator=equal, classification=classification)


def main():
    if OUTPUT.exists():
        raise FileExistsError("preserve the existing posthoc group-image audit")
    source_hash = digest(SOURCE)
    payload = json.loads(SOURCE.read_text())
    if payload["status"] != "finished" or not payload["all_six_attempts_and_outputs_frozen_before_truth_open"]:
        raise ValueError("requires the completed all-six frozen fresh audit")
    rows = []
    for row, entry in zip(payload["rows"], payload["manifest"]):
        if row["run_status"] != "finished":
            raise ValueError("requires all six completed frozen cases")
        metadata = row["module_truth_only"]
        if metadata["status"] != "scored_unique_bounded_catalog" or not metadata["relations_valid"]:
            raise ValueError("ambiguous physical chart: no exact image conclusion")
        p = Matrix(metadata["physical_basis"])
        with np.load(entry["evaluation"], allow_pickle=False) as evaluation:
            visible = row_hnf(evaluation["group_modes"].tolist(), p.cols)
        recovered = row_hnf(p.tolist(), p.cols)
        module_complete = recovered == visible
        kernel = integer_kernel_rows(p)
        kernel_snf = presentation(kernel.tolist(), p.rows)
        if kernel_snf["finite_invariant_factors"]:
            raise ArithmeticError("image of a connected physical torus cannot have components")
        proposed = row["compact"]["group_proposal"].get("relation", [])
        proposed_rows = [proposed] if proposed else []
        selected = row["compact"]["selected_model"]
        selected_rows = proposed_rows if selected == "compact" else []
        selected_assessment = relation_assessment(selected_rows, p, kernel)
        proposed_assessment = relation_assessment(proposed_rows, p, kernel)
        record = dict(run=row["run"], physical_character_map_P=p.tolist(),
            ambient_learned_phase_dimension=p.rows, physical_visible_module_rank=visible.rows,
            recovered_physical_module_HNF=recovered.tolist(), full_visible_module_HNF=visible.tolist(),
            complete_visible_character_module=module_complete,
            true_phase_image=kernel_snf, kernel_times_P_exact=(kernel*p).tolist(),
            proposed_relation=proposed_assessment, selected_relation=selected_assessment,
            selected_model=selected,
            selected_group_image_correct=selected_assessment["equals_full_image_annihilator"],
            selected_full_observable_state_faithful=bool(module_complete and selected_assessment["equals_full_image_annihilator"]),
            proposed_group_would_be_full_observable_state_faithful=bool(module_complete and proposed_assessment["equals_full_image_annihilator"]),
            old_frozen_full_state_boolean=row["full_physical_module_and_group_correct_truth_only"],
            kernel_construction="successive exact unimodular intersections of Z^p; then HNF",
            no_sampling_resonance_assumed_by_evaluation="predeclared intended nonresonant physical alpha")
        if proposed:
            compact = row["compact"]
            if digest(compact["frozen_parameters"]) != compact["frozen_parameters_sha256"]:
                raise ValueError("changed frozen compact parameters")
            discovery = row["discovery"]
            if digest(discovery["source_model"]) != discovery["source_model_sha256"]:
                raise ValueError("changed frozen discovery parameters")
            with np.load(discovery["source_model"], allow_pickle=False) as archive:
                anchors, products = archive["acquired_labels"], archive["phase_products"]
            details = {}
            with np.load(compact["frozen_parameters"], allow_pickle=False) as archive:
                for family, cv in compact["validation_cv"].items():
                    labels = archive[family+"__labels"]
                    coefficient = archive[family+"__coefficients_mass"]
                    alpha = archive[family+"__frequencies"]
                    _, condition = conditioning(labels, coefficient, alpha, anchors, products,
                        relation=proposed if family == "compact" else None, lag_count=128)
                    details[family] = dict(validation_mass_mse=cv["mass_mse_per_field_snapshot"],
                        sigma0_numerical=condition["sigma0_numerical"],
                        conditioning_status=condition["conditioning_status"],
                        learned_template_bound_constant=condition["phase_bound_constant_rad_per_mass_window_norm"],
                        physical_dictionary_labels=(Matrix(labels.tolist())*p).tolist())
            record["frozen_candidate_comparison_no_refit"] = details
        rows.append(record)
    if digest(SOURCE) != source_hash:
        raise ValueError("frozen fresh JSON changed during read-only audit")
    result = dict(status="finished", created_at=datetime.now(timezone.utc).isoformat(),
        source=str(SOURCE), source_sha256=source_hash, audit_source_sha256=digest(__file__),
        frozen_fresh_sources_or_results_modified=False, no_candidate_refit_or_selection_retuning=True,
        all_truth_used_posthoc_only=True, rows=rows,
        summary=dict(runs=len(rows), correct_selected_phase_image=sum(row["selected_group_image_correct"] for row in rows),
            complete_visible_character_module=sum(row["complete_visible_character_module"] for row in rows),
            correct_selected_full_observable_state=sum(row["selected_full_observable_state_faithful"] for row in rows),
            proposed_group_full_observable_state=sum(row["proposed_group_would_be_full_observable_state_faithful"] for row in rows)),
        criterion="annihilator(selected group)=ker_Z(P.T), plus row_Z(P)=full visible physical character module for faithfulness",
        scope="exact arithmetic conditional on unique posthoc bounded-catalog physical-label matching; not observation-only topology certification")
    OUTPUT.write_text(json.dumps(result, indent=2, allow_nan=False, default=int)+"\n")
    print(json.dumps(result["summary"]))


if __name__ == "__main__":
    main()
