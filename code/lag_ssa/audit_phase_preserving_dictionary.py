"""Read-only post-fit audit of the fixed20 dictionary repair.

Scored character-basis completeness is separated from correctness of an
imposed sampled group. Evaluation metadata is read only after all inference
and outputs are frozen. No audit result changes a model or candidate rule.
"""

from project_paths import RESULTS_DIR

import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
SOURCE = RESULTS_DIR / 'phase_preserving_dictionary.json'
OUTPUT = RESULTS_DIR / 'phase_preserving_dictionary_audit.json'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def composition(labels, coefficients, phase):
    """Independent signed-complex evaluation of the saved real coefficients."""
    count = len(labels)
    character = np.exp(1j*np.einsum("mr,rt->mt", labels, phase, optimize=False))
    complex_coefficient = (coefficients[1:1+count]-1j*coefficients[1+count:])/2
    return coefficients[0, :, None]+2*np.real(np.einsum("mq,mt->qt", complex_coefficient,
                                                     character, optimize=False))


def main():
    if OUTPUT.exists():
        raise FileExistsError("preserve the independent post-fit audit")
    payload = json.loads(SOURCE.read_text())
    if payload["status"] != "finished" or not payload["all20_attempts_outputs_and_parameters_frozen_before_truth_open"]:
        raise ValueError("audit requires all completed and frozen inference attempts")
    rows = []
    for entry, row in zip(payload["manifest"], payload["rows"]):
        with np.load(row["frozen_parameters"], allow_pickle=False) as p:
            parameters = {key:p[key] for key in p.files}
        with np.load(row["frozen_output"], allow_pickle=False) as p:
            phases = p["phase_at_window_end"];prediction = p["prediction_endpoint"]
        mass_field = composition(parameters["labels"], parameters["coefficients_mass"], phases)
        field = np.einsum("ij,jt->it", parameters["mass_to_sensor"], mass_field, optimize=False)
        phase_metadata = row.get("phase_truth_only", {})
        errors = phase_metadata.get("phase_errors_in_recovered_factor_coordinates", {})
        old_mae = errors.get("old", {}).get("mae_rad")
        new_mae = errors.get("new", {}).get("mae_rad")
        relation = parameters["relation"]
        relation_information = dict(imposed_relation=relation.tolist(),
            exact_physical_group_certified_by_algorithm=False,
            character_basis_completeness_is_physical_group_correctness=False,
            physical_latent_group_correct=None)
        if len(relation):
            with np.load(entry["evaluation"], allow_pickle=False) as evaluation:
                if entry["kind"] == "wave":
                    with np.load(entry["input"]) as public:
                        step = float(public["t"][1]-public["t"][0])
                    physical_rotation = 2*np.pi*evaluation["source_frequencies_hz"]*step
                else:
                    physical_rotation = evaluation["alpha"]
            basis = phase_metadata.get("physical_basis_truth_only")
            if basis is not None:
                presented_rotation = np.einsum("ij,j->i", np.asarray(basis, dtype=int),
                                               physical_rotation, optimize=False)
                residue = float(abs(np.angle(np.exp(1j*np.dot(relation, presented_rotation)))))
                relation_information.update(generator_rotation_relation_residue_rad_per_sample=residue,
                    residue_assessment_is_postfit_generator_check_not_inference=True,
                    imposed_relation_supported_by_generator_to_numerical_tolerance=residue <= 1e-12)
            if entry["case"] == "unresolvable_detuning":
                relation_information.update(physical_latent_group_correct=False,
                    reason="The selected order50 relation is false for this declared detuned generator; finite-window prediction remains accurate.",
                    actual_orbit_closure_claimed=False)
        if entry["case"] == "wave_1d_source_noise":
            relation_information.update(carrier_coordinates_only=True,
                completeness_claim_for_full_field=False, physical_latent_group_correct=False,
                reason="Source-noise control lies outside the pure phase-field class.")
        rows.append(dict(run=row["run"],
            parameter_hash_unchanged=digest(row["frozen_parameters"]) == row["frozen_parameters_sha256"],
            output_hash_unchanged=digest(row["frozen_output"]) == row["frozen_output_sha256"],
            independent_complex_endpoint_composition_max_difference=float(np.max(abs(field-prediction))),
            selected_threshold=row["selected_threshold"], original_count=row["original_count"], retained_count=row["selected_count"],
            character_basis_complete_truth_only=bool(phase_metadata.get("physical_module_complete", False)),
            physical_group_qualification=relation_information,
            recovered_factor_phase_mae_old=old_mae, recovered_factor_phase_mae_new=new_mae,
            phase_mae_change_rad=None if new_mae is None or old_mae is None else new_mae-old_mae,
            phase_mae_strictly_improved=None if new_mae is None or old_mae is None else new_mae < old_mae))
    sources_unchanged = all(digest(HERE/name) == expected for name, expected in payload["source_fingerprints"].items())
    result = dict(status="finished", source=str(SOURCE), source_sha256=digest(SOURCE),
        audit_source_sha256=digest(__file__), source_fingerprints_unchanged=sources_unchanged,
        all_parameter_and_output_hashes_unchanged=all(row["parameter_hash_unchanged"] and row["output_hash_unchanged"] for row in rows),
        all20_frozen_before_truth_open=payload["all20_attempts_outputs_and_parameters_frozen_before_truth_open"],
        rows=rows, summary=payload["summary"],
        maximum_independent_complex_endpoint_composition_difference=max(row["independent_complex_endpoint_composition_max_difference"] for row in rows),
        optimizer_gradient_converged_windows=sum(row["optimizer"]["gradient_converged_count"] for row in payload["rows"]),
        total_future_windows=sum(row["optimizer"]["window_count"] for row in payload["rows"]),
        maximum_final_gradient=max(row["optimizer"]["final_gradient_max"] for row in payload["rows"]),
        conditioning_guard_rejections=sum(not candidate["eligible_for_validation_selection"] for row in payload["rows"] for candidate in row["candidate_validation"]),
        conditioning_guard_empirical_benefit_demonstrated=False,
        all_acquired_anchors_preserved_and_character_lattice_index1=all(candidate["all_original_observed_anchors_preserved"] and candidate["character_lattice_index"] == 1 for row in payload["rows"] for candidate in row["candidate_validation"]),
        all_phase_errors_improved=False,
        interpretation="All four prior in-class field losses are repaired, while phase MAE has tiny increases for r3 no-fundamentals and the already incomplete weak-frontier factor. The tiny-detuning compact relation remains physically false despite finite-horizon accuracy. Source-noise remains an out-of-class failure. The numerical conditioning guard rejected no candidate here.",
        fixed_hyperparameters_changed_by_audit=False, fresh_external_test_claimed=False)
    OUTPUT.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({key:result[key] for key in ("source_fingerprints_unchanged", "all_parameter_and_output_hashes_unchanged",
        "maximum_independent_complex_endpoint_composition_difference", "optimizer_gradient_converged_windows", "total_future_windows",
        "conditioning_guard_rejections")}))


if __name__ == "__main__":
    main()
