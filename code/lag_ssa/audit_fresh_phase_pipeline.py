"""Read-only provenance, composition and rejection audit of fresh6.

No re-fitting, alternative hyperparameters or corrected predictions. Exact
group-image semantics are documented separately by fresh_group_image_audit.
"""

from project_paths import RESULTS_DIR

import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
SOURCE = RESULTS_DIR / 'fresh_phase_pipeline.json'
GROUP_AUDIT = RESULTS_DIR / 'fresh_group_image_audit.json'
OUTPUT = RESULTS_DIR / 'fresh_phase_pipeline_audit.json'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def complex_field(labels, coefficient, phase):
    count = len(labels)
    character = np.exp(1j*np.einsum("ij,jt->it", labels, phase, optimize=False))
    c = (coefficient[1:1+count]-1j*coefficient[1+count:])/2
    return coefficient[0, :, None]+2*np.real(np.einsum("mq,mt->qt", c, character, optimize=False))


def main():
    if OUTPUT.exists():
        raise FileExistsError("preserve fresh6 audit")
    p = json.loads(SOURCE.read_text())
    if p["status"] != "finished" or not p["all_six_attempts_and_outputs_frozen_before_truth_open"]:
        raise ValueError("requires completed frozen inference before audit")
    rows = [];all_hashes = True;number_hashes = 0
    for entry, row in zip(p["manifest"], p["rows"]):
        archive_checks = {"input":digest(entry["input"]) == entry["input_sha256"],
                          "evaluation":digest(entry["evaluation"]) == entry["evaluation_sha256"]}
        for stage in ("discovery", "compact", "matched_mssa", "dictionary"):
            for key in ("source_model", "frozen_parameters", "frozen_output"):
                if key in row[stage]:
                    archive_checks[stage+"_"+key] = digest(row[stage][key]) == row[stage][key+"_sha256"]
        all_hashes = all_hashes and all(archive_checks.values())
        number_hashes += len(archive_checks)
        with np.load(row["dictionary"]["frozen_parameters"]) as archive:
            parameters = {key:archive[key] for key in archive.files}
        with np.load(row["dictionary"]["frozen_output"]) as archive:
            arrays = {key:archive[key] for key in archive.files}
        endpoint_mass = complex_field(parameters["labels"], parameters["coefficients_mass"], arrays["phase_at_window_end"])
        endpoint = np.einsum("ij,jt->it", parameters["mass_to_sensor"], endpoint_mass, optimize=False)
        start = arrays["phase_at_window_start"];count = start.shape[1];window = 128
        folded = np.zeros((endpoint_mass.shape[0], count+window-1));counts = np.zeros(count+window-1)
        for lag in range(window):
            column = complex_field(parameters["labels"], parameters["coefficients_mass"],
                                   start+parameters["frequencies"][:, None]*lag)
            folded[:, lag:lag+count] += column;counts[lag:lag+count] += 1
        diagonal = np.einsum("ij,jt->it", parameters["mass_to_sensor"],
                             (folded/counts[None, :])[:, window-1:], optimize=False)
        phase_advance = np.angle(np.exp(1j*(arrays["phase_at_window_end"]-start
                                          -parameters["frequencies"][:, None]*(window-1))))
        stages = row["discovery"]["inference"]["stages"]
        diagnostics = []
        for stage in stages:
            front = stage.get("frontend", {})
            diagnostics.append({key:value for key,value in dict(step=stage.get("step"), status=stage.get("status"),
                candidate_frequency=front.get("candidate_frequency"), frontend_reason=front.get("reason"),
                validation_mse_improvement=stage.get("validation_mse_improvement"),
                validation_gain_positive=stage.get("validation_gain_positive"),
                old_relation_rejection_reason=stage.get("old_relation_rejection_reason"),
                accepted_state_rolled_back=stage.get("accepted_state_rolled_back")).items() if value is not None})
        module = row["module_truth_only"]
        rows.append(dict(run=row["run"], archive_hashes=archive_checks,
            learned_ambient_phase_count=row["discovery"]["fitted_phase_count"], physical_phase_count_truth_only=module["physical_rank"],
            learned_character_physical_basis_truth_only=module.get("physical_basis"),
            missing_visible_character_mass_energy_fraction_truth_only=module.get("missing_visible_character_mass_energy_fraction_truth_only"),
            selected_group_family=row["compact"]["selected_model"],
            group_validation_losses={key:value["mass_mse_per_field_snapshot"] for key,value in row["compact"]["validation_cv"].items()},
            phase_dictionary_before=row["dictionary"]["original_count"], phase_dictionary_after=row["dictionary"]["selected_count"],
            independent_complex_endpoint_max_difference=float(np.max(abs(endpoint-arrays["prediction_endpoint"]))),
            independent_complex_diagonal_max_difference=float(np.max(abs(diagonal-arrays["prediction_diagonal"]))),
            wrapped_relative_phase_advance_max_difference=float(np.max(abs(phase_advance))),
            diagonal_available_counts_match=bool(np.array_equal(counts[window-1:], arrays["diagonal_available_counts"])),
            discovery_stage_diagnostics=diagnostics,
            all_acquired_anchors_preserved_and_lattice_index1=all(candidate["all_original_observed_anchors_preserved"]
                and candidate["character_lattice_index"] == 1 for candidate in row["dictionary"]["candidate_validation"])))
    sources_ok = all(digest(HERE/name) == expected for name,expected in p["source_fingerprints"].items())
    result = dict(status="finished", source=str(SOURCE), source_sha256=digest(SOURCE),
        audit_source_sha256=digest(__file__), source_fingerprints_unchanged=sources_ok,
        all_public_evaluation_parameter_and_output_hashes_unchanged=all_hashes,
        archive_hashes_checked=number_hashes, rows=rows,
        all_six_attempts_outputs_before_truth_open=p["all_six_attempts_and_outputs_frozen_before_truth_open"],
        independent_complex_endpoint_max_difference=max(row["independent_complex_endpoint_max_difference"] for row in rows),
        independent_complex_diagonal_max_difference=max(row["independent_complex_diagonal_max_difference"] for row in rows),
        gradient_converged_windows=sum(row["dictionary"]["optimizer"]["gradient_converged_count"] for row in p["rows"]),
        total_windows=sum(row["dictionary"]["optimizer"]["window_count"] for row in p["rows"]),
        final_gradient_max=max(row["dictionary"]["optimizer"]["final_gradient_max"] for row in p["rows"]),
        summary=p["summary"], no_source_or_output_changed=True, no_model_refitted=True,
        honest_conclusion="The fresh full pipeline succeeds on the full physical state only1/6 cases. Field error beats matched MSSA4/6 but does not certify missing phases. Pruning cannot repair failed or redundant temporal/module discovery.",
        group_image_semantics="Nonzero relation may encode redundant acquired coordinates even for nonresonant physical rotations. Exact image requires the complete integer kernel of the physical character matrix; full physical state also requires its row module to contain every visible generator.",
        group_image_audit=str(GROUP_AUDIT) if GROUP_AUDIT.exists() else None,
        group_image_audit_sha256=digest(GROUP_AUDIT) if GROUP_AUDIT.exists() else None)
    OUTPUT.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({key:result[key] for key in ("source_fingerprints_unchanged", "all_public_evaluation_parameter_and_output_hashes_unchanged",
        "archive_hashes_checked", "independent_complex_endpoint_max_difference", "independent_complex_diagonal_max_difference",
        "gradient_converged_windows", "total_windows")}))


if __name__ == "__main__":
    main()
