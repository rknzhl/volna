"""Independent read-only audit of the fixed six-case candidate-search replay.

Checks saved observations, sources, transactions and nonlinear SSA folding.
No model is fitted, no output is changed, and no truth chooses a parameter.
"""

from project_paths import RESULTS_DIR

import hashlib
import json
from pathlib import Path

import numpy as np

from audit_fresh_phase_pipeline import complex_field


HERE = Path(__file__).resolve().parent
SOURCE = RESULTS_DIR / 'fresh_alternative_phase_pipeline.json'
OUTPUT = RESULTS_DIR / 'fresh_alternative_phase_pipeline_audit.json'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if OUTPUT.exists():
        raise FileExistsError("preserve completed replay audit")
    payload = json.loads(SOURCE.read_text())
    if payload["status"] != "finished" or not payload["all_six_attempts_and_outputs_frozen_before_truth_open"]:
        raise ValueError("requires all six frozen attempts and completed evaluation")
    original = json.loads(Path(payload["original_fresh_source"]).read_text())
    rows = []
    for entry, row, old_row in zip(payload["manifest"], payload["rows"], original["rows"]):
        if row["run_status"] != "finished":
            rows.append(dict(run=row["run"], status=row["run_status"], exception=row.get("exception")))
            continue
        hashes = {"input": digest(entry["input"]) == entry["input_sha256"],
                  "evaluation": digest(entry["evaluation"]) == entry["evaluation_sha256"]}
        for stage in ("discovery", "compact", "matched_mssa", "dictionary"):
            for key in ("source_model", "frozen_parameters", "frozen_output"):
                if key in row[stage]:
                    hashes[stage + "_" + key] = digest(row[stage][key]) == row[stage][key + "_sha256"]
        with np.load(row["dictionary"]["frozen_parameters"], allow_pickle=False) as archive:
            parameters = {key: archive[key] for key in archive.files}
        with np.load(row["dictionary"]["frozen_output"], allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        endpoint_mass = complex_field(parameters["labels"], parameters["coefficients_mass"],
                                      arrays["phase_at_window_end"])
        endpoint = np.einsum("ij,jt->it", parameters["mass_to_sensor"], endpoint_mass, optimize=False)
        start = arrays["phase_at_window_start"]
        count = start.shape[1]
        window = 128
        folded = np.zeros((endpoint_mass.shape[0], count + window - 1))
        counts = np.zeros(count + window - 1)
        for lag in range(window):
            column = complex_field(parameters["labels"], parameters["coefficients_mass"],
                                   start + parameters["frequencies"][:, None] * lag)
            folded[:, lag:lag + count] += column
            counts[lag:lag + count] += 1
        diagonal = np.einsum("ij,jt->it", parameters["mass_to_sensor"],
                             (folded / counts[None, :])[:, window - 1:], optimize=False)
        stages = row["discovery"]["inference"]["stages"]
        attempts = [stage for stage in stages if "alternative_index" in stage]
        rollbacks = [stage for stage in stages if "candidate_transaction_rollback" in stage]
        rollback_valid = all(stage["candidate_transaction_rollback"]["exact_cache_snapshot_restored"]
            and stage["candidate_transaction_rollback"]["checkpoint_sha256"] ==
                stage["candidate_transaction_rollback"]["restored_sha256"]
            and stage["accepted_inference_state_restored_before_next_alternative"] for stage in rollbacks)
        rows.append(dict(run=row["run"], status="finished", hashes=hashes,
            alternative_attempts=len(attempts), rejected_transactions=len(rollbacks),
            all_transaction_snapshots_restored=rollback_valid,
            accepted_alternative_indices=[stage["alternative_index"] for stage in attempts
                if stage["status"].startswith("accepted_")],
            physical_module_and_image_correct=row["full_physical_module_and_group_correct_truth_only"],
            endpoint_field=row["endpoint_field"], diagonal_field=row["diagonal_field"],
            mssa_endpoint_score_exactly_unchanged=row["endpoint_field"]["MSSA"] == old_row["endpoint_field"]["MSSA"],
            mssa_diagonal_score_exactly_unchanged=row["diagonal_field"]["MSSA"] == old_row["diagonal_field"]["MSSA"],
            independent_complex_endpoint_max_difference=float(np.max(abs(endpoint - arrays["prediction_endpoint"]))),
            independent_complex_diagonal_max_difference=float(np.max(abs(diagonal - arrays["prediction_diagonal"]))),
            diagonal_available_counts_match=bool(np.array_equal(counts[window - 1:], arrays["diagonal_available_counts"])),
            gradient_converged_windows=row["dictionary"]["optimizer"]["gradient_converged_count"],
            total_windows=row["dictionary"]["optimizer"]["window_count"]))
    finished = [row for row in rows if row["status"] == "finished"]
    result = dict(status="finished", source=str(SOURCE), source_sha256=digest(SOURCE),
        audit_source_sha256=digest(__file__), rows=rows, summary=payload["summary"],
        sources_unchanged=all(digest(HERE / name) == expected for name, expected in payload["source_fingerprints"].items()),
        original_fresh_result_unchanged=digest(payload["original_fresh_source"]) == payload["original_fresh_source_sha256"],
        all_archive_hashes_unchanged=all(all(row["hashes"].values()) for row in finished),
        all_rejected_transactions_restored=all(row["all_transaction_snapshots_restored"] for row in finished),
        same_original_six_inputs=payload["manifest"] == original["manifest"],
        all_mssa_scores_unchanged=all(row["mssa_endpoint_score_exactly_unchanged"] and
                                    row["mssa_diagonal_score_exactly_unchanged"] for row in finished),
        all_six_attempts_outputs_before_truth_open=payload["all_six_attempts_and_outputs_frozen_before_truth_open"],
        independent_complex_endpoint_max_difference=max(row["independent_complex_endpoint_max_difference"] for row in finished),
        independent_complex_diagonal_max_difference=max(row["independent_complex_diagonal_max_difference"] for row in finished),
        gradient_converged_windows=sum(row["gradient_converged_windows"] for row in finished),
        total_windows=sum(row["total_windows"] for row in finished),
        independent_external_validation=False, posthoc_repair_on_previously_examined_six_cases=True,
        no_source_or_output_changed=True, no_model_refitted=True, novelty_claimed=False)
    OUTPUT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key not in ("rows",)}))


if __name__ == "__main__":
    main()
