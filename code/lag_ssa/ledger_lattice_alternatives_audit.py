"""Post-freeze exact coverage audit; no true state chosen for inference.

Reuses the separately frozen raw-ledger posthoc physical kernel only after
all six public-only alternative searches have finished and hashes verify.
No fitting, new validation score, rank selection or source modification.
"""

from project_paths import RESULTS_DIR

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from sympy import Matrix

from ledger_group_selection import row_hnf


HERE = Path(__file__).resolve().parent
SOURCE = RESULTS_DIR / 'ledger_lattice_alternatives.json'
TRUTH_AUDIT = RESULTS_DIR / 'ledger_group_selection_audit.json'
OUTPUT = RESULTS_DIR / 'ledger_lattice_alternatives_audit.json'


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if OUTPUT.exists():raise FileExistsError("preserve previous alternatives audit")
    source = json.loads(SOURCE.read_text())
    if (source["status"] != "finished_search_only" or not source["all_six_attempts_frozen_before_truth"]
            or source["evaluator_files_opened"] or not source["sources_and_inputs_unchanged"]
            or source["summary"]["finished_runs"] != 6):
        raise ValueError("completed all6 public-only search freeze required")
    source_hash = digest(SOURCE)
    for name, expected in source["sources"].items():
        if digest(HERE/name) != expected:raise ValueError("frozen search source changed")
    for entry in source["manifest"]:
        if (digest(entry["input"]) != entry["input_sha256"]
                or digest(entry["ledger_parameters"]) != entry["ledger_parameters_sha256"]):
            raise ValueError("frozen public/ledger input changed")
    for row in source["rows"]:
        if digest(row["alternative_states"]) != row["alternative_states_sha256"]:
            raise ValueError("frozen variants changed before truth coverage read")
    # First truth-derived file access occurs after the complete freeze checks.
    truth_hash = digest(TRUTH_AUDIT)
    truth = json.loads(TRUTH_AUDIT.read_text())
    truth_by_run = {row["run"]:row for row in truth["rows"]}
    rows = []
    for source_row in source["rows"]:
        candidates = json.loads(Path(source_row["alternative_states"]).read_text())
        observed = truth_by_run[source_row["run"]]
        dimension = candidates["observed_line_count"]
        kernel_rows = observed["true_image_kernel_HNF_truth_only"]
        kernel = Matrix(len(kernel_rows), dimension, [int(v) for row in kernel_rows for v in row])
        if row_hnf(kernel_rows, dimension) != kernel:
            raise ArithmeticError("posthoc target kernel is not canonical exact HNF")
        physical_rows = observed["physical_basis_P_truth_only"]
        physical = Matrix(len(physical_rows), len(physical_rows[0]), [int(v) for row in physical_rows for v in row])
        correct, all_counts, final_passing_counts, final_passing_valid_counts = [], {}, {}, {}
        all_hnf_canonical = True
        for state in candidates["all_registered_states"]:
            relation = row_hnf(state["relation_rows_HNF"], dimension)
            canonical = relation.tolist() == state["relation_rows_HNF"]
            all_hnf_canonical &= canonical
            if not canonical:raise ArithmeticError("registered row matrix differs from its exact unsaturated HNF")
            exact_image = relation == kernel
            valid = relation*physical == Matrix.zeros(relation.rows, physical.cols)
            rank = str(state["relation_rank"])
            all_counts[rank] = all_counts.get(rank, 0)+1
            if state["retained_in_final_beam"] and state["representation_sieve_passed"]:
                final_passing_counts[rank] = final_passing_counts.get(rank, 0)+1
                if valid:final_passing_valid_counts[rank] = final_passing_valid_counts.get(rank, 0)+1
            if exact_image:
                correct.append(dict(state_id=state["state_id"], generated_stage=state["first_generated_stage"],
                    retained_final=state["retained_in_final_beam"], sieve_passing=state["representation_sieve_passed"],
                    whole_representation_complexity=state["whole_representation_complexity"],
                    complexity_budget=state["whole_representation_budget"],
                    finite_component_count=state["finite_component_count"], free_dimension=state["free_dimension"],
                    integer_windings=state["integer_windings"],
                    canonical_exact_HNF_verified=True))
        rows.append(dict(run=source_row["run"], registered_state_count=len(candidates["all_registered_states"]),
            final_beam_state_count=len(candidates["final_beam_state_ids"]),
            all_registered_relation_rank_counts=all_counts,
            final_sieve_passing_relation_rank_counts=final_passing_counts,
            final_sieve_passing_physically_valid_relation_rank_counts_truth_only=final_passing_valid_counts,
            every_registered_HNF_canonical_exact=all_hnf_canonical,
            correct_observed_kernel_registered=bool(correct),
            correct_observed_kernel_retained_final=any(state["retained_final"] for state in correct),
            correct_observed_kernel_sieve_passing=any(state["sieve_passing"] for state in correct),
            correct_observed_kernel_retained_and_sieve_passing=any(state["retained_final"] and state["sieve_passing"] for state in correct),
            correct_kernel_variants_truth_only=correct,
            ledger_complete_visible_module_truth_only=observed["complete_visible_module_truth_only"],
            faithful_full_state_variant_available_truth_only=observed["complete_visible_module_truth_only"]
                and any(state["retained_final"] and state["sieve_passing"] for state in correct),
            no_true_rank_kernel_or_candidate_returned_to_inference=True))
    if digest(SOURCE) != source_hash or digest(TRUTH_AUDIT) != truth_hash:
        raise ValueError("frozen search or posthoc kernel audit changed")
    for name, expected in source["sources"].items():
        if digest(HERE/name) != expected:raise ValueError("frozen source changed during audit")
    payload = dict(status="finished_read_only_coverage_audit", created_at=datetime.now(timezone.utc).isoformat(),
        source=str(SOURCE), source_sha256=source_hash, truth_audit=str(TRUTH_AUDIT), truth_audit_sha256=truth_hash,
        audit_source_sha256=digest(Path(__file__)), all_six_search_outputs_frozen_before_truth_coverage=True,
        no_fit_new_validation_score_or_rank_choice=True, exact_unsaturated_lattice_equality_not_Q_span=True,
        sources_unchanged=True, rows=rows,
        summary=dict(runs=6, correct_kernel_registered=sum(row["correct_observed_kernel_registered"] for row in rows),
            correct_kernel_retained_final=sum(row["correct_observed_kernel_retained_final"] for row in rows),
            correct_kernel_sieve_passing=sum(row["correct_observed_kernel_sieve_passing"] for row in rows),
            correct_kernel_retained_and_sieve_passing=sum(row["correct_observed_kernel_retained_and_sieve_passing"] for row in rows),
            complete_visible_module=sum(row["ledger_complete_visible_module_truth_only"] for row in rows),
            faithful_full_state_variant_available=sum(row["faithful_full_state_variant_available_truth_only"] for row in rows),
            unknown_group_or_phase_rank_actually_selected=False))
    OUTPUT.write_text(json.dumps(payload, indent=2, allow_nan=False)+"\n")
    print(json.dumps(payload["summary"]), flush=True)
    for row in rows:
        print(json.dumps(dict(run=row["run"], registered=row["correct_observed_kernel_registered"],
            retained=row["correct_observed_kernel_retained_final"],
            passing=row["correct_observed_kernel_retained_and_sieve_passing"],
            correct_variants=row["correct_kernel_variants_truth_only"],
            final_passing_rank_counts=row["final_sieve_passing_relation_rank_counts"])), flush=True)


if __name__ == "__main__":main()
