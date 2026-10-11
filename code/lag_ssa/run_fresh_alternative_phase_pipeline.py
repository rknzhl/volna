"""Fixed posthoc fresh6 replay with bounded transactional candidate search.

The original six inputs are now previously inspected cases. Only the
discovery orchestration changes. Settings, proposal equations, thresholds,
group/pruning/MSSA inference and raw-window contexts remain fixed. Every
replay output freezes before any evaluator archive is reopened here.
"""

from project_paths import RESULTS_DIR

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np
from sympy import Matrix, Integer

import alternative_cover_selection_ssa as alternative
import run_fresh_phase_pipeline as frozen
from fresh_group_image_audit import row_hnf, integer_kernel_rows, relation_assessment


HERE = Path(__file__).resolve().parent
SOURCE = RESULTS_DIR / 'fresh_phase_pipeline.json'
ROOT = RESULTS_DIR / 'fresh_alternative_phase_pipeline'
OUTPUT = ROOT.with_suffix(".json")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprints():
    result = frozen.fingerprints()
    result.update({name:digest(HERE/name) for name in ("alternative_cover_selection_ssa.py",
        "run_fresh_alternative_phase_pipeline.py", "fresh_group_image_audit.py")})
    return result


def save(payload):
    def symbolic_integer(value):
        if isinstance(value, Integer):
            return int(value)
        raise TypeError("unsupported result type: "+type(value).__name__)
    temporary = OUTPUT.with_suffix(".tmp")
    temporary.write_text(json.dumps(frozen.wave.serializable(payload), indent=2, allow_nan=False,
                                   default=symbolic_integer)+"\n")
    temporary.replace(OUTPUT)


def score_corrected_group_image(row, entry):
    """Reuse field/phase scores, then exact connected-image/faithfulness test."""
    frozen.score(row, entry)
    metadata = row["module_truth_only"]
    metadata["original_square_basis_boolean"] = metadata.get("character_basis_complete")
    if metadata.get("status") != "scored_unique_bounded_catalog" or not metadata.get("relations_valid"):
        row["full_physical_module_and_group_correct_truth_only"] = False
        metadata["group_image_evaluation_status"] = "unresolved_without_unique_physical_character_matching"
        return
    p = Matrix(metadata["physical_basis"])
    with np.load(entry["evaluation"], allow_pickle=False) as evaluation:
        visible = row_hnf(evaluation["group_modes"].tolist(), p.cols)
    recovered = row_hnf(p.tolist(), p.cols)
    complete = recovered == visible
    kernel = integer_kernel_rows(p)
    with np.load(row["dictionary"]["frozen_parameters"], allow_pickle=False) as parameters:
        relation = parameters["relation"]
    selected_rows = [relation.tolist()] if len(relation) else []
    assessment = relation_assessment(selected_rows, p, kernel)
    proposed = row["compact"]["group_proposal"].get("relation", [])
    proposed_assessment = relation_assessment([proposed] if proposed else [], p, kernel)
    faithful = bool(complete and assessment["equals_full_image_annihilator"])
    metadata.update(character_basis_complete=bool(complete), complete_visible_character_module=bool(complete),
        physical_latent_group_correct=assessment["equals_full_image_annihilator"],
        imposed_relation_physically_valid=assessment["every_relation_valid_on_physical_image"],
        selected_group_image_correct=assessment["equals_full_image_annihilator"],
        selected_full_observable_state_faithful=faithful,
        group_image_evaluation_status="exact_kernel_and_visible_module_posthoc",
        recovered_physical_character_module_HNF=recovered.tolist(), visible_physical_character_module_HNF=visible.tolist(),
        true_image_integer_annihilator_HNF=kernel.tolist(), selected_relation_assessment=assessment,
        proposed_relation_assessment=proposed_assessment,
        character_basis_completeness_is_group_correctness=False,
        nonzero_relation_necessarily_false_for_redundant_chart=False,
        exact_group_certified_from_observations=False)
    row["full_physical_module_and_group_correct_truth_only"] = faithful
    row["phase_truth_only"]["incomplete_factor_not_physical_full_state"] = not faithful
    row["group_truth_criterion"] = "chosen relation lattice equals ker_Z(P.T); rowsP equal full visible module"


def main():
    if ROOT.exists() or OUTPUT.exists():
        raise FileExistsError("preserve alternative replay")
    source = json.loads(SOURCE.read_text())
    if source["status"] != "finished" or len(source["manifest"]) != 6:
        raise ValueError("requires completed immutable original fresh6")
    initial = fingerprints();source_hash = digest(SOURCE);started = time.perf_counter();rows = []
    for name in ("discovery_parameters", "compact/parameters", "compact/projected", "matched_mssa/parameters",
                 "matched_mssa/projected", "dictionary/parameters", "dictionary/projected"):
        (ROOT/name).mkdir(parents=True, exist_ok=False)
    manifest = source["manifest"]
    payload = dict(status="running", created_at=datetime.now(timezone.utc).isoformat(), rows=rows, manifest=manifest,
        source_fingerprints=initial, original_fresh_source=str(SOURCE), original_fresh_source_sha256=source_hash,
        settings=asdict(frozen.SETTINGS), group_options=frozen.compact.GROUP_OPTIONS,
        projection_settings=asdict(frozen.compact.PROJECTION),
        protected_dictionary_thresholds=list(frozen.sparse.THRESHOLDS),
        conditioning_preservation_factor=frozen.sparse.CONDITIONING_FACTOR,
        mssa_rank_grid=list(frozen.comparison.RANK_GRID), evaluation_files_opened=False,
        independent_fresh_external_validation=False, posthoc_repair_on_previously_examined_six_cases=True,
        no_generator_regeneration=True, no_old_factor_parameters_given_to_discovery=True,
        method_change="try all original frontend-accepted alternatives in descending represented_energy; full transaction rollback after rejection",
        proposal_frequency_bounds_or_gate_thresholds_changed=False,
        candidate_bank_limit="existing rank/mode proposal count; only exact same rank/mode deduplicated",
        alternative_ranked_by_true_frequency_or_phase_quality=False,
        stale_global_label_or_rank_joint_selection_changed=False,
        exact_group_evaluation="kernel_Z(physical character map.T), with full visible row module for faithful state",
        truth_or_catalog_given_to_inference=False, novelty_claimed=False)
    save(payload)
    previous_root, previous_cover = frozen.ROOT, frozen.cover
    frozen.ROOT, frozen.cover = ROOT, alternative
    try:
        for entry in manifest:
            print("starting alternative-search replay "+entry["run"], flush=True)
            public_entry = {key:entry[key] for key in ("run", "input", "input_sha256", "n_train", "n_validation", "n_future")}
            try:
                row = frozen.fit_public_entry(public_entry)
                row["compact"]["previously_examined_benchmark"] = True
                row["posthoc_repair_on_previously_examined_case"] = True
            except Exception as error:
                row = dict(run=entry["run"], case=entry["case"], run_status="exception",
                    exception=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
            rows.append(row);save(payload)
            stages = row.get("discovery", {}).get("inference", {}).get("stages", [])
            print(json.dumps(dict(run=row["run"], status=row["run_status"],
                rank=row.get("discovery", {}).get("fitted_phase_count"),
                group=row.get("compact", {}).get("selected_model"),
                alternative_attempts=sum("alternative_index" in stage for stage in stages),
                rejected_transactions=sum("candidate_transaction_rollback" in stage for stage in stages),
                seconds=row.get("elapsed_seconds"), exception=row.get("exception"))), flush=True)
    finally:
        frozen.ROOT, frozen.cover = previous_root, previous_cover
    if fingerprints() != initial or digest(SOURCE) != source_hash:
        raise ValueError("immutable replay sources changed during inference")
    if any(digest(entry["input"]) != entry["input_sha256"] for entry in manifest):
        raise ValueError("immutable original observations changed")
    payload.update(all_six_attempts_and_outputs_frozen_before_truth_open=True,
        inference_finished_at=datetime.now(timezone.utc).isoformat(), source_fingerprints_unchanged_before_scoring=True)
    save(payload)
    for row, entry in zip(rows, manifest):
        if row["run_status"] == "fitted_and_frozen":
            score_corrected_group_image(row, entry)
            print(json.dumps(dict(run=row["run"], full_state_faithful=row["full_physical_module_and_group_correct_truth_only"],
                endpoint={key:value["sensor_nrmse"] for key,value in row["endpoint_field"].items()},
                diagonal={key:value["sensor_nrmse"] for key,value in row["diagonal_field"].items()})), flush=True)
    finished = [row for row in rows if row["run_status"] == "finished"]
    payload.update(status="finished", evaluation_files_opened=True, source_fingerprints_unchanged=fingerprints() == initial,
        total_elapsed_seconds=time.perf_counter()-started,
        summary=dict(requested_runs=6, finished_runs=len(finished), exceptions=len(rows)-len(finished),
            complete_visible_character_module=sum(row["module_truth_only"].get("complete_visible_character_module", False) for row in finished),
            correct_selected_phase_image=sum(row["module_truth_only"].get("selected_group_image_correct", False) for row in finished),
            correct_full_observable_state=sum(row["full_physical_module_and_group_correct_truth_only"] for row in finished),
            better_endpoint_than_MSSA=sum(row["endpoint_field"]["pruned"]["sensor_nrmse"] < row["endpoint_field"]["MSSA"]["sensor_nrmse"] for row in finished),
            better_diagonal_than_MSSA=sum(row["diagonal_field"]["pruned"]["sensor_nrmse"] < row["diagonal_field"]["MSSA"]["sensor_nrmse"] for row in finished)))
    save(payload);print(json.dumps(payload["summary"]), flush=True)


if __name__ == "__main__":
    main()
