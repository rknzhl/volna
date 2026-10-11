"""Public-only exact-algebra beam probe; default prepares protocol without search."""

from project_paths import RESULTS_DIR

from dataclasses import asdict
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np

from ledger_lattice_alternatives import SETTINGS, search_alternatives


HERE = Path(__file__).resolve().parent
SOURCE = RESULTS_DIR / 'ledger_group_selection.json'
ROOT = RESULTS_DIR / 'ledger_lattice_alternatives'
OUTPUT = ROOT.with_suffix(".json")
PROTOCOL = RESULTS_DIR / 'ledger_lattice_alternatives_protocol.json'
SOURCES = ("ledger_lattice_alternatives.py", "run_ledger_lattice_alternatives.py", "ledger_group_selection.py")


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def source_inputs():
    source = json.loads(SOURCE.read_text())
    if source["status"] != "finished_inference_only" or not source["all_six_attempts_and_outputs_frozen_before_truth"]:
        raise ValueError("completed public inference-only group proposals required")
    row_by_run = {row["run"]:row for row in source["rows"]}
    result = []
    for entry in source["manifest"]:
        row = row_by_run[entry["run"]]
        if row["run_status"] != "fitted_and_frozen":raise ValueError("all six public proposal rows required")
        filtered = {key:entry[key] for key in ("run", "input", "input_sha256", "ledger_parameters", "ledger_parameters_sha256", "n_train")}
        if digest(filtered["input"]) != filtered["input_sha256"] or digest(filtered["ledger_parameters"]) != filtered["ledger_parameters_sha256"]:
            raise ValueError("public or observed-ledger inputs changed")
        filtered["eligible_proposals"] = row["inference"]["proposals"]["eligible"]
        result.append(filtered)
    return result


def declaration():
    return dict(status="prepared_not_run", created_at=datetime.now(timezone.utc).isoformat(),
        sources={name:digest(HERE/name) for name in SOURCES}, source_group_json=str(SOURCE), source_group_json_sha256=digest(SOURCE),
        manifest=source_inputs(), settings=asdict(SETTINGS), source_math_review_required=True,
        prior_cases_examined=True, exact_integer_search_probe_not_general_method=True,
        source_proposals_only=True, new_fit_or_new_VAL_score=False, evaluator_or_catalog_loaded=False,
        gauge="Vred=V diag(Irho,Tcol), Vinvred=diag(Irho,Tcol^-1)Vinv; Tcol from LLL(A.T).transform.T",
        complexity="max_row(l1(reduced free weights)+sum centered finite weights) <= floor(sqrt(N_train))",
        finite_component_count_budget="N_train", GL_global_minimum_or_necessary_condition_claimed=False,
        exact_unsaturated_HNF_and_source_integer_winding_lift=True,
        search="fixed global64 skip/include beam, rank-balanced quotas; ineligible intermediate states remain expandable",
        rank_not_selected_and_different_rank_numeric_scores_not_compared=True,
        coverage_assessment_only_after_all_six_alternative_outputs_frozen=True, novelty_claimed=False)


def main(run=False):
    prepared_now = declaration()
    if not run:
        if PROTOCOL.exists():raise FileExistsError("preserve protocol")
        write_json(PROTOCOL, prepared_now)
        print("Prepared lattice-alternatives protocol; search not run: "+str(PROTOCOL), flush=True)
        return
    if not PROTOCOL.exists():raise FileNotFoundError("prepare/review protocol before --run")
    previous = json.loads(PROTOCOL.read_text())
    for key in ("sources", "source_group_json_sha256", "manifest", "settings"):
        current = json.loads(json.dumps(prepared_now[key]))
        if previous[key] != current:raise ValueError("sources/public/settings changed: "+key)
    if ROOT.exists() or OUTPUT.exists():raise FileExistsError("preserve previous search")
    ROOT.mkdir(parents=True)
    started, rows = time.perf_counter(), []
    payload = dict(prepared_now, status="running", rows=rows, all_six_attempts_frozen_before_truth=False,
                   protocol_sha256=digest(PROTOCOL), evaluator_files_opened=False)
    write_json(OUTPUT, payload)
    for entry in prepared_now["manifest"]:
        begin = time.perf_counter()
        try:
            with np.load(entry["ledger_parameters"], allow_pickle=False) as archive:
                frequency, covariance = archive["frequencies"], archive["conditional_frequency_covariance"]
            if covariance.shape != (len(frequency), len(frequency)) or not np.isfinite(covariance).all():
                raise ValueError("frozen full frequency covariance required")
            result = search_alternatives(entry["eligible_proposals"], len(frequency), entry["n_train"], SETTINGS,
                progress=lambda status:print(json.dumps(dict(run=entry["run"], **status)), flush=True))
            path = ROOT/(entry["run"]+".json")
            write_json(path, result)
            row = dict(run=entry["run"], run_status="searched_and_frozen", alternative_states=str(path),
                alternative_states_sha256=digest(path), registered_states=len(result["all_registered_states"]),
                sieve_passing_states=len(result["registered_sieve_passing_state_ids"]),
                final_beam_states=len(result["final_beam_state_ids"]), elapsed_seconds=time.perf_counter()-begin)
        except Exception as error:
            row = dict(run=entry["run"], run_status="exception", exception=f"{type(error).__name__}: {error}",
                       traceback=traceback.format_exc(), elapsed_seconds=time.perf_counter()-begin)
        rows.append(row)
        write_json(OUTPUT, payload)
        print(json.dumps(row), flush=True)
    if {name:digest(HERE/name) for name in SOURCES} != prepared_now["sources"] or digest(SOURCE) != prepared_now["source_group_json_sha256"]:
        raise ValueError("immutable search sources changed")
    for entry in prepared_now["manifest"]:
        if digest(entry["input"]) != entry["input_sha256"] or digest(entry["ledger_parameters"]) != entry["ledger_parameters_sha256"]:
            raise ValueError("immutable public/ledger inputs changed")
    finished = [row for row in rows if row["run_status"] == "searched_and_frozen"]
    if any(digest(row["alternative_states"]) != row["alternative_states_sha256"] for row in finished):
        raise ValueError("frozen variants changed")
    payload.update(status="finished_search_only", all_six_attempts_frozen_before_truth=True,
        inference_finished_at=datetime.now(timezone.utc).isoformat(), sources_and_inputs_unchanged=True,
        evaluator_files_opened=False, total_elapsed_seconds=time.perf_counter()-started,
        summary=dict(requested_runs=6, finished_runs=len(finished), exceptions=6-len(finished),
                     true_kernel_coverage_not_yet_scored=True))
    write_json(OUTPUT, payload)
    print(json.dumps(payload["summary"]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="run search only after source/math gate")
    main(parser.parse_args().run)
