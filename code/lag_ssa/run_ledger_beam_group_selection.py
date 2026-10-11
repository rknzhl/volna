"""Posthoc inference-only refit of every frozen eligible final-beam state.

Default declares the immutable public input/candidate/source protocol. --run
requires the reviewed same protocol. All six attempts freeze before any later
evaluation. The only selection score is RAW sensor clock-forecast VAL MSE.
"""

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

from ledger_beam_group_selection import SETTINGS, select_beam_group
from ledger_group_selection import parameter_arrays


HERE = Path(__file__).resolve().parent
SOURCE = RESULTS_DIR / 'ledger_lattice_alternatives.json'
GROUP_SOURCE = RESULTS_DIR / 'ledger_group_selection.json'
ROOT = RESULTS_DIR / 'ledger_beam_group_selection'
OUTPUT = ROOT.with_suffix(".json")
PROTOCOL_PATH = RESULTS_DIR / 'ledger_beam_group_selection_protocol_v2.json'
SOURCES = ("ledger_beam_group_selection.py", "run_ledger_beam_group_selection.py",
           "ledger_group_selection.py", "raw_spectral_ledger.py", "observed_phase_group.py",
           "ledger_lattice_alternatives.py", "run_ledger_lattice_alternatives.py")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, payload):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def fingerprints():
    return {name:digest(HERE/name) for name in SOURCES}


def source_manifest():
    alternatives = json.loads(SOURCE.read_text())
    group = json.loads(GROUP_SOURCE.read_text())
    if (alternatives["status"] != "finished_search_only"
            or not alternatives["all_six_attempts_frozen_before_truth"]
            or alternatives["evaluator_files_opened"] or len(alternatives["rows"]) != 6
            or group["status"] != "finished_inference_only"
            or not group["all_six_attempts_and_outputs_frozen_before_truth"]
            or group["evaluation_files_opened"]):
        raise ValueError("requires six frozen inference-only alternatives and group inputs")
    if alternatives["source_group_json_sha256"] != digest(GROUP_SOURCE):
        raise ValueError("alternatives refer to another frozen group source")
    for declaration,key in ((alternatives,"sources"), (group,"source_fingerprints")):
        for name,expected in declaration[key].items():
            if digest(HERE/name) != expected:raise ValueError("changed old source: "+name)
    if digest(group["raw_ledger_source"]) != group["raw_ledger_source_sha256"]:
        raise ValueError("changed original raw-ledger inference source")
    public_by_run = {entry["run"]:entry for entry in group["manifest"]}
    group_rows = {row["run"]:row for row in group["rows"]}
    manifest, protected = [], []
    for row in group["rows"]:
        for model in row["model_files"]:
            if digest(model["parameters"]) != model["parameters_sha256"]:
                raise ValueError("changed old group-model parameter archive")
            protected.append({key:model[key] for key in ("parameters", "parameters_sha256")})
    for row in alternatives["rows"]:
        if row["run_status"] != "searched_and_frozen":raise ValueError("all six frozen searches required")
        if digest(row["alternative_states"]) != row["alternative_states_sha256"]:
            raise ValueError("changed frozen beam-state archive")
        states = json.loads(Path(row["alternative_states"]).read_text())
        by_id = {state["state_id"]:state for state in states["all_registered_states"]}
        final_ids = states["final_beam_state_ids"]
        if len(set(final_ids)) != len(final_ids):raise ValueError("duplicate final beam state")
        final = [by_id[identifier] for identifier in final_ids if by_id[identifier]["representation_sieve_passed"]]
        if any(not state["retained_in_final_beam"] for state in final):raise ValueError("inconsistent final-beam scope")
        public = public_by_run[row["run"]]
        entry = {key:public[key] for key in ("run", "input", "input_sha256", "n_train", "n_validation", "n_future",
                                              "ledger_parameters", "ledger_parameters_sha256")}
        if digest(entry["input"]) != entry["input_sha256"] or digest(entry["ledger_parameters"]) != entry["ledger_parameters_sha256"]:
            raise ValueError("changed public input or original raw-ledger parameters")
        entry.update(alternative_states=row["alternative_states"], alternative_states_sha256=row["alternative_states_sha256"],
                     final_state_count=len(final_ids), final_sieve_candidate_count=len(final), final_sieve_states=final,
                     noise_estimate_train_only=group_rows[row["run"]]["noise_estimate_train_only"])
        manifest.append(entry)
    return manifest, protected


def protocol():
    manifest, protected = source_manifest()
    return dict(status="prepared_not_run", created_at=datetime.now(timezone.utc).isoformat(),
        source_fingerprints=fingerprints(), source_alternatives=str(SOURCE), source_alternatives_sha256=digest(SOURCE),
        source_group=str(GROUP_SOURCE), source_group_sha256=digest(GROUP_SOURCE),
        manifest=manifest, protected_original_group_models=protected, settings=asdict(SETTINGS),
        source_math_review_required_before_run=True,
        posthoc_repair_after_prior_evaluation=True, independent_external_test=False,
        candidate_scope="ALL frozen final-beam states passing representation_sieve, in frozen final-beam order",
        truthmatched_candidate_subset_used=False, true_r_catalogue_or_PDE_given_to_fit=False,
        candidate_windings="exact frozen state integer homomorphism; no re-rounding HNF times floating frequencies",
        candidate_gauge="frozen U,D,V_reduced,V_inverse_reduced; exact URV=D, unimodular inverse and all characters checked",
        computational_sieve="existing floor(sqrt(N_train)) character-weight and N_train finite-component budgets, unchanged",
        fit_backend="unchanged ledger_group_selection.fit_tied_lines/_prediction/GroupSettings",
        fit_scope="RAW TRAIN full OLS; SLSQP100; independent one-DFT-bin cells around original frozen raw-ledger frequencies",
        baseline="untouched original raw ledger AND same-backend unconstrained refit",
        selection="strict minimum RAW sensor VAL clock-forecast MSE; exact ties retain earlier/free model",
        phase_gauge="phi_n=nu*n; c_origin=c_center exp(-i nu time_center)",
        clock_validation_only=True, observed_window_phase_or_MSSA_result_claimed=False,
        representation_budget_or_validation_gain_certifies_physical_rank=False,
        exact_rationality_free_density_or_relation_completeness_certified=False,
        all_six_freeze_before_later_evaluation=True, evaluation_files_opened=False,
        future_values_given_to_fit=False, no_new_lines_or_proposals_generated=True, novelty_claimed=False)


def validate_prepared(prepared):
    current = protocol()
    for key in ("source_fingerprints", "source_alternatives_sha256", "source_group_sha256",
                "manifest", "protected_original_group_models", "settings"):
        normalized = json.loads(json.dumps(current[key]))
        if normalized != prepared[key]:raise ValueError("protocol/source/public/candidate settings changed: "+key)


def fit_entry(entry):
    begin = time.perf_counter()
    for path,key in ((entry["input"],"input_sha256"), (entry["ledger_parameters"],"ledger_parameters_sha256"),
                     (entry["alternative_states"],"alternative_states_sha256")):
        if digest(path) != entry[key]:raise ValueError("changed frozen entry input")
    with np.load(entry["input"], allow_pickle=False) as public:
        if set(public.files) != {"Y","s","t"}:raise ValueError("exactly public Y,s,t required")
        y, positions, times = public["Y"], public["s"], public["t"]
    with np.load(entry["ledger_parameters"], allow_pickle=False) as archive:
        parameters = {key:archive[key] for key in archive.files}
    n_train, cutoff = entry["n_train"], entry["n_train"]+entry["n_validation"]
    if (not np.array_equal(times, np.arange(cutoff+entry["n_future"]))
            or not np.array_equal(positions, parameters["sensor_positions"])
            or y.shape != (len(positions), len(times))):
        raise ValueError("public frame/geometry differs from original frozen ledger")
    fitted = select_beam_group(y[:, :n_train], y[:, n_train:cutoff], times[:n_train], times[n_train:cutoff],
        parameters, float(entry["noise_estimate_train_only"]["sigma"]), entry["final_sieve_states"], SETTINGS,
        progress=lambda event:print(json.dumps(dict(run=entry["run"], **event)), flush=True))
    model_files = []
    for index, model in enumerate(fitted["models"]):
        arrays = parameter_arrays(model)
        arrays["physical_mass"], arrays["spatial_mass_sqrt"] = parameters["physical_mass"], parameters["spatial_mass_sqrt"]
        arrays["sensor_positions"] = positions
        path = ROOT/"parameters"/(entry["run"]+f"__model{index:03d}.npz")
        np.savez_compressed(path, **arrays)
        model_files.append(dict(model_index=index, model_kind=model["model_kind"],
            beam_state_id=model.get("beam_state_id"), parameters=str(path), parameters_sha256=digest(path),
            validation_raw_sensor_mse=model["validation_raw_sensor_mse"], presentation=model["presentation"]))
    selected = fitted["selected_model_index"]
    return dict(run=entry["run"], run_status="fitted_and_frozen", inference=fitted["audit"], model_files=model_files,
        selected_model_index=selected, selected_parameters=model_files[selected]["parameters"],
        selected_parameters_sha256=model_files[selected]["parameters_sha256"],
        noise_estimate_train_only=entry["noise_estimate_train_only"], sensor_count=len(positions),
        spatial_dimension=positions.shape[1], evaluation_file_opened=False,
        true_rank_or_physical_labels_loaded=False, elapsed_seconds=time.perf_counter()-begin)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    arguments = parser.parse_args()
    if not arguments.run:
        if ROOT.exists() or OUTPUT.exists() or PROTOCOL_PATH.exists():raise FileExistsError("preserve existing beam-refit artifacts")
        prepared = protocol()
        write_json(PROTOCOL_PATH, prepared)
        print(json.dumps(dict(status=prepared["status"], protocol=str(PROTOCOL_PATH),
            candidates={entry["run"]:entry["final_sieve_candidate_count"] for entry in prepared["manifest"]},
            source_fingerprints=prepared["source_fingerprints"])), flush=True)
        return
    prepared = json.loads(PROTOCOL_PATH.read_text())
    validate_prepared(prepared)
    if ROOT.exists() or OUTPUT.exists():raise FileExistsError("preserve previous beam-group fit")
    (ROOT/"parameters").mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    payload = dict(prepared, status="running_inference_only", rows=[], protocol_file=str(PROTOCOL_PATH),
        protocol_sha256=digest(PROTOCOL_PATH), all_six_attempts_and_outputs_frozen_before_truth=False)
    write_json(OUTPUT, payload)
    for entry in prepared["manifest"]:
        print("starting frozen beam refit "+entry["run"], flush=True)
        try:
            row = fit_entry(entry)
        except Exception as error:
            row = dict(run=entry["run"], run_status="exception", exception=f"{type(error).__name__}: {error}",
                       traceback=traceback.format_exc())
        payload["rows"].append(row)
        write_json(OUTPUT, payload)
        print(json.dumps(dict(run=row["run"], status=row["run_status"], selected=row.get("selected_model_index"),
            dimension=row.get("inference",{}).get("selected_presentation",{}).get("continuous_dimension"),
            components=row.get("inference",{}).get("selected_presentation",{}).get("component_count"),
            elapsed_seconds=row.get("elapsed_seconds"), exception=row.get("exception"))), flush=True)
    validate_prepared(prepared)
    finished = [row for row in payload["rows"] if row["run_status"] == "fitted_and_frozen"]
    for row in finished:
        if any(digest(model["parameters"]) != model["parameters_sha256"] for model in row["model_files"]):
            raise ValueError("changed newly frozen model archive")
    payload.update(status="finished_inference_only", all_six_attempts_and_outputs_frozen_before_truth=True,
        finished_at=datetime.now(timezone.utc).isoformat(), source_and_input_hashes_unchanged=True,
        evaluation_files_opened=False, total_elapsed_seconds=time.perf_counter()-started,
        summary=dict(requested_runs=6, fitted_runs=len(finished), exceptions=6-len(finished),
            candidate_attempts=sum(len(entry["final_sieve_states"]) for entry in prepared["manifest"]),
            selected_groups={row["run"]:dict(dimension=row["inference"]["selected_presentation"]["continuous_dimension"],
                components=row["inference"]["selected_presentation"]["component_count"]) for row in finished},
            true_group_or_rank_not_scored=True, observed_window_phase_or_MSSA_not_scored=True))
    write_json(OUTPUT, payload)
    print(json.dumps(dict(status=payload["status"], summary=payload["summary"], output=str(OUTPUT))), flush=True)


if __name__ == "__main__":
    main()
