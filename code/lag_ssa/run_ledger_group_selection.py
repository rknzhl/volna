"""Inference-only group-model assay from frozen observed spectral ledgers.

Default prepares a public/ledger-only protocol. --run requires independently
reviewed sources and the same prepared hashes/settings. No evaluator, physical
frequency catalogue, old harmonic label, generator law or true rank is loaded.
Every selected/admissible model of all6 attempts freezes before later scoring.
This bounded computational probe is not a general group-recovery certificate.
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

from ledger_group_selection import SETTINGS, select_ledger_group, parameter_arrays


HERE = Path(__file__).resolve().parent
SOURCE = RESULTS_DIR / 'raw_spectral_ledger.json'
ROOT = RESULTS_DIR / 'ledger_group_selection'
OUTPUT = ROOT.with_suffix(".json")
PROTOCOL_PATH = RESULTS_DIR / 'ledger_group_selection_protocol.json'
SOURCES = ("ledger_group_selection.py", "run_ledger_group_selection.py",
           "raw_spectral_ledger.py", "observed_phase_group.py")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key:serializable(item) for key,item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(item) for item in value]
    return value


def write_json(path, payload):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(serializable(payload), indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def fingerprints():
    return {name:digest(HERE/name) for name in SOURCES}


def source_manifest():
    source = json.loads(SOURCE.read_text())
    if (source["status"] != "finished_inference_only"
            or not source["all_six_attempts_and_outputs_frozen_before_truth"]
            or source["evaluation_files_opened"] or len(source["manifest"]) != 6):
        raise ValueError("requires original completed inference-only six public ledgers")
    rows = {row["run"]:row for row in source["rows"]}
    result = []
    for public in source["manifest"]:
        row = rows[public["run"]]
        if row["run_status"] != "fitted_and_frozen":
            raise ValueError("all six frozen observed ledgers required")
        entry = {key:public[key] for key in ("run", "input", "input_sha256", "n_train", "n_validation", "n_future")}
        entry.update(ledger_parameters=row["frozen_parameters"],
            ledger_parameters_sha256=row["frozen_parameters_sha256"],
            noise_estimate_train_only=row["noise_estimate_train_only"])
        if (digest(entry["input"]) != entry["input_sha256"]
                or digest(entry["ledger_parameters"]) != entry["ledger_parameters_sha256"]):
            raise ValueError("frozen public input or ledger parameters changed")
        result.append(entry)
    return result


def protocol():
    return dict(status="prepared_not_run", created_at=datetime.now(timezone.utc).isoformat(),
        source_fingerprints=fingerprints(), raw_ledger_source=str(SOURCE), raw_ledger_source_sha256=digest(SOURCE),
        manifest=source_manifest(), settings=asdict(SETTINGS), prior_case_inspection=True,
        independent_external_test=False, diagnostic_computational_probe=True,
        mathematically_general_method_or_search_completeness_established=False,
        input_scope="frozen raw-ledger inference parameters and public TRAIN/VAL Y,s,t only",
        source_math_review_required_before_run=True, true_rank_law_frequency_or_character_catalog_given=False,
        old_harmonic_labels_reused=False, new_lines_or_harmonics_added=False,
        covariance_screen="wrapped actual a.nu / sqrt(a.Cov.nu.a) <=3; conditional diagnostic, not CI",
        observed_product_rule="actual stored TRAIN/VAL products at.95 are reported only; no acceptance gate",
        primitive_divisor_rule="independent candidate with complete re-screen; never normalize or saturate acceptedR",
        integer_algebra="unsaturated HNF; exact U R V=D, phi=V z, V_inverse exact integer",
        finite_components="all retained; unvisited fitted-clock finite components reject candidate; count>100 unavailable for later observer",
        fit="SLSQP100, full unpenalized RAW sensor OLS, independent one-bin cells about final observed frequencies",
        baseline="untouched free ledger plus matched unconstrained SLSQP/one-bin/full-OLS refit",
        selection="strict lowest RAW sensor VAL MSE, no physical-rank target or tie preference for constraints",
        phase_gauge="phi_n=nu*n; profiles_origin=profiles_center exp(-i nu center)",
        group_uncertainty="exact rationality/completeness not certified; all near-relation/model alternatives recorded",
        evaluation_files_opened=False, future_values_given_to_fit=False, novelty_claimed=False)


def fit_entry(entry):
    started = time.perf_counter()
    for path,key in ((entry["input"], "input_sha256"), (entry["ledger_parameters"], "ledger_parameters_sha256")):
        if digest(path) != entry[key]:
            raise ValueError("changed frozen inference input")
    with np.load(entry["input"], allow_pickle=False) as public:
        if set(public.files) != {"Y", "s", "t"}:
            raise ValueError("exactly public Y,s,t required")
        y, positions, times = public["Y"], public["s"], public["t"]
    with np.load(entry["ledger_parameters"], allow_pickle=False) as archive:
        parameters = {key:archive[key] for key in archive.files}
    train, cutoff = entry["n_train"], entry["n_train"]+entry["n_validation"]
    if (not np.array_equal(times, np.arange(train+entry["n_validation"]+entry["n_future"]))
            or not np.array_equal(positions, parameters["sensor_positions"])
            or y.shape[1] != len(times)):
        raise ValueError("public times or sensor geometry differ from frozen ledger")
    sigma = float(entry["noise_estimate_train_only"]["sigma"])
    fitted = select_ledger_group(y[:, :train], y[:, train:cutoff], times[:train], times[train:cutoff],
                                 parameters, sigma, SETTINGS)
    model_files = []
    for index, model in enumerate(fitted["models"]):
        arrays = parameter_arrays(model)
        arrays["physical_mass"] = parameters["physical_mass"]
        arrays["spatial_mass_sqrt"] = parameters["spatial_mass_sqrt"]
        arrays["sensor_positions"] = positions
        path = ROOT/"parameters"/(entry["run"]+f"__model{index:03d}.npz")
        np.savez_compressed(path, **arrays)
        model_files.append(dict(model_index=index, model_kind=model["model_kind"],
            parameters=str(path), parameters_sha256=digest(path),
            validation_raw_sensor_mse=model["validation_raw_sensor_mse"], presentation=model["presentation"]))
    selected = fitted["selected_model_index"]
    return dict(run=entry["run"], run_status="fitted_and_frozen", inference=fitted["audit"],
        model_files=model_files, selected_model_index=selected,
        selected_parameters=model_files[selected]["parameters"],
        selected_parameters_sha256=model_files[selected]["parameters_sha256"],
        noise_estimate_train_only=entry["noise_estimate_train_only"], sensor_count=len(positions),
        spatial_dimension=positions.shape[1], evaluation_file_opened=False,
        true_rank_or_physical_labels_loaded=False, elapsed_seconds=time.perf_counter()-started)


def main(run=False):
    declaration = protocol()
    if not run:
        if PROTOCOL_PATH.exists():
            raise FileExistsError("preserve existing group-stage protocol")
        write_json(PROTOCOL_PATH, declaration)
        print("Prepared bounded group protocol; no model fitted: "+str(PROTOCOL_PATH), flush=True)
        return
    if not PROTOCOL_PATH.exists():
        raise FileNotFoundError("prepare and review protocol before --run")
    prepared = json.loads(PROTOCOL_PATH.read_text())
    for key in ("source_fingerprints", "raw_ledger_source_sha256", "manifest", "settings"):
        if prepared[key] != serializable(declaration[key]):
            raise ValueError("group source/input/settings changed since preparation: "+key)
    if ROOT.exists() or OUTPUT.exists():
        raise FileExistsError("preserve previous group-stage results")
    (ROOT/"parameters").mkdir(parents=True, exist_ok=False)
    started, rows = time.perf_counter(), []
    payload = dict(declaration, status="running", rows=rows,
        protocol_file=str(PROTOCOL_PATH), protocol_sha256=digest(PROTOCOL_PATH),
        all_six_attempts_and_outputs_frozen_before_truth=False)
    write_json(OUTPUT, payload)
    for entry in declaration["manifest"]:
        print("starting bounded whole-ledger group "+entry["run"], flush=True)
        try:
            row = fit_entry(entry)
        except Exception as error:
            row = dict(run=entry["run"], run_status="exception",
                exception=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        rows.append(row)
        write_json(OUTPUT, payload)
        print(json.dumps(dict(run=row["run"], status=row["run_status"],
            selected=row.get("selected_model_index"),
            presentation=row.get("inference", {}).get("selected_presentation", {}).get("continuous_dimension"),
            seconds=row.get("elapsed_seconds"), exception=row.get("exception"))), flush=True)
    if fingerprints() != declaration["source_fingerprints"] or digest(SOURCE) != declaration["raw_ledger_source_sha256"]:
        raise ValueError("frozen group-stage source changed")
    for entry in declaration["manifest"]:
        if (digest(entry["input"]) != entry["input_sha256"]
                or digest(entry["ledger_parameters"]) != entry["ledger_parameters_sha256"]):
            raise ValueError("public or raw-ledger inference parameters changed")
    finished = [row for row in rows if row["run_status"] == "fitted_and_frozen"]
    for row in finished:
        if any(digest(model["parameters"]) != model["parameters_sha256"] for model in row["model_files"]):
            raise ValueError("frozen group-model parameters changed")
    payload.update(status="finished_inference_only", all_six_attempts_and_outputs_frozen_before_truth=True,
        inference_finished_at=datetime.now(timezone.utc).isoformat(), source_and_input_hashes_unchanged=True,
        evaluation_files_opened=False, total_elapsed_seconds=time.perf_counter()-started,
        summary=dict(requested_runs=6, fitted_runs=len(finished), exceptions=6-len(finished),
            selected_candidate_continuous_dimensions={row["run"]:row["inference"]["selected_presentation"]["continuous_dimension"]
                                                       for row in finished},
            physical_rank_group_or_phase_recovery_not_scored=True))
    write_json(OUTPUT, payload)
    print(json.dumps(payload["summary"]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="fit only after independent source/math review")
    main(parser.parse_args().run)
