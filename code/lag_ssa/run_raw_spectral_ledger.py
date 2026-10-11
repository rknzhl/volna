"""Public-only six-case free-line ledger hypothesis probe.

Default execution prepares the predeclared protocol and public-input hashes.
Fitting requires --run after the independent source/math review. This runner
never loads an evaluator, analytic generator, true rank/catalogue, old factor
parameters or old result JSON. All six attempts freeze before any later score.
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

from raw_spectral_ledger import SETTINGS, fit_raw_spectral_ledger, frozen_parameter_arrays
from run_actual_wave_iterative import noise_from_training_high_band


HERE = Path(__file__).resolve().parent
PUBLIC_ROOT = RESULTS_DIR / 'fresh_phase_pipeline/inputs'
ROOT = RESULTS_DIR / 'raw_spectral_ledger'
OUTPUT = ROOT.with_suffix(".json")
PROTOCOL_PATH = RESULTS_DIR / 'raw_spectral_ledger_protocol.json'
RUN_NAMES = ("fresh_r2_primitive_cover6", "fresh_r3_infinite_weak", "fresh_r3_nofund_cover3",
             "fresh_r2_irrational_travelling_1d", "fresh_weak_phase06", "fresh_weak_phase008")
SOURCES = ("raw_spectral_ledger.py", "run_raw_spectral_ledger.py",
    "alternative_cover_selection_ssa.py", "cover_selection_ssa.py", "iterative_ssa.py",
    "ssa_character.py", "nodal_spatial.py", "run_actual_wave_iterative.py")


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


def public_manifest():
    result = []
    for run in RUN_NAMES:
        path = PUBLIC_ROOT/(run+".npz")
        with np.load(path, allow_pickle=False) as public:
            if set(public.files) != {"Y", "s", "t"}:
                raise ValueError("exactly public Y,s,t required")
            expected = SETTINGS.n_train+SETTINGS.n_validation+SETTINGS.n_future
            if public["Y"].shape[1] != expected or not np.array_equal(public["t"], np.arange(expected)):
                raise ValueError("predeclared public sample counts changed")
        result.append(dict(run=run, input=str(path), input_sha256=digest(path),
                           n_train=SETTINGS.n_train, n_validation=SETTINGS.n_validation,
                           n_future=SETTINGS.n_future))
    return result


def protocol():
    return dict(created_at=datetime.now(timezone.utc).isoformat(), status="prepared_not_run",
        source_fingerprints=fingerprints(), manifest=public_manifest(), settings=asdict(SETTINGS),
        source_math_review_required_before_run=True, prior_case_inspection=True,
        independent_external_test=False, truth_or_cached_factors_given_to_inference=False,
        future_values_given_to_fitter=False, no_evaluator_or_generation_file_loaded=True,
        inference_objective="unpenalized joint RAW sensor RSS/(q*N), all line frequencies independent",
        source_of_proposals="unchanged original supported SSA/DMD bank, mass-coordinate residual",
        candidate_initializer="observed SSA phase slope, not true frequency or a harmonic of another line",
        frequency_search="all frequency parameters independently within one DFT bin of current initializer",
        duplicate_rule="observed initializers within quarter DFT bin merge stably; report unresolved pairs",
        line_acceptance="viable finite full-rank converged fit and positive RAW sensor VAL gain",
        line_search_order="first unchanged supported proposal, descending represented_energy",
        group_or_integer_labels_selected=False, harmonics_synthesized=False,
        raw_iid_and_mass_noise_distinguished=True, full_gram=True, frequency_covariance_calibrated=False,
        adaptive_validation_is_not_unbiased_risk=True, maximum_lines_has_no_infinite_tail_certificate=True,
        hypothesis="free observed-line ledger may avoid false harmonic assignments and keep weak independent lines",
        general_working_method_or_new_SSA_claimed=False)


def fit_entry(entry):
    started = time.perf_counter()
    if digest(entry["input"]) != entry["input_sha256"]:
        raise ValueError("public observations changed")
    with np.load(entry["input"], allow_pickle=False) as public:
        if set(public.files) != {"Y", "s", "t"}:
            raise ValueError("only public Y,s,t allowed")
        y, positions, times = public["Y"], public["s"], public["t"]
    train, cutoff = SETTINGS.n_train, SETTINGS.n_train+SETTINGS.n_validation
    noise = noise_from_training_high_band(y[:, :train])
    fitted = fit_raw_spectral_ledger(y[:, :train], y[:, train:cutoff], times[:train], times[train:cutoff],
                                   positions, noise, SETTINGS)
    arrays = frozen_parameter_arrays(fitted, times[:train], times[train:cutoff])
    path = ROOT/"parameters"/(entry["run"]+".npz")
    np.savez_compressed(path, **arrays)
    return dict(run=entry["run"], run_status="fitted_and_frozen", inference=fitted["audit"],
        frozen_parameters=str(path), frozen_parameters_sha256=digest(path),
        frequency_count=len(fitted["model"].frequencies), sensor_count=len(positions),
        spatial_dimension=positions.shape[1], noise_estimate_train_only=noise,
        evaluation_file_opened=False, old_factor_parameters_loaded=False,
        fit_inputs="only TRAIN/VAL Y,s,t; no future values", elapsed_seconds=time.perf_counter()-started)


def main(run=False):
    declaration = protocol()
    if not run:
        if PROTOCOL_PATH.exists():
            raise FileExistsError("preserve existing ledger protocol")
        write_json(PROTOCOL_PATH, declaration)
        print("Protocol prepared; no model fitted: "+str(PROTOCOL_PATH), flush=True)
        return
    if not PROTOCOL_PATH.exists():
        raise FileNotFoundError("prepare and review protocol before --run")
    prepared = json.loads(PROTOCOL_PATH.read_text())
    if (prepared["source_fingerprints"] != declaration["source_fingerprints"]
            or prepared["manifest"] != declaration["manifest"]
            or prepared["settings"] != serializable(declaration["settings"])):
        raise ValueError("source/public/settings changed after protocol preparation")
    if ROOT.exists() or OUTPUT.exists():
        raise FileExistsError("preserve prior ledger outputs")
    (ROOT/"parameters").mkdir(parents=True, exist_ok=False)
    started, rows = time.perf_counter(), []
    payload = dict(declaration, status="running", rows=rows,
        protocol_file=str(PROTOCOL_PATH), protocol_sha256=digest(PROTOCOL_PATH),
        evaluation_files_opened=False, all_six_attempts_and_outputs_frozen_before_truth=False)
    write_json(OUTPUT, payload)
    for entry in declaration["manifest"]:
        print("starting RAW observed-line ledger "+entry["run"], flush=True)
        try:
            row = fit_entry(entry)
        except Exception as error:
            row = dict(run=entry["run"], run_status="exception",
                exception=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        rows.append(row)
        write_json(OUTPUT, payload)
        print(json.dumps(dict(run=row["run"], status=row["run_status"],
            lines=row.get("frequency_count"), termination=row.get("inference", {}).get("status"),
            seconds=row.get("elapsed_seconds"), exception=row.get("exception"))), flush=True)
    if fingerprints() != declaration["source_fingerprints"]:
        raise ValueError("source changed during fitting")
    if any(digest(entry["input"]) != entry["input_sha256"] for entry in declaration["manifest"]):
        raise ValueError("public inputs changed during fitting")
    fitted_rows = [row for row in rows if row["run_status"] == "fitted_and_frozen"]
    if any(digest(row["frozen_parameters"]) != row["frozen_parameters_sha256"] for row in fitted_rows):
        raise ValueError("frozen ledger output changed")
    payload.update(status="finished_inference_only", all_six_attempts_and_outputs_frozen_before_truth=True,
        inference_finished_at=datetime.now(timezone.utc).isoformat(), sources_unchanged=True,
        public_inputs_unchanged=True, evaluation_files_opened=False,
        total_elapsed_seconds=time.perf_counter()-started,
        summary=dict(requested_runs=6, fitted_runs=len(fitted_rows), exceptions=6-len(fitted_rows),
            observed_line_counts={row["run"]:row["frequency_count"] for row in fitted_rows},
            physical_phase_rank_or_true_lines_scored=False))
    write_json(OUTPUT, payload)
    print(json.dumps(payload["summary"]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="run only after independent source/math review")
    main(parser.parse_args().run)
