"""Actual-window encoding with a strictly raw-field, clock-free phase seed.

Learned frequencies, windows, labels, field coefficients and optimizer settings
remain frozen. Hann profiles and their gauge are estimated from training only;
future seed readouts use the raw observed field without clock subtraction.
"""

from project_paths import RESULTS_DIR

import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np

import run_general_robustness_suite as suite
import run_hankel_torus_projection as hankel
from phase_readout_refinement import fit_field_character_observer
from ssa_character import aligned_character_rows


HERE = Path(__file__).resolve().parent
ROOT = RESULTS_DIR / 'raw_seed_hankel_audit'
OUTPUT = ROOT.with_suffix(".json")
SOURCE = RESULTS_DIR / 'general_robustness_factor_cover.json'
SOURCE_NAMES = ("run_raw_seed_hankel_audit.py", "hankel_torus_projection.py",
                "run_hankel_torus_projection.py", "phase_readout_refinement.py",
                "toric_phase_projection.py", "ssa_character.py")


def hashes():
    return {name: hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in SOURCE_NAMES}


def save(payload):
    temporary = OUTPUT.with_suffix(".tmp")
    temporary.write_text(json.dumps(suite.serializable(payload), indent=2, allow_nan=False)+"\n")
    temporary.replace(OUTPUT)


def freeze_raw_seed(entry, old_row):
    with np.load(entry["input"]) as public:
        if set(public.files) != {"Y", "s", "t"}:
            raise ValueError("unexpected public keys")
        observed = public["Y"]
    source_path = Path(old_row["frozen_arrays"])
    if hashlib.sha256(source_path.read_bytes()).hexdigest() != old_row["frozen_arrays_sha256"]:
        raise ValueError("frozen fitted model changed")
    with np.load(source_path) as archive:
        arrays = {key: archive[key] for key in archive.files}
    transform = arrays["spatial_mass_sqrt"]
    training = np.einsum("ij,jt->it", transform, observed[:, :suite.TRAIN], optimize=False)
    future = np.einsum("ij,jt->it", transform, observed[:, -suite.FUTURE:], optimize=False)
    training_phases, future_phases, metadata = [], [], []
    for frequency, window in zip(arrays["acquired_frequencies"], arrays["readout_windows"]):
        observer = fit_field_character_observer(training, float(frequency), int(window))
        training_phases.append(observer.apply(training)["phase"])
        future_phases.append(observer.apply(future)["phase"])
        metadata.append(observer.metadata())
    training_phases = aligned_character_rows(training_phases)
    future_phases = aligned_character_rows(future_phases)
    times = np.arange(training_phases.shape[1])
    origins = np.angle(np.mean(np.exp(1j*(training_phases-arrays["acquired_frequencies"][:, None]*times)), axis=1))
    train_factor = np.einsum("ij,jt->it", arrays["phase_products"], training_phases-origins[:, None], optimize=False)
    phase = np.einsum("ij,jt->it", arrays["phase_products"], future_phases-origins[:, None], optimize=False)
    angles = np.einsum("ij,jt->it", arrays["fit_labels"], phase, optimize=False)
    features = np.column_stack((np.ones(phase.shape[1]), np.cos(angles.T), np.sin(angles.T)))
    mass_prediction = np.einsum("nj,jq->qn", features, arrays["fit_coefficients"], optimize=False)
    arrays.update(train_phase=training_phases, future_phase=future_phases,
        observed_factor_phase=phase, phase_origin_train_only=origins,
        raw_train_factor_phase=train_factor,
        observed_phase_decoder_prediction=np.einsum("ij,jt->it", arrays["mass_to_sensor"], mass_prediction, optimize=False),
        readout_sources=np.asarray(["raw_field"]*len(origins)))
    path = ROOT/"parameters"/(entry["run"]+".npz")
    np.savez_compressed(path, **arrays)
    row = {**old_row, "frozen_arrays": str(path), "frozen_arrays_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return row, metadata


def phase_score(row, entry, fitted_row):
    basis = fitted_row["module_truth_only"].get("phase_basis_in_physical_coordinates")
    if basis is None:
        return {"status": "unscored_no_unique_posthoc_physical_basis"}
    with np.load(entry["evaluation"]) as truth:
        theta = truth["theta"]
    with np.load(row["raw_parameter_archive"]) as parameters:
        train_factor = parameters["raw_train_factor_phase"]
    with np.load(row["frozen_output"]) as result:
        phase = result["hankel_phase"]
    basis = np.asarray(basis, dtype=int)
    target_train = np.einsum("ij,jt->it", basis, theta[:, :train_factor.shape[1]], optimize=False)
    origin = np.angle(np.mean(np.exp(1j*(train_factor-target_train)), axis=1))
    start, count = row["time_start"], row["common_time_start_count"]
    target_future = np.einsum("ij,jt->it", basis, theta[:, start:start+count], optimize=False)
    error = np.angle(np.exp(1j*(phase-origin[:, None]-target_future)))
    return {"future_phase_mae_rad": float(np.mean(abs(error))),
        "future_phase_rmse_rad": float(np.sqrt(np.mean(error**2))),
        "basis_from_posthoc_frozen_frequency_catalog": basis.tolist(),
        "origin_aligned_on_training_truth_only": origin.tolist(),
        "physical_module_complete": fitted_row["module_truth_only"].get("correct_module_rank_relations", False),
        "truth_used_for_inference": False}


def main():
    if OUTPUT.exists():
        raise FileExistsError("preserve the existing raw-seed audit")
    for folder in (ROOT/"parameters", ROOT/"projected"):
        folder.mkdir(parents=True, exist_ok=True)
    source = json.loads(SOURCE.read_text())
    source_rows = {row["run"]: row for row in source["rows"]}
    initial = hashes()
    rows = []
    started = time.perf_counter()
    payload = {"status": "running", "rows": rows, "manifest": source["manifest"],
        "source_fingerprints": initial, "projection_settings": hankel.SETTINGS.__dict__,
        "source_model_archive": str(SOURCE), "raw_profile_training_samples": suite.TRAIN,
        "future_seed_clock_or_prefix_subtraction_used": False,
        "source_model_and_projection_settings_changed": False,
        "truth_opening_scope": "after all eighteen raw-seed parameters and outputs freeze",
        "evaluation_files_opened": False}
    save(payload)
    for entry in source["manifest"]:
        print("starting raw-seed Hankel "+entry["run"], flush=True)
        try:
            fitted_row, metadata = freeze_raw_seed(entry, source_rows[entry["run"]])
            row = hankel.evaluate_observations(entry, fitted_row, ROOT/"projected")
            row.update(raw_parameter_archive=fitted_row["frozen_arrays"], raw_observers_training_only=metadata,
                seed_source="raw observed field Hann windows; training-only origin; no prefix-clock subtraction")
        except Exception as error:
            row = {"run": entry["run"], "case": entry["case"], "seed": entry["seed"],
                "run_status": "exception", "exception": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc()}
        rows.append(row)
        save(payload)
    payload["all_outputs_frozen_before_truth_open"] = True
    payload["source_fingerprints_unchanged_before_scoring"] = hashes() == initial
    save(payload)
    for row, entry in zip(rows, source["manifest"]):
        if row["run_status"] == "exception":
            continue
        hankel.score(row, entry)
        row["phase_truth_only"] = phase_score(row, entry, source_rows[entry["run"]])
        print(json.dumps({"run": row["run"], "nrmse": row["future_sensor_nrmse"],
            "phase_mae": row["phase_truth_only"].get("future_phase_mae_rad")}), flush=True)
    finished = [row for row in rows if row["run_status"] == "finished"]
    payload.update(status="finished", evaluation_files_opened=True,
        source_fingerprints_unchanged=hashes() == initial, total_elapsed_seconds=time.perf_counter()-started,
        summary={"requested_runs": len(source["manifest"]), "finished_runs": len(finished),
            "exceptions": len(rows)-len(finished),
            "hankel_improved_vs_free_linear": sum(row["hankel_over_free_linear_error_ratio"] < 1 for row in finished),
            "hankel_sensor_nrmse_median": float(np.median([row["future_sensor_nrmse"]["hankel"] for row in finished])) if finished else None})
    save(payload)
    print(json.dumps(payload["summary"]), flush=True)


if __name__ == "__main__":
    main()
