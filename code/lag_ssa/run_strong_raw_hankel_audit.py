"""Matched L128 encoding: raw strong-character seed, nonlinear torus, free span.

No future timestamps or clock subtraction initialize the optimizer. Learned
field parameters and all optimization settings remain unchanged. Predictions
are frozen on all eighteen public fields before truth scoring.
"""

from project_paths import RESULTS_DIR

import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np

from hankel_torus_projection import HankelTorusProjector, ProjectionSettings
from raw_hankel_seed import initialize_from_raw_strong_characters


HERE = Path(__file__).resolve().parent
ROOT = RESULTS_DIR / 'strong_raw_hankel_audit'
OUTPUT = ROOT.with_suffix(".json")
SOURCE = RESULTS_DIR / 'general_robustness_factor_cover.json'
SETTINGS = ProjectionSettings()
NAMES = ("run_strong_raw_hankel_audit.py", "raw_hankel_seed.py", "hankel_torus_projection.py",
         "phase_readout_refinement.py")


def hashes():
    return {name: hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in NAMES}


def save(payload):
    temporary = OUTPUT.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False)+"\n")
    temporary.replace(OUTPUT)


def freeze(entry, fitted):
    started = time.perf_counter()
    if hashlib.sha256(Path(entry["input"]).read_bytes()).hexdigest() != entry["input_sha256"]:
        raise ValueError("public input changed")
    parameter_path = Path(fitted["frozen_arrays"])
    if hashlib.sha256(parameter_path.read_bytes()).hexdigest() != fitted["frozen_arrays_sha256"]:
        raise ValueError("frozen fitted model changed")
    with np.load(entry["input"]) as public:
        if set(public.files) != {"Y", "s", "t"}:
            raise ValueError("unexpected input keys")
        observed = public["Y"]
    with np.load(parameter_path) as archive:
        parameters = {name: archive[name] for name in archive.files}
    start = observed.shape[1]-parameters["prediction"].shape[1]
    training = np.einsum("ij,jt->it", parameters["spatial_mass_sqrt"], observed[:, :2400], optimize=False)
    future = np.einsum("ij,jt->it", parameters["spatial_mass_sqrt"], observed[:, start:], optimize=False)
    observer = HankelTorusProjector(parameters["fit_labels"], parameters["fit_coefficients"], parameters["fit_frequencies"], SETTINGS)
    seed = initialize_from_raw_strong_characters(training, future, observer)
    projected = observer.project(future, seed["phase"])
    count = projected["phase"].shape[1]
    free = observer.free_linear_projection(future, count)
    mapping = parameters["mass_to_sensor"]
    arrays = {"hankel_phase": projected["phase"], "raw_seed_phase": seed["phase"],
        "raw_training_seed_phase": seed["training_phase"],
        "hankel_prediction": np.einsum("ij,jt->it", mapping, projected["mass_field"], optimize=False),
        "free_linear_prediction": np.einsum("ij,jt->it", mapping, free["mass_field"], optimize=False),
        "clock_prediction": parameters["prediction"][:, :count],
        "time_starts": np.arange(start, start+count),
        "window_cost": np.asarray([item["final_cost"] for item in projected["diagnostics"]]),
        "selected_deck_start": np.asarray([item["selected_start"] for item in projected["diagnostics"]])}
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise FloatingPointError("nonfinite frozen outputs")
    destination = ROOT/(entry["run"]+".npz")
    np.savez_compressed(destination, **arrays)
    info = projected["diagnostics"]
    return {"run": entry["run"], "case": entry["case"], "seed": entry["seed"],
        "run_status": "fitted_and_frozen", "frozen_output": str(destination),
        "frozen_output_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "source_model": str(parameter_path), "source_model_sha256": fitted["frozen_arrays_sha256"],
        "time_start": start, "window_count": count, "raw_seed": seed["metadata"],
        "multistart": projected["multistart"], "free_linear": free["metadata"],
        "optimizer": {"gradient_converged_count": sum(item["converged_gradient"] for item in info),
            "final_gradient_max": max(item["final_gradient_max"] for item in info),
            "selected_nonzero_deck_start_count": sum(item["selected_start"] != 0 for item in info),
            "cost_above_seed_count": sum(not item["cost_not_above_product_seed"] for item in info),
            "global_optimum_certified": False},
        "same_total_raw_context_as_free_linear": True, "elapsed_seconds": time.perf_counter()-started}


def score(row, entry, fitted):
    with np.load(entry["evaluation"]) as truth:
        clean, theta = truth["clean"], truth["theta"]
    with np.load(row["frozen_output"]) as output:
        arrays = {name: output[name] for name in output.files}
    start, count = row["time_start"], row["window_count"]
    target = clean[:, start:start+count]
    denominator = np.linalg.norm(target)
    row["sensor_nrmse"] = {name: float(np.linalg.norm(arrays[name+"_prediction"]-target)/denominator)
        for name in ("hankel", "free_linear", "clock")}
    physical_basis = fitted["module_truth_only"].get("phase_basis_in_physical_coordinates")
    if physical_basis is not None:
        basis = np.asarray(physical_basis, dtype=int)
        with np.load(row["source_model"]) as parameters:
            frequency = parameters["fit_frequencies"]
        train_clock = frequency[:, None]*np.arange(2400)
        complete = bool(fitted["module_truth_only"].get("correct_module_rank_relations", False))
        if complete and basis.shape[0] == basis.shape[1]:
            inverse = np.linalg.inv(basis)
            if not np.allclose(inverse, np.rint(inverse), atol=1e-10):
                raise ArithmeticError("complete integer basis must have unimodular inverse")
            inverse = np.rint(inverse).astype(int)
            phase = np.einsum("ij,jt->it", inverse, arrays["hankel_phase"], optimize=False)
            calibrated_clock = np.einsum("ij,jt->it", inverse, train_clock, optimize=False)
            train_target = theta[:, :2400]
            future_target = theta[:, start:start+count]
            phase_scope = "canonical physical generator coordinates, fixed before future scoring"
        else:
            phase, calibrated_clock = arrays["hankel_phase"], train_clock
            train_target = np.einsum("ij,jt->it", basis, theta[:, :2400], optimize=False)
            future_target = np.einsum("ij,jt->it", basis, theta[:, start:start+count], optimize=False)
            phase_scope = "recovered factor only; incomplete physical module retained as failure"
        origin = np.angle(np.mean(np.exp(1j*(calibrated_clock-train_target)), axis=1))
        error = np.angle(np.exp(1j*(phase-origin[:, None]-future_target)))
        row["phase_truth_only"] = {"physical_module_complete": complete,
            "mae_rad": float(np.mean(abs(error))), "rmse_rad": float(np.sqrt(np.mean(error**2))),
            "scope": phase_scope, "gauge_calibration": "shared decoder clock gauge against training truth only",
            "future_truth_used_to_choose_basis_or_origin": False}
    row["run_status"] = "finished"


def main():
    if OUTPUT.exists():
        raise FileExistsError("preserve the existing matched-context audit")
    ROOT.mkdir(parents=True, exist_ok=True)
    source = json.loads(SOURCE.read_text())
    fitted_rows = {row["run"]: row for row in source["rows"]}
    initial = hashes()
    rows = []
    started = time.perf_counter()
    payload = {"status": "running", "rows": rows, "manifest": source["manifest"],
        "source_fingerprints": initial, "projection_settings": SETTINGS.__dict__,
        "source_model_archive": str(SOURCE), "future_absolute_time_used_as_seed": False,
        "future_prefix_clock_subtraction_used": False,
        "nonlinear_and_free_linear_total_context": "identical raw 128-sample window",
        "source_field_model_and_optimizer_settings_changed": False,
        "truth_opening_scope": "after all eighteen matched-context predictions freeze", "evaluation_files_opened": False}
    save(payload)
    for entry in source["manifest"]:
        print("starting strong-raw128 "+entry["run"], flush=True)
        try:
            row = freeze(entry, fitted_rows[entry["run"]])
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
        score(row, entry, fitted_rows[entry["run"]])
        print(json.dumps({"run": row["run"], "nrmse": row["sensor_nrmse"], "phase": row.get("phase_truth_only")}), flush=True)
    finished = [row for row in rows if row["run_status"] == "finished"]
    payload.update(status="finished", evaluation_files_opened=True, source_fingerprints_unchanged=hashes() == initial,
        total_elapsed_seconds=time.perf_counter()-started,
        summary={"requested_runs": len(source["manifest"]), "finished_runs": len(finished),
            "exceptions": len(rows)-len(finished),
            "hankel_improved_vs_free_linear": sum(row["sensor_nrmse"]["hankel"] < row["sensor_nrmse"]["free_linear"] for row in finished),
            "hankel_sensor_nrmse_median": float(np.median([row["sensor_nrmse"]["hankel"] for row in finished])) if finished else None})
    save(payload)
    print(json.dumps(payload["summary"]), flush=True)


if __name__ == "__main__":
    main()
