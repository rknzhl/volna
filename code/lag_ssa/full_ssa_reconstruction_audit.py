"""Complete frozen torus/MSSA reconstruction chains on the same six fields.

Torus phases, character dictionary, coefficients, frequencies, and classical
TRAIN MSSA basis are reused unchanged. Both projected column matrices are
diagonally averaged. The MSSA rank is selected using validation interior SURE
for that reconstruction; future observations and clean fields never select
the rank. All six reconstructions are frozen before evaluation files open.
"""

from project_paths import RESULTS_DIR

import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np
from scipy.linalg import solve
from scipy.linalg.blas import dgemm

from mssa_column_encoding import RANK_GRID, trajectory


HERE = Path(__file__).resolve().parent
ROOT = RESULTS_DIR / 'full_ssa_reconstruction_audit'
OUTPUT = ROOT.with_suffix(".json")
SOURCE_TORUS = RESULTS_DIR / 'robust_strong_raw_hankel_audit.json'
SOURCE_MSSA = RESULTS_DIR / 'mssa_column_encoding_audit.json'
SOURCE_INPUTS = RESULTS_DIR / 'general_robustness_factor_cover.json'
TRAIN, VALIDATION, FUTURE, WINDOW = 2400, 1200, 1200, 128
FIRST_SEED = 202610081


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def product(first, second, transpose_first=False):
    return dgemm(1., np.asfortranarray(first), np.asfortranarray(second),
                 trans_a=transpose_first)


def mass_trajectory(mass_values, window):
    count = mass_values.shape[1]-window+1
    return np.asfortranarray(np.vstack([mass_values[:, lag:lag+count]
                              for lag in range(window)])/np.sqrt(window))


def diagonal_average(matrix, sensors, window):
    """Inverse uniform-delay embedding; matrix has normalized lag blocks."""
    count = matrix.shape[1]
    blocks = matrix.reshape(window, sensors, count)
    result = np.zeros((sensors, count+window-1))
    multiplicity = np.zeros(count+window-1)
    for lag in range(window):
        result[:, lag:lag+count] += blocks[lag]*np.sqrt(window)
        multiplicity[lag:lag+count] += 1
    return result/multiplicity[None, :], multiplicity


def column_diagnostics(matrix, averaged_mass, sensors, window):
    rehankelized = mass_trajectory(averaged_mass, window)
    difference = matrix-rehankelized
    denominator = float(np.sum(matrix**2))
    return {"relative_frobenius_Hankelization_change_centered_mass":
            float(np.sqrt(np.sum(difference**2)/denominator)),
        "mean_squared_Hankelization_change_centered_mass": float(np.mean(difference**2)),
        "projected_columns_exactly_Hankel_at_absolute_tolerance_1e_10": bool(np.max(abs(difference))<1e-10),
        "Hankelization_is_a_projection_onto_column_torus": False}


def torus_columns(labels, coefficients, frequencies, phases, window):
    """Evaluate the frozen learned field at actual frozen observed phases."""
    count = len(labels)
    line_frequency = np.einsum("ij,j->i", labels, frequencies, optimize=False)
    lag_angle = line_frequency[:, None]*np.arange(window)[None, :]
    a, b = coefficients[1:1+count], coefficients[1+count:]
    first = a[:, :, None]*np.cos(lag_angle)[:, None, :]+b[:, :, None]*np.sin(lag_angle)[:, None, :]
    second = -a[:, :, None]*np.sin(lag_angle)[:, None, :]+b[:, :, None]*np.cos(lag_angle)[:, None, :]
    bank = np.concatenate((first, second), axis=0)
    phase_angle = np.einsum("ij,jn->in", labels, phases, optimize=False)
    features = np.vstack((np.cos(phase_angle), np.sin(phase_angle)))
    sensors, columns = coefficients.shape[1], phases.shape[1]
    centered = product(bank.reshape(2*count, sensors*window), features, True).reshape(sensors, window, columns)
    matrix = np.asfortranarray(centered.transpose(1, 0, 2).reshape(window*sensors, columns)/np.sqrt(window))
    return matrix, coefficients[0]


def phase_shift_diagnostics(phases, frequencies):
    discrepancy = np.angle(np.exp(1j*(phases[:, 1:]-phases[:, :-1]-frequencies[:, None])))
    return {"formula": "wrap(theta_hat[n+1]-theta_hat[n]-alpha_hat)",
        "scope": "actual frozen observed phase coordinates; diagnostic only, no phase correction",
        "rmse_radians_by_phase": np.sqrt(np.mean(discrepancy**2, axis=1)).tolist(),
        "mean_absolute_radians_by_phase": np.mean(abs(discrepancy), axis=1).tolist(),
        "maximum_absolute_radians_by_phase": np.max(abs(discrepancy), axis=1).tolist(),
        "joint_wrapped_rmse_radians": float(np.sqrt(np.mean(np.sum(discrepancy**2, axis=0)))),
        "exact_rotation_at_absolute_tolerance_1e_10": bool(np.max(abs(discrepancy))<1e-10)}


def select_diagonal_rank(validation, mean, mass_sqrt, basis, sigma):
    sensors = len(mean)
    target_times = np.arange(WINDOW-1, validation.shape[1]-WINDOW+1)
    if not len(target_times):
        raise ValueError("validation block requires at least 2*L-1 samples")
    hankel = trajectory(validation, mean, mass_sqrt, WINDOW)
    coordinates = product(basis, hankel, True)
    maximum = basis.shape[1]
    grid = sorted({min(rank, maximum) for rank in RANK_GRID})
    projected = np.zeros(hankel.shape, order="F")
    previous, selection = 0, []
    for rank in grid:
        projected += product(basis[:, previous:rank], coordinates[previous:rank])
        averaged_mass, multiplicity = diagonal_average(projected, sensors, WINDOW)
        reconstructed = solve(mass_sqrt, averaged_mass, assume_a="pos", check_finite=True)+mean[:, None]
        observed_mse = float(np.mean(np.sum((reconstructed[:, target_times]-validation[:, target_times])**2, axis=0)))
        # Every interior anti-diagonal contains every lag once. Spatial
        # conjugation preserves trace, so divergence equals trace(P_k)/L.
        trace_per_target = float(np.sum(basis[:, :rank]**2)/WINDOW)
        sure = observed_mse+2*sigma**2*trace_per_target-sensors*sigma**2
        selection.append({"rank": rank,
            "validation_observed_interior_raw_mse_sum_over_sensors": observed_mse,
            "trace_per_interior_target": trace_per_target,
            "rank_over_window_reference": rank/WINDOW,
            "validation_SURE_sum_over_sensors": sure,
            "validation_interior_targets": len(target_times),
            "minimum_interior_multiplicity": int(multiplicity[target_times].min()),
            "maximum_interior_multiplicity": int(multiplicity[target_times].max()),
            "rank_grid_value_clipped_to_frozen_basis_max": rank not in RANK_GRID})
        previous = rank
    selected = min(selection, key=lambda row: (row["validation_SURE_sum_over_sensors"], row["rank"]))
    return selected["rank"], selection


def fit_public_case(torus_row, mssa_row, entry):
    started = time.perf_counter()
    with np.load(entry["input"]) as public:
        observed, times = public["Y"], public["t"]
    validation = observed[:, TRAIN:TRAIN+VALIDATION]
    future_start = TRAIN+VALIDATION
    future = observed[:, future_start:future_start+FUTURE]
    with np.load(mssa_row["frozen_output"]) as source:
        mean, basis, mass_sqrt = source["mean"], source["basis"], source["mass_sqrt"]
    sigma = float(mssa_row["noise_estimate_train_only"]["sigma"])
    selected_rank, selection = select_diagonal_rank(validation, mean, mass_sqrt, basis, sigma)
    with np.load(torus_row["frozen_output"]) as source:
        phases, column_times = source["hankel_phase"], source["time_starts"]
        frozen_lag0_prediction = source["hankel_prediction"]
    if phases.shape[1] != FUTURE-WINDOW+1 or not np.array_equal(column_times, times[future_start:future_start+phases.shape[1]]):
        raise ValueError("the declared full 1073 contiguous observed-phase columns are required")
    with np.load(torus_row["source_model"]) as model:
        labels, coefficients = model["fit_labels"], model["fit_coefficients"]
        frequencies, mapping = model["fit_frequencies"], model["mass_to_sensor"]
        torus_mass_sqrt = model["spatial_mass_sqrt"]
    if not np.allclose(mass_sqrt, torus_mass_sqrt, rtol=1e-12, atol=1e-14):
        raise ValueError("MSSA and torus must use exactly the same spatial mass")
    torus_matrix, torus_mean_mass = torus_columns(labels, coefficients, frequencies, phases, WINDOW)
    torus_averaged_mass, multiplicity = diagonal_average(torus_matrix, len(mean), WINDOW)
    torus_reconstruction = product(mapping, torus_averaged_mass+torus_mean_mass[:, None])
    torus_lag0 = product(mapping, np.sqrt(WINDOW)*torus_matrix[:len(mean)]+torus_mean_mass[:, None])
    lag0_difference = float(np.max(abs(torus_lag0-frozen_lag0_prediction)))
    if lag0_difference>1e-8:
        raise ValueError("frozen phase/decoder gauge inconsistent with source torus prediction")
    future_hankel = trajectory(future, mean, mass_sqrt, WINDOW)
    u = basis[:, :selected_rank]
    mssa_matrix = product(u, product(u, future_hankel, True))
    mssa_averaged_mass, mssa_multiplicity = diagonal_average(mssa_matrix, len(mean), WINDOW)
    mssa_reconstruction = solve(mass_sqrt, mssa_averaged_mass, assume_a="pos", check_finite=True)+mean[:, None]
    target_indices = np.arange(WINDOW-1, FUTURE-WINDOW+1)
    if len(target_indices)!=946 or not np.array_equal(multiplicity, mssa_multiplicity):
        raise ValueError("the declared 946 matched interior targets are required")
    identity_mass, _ = diagonal_average(future_hankel, len(mean), WINDOW)
    identity = solve(mass_sqrt, identity_mass, assume_a="pos", check_finite=True)+mean[:, None]
    identity_error = float(np.max(abs(identity-future)))
    arrays = {"torus_diagonal_prediction": torus_reconstruction,
        "mssa_diagonal_prediction": mssa_reconstruction,
        "hankel_phase": phases, "fit_frequencies": frequencies,
        "target_indices_in_future": target_indices, "target_times": times[future_start+target_indices],
        "column_time_starts": column_times, "anti_diagonal_multiplicity": multiplicity,
        "selected_mssa_rank": np.asarray(selected_rank),
        "validation_rank_grid": np.asarray([row["rank"] for row in selection]),
        "validation_SURE_curve": np.asarray([row["validation_SURE_sum_over_sensors"] for row in selection]),
        "physical_mass": product(mass_sqrt, mass_sqrt)}
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise FloatingPointError("nonfinite frozen reconstruction array")
    path = ROOT/(torus_row["run"]+".npz")
    np.savez_compressed(path, **arrays)
    return {"run": torus_row["run"], "case": torus_row["case"], "seed": torus_row["seed"],
        "run_status": "finished", "source_input": entry["input"],
        "source_input_sha256": digest(entry["input"]),
        "source_torus_observed_phases": torus_row["frozen_output"],
        "source_torus_observed_phases_sha256": digest(torus_row["frozen_output"]),
        "source_torus_model": torus_row["source_model"],
        "source_torus_model_sha256": digest(torus_row["source_model"]),
        "source_MSSA_training_basis": mssa_row["frozen_output"],
        "source_MSSA_training_basis_sha256": digest(mssa_row["frozen_output"]),
        "noise_estimate_train_only": mssa_row["noise_estimate_train_only"],
        "source_MSSA_numeric_rank": mssa_row["encoder"]["numeric_training_hankel_rank"],
        "frozen_MSSA_basis_columns": basis.shape[1], "selected_MSSA_rank": selected_rank,
        "validation_diagonal_SURE_curve": selection,
        "MSSA_rank_selection": "minimum VAL diagonal-average interior plug-in SURE; exact ties choose lower rank",
        "MSSA_SURE_formula": "mean_t sum_s (reconstruction-Y)^2 + 2*sigma_train^2*trace(P_k)/L - q*sigma_train^2",
        "SURE_reference_assumption": "independent TRAIN-frozen basis/mean; iid raw sensor Gaussian noise with common variance; TRAIN high-band sigma plug-in, not exact calibration",
        "torus_phase_rank": phases.shape[0], "time_start": future_start,
        "observed_window_columns": phases.shape[1], "interior_target_count": len(target_indices),
        "interior_future_indices_first_last": [int(target_indices[0]), int(target_indices[-1])],
        "matched_context_samples_per_interior_target": 2*WINDOW-1,
        "torus_Hankelization": column_diagnostics(torus_matrix, torus_averaged_mass, len(mean), WINDOW),
        "MSSA_Hankelization": column_diagnostics(mssa_matrix, mssa_averaged_mass, len(mean), WINDOW),
        "observed_phase_shift_discrepancy": phase_shift_diagnostics(phases, frequencies),
        "torus_lag0_matches_frozen_prediction_max_abs_difference": lag0_difference,
        "diagonal_average_of_raw_embedding_identity_max_abs_difference": identity_error,
        "frozen_output": str(path), "frozen_output_sha256": digest(path),
        "all_frozen_arrays_finite": True, "elapsed_seconds": time.perf_counter()-started}


def score_frozen_case(row, entry):
    with np.load(row["frozen_output"]) as frozen:
        targets = frozen["target_indices_in_future"]
        predictions = {"torus_diagonal": frozen["torus_diagonal_prediction"][:, targets],
                       "MSSA_diagonal": frozen["mssa_diagonal_prediction"][:, targets]}
        mass = frozen["physical_mass"]
    with np.load(entry["evaluation"]) as truth:
        clean = truth["clean"][:, row["time_start"]+targets]
    raw_denominator = float(np.sum(clean**2))
    mass_denominator = float(np.einsum("in,ij,jn", clean, mass, clean, optimize=False))
    row["sensor_NRMSE_same_interior_targets"] = {name: float(np.sqrt(np.sum((value-clean)**2)/raw_denominator))
                                               for name, value in predictions.items()}
    row["field_mass_NRMSE_same_interior_targets"] = {name: float(np.sqrt(np.einsum("in,ij,jn", value-clean,
        mass, value-clean, optimize=False)/mass_denominator)) for name, value in predictions.items()}
    row["scoring_scope"] = "same sensors, same 946 interior future times, same 255 observed-context samples; complete diagonal-reconstruction chains only"


def fingerprints():
    return {name: digest(HERE/name) for name in ("full_ssa_reconstruction_audit.py", "mssa_column_encoding.py")}


def save(payload):
    tmp = OUTPUT.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
    tmp.replace(OUTPUT)


def main():
    started = time.perf_counter()
    ROOT.mkdir(exist_ok=True)
    torus_source = json.loads(SOURCE_TORUS.read_text())
    mssa_source = json.loads(SOURCE_MSSA.read_text())
    input_source = json.loads(SOURCE_INPUTS.read_text())
    torus_rows = {row["run"]: row for row in torus_source["rows"]}
    entries = {entry["run"]: entry for entry in input_source["manifest"]}
    selected = [row for row in mssa_source["rows"] if row["seed"] == FIRST_SEED]
    if len(selected)!=6:
        raise ValueError("the same six first-seed generic fields are required")
    payload = {"status": "running", "rows": [],
        "manifest": [entries[row["run"]] for row in selected],
        "source_fingerprints": fingerprints(),
        "source_archives": {str(path): digest(path) for path in (SOURCE_TORUS, SOURCE_MSSA, SOURCE_INPUTS)},
        "settings": {"train": TRAIN, "validation": VALIDATION, "future": FUTURE, "window": WINDOW,
            "MSSA_rank_grid": RANK_GRID, "interior_targets": 946, "observed_columns": 1073,
            "matched_observed_context": 255},
        "torus_chain": "all frozen observed-phase Phi_L columns -> standard diagonal averaging",
        "MSSA_chain": "frozen TRAIN joint-Hankel basis -> VAL diagonal-SURE rank selection -> column projection -> standard diagonal averaging",
        "phase_optimizer_or_field_model_refitted": False,
        "future_clocks_used_for_phase_or_reconstruction": False,
        "truth_or_future_used_to_choose_MSSA_rank": False,
        "full_torus_and_Hankel_intersection_enforced": False,
        "field_reconstruction_does_not_certify_phase_completeness": True,
        "comparison_with_point_column_encoder_performed": False,
        "evaluation_files_opened": False,
        "truth_opening_scope": "all six complete reconstructions saved before first evaluation NPZ is opened"}
    save(payload)
    for mssa_row in selected:
        run = mssa_row["run"]
        try:
            row = fit_public_case(torus_rows[run], mssa_row, entries[run])
        except Exception as error:
            row = {"run": run, "case": mssa_row["case"], "seed": mssa_row["seed"],
                "run_status": "exception", "exception": repr(error), "traceback": traceback.format_exc()}
        payload["rows"].append(row)
        save(payload)
        print(json.dumps({"run": run, "status": row["run_status"],
                         "rank": row.get("selected_MSSA_rank"), "elapsed_seconds": row.get("elapsed_seconds")}), flush=True)
    payload["all_complete_reconstructions_frozen_before_truth_opening"] = all(row["run_status"]=="finished" for row in payload["rows"])
    payload["source_fingerprints_unchanged_before_scoring"] = payload["source_fingerprints"] == fingerprints()
    save(payload)
    for row in payload["rows"]:
        if row["run_status"] == "finished":
            score_frozen_case(row, entries[row["run"]])
    payload["evaluation_files_opened"] = True
    payload["source_fingerprints_unchanged"] = payload["source_fingerprints"] == fingerprints()
    payload["source_archives_unchanged"] = all(digest(path)==value for path,value in payload["source_archives"].items())
    payload["all_frozen_output_hashes_unchanged"] = all(digest(row["frozen_output"])==row["frozen_output_sha256"]
        for row in payload["rows"] if row["run_status"]=="finished")
    finished = [row for row in payload["rows"] if row["run_status"]=="finished"]
    payload["status"] = "finished"
    payload["total_elapsed_seconds"] = time.perf_counter()-started
    payload["summary"] = {"finished": len(finished), "exceptions": len(payload["rows"])-len(finished),
        "torus_better_raw_sensor_reconstruction_cases": sum(row["sensor_NRMSE_same_interior_targets"]["torus_diagonal"]
            <row["sensor_NRMSE_same_interior_targets"]["MSSA_diagonal"] for row in finished)}
    save(payload)
    print(json.dumps({"summary": payload["summary"], "elapsed_seconds": payload["total_elapsed_seconds"],
        "rows": [{"run":row["run"], "rank":row["selected_MSSA_rank"],
                  "sensor_NRMSE":row["sensor_NRMSE_same_interior_targets"],
                  "torus_Hankel_change":row["torus_Hankelization"]["relative_frobenius_Hankelization_change_centered_mass"],
                  "phase_shift_RMSE":row["observed_phase_shift_discrepancy"]["joint_wrapped_rmse_radians"]}
                 for row in finished]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
