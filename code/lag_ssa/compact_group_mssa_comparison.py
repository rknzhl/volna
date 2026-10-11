"""Same-window endpoint and diagonal-fold comparison with classical MSSA.

Group models and observed window-start phases are already frozen. Classical
joint weighted-MSSA uses TRAIN mean/basis and separately VAL-selected endpoint
and diagonal plug-in SURE ranks. All twenty comparisons freeze before this
runner opens any evaluation archive. No physical noise/frequency/phase enters.
"""

from project_paths import RESULTS_DIR

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np
from scipy.linalg import eigh, solve
from scipy.linalg.blas import dgemm

from mssa_column_encoding import trajectory, RANK_GRID
import run_actual_wave_iterative as wave


HERE = Path(__file__).resolve().parent
SOURCE = RESULTS_DIR / 'compact_group_hankel.json'
ROOT = RESULTS_DIR / 'compact_group_mssa_comparison'
OUTPUT = ROOT.with_suffix(".json")
WINDOW = 128


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def hashes():
    return {name: digest(HERE/name) for name in ("compact_group_mssa_comparison.py",
        "mssa_column_encoding.py", "run_actual_wave_iterative.py")}


def save(payload):
    temporary = OUTPUT.with_suffix(".tmp")
    temporary.write_text(json.dumps(wave.serializable(payload), indent=2, allow_nan=False)+"\n")
    temporary.replace(OUTPUT)


def diagonal_average(column_blocks):
    """L x q x C columns -> q x (C+L-1) signal and available counts."""
    blocks = np.asarray(column_blocks)
    lags, sensors, columns = blocks.shape
    result = np.zeros((sensors, columns+lags-1))
    for lag in range(lags):
        result[:, lag:lag+columns] += blocks[lag]
    counts = np.convolve(np.ones(columns), np.ones(lags))
    return result/counts[None, :], counts


def diagonal_divergence(block_traces, columns):
    counts = np.convolve(np.ones(columns), np.ones(len(block_traces)))
    return np.convolve(np.ones(columns), block_traces)/counts, counts


def fit_training_basis(training, mass_sqrt, window=WINDOW):
    mean = training.mean(axis=1)
    hankel = trajectory(training, mean, mass_sqrt, window)
    if hankel.shape[0] <= hankel.shape[1]:
        covariance = dgemm(1., hankel, hankel, trans_b=True)
        eigenvalues, eigenvectors = eigh(covariance, driver="evr")
        order = np.argsort(eigenvalues)[::-1]
        singular = np.sqrt(np.maximum(0., eigenvalues[order]))
        threshold = float(singular[0]*max(hankel.shape)*np.finfo(float).eps)
        numerical_rank = int(np.sum(singular > threshold))
        maximum = min(max(RANK_GRID), numerical_rank, min(hankel.shape))
        basis = np.asfortranarray(eigenvectors[:, order[:maximum]])
        branch = "ambient covariance eigenbasis"
    else:
        gram = dgemm(1., hankel, hankel, trans_a=True)
        eigenvalues, eigenvectors = eigh(gram, driver="evr")
        order = np.argsort(eigenvalues)[::-1]
        singular = np.sqrt(np.maximum(0., eigenvalues[order]))
        threshold = float(singular[0]*max(hankel.shape)*np.finfo(float).eps)
        numerical_rank = int(np.sum(singular > threshold))
        maximum = min(max(RANK_GRID), numerical_rank, min(hankel.shape))
        basis = dgemm(1., hankel, np.asfortranarray(eigenvectors[:, order[:maximum]]))/singular[:maximum][None, :]
        basis = np.asfortranarray(basis)
        branch = "training-column Gram eigenbasis"
    if maximum <= 0:
        raise ValueError("nonzero training Hankel basis required")
    information = {"mean_training_samples": training.shape[1], "window": window,
        "training_hankel_shape": list(hankel.shape), "numerical_training_rank": numerical_rank,
        "singular_value_threshold": threshold, "maximum_saved_basis_rank": maximum,
        "eigendecomposition": branch, "basis_orthogonality_max_absolute_defect": float(np.max(abs(
            np.einsum("ji,jk->ik", basis, basis, optimize=False)-np.eye(maximum)))),
        "basis_training_scope": "TRAIN only; neither VAL nor future enters mean/eigenbasis"}
    return mean, basis, singular, information


def _decode_mass_columns(projected, mean, mass_sqrt, window):
    sensors = len(mean);columns = projected.shape[1]
    blocks = np.sqrt(window)*projected.reshape(window, sensors, columns)
    endpoint = solve(mass_sqrt, blocks[-1], assume_a="pos")+mean[:, None]
    diagonal_mass, counts = diagonal_average(blocks)
    diagonal = solve(mass_sqrt, diagonal_mass, assume_a="pos")+mean[:, None]
    return endpoint, diagonal, counts


def select_ranks(training, validation_context, validation_target, mass_sqrt, sigma, window=WINDOW):
    mean, basis, singular, information = fit_training_basis(training, mass_sqrt, window)
    hankel = trajectory(validation_context, mean, mass_sqrt, window)
    if hankel.shape[1] != validation_target.shape[1]:
        raise ValueError("VAL context must provide one trailing column per VAL target")
    coordinates = dgemm(1., basis, hankel, trans_a=True)
    grid = sorted({min(rank, basis.shape[1]) for rank in RANK_GRID})
    projected = np.zeros_like(hankel, order="F")
    records = [];previous = 0
    sensors = training.shape[0]
    for rank in grid:
        addition = dgemm(1., np.asfortranarray(basis[:, previous:rank]), np.asfortranarray(coordinates[previous:rank]))
        projected += addition;previous = rank
        endpoint, diagonal, counts = _decode_mass_columns(projected, mean, mass_sqrt, window)
        diagonal_targets = diagonal[:, window-1:]
        block_trace = np.sum(basis[:, :rank].reshape(window, sensors, rank)**2, axis=(1, 2))
        divergence, _ = diagonal_divergence(block_trace, hankel.shape[1])
        divergence_targets = divergence[window-1:]
        endpoint_loss = float(np.mean(np.sum((endpoint-validation_target)**2, axis=0)))
        diagonal_loss = float(np.mean(np.sum((diagonal_targets-validation_target)**2, axis=0)))
        endpoint_trace = float(block_trace[-1])
        diagonal_trace = float(np.mean(divergence_targets))
        records.append({"rank": rank,
            "endpoint_raw_validation_observed_loss_sum_over_sensors": endpoint_loss,
            "endpoint_trace_last_lag_block": endpoint_trace,
            "endpoint_plugin_SURE_sum_over_sensors": endpoint_loss+2*sigma**2*endpoint_trace-sensors*sigma**2,
            "diagonal_raw_validation_observed_loss_sum_over_sensors": diagonal_loss,
            "diagonal_average_target_divergence": diagonal_trace,
            "diagonal_plugin_SURE_sum_over_sensors": diagonal_loss+2*sigma**2*diagonal_trace-sensors*sigma**2,
            "VAL_target_count": validation_target.shape[1],
            "diagonal_available_count_min": float(counts[window-1:].min()),
            "diagonal_available_count_max": float(counts[window-1:].max()),
            "grid_value_clipped_to_training_capacity": rank not in RANK_GRID})
    endpoint_rank = min(records, key=lambda item: (item["endpoint_plugin_SURE_sum_over_sensors"], item["rank"]))["rank"]
    diagonal_rank = min(records, key=lambda item: (item["diagonal_plugin_SURE_sum_over_sensors"], item["rank"]))["rank"]
    return {"mean": mean, "basis": basis, "singular": singular, "metadata": information,
        "endpoint_rank": endpoint_rank, "diagonal_rank": diagonal_rank, "selection": records}


def raw_operator_checks():
    """Independent small impulse checks for endpoint/D divergence and folding."""
    rng = np.random.default_rng(402611)
    sensors, window, columns, rank = 2, 4, 5, 3
    raw = rng.standard_normal((sensors, columns+window-1))
    mass_sqrt = np.asarray([[1.2, .17], [.17, .8]])
    mean = np.zeros(sensors)
    basis = np.linalg.qr(rng.standard_normal((sensors*window, rank)))[0]
    h = trajectory(raw, mean, mass_sqrt, window)
    _, restored, _ = _decode_mass_columns(h, mean, mass_sqrt, window)
    raw_difference = float(np.max(abs(restored-raw)))
    trace = np.sum(basis.reshape(window, sensors, rank)**2, axis=(1, 2))
    expected_diagonal, counts = diagonal_divergence(trace, columns)
    measured_diagonal = np.zeros(columns+window-1)
    measured_endpoint = np.zeros(columns)
    for timestamp in range(columns+window-1):
        for sensor in range(sensors):
            impulse = np.zeros_like(raw);impulse[sensor, timestamp] = 1.
            impulse_h = trajectory(impulse, mean, mass_sqrt, window)
            coordinate = np.einsum("ji,jt->it", basis, impulse_h, optimize=False)
            projected = np.einsum("ij,jt->it", basis, coordinate, optimize=False)
            endpoint, diagonal, _ = _decode_mass_columns(projected, mean, mass_sqrt, window)
            measured_diagonal[timestamp] += diagonal[sensor, timestamp]
            column = timestamp-window+1
            if 0 <= column < columns:
                measured_endpoint[column] += endpoint[sensor, column]
    return {"D_H_raw_max_absolute_difference": raw_difference,
        "D_divergence_impulse_trace_max_absolute_difference": float(np.max(abs(measured_diagonal-expected_diagonal))),
        "endpoint_divergence_impulse_trace_max_absolute_difference": float(np.max(abs(measured_endpoint-trace[-1]))),
        "small_operator_shape": {"sensors": sensors, "window": window, "columns": columns, "rank": rank},
        "small_available_counts": counts.tolist()}


def group_columns(labels, coefficients, frequencies, phase, window=WINDOW):
    base = np.einsum("ij,jt->it", labels, phase, optimize=False)
    rotation = np.einsum("ij,j->i", labels, frequencies, optimize=False)
    columns = np.empty((window, coefficients.shape[1], phase.shape[1]))
    count = len(labels)
    for lag in range(window):
        angles = base+rotation[:, None]*lag
        features = np.column_stack((np.ones(phase.shape[1]), np.cos(angles.T), np.sin(angles.T)))
        columns[lag] = np.einsum("tn,nq->qt", features, coefficients, optimize=False)
    return columns


def fit_and_freeze(entry, source_row):
    started = time.perf_counter()
    if digest(entry["input"]) != entry["input_sha256"]:
        raise ValueError("public observation file changed")
    if digest(source_row["frozen_parameters"]) != source_row["frozen_parameters_sha256"] or digest(source_row["frozen_output"]) != source_row["frozen_output_sha256"]:
        raise ValueError("selected group source changed")
    with np.load(entry["input"]) as public:
        if set(public.files) != {"Y", "s", "t"}:
            raise ValueError("unexpected public input keys")
        observed = public["Y"]
    with np.load(source_row["frozen_parameters"]) as archive:
        parameters = {key: archive[key] for key in archive.files}
    selected = source_row["selected_model"]
    with np.load(source_row["frozen_output"]) as archive:
        phase = archive[selected+"__phase_at_window_start"]
        old_group_endpoint = archive["selected_prediction"]
    n_train, n_validation = entry["n_train"], entry["n_validation"]
    cutoff = n_train+n_validation
    sigma = wave.noise_from_training_high_band(observed[:, :n_train])
    mass_sqrt = parameters["spatial_mass_sqrt"]
    fitted = select_ranks(observed[:, :n_train], observed[:, n_train-WINDOW+1:cutoff],
        observed[:, n_train:cutoff], mass_sqrt, sigma["sigma"])
    future_values = observed[:, cutoff-WINDOW+1:]
    h = trajectory(future_values, fitted["mean"], mass_sqrt, WINDOW)
    if h.shape[1] != phase.shape[1] or phase.shape[1] != old_group_endpoint.shape[1]:
        raise ValueError("classical and group future column blocks differ")
    count = phase.shape[1]
    coordinate = dgemm(1., fitted["basis"], h, trans_a=True)
    predictions = {};divergences = {}
    for method, rank in (("endpoint", fitted["endpoint_rank"]), ("diagonal", fitted["diagonal_rank"])):
        projection = dgemm(1., np.asfortranarray(fitted["basis"][:, :rank]), np.asfortranarray(coordinate[:rank]))
        endpoint, diagonal, counts = _decode_mass_columns(projection, fitted["mean"], mass_sqrt, WINDOW)
        block_trace = np.sum(fitted["basis"][:, :rank].reshape(WINDOW, len(fitted["mean"]), rank)**2, axis=(1, 2))
        full_divergence, _ = diagonal_divergence(block_trace, count)
        if method == "endpoint":
            predictions["mssa_endpoint"] = endpoint
            divergences["endpoint"] = np.full(count, block_trace[-1])
        else:
            predictions["mssa_diagonal"] = diagonal[:, WINDOW-1:]
            divergences["diagonal"] = full_divergence[WINDOW-1:]
    selected_columns = group_columns(parameters[selected+"__labels"], parameters[selected+"__coefficients_mass"],
        parameters[selected+"__frequencies"], phase)
    mapping = parameters["mass_to_sensor"]
    group_endpoint = np.einsum("ij,jt->it", mapping, selected_columns[-1], optimize=False)
    composition_difference = float(np.max(abs(group_endpoint-old_group_endpoint)))
    if composition_difference > 1e-8*max(1., float(np.max(abs(old_group_endpoint)))):
        raise ArithmeticError("saved phase composition does not reproduce the selected group endpoint")
    group_diagonal_mass, group_counts = diagonal_average(selected_columns)
    group_diagonal = np.einsum("ij,jt->it", mapping, group_diagonal_mass[:, WINDOW-1:], optimize=False)
    if not np.array_equal(group_counts, counts):
        raise ArithmeticError("diagonal averaging uses different available pairs")
    available = counts[WINDOW-1:]
    context_min = np.maximum(0, np.arange(WINDOW-1, WINDOW-1+count)-WINDOW+1)
    context_max = np.minimum(count-1, np.arange(WINDOW-1, WINDOW-1+count))+WINDOW-1
    context_width = context_max-context_min+1
    arrays = {**predictions, "group_endpoint": old_group_endpoint, "group_diagonal": group_diagonal,
        "endpoint_divergence": divergences["endpoint"], "diagonal_divergence": divergences["diagonal"],
        "diagonal_available_counts": available, "diagonal_raw_context_width": context_width,
        "time_targets": np.arange(cutoff, cutoff+count)}
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise FloatingPointError("nonfinite frozen comparison outputs")
    output_path = ROOT/"projected"/(entry["run"]+".npz")
    parameter_path = ROOT/"parameters"/(entry["run"]+".npz")
    np.savez_compressed(output_path, **arrays)
    np.savez_compressed(parameter_path, basis_available_to_grid=fitted["basis"], mean_train=fitted["mean"],
        mass_sqrt=mass_sqrt, singular_values=fitted["singular"],
        endpoint_rank=np.asarray(fitted["endpoint_rank"]), diagonal_rank=np.asarray(fitted["diagonal_rank"]),
        physical_mass=parameters["physical_mass"])
    return {"run": entry["run"], "case": entry["case"], "kind": entry["kind"], "run_status": "fitted_and_frozen",
        "selected_group_family_frozen": selected, "group_selection_retuned": False,
        "group_source_output": source_row["frozen_output"], "group_source_output_sha256": source_row["frozen_output_sha256"],
        "group_phase_composition_endpoint_max_absolute_difference": composition_difference,
        "classical_basis": fitted["metadata"], "noise_estimate_train_only": sigma,
        "declared_rank_grid": list(RANK_GRID), "rank_selection_curves": fitted["selection"],
        "endpoint_selected_rank": fitted["endpoint_rank"], "diagonal_selected_rank": fitted["diagonal_rank"],
        "VAL_trajectory_context": "allVAL plus preceding127 TRAIN samples; score only VAL target timestamps",
        "future_start": cutoff, "future_count": count, "same_future_window_column_block": True,
        "diagonal_counts_range": [float(available.min()), float(available.max())],
        "diagonal_context_width_range": [int(context_width.min()), int(context_width.max())],
        "full_255_context_target_count": int(np.sum(context_width == 2*WINDOW-1)),
        "short_future_block_has_255_interior": bool(np.any(context_width == 2*WINDOW-1)),
        "frozen_output": str(output_path), "frozen_output_sha256": digest(output_path),
        "frozen_parameters": str(parameter_path), "frozen_parameters_sha256": digest(parameter_path),
        "elapsed_seconds": time.perf_counter()-started}


def score(row, entry):
    if digest(row["frozen_output"]) != row["frozen_output_sha256"]:
        raise ValueError("frozen comparison changed before scoring")
    with np.load(entry["evaluation"]) as archive:
        clean = archive["clean_Y"] if entry["kind"] == "wave" else archive["clean"]
    target = clean[:, row["future_start"]:row["future_start"]+row["future_count"]]
    with np.load(row["frozen_parameters"]) as archive:
        mass = archive["physical_mass"]
    with np.load(row["frozen_output"]) as archive:
        endpoint = {"group": wave.field_score(archive["group_endpoint"], target, mass),
                    "MSSA": wave.field_score(archive["mssa_endpoint"], target, mass)}
        diagonal = {"group": wave.field_score(archive["group_diagonal"], target, mass),
                    "MSSA": wave.field_score(archive["mssa_diagonal"], target, mass)}
    row.update(run_status="finished", endpoint_field=endpoint, diagonal_field=diagonal)


def main():
    if OUTPUT.exists():
        raise FileExistsError("preserve the existing comparison")
    for directory in (ROOT/"projected", ROOT/"parameters"):
        directory.mkdir(parents=True, exist_ok=True)
    source = json.loads(SOURCE.read_text())
    if source["status"] != "finished" or len(source["manifest"]) != 20 or any(row["run_status"] != "finished" for row in source["rows"]):
        raise ValueError("all twenty selected group models must already be frozen and complete")
    initial = hashes();started = time.perf_counter();rows = []
    checks = raw_operator_checks()
    if max(checks[key] for key in ("D_H_raw_max_absolute_difference", "D_divergence_impulse_trace_max_absolute_difference",
            "endpoint_divergence_impulse_trace_max_absolute_difference")) > 1e-10:
        raise ArithmeticError("raw folding or trace formula check failed")
    payload = {"status": "running", "rows": rows, "manifest": source["manifest"],
        "source_fingerprints": initial, "selected_group_source": str(SOURCE), "selected_group_source_sha256": digest(SOURCE),
        "rules_declared_before_new_scoring": {"window": WINDOW, "rank_grid": list(RANK_GRID),
            "rank_selection": "separate endpoint and fullD validation plug-in SURE, ties lower rank",
            "group_selection": "frozen from original masked-VAL comparison; not rerun or retuned",
            "future_context": "identical observed trailing128 column block; D uses all overlapping columns from that same block"},
        "numeric_operator_checks": checks, "evaluation_files_opened": False,
        "SURE_disclaimer": "training-estimated noise under iid raw-sensor variance assumption; no calibration; overlapping VAL targets; classical basis/mean TRAIN only; group dictionary/family previously VAL-selected",
        "previously_seen_benchmarks_not_fresh_external_test": True, "novelty_claimed": False}
    save(payload)
    for entry, source_row in zip(source["manifest"], source["rows"]):
        print("starting matched classical MSSA "+entry["run"], flush=True)
        try:
            row = fit_and_freeze(entry, source_row)
        except Exception as error:
            row = {"run": entry["run"], "case": entry["case"], "kind": entry["kind"], "run_status": "exception",
                "exception": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()}
        rows.append(row);save(payload)
        print(json.dumps({"run": row["run"], "status": row["run_status"],
            "endpoint_rank": row.get("endpoint_selected_rank"), "diagonal_rank": row.get("diagonal_selected_rank"),
            "seconds": row.get("elapsed_seconds"), "exception": row.get("exception")}), flush=True)
    payload.update(all20_attempts_and_comparison_outputs_frozen_before_truth_open=True,
        source_fingerprints_unchanged_before_scoring=hashes() == initial)
    save(payload)
    if hashes() != initial:
        raise ValueError("comparison sources changed before scoring")
    for row, entry in zip(rows, source["manifest"]):
        if row["run_status"] == "exception":
            continue
        score(row, entry)
        print(json.dumps({"run": row["run"], "endpoint_sensor_nrmse":
            {key: value["sensor_nrmse"] for key, value in row["endpoint_field"].items()},
            "diagonal_sensor_nrmse": {key: value["sensor_nrmse"] for key, value in row["diagonal_field"].items()}}), flush=True)
    finished = [row for row in rows if row["run_status"] == "finished"]
    payload.update(status="finished", evaluation_files_opened=True, source_fingerprints_unchanged=hashes() == initial,
        total_elapsed_seconds=time.perf_counter()-started,
        summary={"requested_runs": 20, "finished_runs": len(finished), "exceptions": len(rows)-len(finished),
            "group_better_endpoint_sensor_nrmse": sum(row["endpoint_field"]["group"]["sensor_nrmse"] < row["endpoint_field"]["MSSA"]["sensor_nrmse"] for row in finished),
            "group_better_diagonal_sensor_nrmse": sum(row["diagonal_field"]["group"]["sensor_nrmse"] < row["diagonal_field"]["MSSA"]["sensor_nrmse"] for row in finished)})
    save(payload);print(json.dumps(payload["summary"]), flush=True)


if __name__ == "__main__":
    main()
