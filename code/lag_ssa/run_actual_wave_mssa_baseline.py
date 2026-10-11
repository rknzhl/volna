"""Blind baselines for the actual sensor-wave inputs.

Inference reads only Y,s,t. All 11 fits finish before evaluation files are
opened. Model selection uses train/validation data; future data is scored only.
The notebook MSSA recurrence is executed from its original pure function
definitions, without importing its scenario/PDE dependencies.
"""

from __future__ import annotations

from project_paths import RESULTS_DIR

import ast
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from frequency_fit import fit_joint_frequencies, sinusoid_design


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = RESULTS_DIR / 'wave_actual_data'
OUT = RESULTS_DIR
DESTINATION = OUT/"actual_wave_baseline.json"
MODEL_OUT = OUT/"actual_wave_baseline_models"
RANKS = (2, 4, 8, 16)


def _safe_product(first, second):
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        value = first@second
    if not np.isfinite(value).all():
        raise FloatingPointError("nonfinite SSA multiplication")
    return value


def trajectory(y: np.ndarray, lag: int) -> np.ndarray:
    q = y.shape[0]
    view = np.lib.stride_tricks.sliding_window_view(y, lag, axis=1)
    return view.transpose(2, 0, 1).reshape(lag*q, -1)/np.sqrt(lag*q)


def _randomized_svd(h: np.ndarray, rank: int = 24):
    count = min(rank+8, min(h.shape))
    omega = np.random.default_rng(7191).standard_normal((h.shape[1], count))
    basis, _ = np.linalg.qr(_safe_product(h, omega), mode="reduced")
    for _ in range(2):
        basis, _ = np.linalg.qr(_safe_product(h, _safe_product(h.T, basis)), mode="reduced")
    small = _safe_product(basis.T, h)
    left, singular, _ = np.linalg.svd(small, full_matrices=False)
    return _safe_product(basis, left), singular


def _hann_refine(y: np.ndarray, center: float) -> tuple[float, float]:
    n = y.shape[1]
    transformed = np.fft.rfft((y-y.mean(axis=1, keepdims=True))*np.hanning(n),
                              n=8*n, axis=1)
    power = np.sum(abs(transformed)**2, axis=0)
    frequencies = np.fft.rfftfreq(8*n)*2*np.pi
    allowed = np.flatnonzero((frequencies > 0) & (frequencies < np.pi)
        & (abs(frequencies-center) <= 2*np.pi/n))
    if not len(allowed):
        return center, 0.
    selected = int(allowed[np.argmax(power[allowed])])
    return float(frequencies[selected]), float(power[selected])


def candidates_from_rank(y: np.ndarray, h: np.ndarray, u: np.ndarray,
                          rank: int) -> tuple[list[dict], dict]:
    subspace = u[:, :rank]
    projected = _safe_product(subspace.T, h)
    previous, following = projected[:, :-1], projected[:, 1:]
    gram = np.einsum("it,jt->ij", previous, previous, optimize=False)
    cross = np.einsum("it,jt->ij", following, previous, optimize=False)
    shift = np.linalg.solve(gram.T, cross.T).T
    values, vectors = np.linalg.eig(shift)
    readout = np.linalg.solve(vectors, subspace.T)
    coordinates = np.einsum("ij,jt->it", readout, h, optimize=False)
    proposals = []
    for index, value in enumerate(values):
        frequency = float(np.angle(value))
        if not (np.imag(value) > 1e-6 and 0 < frequency < np.pi):
            continue
        z = coordinates[index]
        radii = abs(z)
        normalized = z/np.maximum(radii, 1e-15)
        coherence = float(abs(np.mean(normalized[1:]*np.conj(normalized[:-1]))))
        radius_cv = float(np.std(radii)/max(np.mean(radii), 1e-15))
        candidate, power = _hann_refine(y, frequency)
        if abs(value) < .85 or abs(value) > 1.05 or coherence < .85 or radius_cv > 1.2:
            continue
        proposals.append({"dmd_index": index, "frequency_dmd": frequency,
            "frequency_start": candidate, "eigenvalue_real_imag": [float(value.real), float(value.imag)],
            "eigenvalue_modulus": float(abs(value)), "phase_step_coherence": coherence,
            "radius_cv": radius_cv, "training_hann_power": power,
            "filter": readout[index].copy(), "train_phase": np.angle(z)})
    proposals.sort(key=lambda row: -row["training_hann_power"])
    unique = []
    for proposal in proposals:
        if all(abs(proposal["frequency_start"]-old["frequency_start"]) > 2*np.pi/y.shape[1]
               for old in unique):
            unique.append(proposal)
    return unique, {"rank": rank, "eigenvalues_real_imag":
                   [[float(value.real), float(value.imag)] for value in values],
                   "retained_candidate_count": len(unique),
                   "candidate_quality_is_heuristic": True}


def _predict_sinusoids(fit: dict, indices: np.ndarray) -> np.ndarray:
    design = sinusoid_design(fit["frequencies"], indices-fit["times_center"])
    return np.einsum("nj,ji->in", design, fit["coefficients"], optimize=False)


def _criterion(residual: np.ndarray, parameters: int, train_count: int) -> float:
    count = residual.size
    rss = max(float(np.sum(residual**2)), 1e-30)
    return float(count*np.log(rss/count)+parameters*np.log(train_count))


def _nrmse(predicted: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(predicted-reference)/max(np.linalg.norm(reference), 1e-30))


def predict_joint_decoder(model: dict, times: np.ndarray,
                          coefficients: np.ndarray | None = None) -> np.ndarray:
    """Predict at times; interpolated coefficient columns may be supplied."""
    decoder = model["decoder"]
    indices = (np.asarray(times)-decoder["time_origin"])/decoder["sample_step"]
    design = sinusoid_design(np.asarray(model["frequencies_rad_per_sample"]),
                             indices-decoder["times_center_index"])
    values = decoder["coefficients"] if coefficients is None else coefficients
    return np.einsum("nj,ji->in", design, np.asarray(values), optimize=False)


def fit_joint_baseline(y: np.ndarray, times: np.ndarray) -> dict:
    q, n = y.shape
    n_train, n_validation = int(.6*n), int(.2*n)
    future_start = n_train+n_validation
    lag = min(64, n_train//4)
    train = y[:, :n_train]
    h = trajectory(train, lag)
    center = h.mean(axis=1, keepdims=True)
    centered = h-center
    u, singular = _randomized_svd(centered)
    validation_indices = np.arange(n_train, future_start)
    future_indices = np.arange(future_start, n)
    mean_prediction = np.broadcast_to(train.mean(axis=1, keepdims=True), (q, n_validation))
    mean_score = _criterion(y[:, n_train:future_start]-mean_prediction, q, train.size)
    fitted_models = [{"score": mean_score, "rank": 0, "line_count": 0,
                      "validation_prediction": mean_prediction, "fit": None, "candidates": []}]
    grid, rank_audits = [{"rank": 0, "line_count": 0, "validation_selection_criterion": mean_score}], []
    for rank in RANKS:
        if rank >= min(centered.shape) or rank > len(singular):
            continue
        candidates, rank_audit = candidates_from_rank(train, centered, u, rank)
        rank_audits.append(rank_audit)
        for count in range(1, len(candidates)+1):
            starting = np.array([row["frequency_start"] for row in candidates[:count]])
            try:
                fitted = fit_joint_frequencies(train, starting, half_width_cycles=.75)
                validation_prediction = _predict_sinusoids(fitted, validation_indices)
                residual = y[:, n_train:future_start]-validation_prediction
                parameters = q*(2*count+1)+count
                score = _criterion(residual, parameters, train.size)
                grid.append({"rank": rank, "line_count": count,
                    "frequency_rad_per_sample": fitted["frequencies"].tolist(),
                    "validation_selection_criterion": score,
                    "validation_observed_nrmse": _nrmse(validation_prediction, y[:, n_train:future_start]),
                    "training_bic": float(train.size*np.log(fitted["rss"]/train.size)+parameters*np.log(train.size)),
                    "optimizer_success": fitted["optimizer_success"]})
                fitted_models.append({"score": score, "rank": rank, "line_count": count,
                    "validation_prediction": validation_prediction, "fit": fitted,
                    "candidates": candidates[:count]})
            except (ValueError, np.linalg.LinAlgError, FloatingPointError, RuntimeError) as error:
                grid.append({"rank": rank, "line_count": count,
                             "failure": f"{type(error).__name__}: {error}"})
    best_score = min(model["score"] for model in fitted_models)
    # With indistinguishable forecast scores prefer the smaller SSA subspace.
    selected = min((model for model in fitted_models if model["score"] <= best_score+2),
                   key=lambda model: (model["rank"], model["line_count"], model["score"]))
    fit = selected["fit"]
    frequencies = np.empty(0) if fit is None else fit["frequencies"]
    if fit is None:
        future_prediction = np.broadcast_to(train.mean(axis=1, keepdims=True), (q, n-future_start)).copy()
        filters = np.empty((0, h.shape[0]), dtype=complex)
        train_phase = np.empty((0, h.shape[1]))
        future_phase = np.empty((0, n-future_start-lag+1))
        clock_train = train_phase.copy()
        clock_future = future_phase.copy()
    else:
        future_prediction = _predict_sinusoids(fit, future_indices)
        filters = np.stack([candidate["filter"] for candidate in selected["candidates"]])
        train_phase = np.stack([candidate["train_phase"] for candidate in selected["candidates"]])
        future_h = trajectory(y[:, future_start:], lag)-center
        future_phase = np.angle(np.einsum("ij,jt->it", filters, future_h, optimize=False))
        train_column_indices = np.arange(train_phase.shape[1])
        origins = np.angle(np.mean(np.exp(1j*(train_phase-frequencies[:, None]
                            *train_column_indices[None, :])), axis=1))
        clock_train = frequencies[:, None]*train_column_indices[None, :]+origins[:, None]
        future_column_indices = np.arange(future_start, future_start+future_phase.shape[1])
        clock_future = frequencies[:, None]*future_column_indices[None, :]+origins[:, None]
    decoder = {"coefficients": train.mean(axis=1)[None, :] if fit is None else fit["coefficients"],
               "times_center_index": (n_train-1)/2 if fit is None else fit["times_center"],
               "time_origin": float(times[0]), "sample_step": float(np.median(np.diff(times))),
               "coefficient_row_order": "constant, all cosines, all sines",
               "trained_on": "first60 percent only"}
    return {"selected_ssa_rank": selected["rank"], "selected_oscillation_line_count": selected["line_count"],
        "frequencies_rad_per_sample": frequencies,
        "frequencies_hz": frequencies/(2*np.pi*float(np.median(np.diff(times)))),
        "lag_window": lag, "joint_trajectory_shape": list(h.shape),
        "singular_values": singular.tolist(), "rank_audits": rank_audits, "selection_grid": grid,
        "selection_rule": "validation Gaussian RSS criterion with BIC-style parameter penalty; score tolerance2 prefers smaller SSA rank",
        "phase_rank_is_certified": False,
        "validation_observed_nrmse": _nrmse(selected["validation_prediction"], y[:, n_train:future_start]),
        "future_observed_nrmse": _nrmse(future_prediction, y[:, future_start:]),
        "future_prediction": future_prediction, "filters": filters, "hankel_center": center,
        "train_ssa_phase": train_phase, "future_ssa_phase": future_phase,
        "train_clock_phase": clock_train, "future_clock_phase": clock_future,
        "train_phase_indices": np.arange(train_phase.shape[1]),
        "future_phase_indices": np.arange(future_start, future_start+future_phase.shape[1]),
        "n_train": n_train, "n_validation": n_validation, "future_start": future_start,
        "decoder": decoder}


def original_mssa(y: np.ndarray, times: np.ndarray, sensors: np.ndarray) -> dict:
    source_path = ROOT/"code/waves/synthetic/models.py"
    source = source_path.read_text()
    names = {"hankel", "subspace", "recurrence", "smooth", "extend", "forecast_mssa"}
    nodes = [node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {"np": np}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source_path), "exec"), namespace)
    capture = {}
    actual_subspace = namespace["subspace"]

    def capture_subspace(matrix):
        capture["trajectory_shape"] = list(matrix.shape)
        result = actual_subspace(matrix)
        capture["rank"] = result.shape[1]
        return result

    namespace["subspace"] = capture_subspace
    n_train, n_validation = int(.6*y.shape[1]), int(.2*y.shape[1])
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        prediction = namespace["forecast_mssa"](times[:n_train], y[:, :n_train],
                                                sensors, times[n_train:], sc=None)
    if not np.isfinite(prediction).all():
        raise FloatingPointError("original MSSA forecast is nonfinite")
    return {"method": "original notebook horizontal MSSA with recurrence",
        "source_path": str(source_path), "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "implementation": "unmodified pure source functions executed with NumPy; scenario module not imported",
        "lag_window": n_train//2, "selected_ssa_rank": capture["rank"],
        "trajectory_shape": capture["trajectory_shape"],
        "rank_selection": "original largest singular-value ratio among first20",
        "validation_observed_nrmse": _nrmse(prediction[:, :n_validation], y[:, n_train:n_train+n_validation]),
        "future_observed_nrmse": _nrmse(prediction[:, n_validation:], y[:, n_train+n_validation:]),
        "future_prediction": prediction[:, n_validation:]}


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


PRIVATE_ARRAYS = {"future_prediction", "filters", "hankel_center", "train_ssa_phase", "future_ssa_phase",
                  "train_clock_phase", "future_clock_phase", "train_phase_indices", "future_phase_indices"}


def save(rows, fitting_complete=False, scoring_complete=False):
    public = [{**row, "joint_ssa_dmd": {key: value for key, value in row["joint_ssa_dmd"].items()
                                      if key not in PRIVATE_ARRAYS},
               "original_mssa": {key: value for key, value in row["original_mssa"].items()
                                  if key != "future_prediction"}} for row in rows]
    payload = {"created_utc": datetime.now(timezone.utc).isoformat(),
        "input_directory": str(DATA/"inputs"), "fit_inputs": ["Y", "s", "t"],
        "true_frequencies_phases_rank_pde_supplied_to_fit": False,
        "evaluation_read_only_after_all_fits": True,
        "train_validation_future_fraction": [.6, .2, .2],
        "ssa_rank_candidates": list(RANKS), "spatial_metric": "uniform sensor weights",
        "fitting_complete": fitting_complete, "scoring_complete": scoring_complete,
        "rows": public,
        "limits": ["selected line count is not certified latent phase rank",
                   "DMD coherence and predictive BIC-style selection are diagnostics",
                   "frozen SSA phases read observed future data; clock forecast uses training-only parameters",
                   "no wave equation or analytic spatial profiles enter fitting"]}
    DESTINATION.write_text(json.dumps(_jsonable(payload), indent=2, allow_nan=False)+"\n")


def score_frozen_models(rows: list[dict]) -> None:
    """Read evaluation only once all train/validation models have been frozen."""
    for row in rows:
        joint = row["joint_ssa_dmd"]
        future_start = joint["future_start"]
        with np.load(DATA/"evaluation"/f"{row['case']}.npz", allow_pickle=False) as truth:
            clean = truth["clean_Y"]
            true_frequencies = truth["source_frequencies_hz"]
            physical_times = truth["physical_t"]
            noise_sigma = float(truth["sensor_noise_standard_deviation"])
            scenario = str(truth["scenario_name"])
        future_clean = clean[:, future_start:]
        scoring = {"scenario_name_evaluation_only": scenario,
            "true_source_frequencies_hz_evaluation_only": true_frequencies.tolist(),
            "joint_future_clean_nrmse": _nrmse(joint["future_prediction"], future_clean),
            "original_mssa_future_clean_nrmse": (
                _nrmse(row["original_mssa"]["future_prediction"], future_clean)
                if "future_prediction" in row["original_mssa"] else None),
            "expected_sensor_noise_relative_rms": float(noise_sigma*np.sqrt(future_clean.size)
                                                        /max(np.linalg.norm(future_clean), 1e-30)),
            "future_phase_kind": "frozen train SSA filter applied to observed future record",
            "future_clock_kind": "training-only learned sinusoid clock",
            "phase_calibration": "one constant origin per matched coordinate estimated on training only",
            "coordinate_scores": []}
        estimated_hz = np.asarray(joint["frequencies_hz"])
        train_indices = np.asarray(joint["train_phase_indices"], dtype=int)
        future_indices = np.asarray(joint["future_phase_indices"], dtype=int)
        for coordinate, frequency in enumerate(estimated_hz):
            index = int(np.argmin(abs(true_frequencies-frequency)))
            truth_train = 2*np.pi*true_frequencies[index]*physical_times[train_indices]
            truth_future = 2*np.pi*true_frequencies[index]*physical_times[future_indices]
            values = {"estimated_frequency_hz": float(frequency),
                      "nearest_true_source_frequency_hz": float(true_frequencies[index]),
                      "frequency_error_hz": float(frequency-true_frequencies[index])}
            for name in ("ssa", "clock"):
                train_phase = joint[f"train_{name}_phase"][coordinate]
                future_phase = joint[f"future_{name}_phase"][coordinate]
                origin = float(np.angle(np.mean(np.exp(1j*(train_phase-truth_train)))))
                errors = abs(np.angle(np.exp(1j*(future_phase-origin-truth_future))))
                values[f"future_{name}_train_calibrated_circular_mae_rad"] = float(np.mean(errors))
                values[f"{name}_training_origin_rad_scoring_only"] = origin
            scoring["coordinate_scores"].append(values)
        row["evaluation_scoring_after_all_fits"] = scoring
        print(row["case"], "clean future joint/original", round(scoring["joint_future_clean_nrmse"],6),
              round(scoring["original_mssa_future_clean_nrmse"],6)
              if scoring["original_mssa_future_clean_nrmse"] is not None else None,
              "phase SSA", [round(score["future_ssa_train_calibrated_circular_mae_rad"],5)
                            for score in scoring["coordinate_scores"]], flush=True)
    save(rows, fitting_complete=True, scoring_complete=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-only", action="store_true",
                        help="score already frozen models, without refitting")
    args = parser.parse_args()
    if args.score_only:
        payload = json.loads(DESTINATION.read_text())
        if not payload["fitting_complete"] or len(payload["rows"]) != 11:
            raise ValueError("all11 fits must finish before evaluation")
        rows = payload["rows"]
        for row in rows:
            with np.load(row["model_arrays"], allow_pickle=False) as saved:
                for key in PRIVATE_ARRAYS:
                    row["joint_ssa_dmd"][key] = saved[key]
                if "original_mssa_future_prediction" in saved:
                    row["original_mssa"]["future_prediction"] = saved["original_mssa_future_prediction"]
        score_frozen_models(rows)
        return
    paths = sorted((DATA/"inputs").glob("*.npz"))
    if len(paths) != 11:
        raise ValueError("expected all11 actual-wave input records")
    MODEL_OUT.mkdir(exist_ok=True)
    rows = []
    for path in paths:
        started = time.monotonic()
        with np.load(path, allow_pickle=False) as record:
            if set(record.files) != {"Y", "s", "t"}:
                raise ValueError("fit input archive must contain only Y,s,t")
            y, sensors, times = record["Y"], record["s"], record["t"]
        joint = fit_joint_baseline(y, times)
        try:
            original = original_mssa(y, times, sensors)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError, RuntimeError) as error:
            original = {"failure": f"{type(error).__name__}: {error}"}
        model_path = MODEL_OUT/f"{path.stem}.npz"
        arrays = {key: value for key, value in joint.items() if key in PRIVATE_ARRAYS}
        arrays["frequencies_rad_per_sample"] = joint["frequencies_rad_per_sample"]
        arrays["decoder_coefficients"] = joint["decoder"]["coefficients"]
        if "future_prediction" in original:
            arrays["original_mssa_future_prediction"] = original["future_prediction"]
        np.savez_compressed(model_path, **arrays)
        row = {"case": path.stem, "input_file": str(path), "model_arrays": str(model_path),
               "sensor_count": y.shape[0], "sample_count": y.shape[1],
               "joint_ssa_dmd": joint, "original_mssa": original,
               "elapsed_seconds": time.monotonic()-started}
        rows.append(row)
        save(rows)
        print(path.stem, "joint rank", joint["selected_ssa_rank"], "lines", joint["selected_oscillation_line_count"],
              "Hz", joint["frequencies_hz"], "future observed", round(joint["future_observed_nrmse"],5),
              "original", round(original.get("future_observed_nrmse", float("nan")),5),
              "seconds", round(row["elapsed_seconds"],2), flush=True)
    save(rows, fitting_complete=True)
    # Evaluation is intentionally opened only after every train/validation fit.
    score_frozen_models(rows)


if __name__ == "__main__":
    main()
