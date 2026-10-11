"""Law-blind viability assay on the original notebook's cached wave fields.

Inference loads inputs/Y,s,t only.  The nodal P1 space uses sensor geometry,
not a wave equation, true frequencies, prescribed rank, or polynomial V.
This is an adapter baseline, not a new multidimensional Hankel operator.
Each separate evaluation archive is opened only after that case's model fits,
forecasts, and frozen future readouts.  Settings are fixed before scoring and
are not adapted between cases.  A non-source line may be a transient rather
than false.
"""

from __future__ import annotations

from project_paths import RESULTS_DIR

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np

from frequency_fit import sinusoid_design
from iterative_ssa import Settings, fit_iterative_ssa, predict_field
from nodal_spatial import NodalSpatialModel
from ssa_character import apply_ssa_character, apply_field_character, aligned_character_rows


HERE = Path(__file__).resolve().parent
DATA = RESULTS_DIR / 'wave_actual_data'
DESTINATION = RESULTS_DIR / 'actual_wave_iterative.json'
CASES = (
    "wave_1d_traveling", "wave_1d_standing", "wave_1d_counterpropagating",
    "wave_1d_layered", "wave_1d_source_noise", "wave_2d_uniform",
    "wave_2d_layered", "wave_2d_lens", "wave_2d_waveguide",
    "wave_3d_uniform", "wave_3d_lens",
)


def serializable(value):
    if isinstance(value, np.ndarray):
        return serializable(value.tolist())
    if isinstance(value, np.generic):
        return serializable(value.item())
    if isinstance(value, dict):
        return {str(key): serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def fingerprints() -> dict:
    names = ("run_actual_wave_iterative.py", "nodal_spatial.py", "iterative_ssa.py",
             "ssa_character.py", "ssa_dynamic_character.py", "frequency_fit.py",
             "nuisance_frequency_fit.py", "phase_lattice.py", "spatial_field.py",
             "phase_factor_regression.py", "phase_readout_refinement.py")
    return {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
            for name in names}


def save(payload: dict, path: Path) -> None:
    payload["last_update_utc"] = datetime.now(timezone.utc).isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(serializable(payload), ensure_ascii=False,
                                    indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def noise_from_training_high_band(y: np.ndarray) -> dict:
    """Pooled periodogram median; finite-band noise dominance is an assumption."""
    n = y.shape[1]
    centered = y - y.mean(axis=1, keepdims=True)
    frequency = 2 * np.pi * np.fft.rfftfreq(n)
    mask = (frequency >= .6 * np.pi) & (frequency <= .95 * np.pi)
    power = abs(np.fft.rfft(centered, axis=1)[:, mask]) ** 2 / n
    sigma_squared = float(np.median(power) / np.log(2))
    if sigma_squared <= 0 or not np.isfinite(sigma_squared):
        raise ValueError("training high-band noise estimate is not positive finite")
    return {"sigma": float(np.sqrt(sigma_squared)),
            "sigma_squared": sigma_squared,
            "method": "pooled median periodogram power / log(2)",
            "band_rad_per_sample": [.6 * np.pi, .95 * np.pi],
            "pooled_bins": int(power.size), "training_samples": n,
            "uses_generator_truth": False,
            "assumption": "high frequency band is dominated by iid Gaussian sensor noise",
            "calibration_status": "assumption_based_not_verified_against_generator_during_fit"}


def settings_for_samples(n: int) -> Settings:
    n_train = int(.6 * n)
    n_validation = int(.2 * n)
    return Settings(n_train=n_train, n_validation=n_validation,
                    n_future=n - n_train - n_validation,
                    lag_window=min(64, n_train // 4), max_steps=6,
                    k_candidates=(16, 32), frontend_mode="dynamic",
                    dynamic_ranks=(4, 8, 16),
                    joint_frequency_mode="harmonic_nuisance",
                    phase_metric_mode="field", temporal_gate_mode="iid_reference")


def direct_fit(train: np.ndarray, frequency: np.ndarray, future_times: np.ndarray) -> dict:
    x = sinusoid_design(frequency, np.arange(train.shape[1]))
    coefficient = np.linalg.lstsq(x, train.T, rcond=1e-10)[0]
    prediction = np.einsum("ij,jk->ik", sinusoid_design(frequency, future_times), coefficient, optimize=False)
    p = len(frequency)
    amplitudes = (coefficient[1:1+p] - 1j * coefficient[1+p:]).T
    return {"prediction": prediction.T, "amplitudes": amplitudes,
            "reference_sensor_indices": np.argmax(abs(amplitudes), axis=0).tolist()
            if p else []}


def classical_frozen(train: np.ndarray, future: np.ndarray,
                     frequencies: np.ndarray, window: int) -> dict:
    """Generic raw MSSA/DMD trained once, targets are inferred lines only."""
    q = train.shape[0]
    def hankel(y):
        count = y.shape[1] - window + 1
        return np.vstack([y[:, lag:lag+count] for lag in range(window)]) / np.sqrt(q * window)
    h_train, h_future = hankel(train), hankel(future)
    center = h_train.mean(axis=1, keepdims=True)
    centered = h_train - center
    x0, x1 = centered[:, :-1], centered[:, 1:]
    rank = min(2 * len(frequencies) + 8, 64, min(x0.shape) - 1)
    sketch_rank = min(min(x0.shape), rank + 16)
    omega = np.random.default_rng(101).standard_normal((x0.shape[1], sketch_rank))
    basis, _ = np.linalg.qr(np.einsum("ij,jk->ik", x0, omega, optimize=False), mode="reduced")
    small_u, singular, vh = np.linalg.svd(np.einsum("ij,jk->ik", basis.T, x0, optimize=False), full_matrices=False)
    rank = min(rank, int(np.sum(singular > singular[0] * 1e-10)))
    u = np.einsum("ij,jk->ik", basis, small_u, optimize=False)[:, :rank]
    singular, vh = singular[:rank], vh[:rank]
    ux1 = np.einsum("ij,jk->ik", u.T, x1, optimize=False)
    shift = np.einsum("ij,jk->ik", ux1, vh.T, optimize=False) / singular[None, :]
    eigenvalues, vectors = np.linalg.eig(shift)
    readout = np.linalg.solve(vectors, u.T)
    used, selected = set(), []
    for frequency in frequencies:
        candidates = np.argsort(abs(eigenvalues - np.exp(1j * frequency)))
        index = next(int(item) for item in candidates if int(item) not in used)
        used.add(index)
        selected.append(index)
    readout = readout[selected]
    return {"train_phase": np.angle(np.einsum("ij,jk->ik", readout, centered, optimize=False)),
            "future_phase": np.angle(np.einsum("ij,jk->ik", readout, h_future - center, optimize=False)),
            "rank": rank, "window": window,
            "selected_eigenvalues": [[float(v.real), float(v.imag)] for v in eigenvalues[selected]],
            "target_frequencies_inferred_only": frequencies.tolist(),
            "future_refit": False, "spatial_weights": "equal raw sensor weights"}


def circular_errors(phase: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return abs(np.angle(np.exp(1j * (phase - reference))))


def phase_score(train: np.ndarray, future: np.ndarray,
                true_train: np.ndarray, true_future: np.ndarray) -> dict:
    zero = float(np.angle(np.mean(np.exp(1j * (train - true_train)))))
    error = circular_errors(future - zero, true_future)
    return {"future_train_origin_aligned_mae_rad": float(error.mean()),
            "train_origin_shift_truth_only_rad": zero,
            "future_columns": len(future), "origin_fit_on_training_only": True}


def field_score(prediction: np.ndarray, clean: np.ndarray, mass: np.ndarray) -> dict:
    error = prediction - clean
    numerator = float(np.sum(error ** 2))
    denominator = float(np.sum(clean ** 2))
    mass_error = float(np.einsum("it,ij,jt->", error, mass, error, optimize=False))
    mass_signal = float(np.einsum("it,ij,jt->", clean, mass, clean, optimize=False))
    return {"sensor_nrmse": float(np.sqrt(numerator / max(denominator, 1e-30))),
            "sensor_rmse": float(np.sqrt(np.mean(error ** 2))),
            "nodal_P1_mass_nrmse": float(np.sqrt(mass_error / max(mass_signal, 1e-30))),
            "spatial_evaluation_scope": "same sensors; nodal interpolation is not unseen-field verification"}


def profile_score(estimated: np.ndarray, reference: np.ndarray, reference_sensor: int) -> dict:
    estimated_relative = np.angle(estimated * np.conj(estimated[reference_sensor]))
    true_relative = np.angle(reference * np.conj(reference[reference_sensor]))
    error = circular_errors(estimated_relative, true_relative)
    visible = abs(reference) >= .05 * abs(reference).max()
    weight = abs(reference[visible]) ** 2
    return {"reference_sensor_selected_from_training": reference_sensor,
            "relative_phase_mae_rad_visible_sensors": float(error[visible].mean()),
            "relative_phase_amplitude_weighted_mae_rad": float(np.sum(weight * error[visible]) / max(weight.sum(), 1e-30)),
            "visible_sensors_truth_only": int(visible.sum()),
            "visibility_threshold_truth_only": "5% of maximum clean fitted amplitude",
            "estimated_relative_phase_rad": estimated_relative.tolist(),
            "clean_relative_phase_rad_truth_only": true_relative.tolist()}


def evaluate(slug: str, data: Path, destination: Path, additional_baselines=None) -> dict:
    started = time.perf_counter()
    input_path = data / "inputs" / (slug + ".npz")
    input_hash = hashlib.sha256(input_path.read_bytes()).hexdigest()
    with np.load(input_path, allow_pickle=False) as archive:
        if set(archive.files) != {"Y", "s", "t"}:
            raise ValueError("model inputs must contain exactly Y,s,t")
        y, s, t = archive["Y"], archive["s"], archive["t"]
    settings = settings_for_samples(y.shape[1])
    if not np.allclose(np.diff(t), np.diff(t).mean(), atol=1e-12, rtol=0):
        raise ValueError("this baseline requires a uniform time grid")
    n_train = settings.n_train
    future_start = n_train + settings.n_validation
    training, validation, future = y[:, :n_train], y[:, n_train:future_start], y[:, future_start:]
    model = NodalSpatialModel.from_positions(s)
    noise = noise_from_training_high_band(training)
    fitted = fit_iterative_ssa(training, validation, settings,
                               sampling_model=model, noise_estimate=noise)
    frequencies = np.asarray(fitted["acquired_frequencies"])
    future_times = np.arange(future_start, y.shape[1])
    prediction = predict_field(fitted["clock_models"][-1], future_times, s,
                               sampling_model=model)
    direct = direct_fit(training, frequencies, future_times)
    mass_future = model.to_mass_coordinates(model.recover(future))
    train_phase, future_phase, quality = [], [], []
    for index, frontend in enumerate(fitted["frozen_frontends"]):
        applied = apply_field_character(frontend, mass_future,
            fitted["clock_models"][index].predict(future_times))
        train = np.asarray(frontend["phase"])
        zero = float(np.angle(np.mean(np.exp(1j * (train - frequencies[index] * np.arange(len(train)))))))
        train = np.angle(np.exp(1j * (train - zero)))
        test = np.angle(np.exp(1j * (np.asarray(applied["phase"]) - zero)))
        radius = np.asarray(applied["radii"])
        normalized = np.asarray(applied["normalized_character"])
        train_phase.append(train)
        future_phase.append(test)
        quality.append({"character": index + 1, "frozen_filter": True,
                        "readout_source": frontend.get("readout_source", "residual"),
                        "window": frontend["window"],
                        "old_model_index": index,
                        "future_columns": len(test),
                        "future_radius_cv": float(radius.std() / max(radius.mean(), 1e-15)),
                        "future_phase_step_coherence": float(abs(np.mean(normalized[1:] * np.conj(normalized[:-1])))),
                        "future_phase_mae_to_inferred_clock_rad": float(circular_errors(test, frequencies[index] * np.arange(future_start, future_start + len(test))).mean()),
                        "train_phase_origin_to_inferred_clock_rad": zero})
    train_phase, future_phase = aligned_character_rows(train_phase), aligned_character_rows(future_phase)
    classical = classical_frozen(training, future, frequencies, settings.lag_window) if len(frequencies) else None
    extra_baselines = (additional_baselines(fitted, training, validation, future_times, model, settings)
                       if additional_baselines is not None else {})
    inference_finished = datetime.now(timezone.utc).isoformat()

    # All inference, amplitude fits, forecasts, and frozen readouts are finished.
    # Separate generator truth is first opened below, exclusively for scoring.
    evaluation_path = data / "evaluation" / (slug + ".npz")
    with np.load(evaluation_path, allow_pickle=False) as archive:
        clean = archive["clean_Y"]
        source_frequency_hz = archive["source_frequencies_hz"]
        physical_t = archive["physical_t"]
        scenario = json.loads(str(archive["scenario_json"]))
        scenario_name = str(archive["scenario_name"])
        applied_sigma = float(archive["sensor_noise_standard_deviation"])
    step = float(np.diff(t).mean())
    source_frequency = 2 * np.pi * source_frequency_hz * step
    true_phase = 2 * np.pi * source_frequency_hz[:, None] * physical_t[None, :]
    tolerance = 2 * np.pi / n_train
    clean_coefficient = np.linalg.lstsq(sinusoid_design(source_frequency, np.arange(n_train)),
                                       clean[:, :n_train].T, rcond=1e-10)[0]
    clean_amplitude = (clean_coefficient[1:3] - 1j * clean_coefficient[3:5]).T
    frequency_scores, label_rows, phase_scores, profile_scores = [], [], [], []
    for index, frequency in enumerate(frequencies):
        matches = []
        nearest = None
        for source_index, target in enumerate(source_frequency):
            for sign in (1, -1):
                gap = float(abs(np.angle(np.exp(1j * (frequency - sign * target)))))
                item = {"source_index": source_index, "sign": sign, "gap_rad_per_sample": gap}
                if nearest is None or gap < nearest["gap_rad_per_sample"]:
                    nearest = item
                if gap <= tolerance:
                    matches.append(item)
        unique = matches[0] if len(matches) == 1 else None
        frequency_scores.append({"acquired_index": index, "frequency_rad_per_sample": float(frequency),
                                 "frequency_hz": float(frequency / (2 * np.pi * step)),
                                 "matches_forcing_tones_truth_only": matches,
                                 "nearest_forcing_tone_truth_only": nearest,
                                 "matching_tolerance_rad_per_sample": tolerance,
                                 "unmatched_does_not_imply_physically_false": True})
        label = [0, 0]
        if unique is None:
            label_rows.append(None)
            continue
        source_index, sign = unique["source_index"], unique["sign"]
        label[source_index] = sign
        label_rows.append(label)
        count = future_phase.shape[1]
        train_count = train_phase.shape[1]
        target_train = sign * true_phase[source_index, :train_count]
        target_future = sign * true_phase[source_index, future_start:future_start+count]
        comparison = {"acquired_index": index, "source_index": source_index, "sign": sign,
                      "frozen_iterative_ssa": phase_score(train_phase[index], future_phase[index], target_train, target_future),
                      "direct_inferred_clock": phase_score(frequency * np.arange(train_count),
                          frequency * np.arange(future_start, future_start+count), target_train, target_future)}
        if classical:
            comparison["frozen_classical_raw_mssa_dmd"] = phase_score(
                classical["train_phase"][index, :train_count], classical["future_phase"][index, :count], target_train, target_future)
        phase_scores.append(comparison)
        reference = clean_amplitude[:, source_index] if sign == 1 else np.conj(clean_amplitude[:, source_index])
        profile_scores.append({"acquired_index": index, "source_index": source_index, "sign": sign,
                               **profile_score(direct["amplitudes"][:, index], reference,
                                               direct["reference_sensor_indices"][index])})
    matched_sources = sorted({item["source_index"] for score in frequency_scores
                              for item in score["matches_forcing_tones_truth_only"]})
    lattice = fitted["audit"].get("final_lattice")
    all_mapped = bool(len(label_rows) and all(label is not None for label in label_rows))
    relation_valid = False
    physical_basis = None
    if lattice and all_mapped:
        labels = np.asarray(label_rows, dtype=int)
        physical_basis = np.asarray(lattice["products"], dtype=int) @ labels
        relation_valid = bool(np.array_equal(np.asarray(lattice["peak_labels"], dtype=int) @ physical_basis, labels))
    stage_summary = [{"step": stage.get("step"), "status": stage.get("status"),
                      "validation_gain_positive": stage.get("validation_gain_positive"),
                      "validation_mse_improvement": stage.get("validation_mse_improvement"),
                      "candidate_frequency": (stage.get("candidate") or {}).get("frequency"),
                      "frontend_status": (stage.get("frontend") or {}).get("status"),
                      "train_phase_coherence": ((stage.get("frontend") or {}).get("metrics") or {}).get("phase_step_coherence"),
                      "train_radius_cv": ((stage.get("frontend") or {}).get("metrics") or {}).get("coordinate_radius_cv")}
                     for stage in fitted["audit"]["stages"]]
    coordinate_path = destination.parent / (destination.stem + "_" + slug + "_coordinates.npz")
    np.savez_compressed(coordinate_path, acquired_frequencies=frequencies,
                        train_phase=train_phase, future_phase=future_phase,
                        classical_train_phase=classical["train_phase"] if classical else np.empty((0, 0)),
                        classical_future_phase=classical["future_phase"] if classical else np.empty((0, 0)),
                        true_phase_truth_only=true_phase,
                        hierarchy_future_sensor_prediction=prediction,
                        direct_sinusoid_future_sensor_prediction=direct["prediction"],
                        future_times=t[future_start:],
                        readout_windows=np.asarray([front["window"] for front in fitted["frozen_frontends"]]),
                        readout_sources=np.asarray([front.get("readout_source", "residual") for front in fitted["frozen_frontends"]]),
                        **{name+"_future_prediction": value["prediction"] for name, value in extra_baselines.items()})
    scored_extra = {name: field_score(value["prediction"], clean[:, future_start:], model.physical_mass)
                    for name, value in extra_baselines.items()}
    return {"case": slug, "scenario_name_truth_only": scenario_name,
            "run_status": "finished", "settings": asdict(settings),
            "inputs": {"path": str(input_path), "sha256": input_hash,
                       "keys": ["Y", "s", "t"], "shape": list(y.shape),
                       "spatial_dimension": s.shape[1], "sample_step_seconds": step},
            "inference_finished_before_truth_open_utc": inference_finished,
            "evaluation_path_truth_only": str(evaluation_path),
            "evaluation_sha256_truth_only": hashlib.sha256(evaluation_path.read_bytes()).hexdigest(),
            "adapter_kind": "coordinate-only nodal P1 sensor-hull baseline",
            "new_multidimensional_hankel_operator": False,
            "generator_law_used_in_inference": False,
            "generator_frequency_or_rank_used_in_inference": False,
            "noise_estimate_train_only": noise,
            "applied_noise_sigma_truth_only": applied_sigma,
            "noise_sigma_estimated_over_applied_truth_only": noise["sigma"] / applied_sigma,
            "inference": fitted["audit"], "stage_summary": stage_summary,
            "accepted_nonpositive_validation_gain_count": sum(stage.get("validation_gain_positive") is False for stage in fitted["audit"]["stages"]),
            "frozen_iterative_readout": quality,
            "frequency_scores_truth_only": frequency_scores,
            "forcing_tone_catalog_scope": "two imposed tones; does not label free-wave transients or stochastic response",
            "matched_forcing_source_indices_truth_only": matched_sources,
            "both_forcing_tones_found_truth_only": matched_sources == [0, 1],
            "accepted_not_uniquely_matched_to_forcing_tones_count": sum(label is None for label in label_rows),
            "inferred_relation_matches_forcing_labels_truth_only": relation_valid,
            "physical_basis_forcing_labels_truth_only": physical_basis,
            "source_character_module_recovered_truth_only": bool(matched_sources == [0, 1] and all_mapped and relation_valid and fitted["audit"].get("r_candidate_conditional") == 2),
            "phase_comparison_for_matched_forcing_tones_truth_only": phase_scores,
            "relative_sensor_phase_profiles_truth_only": profile_scores,
            "classical_baseline": {key: value for key, value in classical.items() if key not in ("train_phase", "future_phase")} if classical else None,
            "additional_lawblind_baselines": {name: {key: value for key, value in row.items() if key != "prediction"}
                                              for name, row in extra_baselines.items()},
            "future_clean_sensor_field": {
                "iterative_hierarchy": field_score(prediction, clean[:, future_start:], model.physical_mass),
                "direct_raw_sinusoid_same_acquired_frequencies": field_score(direct["prediction"], clean[:, future_start:], model.physical_mass),
                **scored_extra},
            "source_noise_truth_only": bool(scenario.get("noise", 0)),
            "phase_only_field_model_status_truth_only": "violated_by_stochastic_source" if scenario.get("noise", 0) else "stationary_two-tone_approximation_transient_bias_not_certified",
            "sampled_orbit_dense_in_full_torus_truth_only": False,
            "sampled_first_source_phase_values_truth_only": 50,
            "torus_geometry_recovered": False,
            "phase_completeness_certified": False,
            "coordinates_npz": str(coordinate_path),
            "elapsed_seconds": time.perf_counter() - started}


def summary(rows: list) -> dict:
    finished = [row for row in rows if row["run_status"] == "finished"]
    return {"finished_cases": len(finished), "exception_cases": len(rows)-len(finished),
            "both_forcing_tones_found": sum(row["both_forcing_tones_found_truth_only"] for row in finished),
            "source_character_module_recovered": sum(row["source_character_module_recovered_truth_only"] for row in finished),
            "accepted_not_source_matched_lines": sum(row["accepted_not_uniquely_matched_to_forcing_tones_count"] for row in finished),
            "accepted_nonpositive_validation_gain_stages": sum(row["accepted_nonpositive_validation_gain_count"] for row in finished),
            "hierarchy_future_sensor_nrmse_median": float(np.median([row["future_clean_sensor_field"]["iterative_hierarchy"]["sensor_nrmse"] for row in finished])) if finished else None,
            "direct_future_sensor_nrmse_median": float(np.median([row["future_clean_sensor_field"]["direct_raw_sinusoid_same_acquired_frequencies"]["sensor_nrmse"] for row in finished])) if finished else None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, default=DESTINATION)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    source_hashes = fingerprints()
    rows = []
    if args.output.exists():
        if not args.resume:
            parser.error("output already exists; use --resume or a separate --output")
        previous = json.loads(args.output.read_text())
        if previous["source_fingerprints"] != source_hashes or previous["data_directory"] != str(args.data.resolve()):
            parser.error("resume rejected: inference source or input data directory changed")
        rows = previous["rows"]
        for row in rows:
            if row["run_status"] == "finished":
                path = Path(row["inputs"]["path"])
                if hashlib.sha256(path.read_bytes()).hexdigest() != row["inputs"]["sha256"]:
                    parser.error("resume rejected: cached inputs changed")
    started = time.perf_counter()
    payload = {"status": "running", "invocation_finished": False,
               "data_directory": str(args.data.resolve()),
               "source_fingerprints": source_hashes, "rows": rows,
               "assay": "original notebook waves, law-blind nodal baseline adapter",
               "split": "60% train / 20% validation / 20% future",
               "training_time_coordinate": "sample index; supplied uniform t sets post-fit units",
               "truth_archive_opening_scope": "per case, after that case's fits, forecasts, and frozen readouts",
               "between_case_hyperparameter_adaptation_from_truth": False,
               "finite_assay_is_algorithm_or_geometry_certificate": False}
    save(payload, args.output)
    completed = {row["case"] for row in rows}
    for slug in args.cases:
        if slug in completed:
            continue
        print(f"starting {slug}", flush=True)
        try:
            row = evaluate(slug, args.data, args.output)
        except Exception as error:
            row = {"case": slug, "run_status": "exception",
                   "exception": f"{type(error).__name__}: {error}",
                   "traceback": traceback.format_exc()}
        rows.append(row)
        payload["summary"] = summary(rows)
        save(payload, args.output)
        if row["run_status"] == "finished":
            print(json.dumps({"case": slug, "accepted": row["inference"]["accepted_character_count"],
                              "r_candidate": row["inference"]["r_candidate_conditional"],
                              "status": row["inference"]["status"],
                              "both_tones": row["both_forcing_tones_found_truth_only"],
                              "not_source_matched": row["accepted_not_uniquely_matched_to_forcing_tones_count"],
                              "negative_validation_gain": row["accepted_nonpositive_validation_gain_count"],
                              "hierarchy_nrmse": row["future_clean_sensor_field"]["iterative_hierarchy"]["sensor_nrmse"],
                              "direct_nrmse": row["future_clean_sensor_field"]["direct_raw_sinusoid_same_acquired_frequencies"]["sensor_nrmse"],
                              "elapsed_seconds": row["elapsed_seconds"]}), flush=True)
        else:
            print(json.dumps({"case": slug, "exception": row["exception"]}), flush=True)
    unchanged = fingerprints() == source_hashes
    payload.update(status="finished" if unchanged else "source_changed_during_run",
                   invocation_finished=True,
                   source_fingerprints_unchanged=unchanged,
                   invocation_runtime_seconds=time.perf_counter()-started,
                   summary=summary(rows))
    save(payload, args.output)
    print(json.dumps({"output": str(args.output), "status": payload["status"],
                      **payload["summary"]}), flush=True)


if __name__ == "__main__":
    main()
