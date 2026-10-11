"""Saved revised phase-family assay, with original waves and five controls.

The core uses learned phase-factor Fourier sieves with validation rollback.
All spatial models are nodal models constructed from sensor coordinates.
Generator truth is used only for data production and post-fit scoring.
Dictionary sizes, windows, rank budgets and noise estimates never use truth.
"""

from __future__ import annotations

from project_paths import RESULTS_DIR

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np

import run_actual_wave_iterative as wave_driver
from iterative_ssa import Settings, fit_iterative_ssa, predict_field
from nodal_spatial import NodalSpatialModel
from ssa_character import apply_ssa_character, apply_field_character, aligned_character_rows


HERE = Path(__file__).resolve().parent
OUTPUT = RESULTS_DIR / 'actual_wave_revised.json'
FIXTURE_OUTPUT = RESULTS_DIR / 'actual_wave_revised_fixtures.json'
FIXTURES = ("r2_poisson_dominant_weak", "r2_infinite_mixed_products", "r2_partial_cover",
            "r1_poisson_negative", "r1_poisson2_cos3")
ORIGINAL_SETTINGS = wave_driver.settings_for_samples


def revised_settings(n: int) -> Settings:
    return replace(ORIGINAL_SETTINGS(n), conditional_model_mode="fourier_sieve",
                   enforce_validation_improvement=True)


def fingerprints() -> dict:
    result = wave_driver.fingerprints()
    for name in ("phase_factor_regression.py", "run_actual_wave_revised.py", "iterative_fields.py"):
        result[name] = hashlib.sha256((HERE / name).read_bytes()).hexdigest()
    return result


def comparison(rows: list) -> dict:
    """Read prior results only after revised case fits have finished."""
    legacy_path = RESULTS_DIR / 'actual_wave_iterative_verified.json'
    mssa_path = RESULTS_DIR / 'actual_wave_baseline.json'
    legacy = {row["case"]: row for row in json.loads(legacy_path.read_text())["rows"]}
    mssa = {row["case"]: row for row in json.loads(mssa_path.read_text())["rows"]}
    for row in rows:
        if row["run_status"] != "finished":
            continue
        slug = row["case"]
        new_error = row["future_clean_sensor_field"]["iterative_hierarchy"]["sensor_nrmse"]
        original_error = legacy[slug]["future_clean_sensor_field"]["iterative_hierarchy"]["sensor_nrmse"]
        classical = mssa[slug]["evaluation_scoring_after_all_fits"]
        row["comparison_with_saved_frozen_assays"] = {
            "legacy_iterative_hierarchy_sensor_nrmse": original_error,
            "revised_iterative_hierarchy_sensor_nrmse": new_error,
            "independent_joint_raw_ssa_dmd_sensor_nrmse": classical["joint_future_clean_nrmse"],
            "original_notebook_mssa_sensor_nrmse": classical["original_mssa_future_clean_nrmse"],
            "legacy_to_revised_error_ratio": original_error / new_error,
            "same_original_cached_inputs_and_time_split": True,
            "comparison_does_not_change_fitting_or_parameters": True}
    return {"legacy_file": str(legacy_path), "independent_classical_file": str(mssa_path)}


def recurrent_mssa(train: np.ndarray, validation: np.ndarray,
                   future_count: int, window: int) -> dict:
    """Independent horizontal MSSA recurrence, validation-selected rank.

    The fixed rank grid allows substantially more temporal harmonic pairs
    than the acquired-generator list.  No frequencies or phase rank enter.
    """
    mean = train.mean(axis=1, keepdims=True)
    values = train - mean
    hankels = [np.lib.stride_tricks.sliding_window_view(y, window).T for y in values]
    covariance = sum(np.einsum("it,jt->ij", h, h, optimize=False) for h in hankels)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    u = eigenvectors[:, order]
    candidates, grid = [], []
    for rank in (2, 4, 8, 16, 32, 48, 64):
        if rank >= window - 2:
            continue
        basis = u[:, :rank]
        denominator = 1 - float(np.sum(basis[-1] ** 2))
        if denominator <= 1e-8:
            grid.append({"rank": rank, "failure": "unstable recurrence denominator"})
            continue
        recurrence = np.einsum("ij,j->i", basis[:-1], basis[-1], optimize=False) / denominator
        smoothed = []
        for h in hankels:
            projection = np.einsum("ij,jt->it", basis,
                np.einsum("ij,it->jt", basis, h, optimize=False), optimize=False)
            n = h.shape[0] + h.shape[1] - 1
            smoothed.append(np.asarray([np.mean(np.diagonal(projection[::-1], offset=k-window+1)) for k in range(n)]))
        smoothed = np.asarray(smoothed)
        history = np.zeros((train.shape[0], train.shape[1] + validation.shape[1] + future_count))
        history[:, :train.shape[1]] = smoothed
        with np.errstate(over="ignore", invalid="ignore"):
            for index in range(train.shape[1], history.shape[1]):
                history[:, index] = np.einsum("i,qi->q", recurrence,
                    history[:, index-len(recurrence):index], optimize=False)
        forecast = history[:, train.shape[1]:] + mean
        if not np.isfinite(forecast).all():
            grid.append({"rank": rank, "failure": "nonfinite recurrent forecast"})
            continue
        error = float(np.mean((forecast[:, :validation.shape[1]] - validation) ** 2))
        grid.append({"rank": rank, "validation_sensor_mse": error})
        candidates.append((error, rank, forecast[:, validation.shape[1]:]))
    if not candidates:
        raise ValueError("no usable independent MSSA recurrence rank")
    _, rank, prediction = min(candidates, key=lambda item: item[:2])
    return {"prediction": prediction, "rank": rank, "window": window,
            "rank_selection_grid": grid,
            "method": "horizontal raw MSSA with recurrence; validation-selected fixed rank grid",
            "true_frequency_or_phase_rank_supplied": False}


def fixture_inputs(name: str):
    """Fixture generator only; no generator labels are returned to fitting."""
    from iterative_fields import AREA, SENSORS, generate_field
    samples = np.arange(6400)
    clean = generate_field(name, samples, SENSORS, n_train=3200)["clean"]
    sigma = .08 * np.sqrt(np.mean(np.sum(AREA[:, None] * clean[:, :3200] ** 2, axis=0)))
    observed = clean + sigma * np.random.default_rng(54000).standard_normal(clean.shape)
    return observed, SENSORS.copy(), samples, float(sigma)


def evaluate_fixture(name: str, destination: Path,
                     phase_relation_mode: str = "clock_diagnostic", settings_override=None,
                     additional_baselines=None) -> dict:
    started = time.perf_counter()
    y, positions, times, generator_sigma = fixture_inputs(name)
    settings = Settings(n_train=3200, n_validation=1600, n_future=1600,
                        lag_window=128, frontend_mode="dynamic", dynamic_ranks=(4, 8, 16),
                        joint_frequency_mode="harmonic_nuisance", phase_metric_mode="field",
                        temporal_gate_mode="iid_reference", k_candidates=(16, 32),
                        conditional_model_mode="fourier_sieve", enforce_validation_improvement=True,
                        phase_relation_mode=phase_relation_mode)
    if settings_override is not None:
        settings = settings_override(settings)
    model = NodalSpatialModel.from_positions(positions)
    noise = wave_driver.noise_from_training_high_band(y[:, :3200])
    fitted = fit_iterative_ssa(y[:, :3200], y[:, 3200:4800], settings,
                               sampling_model=model, noise_estimate=noise)
    frequency = np.asarray(fitted["acquired_frequencies"])
    future_times = times[4800:]
    prediction = predict_field(fitted["clock_models"][-1], future_times, positions,
                               sampling_model=model)
    future_mass = model.to_mass_coordinates(y[:, 4800:])
    train_phase, future_phase, quality = [], [], []
    for index, frontend in enumerate(fitted["frozen_frontends"]):
        readout = apply_field_character(frontend, future_mass, fitted["clock_models"][index].predict(future_times))
        train = np.asarray(frontend["phase"])
        zero = float(np.angle(np.mean(np.exp(1j * (train - frequency[index] * np.arange(len(train)))))))
        train_phase.append(np.angle(np.exp(1j * (train - zero))))
        future_phase.append(np.angle(np.exp(1j * (readout["phase"] - zero))))
        radius = readout["radii"]
        quality.append({"readout_source": frontend.get("readout_source", "residual"), "window": frontend["window"],
                        "future_radius_cv": float(radius.std() / max(radius.mean(), 1e-15)),
                        "future_phase_step_coherence": float(abs(np.mean(readout["normalized_character"][1:] * np.conj(readout["normalized_character"][:-1]))))})
    train_phase, future_phase = aligned_character_rows(train_phase), aligned_character_rows(future_phase)
    classical_phase = wave_driver.classical_frozen(y[:, :3200], y[:, 4800:], frequency, 128) if len(frequency) else None
    recurrent = recurrent_mssa(y[:, :3200], y[:, 3200:4800], 1600, 128)
    extra_baselines = (additional_baselines(fitted, y[:, :3200], y[:, 3200:4800],
                                           future_times, model, settings)
                       if additional_baselines is not None else {})

    # Inference and all baselines finished: begin generator-only scoring.
    from iterative_fields import CASE_BY_NAME, generate_field, score_frequency
    from run_iterative_ssa import _signature, _phase_scores, _products
    generated = generate_field(name, times, positions, n_train=3200)
    clean = generated["clean"]
    fixture = CASE_BY_NAME[name]
    truth_lattice = generated["observable_lattice"]
    scores = [score_frequency(name, float(value), 2 * np.pi / 3200, max_harmonic=32)
              for value in frequency]
    labels = []
    for score in scores:
        distinct = {tuple(match["mode"]) for match in score["matches"]}
        labels.append(next(iter(distinct)) if len(distinct) == 1 else None)
    all_mapped = bool(labels and all(label is not None for label in labels))
    module_complete, relation_valid, phase_comparison = False, False, None
    lattice = fitted["audit"].get("final_lattice")
    if all_mapped and lattice:
        physical_labels = np.asarray(labels, dtype=int)
        signature = _signature(physical_labels, fixture.phase_rank)
        module_complete = signature["rank"] == truth_lattice["rank"] and signature["basis"] == truth_lattice["basis"]
        products = np.asarray(lattice["products"], dtype=int)
        physical_basis = products @ physical_labels
        relation_valid = bool(np.array_equal(np.asarray(lattice["peak_labels"], dtype=int) @ physical_basis, physical_labels))
        count, train_count = future_phase.shape[1], train_phase.shape[1]
        true_train = np.einsum("ij,jt->it", physical_basis, generated["theta"][:, :train_count], optimize=False)
        true_future = np.einsum("ij,jt->it", physical_basis, generated["theta"][:, 4800:4800+count], optimize=False)
        iterative_scores, _ = _phase_scores(_products(train_phase, products), _products(future_phase, products), true_train, true_future)
        mssa_scores, _ = _phase_scores(_products(classical_phase["train_phase"][:, :train_count], products),
                                      _products(classical_phase["future_phase"][:, :count], products), true_train, true_future)
        clock_frequency = np.asarray(lattice["basis"])
        clock_scores, _ = _phase_scores(clock_frequency[:, None] * np.arange(train_count),
                                       clock_frequency[:, None] * np.arange(4800,4800+count), true_train, true_future)
        phase_comparison = {"iterative_frozen_ssa": iterative_scores,
                            "classical_frozen_raw_mssa_dmd": mssa_scores,
                            "direct_inferred_clock": clock_scores}
    np.savez_compressed(destination.parent / (destination.stem + "_" + name + "_coordinates.npz"),
                        acquired_frequencies=frequency, train_phase=train_phase, future_phase=future_phase,
                        hierarchy_future_prediction=prediction, recurrent_mssa_future_prediction=recurrent["prediction"],
                        readout_windows=np.asarray([front["window"] for front in fitted["frozen_frontends"]]),
                        readout_sources=np.asarray([front.get("readout_source", "residual") for front in fitted["frozen_frontends"]]),
                        **{name+"_future_prediction": value["prediction"] for name, value in extra_baselines.items()})
    return {"case": name, "run_status": "finished", "settings": asdict(settings),
            "input_scope": "observed Y, sensor positions and times only; nodal model, no polynomial V",
            "data_seed": 54000, "generator_noise_sigma_truth_only": generator_sigma,
            "noise_estimate_train_only": noise,
            "inference": fitted["audit"], "frequency_scores_truth_only": scores,
            "acquired_labels_truth_only": labels,
            "module_complete_truth_only": module_complete,
            "relation_valid_truth_only": relation_valid,
            "correct_rank_truth_only": fitted["audit"]["r_candidate_conditional"] == truth_lattice["rank"],
            "correct_observable_module_rank_relation_truth_only": bool(module_complete and relation_valid and fitted["audit"]["r_candidate_conditional"] == truth_lattice["rank"]),
            "phase_comparison_truth_only": phase_comparison,
            "frozen_future_character_quality": quality,
            "future_clean_sensor_field": {"revised_phase_family": wave_driver.field_score(prediction, clean[:, 4800:], model.physical_mass),
                                           "independent_recurrent_mssa": wave_driver.field_score(recurrent["prediction"], clean[:, 4800:], model.physical_mass),
                                           **{name: wave_driver.field_score(value["prediction"], clean[:, 4800:], model.physical_mass)
                                              for name, value in extra_baselines.items()}},
            "additional_lawblind_baselines": {name: {key: value for key, value in row.items() if key != "prediction"}
                                              for name, row in extra_baselines.items()},
            "independent_mssa": {key: value for key, value in recurrent.items() if key != "prediction"},
            "phase_completeness_or_spatial_geometry_certified": False,
            "elapsed_seconds": time.perf_counter() - started}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", choices=wave_driver.CASES, default=list(wave_driver.CASES))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-fixtures", action="store_true")
    parser.add_argument("--phase-relations", choices=("clock_diagnostic", "observed"), default="clock_diagnostic")
    parser.add_argument("--fixture-output", type=Path, default=FIXTURE_OUTPUT)
    args = parser.parse_args(argv)
    wave_driver.settings_for_samples = lambda n: replace(revised_settings(n), phase_relation_mode=args.phase_relations)
    hashes = fingerprints()
    rows = []
    if args.output.exists():
        if not args.resume:
            parser.error("output exists; use --resume or another --output")
        previous = json.loads(args.output.read_text())
        if previous["source_fingerprints"] != hashes:
            parser.error("resume rejected: source fingerprints changed")
        rows = previous["rows"]
    payload = {"status": "running", "invocation_finished": False, "source_fingerprints": hashes,
               "rows": rows, "settings_mode": "fourier_sieve with strict validation rollback",
               "phase_relation_mode": args.phase_relations,
               "truth_opening_scope": "per case after all that case's fits/forecasts/readouts",
               "between_case_adaptation_from_truth": False,
               "general_phase_factor_formula_finite_implementation_budgets": True}
    wave_driver.save(payload, args.output)
    started = time.perf_counter()
    completed = {row["case"] for row in rows}
    for name in args.cases:
        if name in completed:
            continue
        print("starting revised " + name, flush=True)
        try:
            row = wave_driver.evaluate(name, wave_driver.DATA, args.output)
        except Exception as error:
            row = {"case": name, "run_status": "exception", "exception": f"{type(error).__name__}: {error}",
                   "traceback": traceback.format_exc()}
        rows.append(row)
        payload["summary"] = wave_driver.summary(rows)
        wave_driver.save(payload, args.output)
        print(json.dumps({"case": name, "status": row.get("inference", {}).get("status", row["run_status"]),
                          "accepted": row.get("inference", {}).get("accepted_character_count"),
                          "nrmse": row.get("future_clean_sensor_field", {}).get("iterative_hierarchy", {}).get("sensor_nrmse"),
                          "elapsed": row.get("elapsed_seconds")}), flush=True)
    payload.update(status="finished", invocation_finished=True,
                   comparison_sources=comparison(rows), summary=wave_driver.summary(rows),
                   source_fingerprints_unchanged=fingerprints() == hashes,
                   invocation_runtime_seconds=time.perf_counter() - started)
    wave_driver.save(payload, args.output)
    if not args.skip_fixtures:
        fixture_rows = []
        fixture_payload = {"status": "running", "rows": fixture_rows, "source_fingerprints": hashes,
                           "input_spatial_model": "nodal P1, no known polynomial V", "truth_for_scoring_only": True}
        fixture_payload["phase_relation_mode"] = args.phase_relations
        wave_driver.save(fixture_payload, args.fixture_output)
        for name in FIXTURES:
            print("starting revised fixture " + name, flush=True)
            try:
                row = evaluate_fixture(name, args.fixture_output, args.phase_relations)
            except Exception as error:
                row = {"case": name, "run_status": "exception", "exception": f"{type(error).__name__}: {error}",
                       "traceback": traceback.format_exc()}
            fixture_rows.append(row)
            wave_driver.save(fixture_payload, args.fixture_output)
            print(json.dumps({"fixture": name, "status": row.get("inference", {}).get("status", row["run_status"]),
                              "rank": row.get("inference", {}).get("r_candidate_conditional"),
                              "correct_module": row.get("correct_observable_module_rank_relation_truth_only"),
                              "field": row.get("future_clean_sensor_field"), "exception": row.get("exception")}), flush=True)
        fixture_payload.update(status="finished", invocation_finished=True,
                               source_fingerprints_unchanged=fingerprints() == hashes)
        wave_driver.save(fixture_payload, args.fixture_output)
    print(json.dumps({"output": str(args.output), "summary": payload["summary"],
                      "source_fingerprints_unchanged": payload["source_fingerprints_unchanged"]}), flush=True)


if __name__ == "__main__":
    main()
