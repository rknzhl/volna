"""Spatial RKHS pilot with an outer sensor excluded from every learned model.

All temporal dictionaries are discovered afresh from retained sensors.  Inner
sensor CV is conditional on that dictionary.  The same trailing raw L128
observer encodes each candidate.  Predictions for all nine declared cases
are frozen before any clean evaluation archive is opened.  No propagation
law, generator parameters, physical phases or prior full-sensor fits enter
the inference functions.
"""

from __future__ import annotations

from project_paths import RESULTS_DIR

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np
from scipy.spatial import Delaunay

import cover_selection_ssa as cover
from factor_ssa import fit_factor_ssa
from hankel_torus_projection import ProjectionSettings
from nodal_spatial import NodalSpatialModel
from raw_hankel_seed import initialize_from_raw_strong_characters
from robust_hankel_torus_projection import RobustHankelTorusProjector
import run_actual_wave_iterative as wave
from run_general_robustness_suite import CASES as GENERAL_CASES, SETTINGS as GENERAL_SETTINGS
from run_original_hankel_torus import evaluate_field, settings_for
from spatial_rkhs_factor import SpatialRKHSSettings, fit_spatial_rkhs_factor


HERE = Path(__file__).resolve().parent
ROOT = RESULTS_DIR / 'spatial_rkhs_hankel'
OUTPUT = ROOT.with_suffix(".json")
ORIGINAL_CASES = (("wave_1d_traveling", 3), ("wave_2d_lens", 5), ("wave_3d_uniform", 13))
GENERAL_SEED = 202610081
PROJECTION = ProjectionSettings()
SPATIAL_SETTINGS = SpatialRKHSSettings()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(payload):
    temporary = OUTPUT.with_suffix(".tmp")
    temporary.write_text(json.dumps(wave.serializable(payload), ensure_ascii=False,
                                    indent=2, allow_nan=False) + "\n")
    temporary.replace(OUTPUT)


def hashes():
    names = ("run_spatial_rkhs_hankel.py", "spatial_rkhs_factor.py", "factor_ssa.py",
             "cover_selection_ssa.py", "nodal_spatial.py", "phase_witness_conditioning.py",
             "phase_factor_regression.py", "hankel_torus_projection.py",
             "raw_hankel_seed.py", "robust_hankel_torus_projection.py")
    return {name: digest(HERE / name) for name in names}


def geometry_only_holdout(positions):
    """Prefer an interior node; explicitly mark extrapolation if none exists."""
    positions = np.asarray(positions)
    distance = np.linalg.norm(positions - positions.mean(axis=0), axis=1)
    for held in np.argsort(distance, kind="stable"):
        retained = np.delete(positions, held, axis=0)
        if positions.shape[1] == 1:
            inside = retained.min() < positions[held, 0] < retained.max()
        else:
            inside = Delaunay(retained).find_simplex(positions[[held]], tol=1e-12)[0] >= 0
        if inside:
            return int(held)
    return int(np.argsort(distance, kind="stable")[0])


def public_manifest():
    specifications = [
        {"case": name, "run": name, "kind": "wave", "held": held,
         "input": wave.DATA / "inputs" / (name + ".npz"),
         "evaluation": wave.DATA / "evaluation" / (name + ".npz")}
        for name, held in ORIGINAL_CASES]
    for spec in GENERAL_CASES:
        run = spec["name"] + "_seed" + str(GENERAL_SEED)
        specifications.append({"case": spec["name"], "run": run, "kind": "general",
            "held": None, "input": RESULTS_DIR / 'general_robustness/inputs' / (run + ".npz"),
            "evaluation": RESULTS_DIR / 'general_robustness/evaluation' / (run + ".npz")})
    manifest = []
    for spec in specifications:
        with np.load(spec["input"], allow_pickle=False) as public:
            if set(public.files) != {"Y", "s", "t"}:
                raise ValueError("public observation archive must contain exactly Y,s,t")
            positions, times = public["s"], public["t"]
            held = spec["held"] if spec["held"] is not None else geometry_only_holdout(positions)
            keep = np.delete(np.arange(len(positions)), held)
            retained = public["Y"][keep]
        path = ROOT / "inputs" / (spec["run"] + ".npz")
        np.savez_compressed(path, Y=retained, s=positions[keep], t=times)
        retained_positions = positions[keep]
        if positions.shape[1] == 1:
            inside = bool(retained_positions.min() <= positions[held, 0] <= retained_positions.max())
        else:
            inside = bool(Delaunay(retained_positions).find_simplex(positions[[held]], tol=1e-12)[0] >= 0)
        manifest.append({"case": spec["case"], "run": spec["run"], "kind": spec["kind"],
            "retained_input": str(path), "retained_input_sha256": digest(path),
            "source_public_input": str(spec["input"]), "source_public_input_sha256": digest(spec["input"]),
            "evaluation": str(spec["evaluation"]), "held_sensor_index": held,
            "retained_sensor_indices": keep.tolist(), "query_position": positions[[held]].tolist(),
            "query_inside_retained_convex_hull": inside,
            "holdout_selection": "predeclared original node; general nearest-centroid interior node, fallback nearest-centroid node when all nodes are hull vertices"})
    return manifest


def encode_candidate(name, sensor_coefficients, query_coefficients, fit, spatial, y, settings):
    mass_coefficients = spatial.coordinates(sensor_coefficients.T).T
    projector = RobustHankelTorusProjector(fit.labels, mass_coefficients, fit.frequencies, PROJECTION)
    cutoff = settings.n_train + settings.n_validation
    start = cutoff - PROJECTION.window + 1
    mass_training = spatial.coordinates(y[:, :settings.n_train])
    observed = spatial.coordinates(y[:, start:])
    seed = initialize_from_raw_strong_characters(mass_training, observed, projector)
    if seed["phase"].shape[1] != y.shape[1] - cutoff:
        raise ValueError("raw trailing-window starts do not match held-out endpoints")
    projected = projector.project(observed, seed["phase"])
    phase = np.angle(np.exp(1j * (projected["phase"] +
                         fit.frequencies[:, None] * (PROJECTION.window - 1))))
    retained_prediction = evaluate_field(fit.labels, sensor_coefficients, phase)
    query_prediction = evaluate_field(fit.labels, query_coefficients, phase)
    diagnostics = projected["diagnostics"]
    arrays = {"phase_at_window_start": projected["phase"], "phase_at_window_end": phase,
              "retained_prediction": retained_prediction, "query_prediction": query_prediction,
              "time_end_indices": np.arange(cutoff, y.shape[1]),
              "window_cost": np.asarray([item["final_cost"] for item in diagnostics]),
              "selected_start": np.asarray([item["selected_start"] for item in diagnostics]),
              "coefficients_sensor": sensor_coefficients, "coefficients_query": query_coefficients}
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise FloatingPointError("nonfinite phase, profile or reconstruction")
    return arrays, {"name": name, "seed": seed["metadata"], "multistart": projected["multistart"],
        "window": PROJECTION.window, "context_scope": "same raw retained-sensor trailing128 window",
        "gradient_converged_count": sum(item["converged_gradient"] for item in diagnostics),
        "window_count": len(diagnostics),
        "maximum_final_gradient": max(item["final_gradient_max"] for item in diagnostics),
        "global_optimum_certified": False, "future_clock_seed_used": False}


def fit_and_freeze(entry):
    started = time.perf_counter()
    source = Path(entry["retained_input"])
    if digest(source) != entry["retained_input_sha256"]:
        raise ValueError("retained observation file changed")
    with np.load(source, allow_pickle=False) as public:
        y, positions, physical_times = public["Y"], public["s"], public["t"]
    query = np.asarray(entry["query_position"])
    if not np.allclose(np.diff(physical_times), np.diff(physical_times).mean(), atol=1e-12, rtol=0):
        raise ValueError("uniform sampling required by this prototype")
    settings = settings_for(entry, y.shape[1]) if entry["kind"] == "wave" else GENERAL_SETTINGS
    cutoff = settings.n_train + settings.n_validation
    spatial = NodalSpatialModel.from_positions(positions)
    # Kernel extension allows extrapolation; P1 is unavailable outside the
    # retained hull.  The clustered case has no interior sensor to exclude.
    if entry["query_inside_retained_convex_hull"]:
        spatial.evaluate(np.ones((len(positions), 1)), query)
    noise = wave.noise_from_training_high_band(y[:, :settings.n_train])
    fitted = fit_factor_ssa(y[:, :settings.n_train], y[:, settings.n_train:cutoff], settings,
                           sampling_model=spatial, noise_estimate=noise, discovery_module=cover)
    global_fit = fitted["clock_models"][-1].fit
    if global_fit is None:
        raise ValueError("fresh retained-sensor discovery produced no phase-factor model")
    acquired = np.asarray(fitted["audit"]["final_lattice"]["peak_labels"], dtype=int)
    products = np.asarray(fitted["audit"]["final_lattice"]["products"], dtype=int)
    kernel_fit = fit_spatial_rkhs_factor(y[:, :settings.n_train], positions,
        np.arange(settings.n_train), global_fit.frequencies, global_fit.labels,
        global_fit.selected["ridge_scale"], acquired_labels=acquired, phase_products=products,
        sampling_model=spatial, settings=SPATIAL_SETTINGS)
    baseline = np.linalg.solve(spatial.mass_sqrt.T, global_fit.coefficients.T).T
    difference = float(np.max(abs(baseline - kernel_fit["baseline_coefficients_sensor"])))
    scale = max(1., float(np.max(abs(baseline))))
    if difference > 1e-7 * scale:
        raise ArithmeticError("spatial lambda0 fit does not reproduce the frozen temporal regression")
    candidates = {}
    if entry["query_inside_retained_convex_hull"]:
        p1_query = spatial.evaluate(baseline.T, query).T
        candidates["p1"] = (baseline, p1_query, {"spatial_method": "complex coefficient P1"})
    for name, key in (("kernel_zero_ridge", "zero_spatial_ridge_model"),
                      ("unconstrained", "unconstrained_model"), ("guarded", "guarded_model")):
        model = kernel_fit.get(key)
        if model is not None:
            candidates[name] = (model.coefficients_sensor, model.query_coefficients(query), model.metadata())
    if "kernel_zero_ridge" not in candidates:
        raise ValueError("CV-selected zero-spatial-ridge kernel comparator missing")
    parameter_path = ROOT / "parameters" / (entry["run"] + ".npz")
    np.savez_compressed(parameter_path, fit_labels=global_fit.labels,
        fit_frequencies=global_fit.frequencies, baseline_coefficients_sensor=baseline,
        acquired_labels=acquired, phase_products=products, spatial_mass_sqrt=spatial.mass_sqrt,
        physical_mass=spatial.physical_mass, query=query,
        retained_sensor_indices=np.asarray(entry["retained_sensor_indices"]))
    prediction_arrays, observers = {}, {}
    cache = []
    for name, (coefficients, query_coefficients, metadata) in candidates.items():
        shared = next((item for item in cache if np.array_equal(item[0], coefficients)
                      and np.array_equal(item[1], query_coefficients)), None)
        if shared is None:
            arrays, observer = encode_candidate(name, coefficients, query_coefficients,
                                               global_fit, spatial, y, settings)
            cache.append((coefficients, query_coefficients, arrays, observer, name))
        else:
            arrays, observer = shared[2], {**shared[3], "name": name, "identical_candidate_reused": shared[4]}
        prediction_arrays.update({name + "__" + key: value for key, value in arrays.items()})
        observers[name] = {**observer, "spatial_model": metadata}
    output_path = ROOT / "projected" / (entry["run"] + ".npz")
    np.savez_compressed(output_path, **prediction_arrays)
    return {"case": entry["case"], "run": entry["run"], "kind": entry["kind"],
        "run_status": "fitted_and_frozen", "settings": asdict(settings),
        "held_sensor_index": entry["held_sensor_index"], "retained_sensor_count": len(positions),
        "query_inside_retained_convex_hull": entry["query_inside_retained_convex_hull"],
        "p1_comparator_scope": "available" if entry["query_inside_retained_convex_hull"] else "unavailable_outside_hull; kernel extrapolation reported separately",
        "learned_phase_dimension": len(global_fit.frequencies),
        "fresh_retained_sensor_temporal_discovery": True, "inference": fitted["audit"],
        "noise_estimate_train_only": noise, "spatial_fit": kernel_fit["metadata"],
        "baseline_conditioning": kernel_fit["baseline_conditioning"],
        "lambda0_maximum_coefficient_difference_from_original_global_fit": difference,
        "inner_sensor_cv_scope": "conditional on dictionary learned from all outer-retained sensors",
        "outer_sensor_signal_used_for_any_learning_or_phase_encoding": False,
        "observers": observers, "future_start": cutoff, "future_count": y.shape[1] - cutoff,
        "frozen_parameters": str(parameter_path), "frozen_parameters_sha256": digest(parameter_path),
        "frozen_output": str(output_path), "frozen_output_sha256": digest(output_path),
        "elapsed_seconds": time.perf_counter() - started}


def nrmse(prediction, reference):
    return float(np.linalg.norm(prediction - reference) / max(np.linalg.norm(reference), 1e-30))


def score(row, entry):
    if digest(row["frozen_output"]) != row["frozen_output_sha256"]:
        raise ValueError("frozen prediction changed before scoring")
    with np.load(entry["evaluation"], allow_pickle=False) as truth:
        clean = truth["clean_Y"] if entry["kind"] == "wave" else truth["clean"]
    target = clean[:, row["future_start"]:row["future_start"] + row["future_count"]]
    keep = np.asarray(entry["retained_sensor_indices"])
    held = entry["held_sensor_index"]
    with np.load(row["frozen_parameters"]) as parameters:
        mass = parameters["physical_mass"]
    with np.load(row["frozen_output"]) as frozen:
        scores = {name: {"held_sensor_nrmse": nrmse(frozen[name + "__query_prediction"], target[[held]]),
                         "retained_field": wave.field_score(frozen[name + "__retained_prediction"], target[keep], mass)}
                  for name in row["observers"]}
    row.update(run_status="finished", future_field=scores)


def main():
    if OUTPUT.exists():
        raise FileExistsError("preserve the completed or running spatial pilot; use a separately declared protocol")
    for directory in (ROOT / "inputs", ROOT / "parameters", ROOT / "projected"):
        directory.mkdir(parents=True, exist_ok=True)
    fingerprints = hashes()
    manifest = public_manifest()
    rows = []
    payload = {"status": "running", "created_at": datetime.now(timezone.utc).isoformat(),
        "manifest": manifest, "rows": rows, "source_fingerprints": fingerprints,
        "spatial_settings": asdict(SPATIAL_SETTINGS), "projection_settings": asdict(PROJECTION),
        "evaluation_files_opened": False, "truth_used_for_between_case_adaptation": False,
        "full_sensor_temporal_fit_reused": False, "novelty_claimed": False,
        "case_scope": "3 original dimensions plus6 general fields, one geometry-only outer sensor per case",
        "temporal_parameter_units": "radians per sample; raw physical times only verify uniform sampling",
        "claim_scope": "finite-dictionary learned-template spatial regularization pilot; no physical inverse certificate"}
    save(payload)
    for entry in manifest:
        print("starting fresh retained-sensor fit " + entry["run"], flush=True)
        try:
            row = fit_and_freeze(entry)
        except Exception as error:
            row = {"case": entry["case"], "run": entry["run"], "kind": entry["kind"],
                "run_status": "exception", "exception": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc()}
        rows.append(row)
        save(payload)
        print(json.dumps({"run": row["run"], "status": row["run_status"],
                          "seconds": row.get("elapsed_seconds"), "exception": row.get("exception")}), flush=True)
    if hashes() != fingerprints:
        raise ValueError("inference source changed during the frozen nine-case pilot")
    payload.update(status="all_inference_attempts_finished_before_truth",
                   inference_finished_at=datetime.now(timezone.utc).isoformat())
    save(payload)
    for row, entry in zip(rows, manifest):
        if row["run_status"] == "fitted_and_frozen":
            score(row, entry)
    payload.update(status="finished", evaluation_files_opened=True,
                   scoring_finished_at=datetime.now(timezone.utc).isoformat())
    save(payload)
    print(json.dumps({"status": "finished", "successful_cases": sum(row["run_status"] == "finished" for row in rows),
                      "failed_cases": sum(row["run_status"] == "exception" for row in rows)}), flush=True)


if __name__ == "__main__":
    main()
