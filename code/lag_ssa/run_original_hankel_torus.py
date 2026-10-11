"""Original 1D/2D/3D tone fields and five controls: trailing Hankel encoding.

The global factor/cover discovery rule and the L128 projector stay fixed.
Strong learned field characters initialize each observer from that exact raw
window. All fits, seed parameters, phases and comparison outputs are saved
before evaluation truth is opened. Generator use is confined to producing
the five synthetic public observation files and subsequent scoring.
"""

from project_paths import RESULTS_DIR

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np

import cover_selection_ssa as cover
import final_phase_factor_assay as final
from factor_ssa import fit_factor_ssa
from hankel_torus_projection import HankelTorusProjector, ProjectionSettings
from iterative_ssa import Settings
from nodal_spatial import NodalSpatialModel
from raw_hankel_seed import initialize_from_raw_strong_characters
import run_actual_wave_iterative as wave
import run_actual_wave_revised as revised


HERE = Path(__file__).resolve().parent
ROOT = RESULTS_DIR / 'original_hankel_torus'
OUTPUT = ROOT.with_suffix(".json")
WAVES = ("wave_1d_traveling", "wave_2d_uniform", "wave_3d_uniform")
PROJECTION = ProjectionSettings()


def hashes():
    result = final.fingerprints()
    for name in ("cover_selection_ssa.py", "factor_ssa.py", "hankel_torus_projection.py",
                 "raw_hankel_seed.py", "run_original_hankel_torus.py"):
        result[name] = hashlib.sha256((HERE/name).read_bytes()).hexdigest()
    return result


def save(payload):
    wave.save(payload, OUTPUT)


def public_manifest():
    """Produce observations; fitting receives the saved Y,s,t only."""
    manifest = []
    for name in WAVES:
        path = wave.DATA/"inputs"/(name+".npz")
        manifest.append({"case": name, "kind": "wave", "input": str(path),
            "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    for name in revised.FIXTURES:
        y, positions, times, _ = revised.fixture_inputs(name)
        path = ROOT/"inputs"/(name+".npz")
        np.savez_compressed(path, Y=y, s=positions, t=times)
        manifest.append({"case": name, "kind": "fixture", "input": str(path),
            "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "observation_generator_scope": "same cached fixture production rule and noise seed54000"})
    return manifest


def settings_for(entry, samples):
    if entry["kind"] == "wave":
        return final.fixed_settings(final.ORIGINAL_SETTINGS(samples))
    return final.fixed_settings(Settings(n_train=3200, n_validation=1600, n_future=1600,
        lag_window=128, frontend_mode="dynamic", dynamic_ranks=(4, 8, 16),
        joint_frequency_mode="harmonic_nuisance", phase_metric_mode="field",
        temporal_gate_mode="iid_reference", k_candidates=(16, 32)))


def evaluate_field(labels, coefficients, phase):
    angles = np.einsum("ij,jt->it", labels, phase, optimize=False)
    features = np.column_stack((np.ones(phase.shape[1]), np.cos(angles.T), np.sin(angles.T)))
    return np.einsum("nj,jq->qn", features, coefficients, optimize=False)


def fit_and_freeze(entry):
    started = time.perf_counter()
    path = Path(entry["input"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != entry["input_sha256"]:
        raise ValueError("public input changed")
    with np.load(path) as public:
        if set(public.files) != {"Y", "s", "t"}:
            raise ValueError("unexpected public input keys")
        y, positions, times = public["Y"], public["s"], public["t"]
    if not np.allclose(np.diff(times), np.diff(times).mean(), atol=1e-12, rtol=0):
        raise ValueError("this fixed observer requires uniformly spaced observations")
    settings = settings_for(entry, y.shape[1])
    cutoff = settings.n_train+settings.n_validation
    spatial = NodalSpatialModel.from_positions(positions)
    noise = wave.noise_from_training_high_band(y[:, :settings.n_train])
    fitted = fit_factor_ssa(y[:, :settings.n_train], y[:, settings.n_train:cutoff], settings,
        sampling_model=spatial, noise_estimate=noise, discovery_module=cover)
    model = fitted["clock_models"][-1]
    if model.fit is None:
        raise ValueError("no inferred phase-factor model")
    fit = model.fit
    projector = HankelTorusProjector(fit.labels, fit.coefficients, fit.frequencies, PROJECTION)
    mass = spatial.coordinates(y)
    # Each future target is the END of one measured L128 window. This includes
    # 127 previous samples for the first held-out target; no samples beyond
    # that target enter either observer. In 3D validation has120 samples, so
    # the first7 windows also include at most7 final training samples.
    window_start = cutoff-(PROJECTION.window-1)
    if window_start < 0:
        raise ValueError("observations shorter than the declared context")
    observed = mass[:, window_start:]
    seed = initialize_from_raw_strong_characters(mass[:, :settings.n_train], observed, projector)
    count = y.shape[1]-cutoff
    if seed["phase"].shape[1] != count:
        raise ValueError("raw seed and trailing observation starts do not align")
    mapping = np.linalg.solve(spatial.mass_sqrt.T, spatial.design.T).T
    parameters = {"fit_labels": fit.labels, "fit_coefficients": fit.coefficients,
        "fit_frequencies": fit.frequencies, "spatial_mass_sqrt": spatial.mass_sqrt,
        "mass_to_sensor": mapping, "physical_mass": spatial.physical_mass,
        "phase_products": np.asarray(fitted["audit"]["final_lattice"]["products"], dtype=int),
        "acquired_frequencies": fitted["acquired_frequencies"],
        "raw_seed_phase_at_window_start": seed["phase"],
        "raw_training_seed_phase_at_window_start": seed["training_phase"],
        "training_only_seed_origins": np.asarray(seed["metadata"]["training_origins"]),
        "strong_character_labels": np.asarray(seed["metadata"]["strong_character_labels"], dtype=int)}
    parameter_path = ROOT/"parameters"/(entry["case"]+".npz")
    np.savez_compressed(parameter_path, **parameters)
    check = projector.compression_check(observed, seed["phase"][:, 0]+.07*np.arange(1, projector.rank+1))
    if check["cost_absolute_difference"] > 1e-8 or check["analytic_gradient_max_difference_from_finite_difference"] > 1e-4:
        raise ArithmeticError("compressed window cost or gradient mismatch")
    projected = projector.project(observed, seed["phase"])
    free = projector.free_linear_projection(observed, count)
    advance = fit.frequencies[:, None]*(PROJECTION.window-1)
    end_phase = np.angle(np.exp(1j*(projected["phase"]+advance)))
    end_seed = np.angle(np.exp(1j*(seed["phase"]+advance)))
    endpoint_mass = evaluate_field(fit.labels, fit.coefficients, end_phase)
    seed_mass = evaluate_field(fit.labels, fit.coefficients, end_seed)
    free_endpoint = fit.coefficients[0, :, None]+np.einsum("dq,dn->qn",
        projector.bank[:, :, PROJECTION.window-1], free["free_coefficients"], optimize=False)
    clock_mass = model.predict(np.arange(cutoff, y.shape[1]))
    arrays = {"hankel_phase": end_phase, "hankel_phase_at_window_start": projected["phase"],
        "raw_seed_phase": end_seed, "raw_seed_phase_at_window_start": seed["phase"],
        "hankel_prediction": np.einsum("ij,jt->it", mapping, endpoint_mass, optimize=False),
        "free_linear_prediction": np.einsum("ij,jt->it", mapping, free_endpoint, optimize=False),
        "raw_seed_prediction": np.einsum("ij,jt->it", mapping, seed_mass, optimize=False),
        "clock_prediction": np.einsum("ij,jt->it", mapping, clock_mass, optimize=False),
        "free_linear_coefficients": free["free_coefficients"],
        "time_end_indices": np.arange(cutoff, y.shape[1]),
        "time_start_indices": np.arange(window_start, y.shape[1]-PROJECTION.window+1),
        "hankel_window_cost": np.asarray([item["final_cost"] for item in projected["diagnostics"]]),
        "raw_seed_window_cost": np.asarray([item["seed_cost"] for item in projected["diagnostics"]]),
        "selected_deck_start": np.asarray([item["selected_start"] for item in projected["diagnostics"]])}
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise FloatingPointError("nonfinite frozen field or phase output")
    output_path = ROOT/"projected"/(entry["case"]+".npz")
    np.savez_compressed(output_path, **arrays)
    diagnostics = projected["diagnostics"]
    return {"case": entry["case"], "kind": entry["kind"], "run_status": "fitted_and_frozen",
        "settings": asdict(settings), "projection_settings": asdict(PROJECTION),
        "inference": fitted["audit"], "noise_estimate_train_only": noise,
        "raw_seed_training_only": seed["metadata"], "multistart": projected["multistart"],
        "frozen_parameters": str(parameter_path),
        "frozen_parameters_sha256": hashlib.sha256(parameter_path.read_bytes()).hexdigest(),
        "frozen_output": str(output_path), "frozen_output_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "public_input_sha256": entry["input_sha256"], "time_end_start": cutoff,
        "time_window_start": window_start, "count": count,
        "compression_check": check, "free_linear": free["metadata"],
        "optimizer": {"cost_exceeds_seed_count": sum(not item["cost_not_above_product_seed"] for item in diagnostics),
            "selected_nonzero_deck_start_count": sum(item["selected_start"] != 0 for item in diagnostics),
            "gradient_converged_count": sum(item["converged_gradient"] for item in diagnostics),
            "final_gradient_max": max(item["final_gradient_max"] for item in diagnostics),
            "iterations_median": float(np.median([item["iterations"] for item in diagnostics])),
            "iterations_max": max(item["iterations"] for item in diagnostics),
            "global_optimum_certified": False},
        "seed_and_objective_and_linear_context_identical": True,
        "future_absolute_time_used_for_phase_seed": False,
        "future_prefix_clock_subtraction_used": False,
        "relative_window_phase_advance": PROJECTION.window-1,
        "context_training_overlap_at_first_target": max(0, settings.n_train-window_start),
        "evaluation_scope": "field at held-out window ends from trailing128 raw observations; first127 windows overlap earlier context",
        "elapsed_seconds": time.perf_counter()-started}


def score(row, entry):
    if entry["kind"] == "wave":
        with np.load(wave.DATA/"evaluation"/(entry["case"]+".npz")) as truth:
            clean = truth["clean_Y"]
            source_frequencies = truth["source_frequencies_hz"]
        # This qualification uses generator metadata only after all outputs
        # are frozen. The sampled first original tone has period50 samples.
        row["sampling_qualification_truth_only"] = {
            "source_frequencies_hz": source_frequencies.tolist(), "time_step_seconds": .01,
            "first_source_rotation": "3/50 turns per sample",
            "sampled_orbit_closure": "50 circles; not a dense continuous two-torus",
            "full_continuous_torus_identified_from_this_sampled_orbit": False}
    else:
        from iterative_fields import generate_field
        with np.load(entry["input"]) as public:
            clean = generate_field(entry["case"], public["t"], public["s"], n_train=3200)["clean"]
    target = clean[:, row["time_end_start"]:row["time_end_start"]+row["count"]]
    with np.load(row["frozen_parameters"]) as parameters:
        mass = parameters["physical_mass"]
    with np.load(row["frozen_output"]) as output:
        errors = {name: wave.field_score(output[name+"_prediction"], target, mass)
                  for name in ("hankel", "free_linear", "raw_seed", "clock")}
    row.update(run_status="finished", heldout_field=errors,
        hankel_over_free_linear_error_ratio=errors["hankel"]["sensor_nrmse"]/errors["free_linear"]["sensor_nrmse"],
        hankel_over_raw_seed_error_ratio=errors["hankel"]["sensor_nrmse"]/errors["raw_seed"]["sensor_nrmse"])


def main():
    if OUTPUT.exists():
        raise FileExistsError("preserve the existing original-case projection audit")
    for directory in (ROOT/"inputs", ROOT/"parameters", ROOT/"projected"):
        directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    initial = hashes()
    manifest = public_manifest()
    rows = []
    payload = {"status": "running", "manifest": manifest, "rows": rows,
        "source_fingerprints": initial, "projection_settings": asdict(PROJECTION),
        "case_scope": "original1Dtraveling,2Duniform,3Duniform fields and same5 controls",
        "settings_rule": "frozen final_phase_factor_assay.fixed_settings; optional cover discovery; one global refit",
        "truth_opening_scope": "evaluation truth after all8 model parameters, raw seeds, nonlinear and free-linear outputs frozen",
        "evaluation_files_opened": False, "between_case_adaptation_from_truth": False,
        "input_scope": "public Y,s,t only; geometry-only nodal spatial model",
        "raw_seed_and_hankel_and_linear_window": 128,
        "no_future_clock_or_prefix_subtraction_in_phase_initialization": True,
        "optimization_settings_changed": False, "novelty_claimed": False}
    save(payload)
    for entry in manifest:
        print("starting original trailing Hankel "+entry["case"], flush=True)
        try:
            row = fit_and_freeze(entry)
        except Exception as error:
            row = {"case": entry["case"], "kind": entry["kind"], "run_status": "exception",
                "exception": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()}
        rows.append(row)
        save(payload)
        print(json.dumps({"case": entry["case"], "status": row["run_status"],
            "rank": row.get("inference", {}).get("r_candidate_conditional"),
            "count": row.get("count"), "exception": row.get("exception")}), flush=True)
    payload.update(all_outputs_frozen_before_truth_open=True,
        source_fingerprints_unchanged_before_scoring=hashes() == initial)
    save(payload)
    for row, entry in zip(rows, manifest):
        if row["run_status"] == "exception":
            continue
        score(row, entry)
        print(json.dumps({"case": row["case"], "sensor_nrmse":
            {name: result["sensor_nrmse"] for name, result in row["heldout_field"].items()}}), flush=True)
    finished = [row for row in rows if row["run_status"] == "finished"]
    payload.update(status="finished", invocation_finished=True, evaluation_files_opened=True,
        source_fingerprints_unchanged=hashes() == initial, total_elapsed_seconds=time.perf_counter()-started,
        summary={"requested_runs": len(manifest), "finished_runs": len(finished),
            "exceptions": len(rows)-len(finished),
            "hankel_improved_vs_free_linear": sum(row["hankel_over_free_linear_error_ratio"] < 1 for row in finished),
            "hankel_improved_vs_raw_seed": sum(row["hankel_over_raw_seed_error_ratio"] < 1 for row in finished),
            "hankel_sensor_nrmse_median": float(np.median([row["heldout_field"]["hankel"]["sensor_nrmse"] for row in finished])) if finished else None})
    save(payload)
    print(json.dumps(payload["summary"]), flush=True)


if __name__ == "__main__":
    main()
