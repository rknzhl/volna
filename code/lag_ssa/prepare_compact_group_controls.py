"""Generate and law-blindly fit three fixed compact-group controls.

Generator metadata is written to separate evaluation archives. The fitting
function reads only Y,s,t and uses the same frozen general-suite settings.
It saves field-model parameters only: no future phase projection, forecast,
truth scoring, or case-dependent adaptation is performed in this script.
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
from factor_ssa import fit_factor_ssa
from final_phase_factor_assay import fixed_settings
from iterative_ssa import Settings
from nodal_spatial import NodalSpatialModel
from run_actual_wave_iterative import noise_from_training_high_band, serializable


HERE = Path(__file__).resolve().parent
ROOT = RESULTS_DIR / 'compact_group_controls'
OUTPUT = ROOT.with_suffix(".json")
TRAIN, VALIDATION, FUTURE = 2400, 1200, 1200
NOISE_SEED = 202610091
POSITION_SEED = 71191
NOISE_RATIO = .08
SETTINGS = fixed_settings(Settings(n_train=TRAIN, n_validation=VALIDATION, n_future=FUTURE,
    lag_window=128, frontend_mode="dynamic", dynamic_ranks=(4, 8, 16),
    joint_frequency_mode="harmonic_nuisance", phase_metric_mode="field",
    temporal_gate_mode="iid_reference", k_candidates=(16, 32)))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprints():
    names = ("prepare_compact_group_controls.py", "factor_ssa.py", "cover_selection_ssa.py",
        "final_phase_factor_assay.py", "iterative_ssa.py", "phase_factor_regression.py",
        "nodal_spatial.py", "run_actual_wave_iterative.py", "ssa_dynamic_character.py",
        "ssa_character.py", "phase_readout_refinement.py", "frequency_fit.py",
        "nuisance_frequency_fit.py", "phase_lattice.py")
    return {name: digest(HERE/name) for name in names}


def save(payload):
    tmp = OUTPUT.with_suffix(".tmp")
    tmp.write_text(json.dumps(serializable(payload), ensure_ascii=False, indent=2, allow_nan=False))
    tmp.replace(OUTPUT)


def generate_inputs():
    """Write public input and separate truth; return paths and hashes only."""
    rng = np.random.default_rng(POSITION_SEED)
    corners = np.asarray([[.04, .04], [.96, .04], [.04, .96], [.96, .96]])
    positions = np.vstack((corners, rng.uniform(.08, .92, (4, 2))))
    times = np.arange(TRAIN+VALIDATION+FUTURE)
    generators = (
        ("mixed_rotation", np.asarray([.271*np.sqrt(2), -.271*np.sqrt(2)+2*np.pi/7])),
        ("resolvable_detuning", np.asarray([2*np.pi*3/50+1e-5, np.sqrt(3)/7])),
        ("unresolvable_detuning", np.asarray([2*np.pi*3/50+1e-7, np.sqrt(3)/7])),
    )
    origins = np.asarray([.31, 1.07])
    modes = np.asarray([[1, 0], [0, 1], [1, 1], [2, 0]])
    amplitudes = np.asarray([1., .4, .2, .15])
    x, y = positions.T
    profiles = np.asarray([.85+.20*np.cos(np.pi*(x+.21*group))
        +.10*np.sin(np.pi*(y+.37*group)) for group in range(4)])
    spatial_phases = np.asarray([2*np.pi*((1.2+.31*group)*x+(.7-.19*group)*y
        +.18*np.sin(2*np.pi*x)*np.cos(np.pi*y))+.41*group for group in range(4)])
    common_noise = np.random.default_rng(NOISE_SEED).standard_normal((len(positions), len(times)))
    manifest = []
    for name, alpha in generators:
        phases = origins[:, None]+alpha[:, None]*times[None, :]
        clean = np.zeros_like(common_noise)
        for mode, amplitude, profile, spatial_phase in zip(modes, amplitudes, profiles, spatial_phases):
            angle = np.einsum("i,it->t", mode, phases, optimize=False)
            clean += amplitude*profile[:, None]*np.cos(angle[None, :]-spatial_phase[:, None])
        sigma = NOISE_RATIO*float(np.sqrt(np.mean(clean[:, :TRAIN]**2)))
        run = name+"_seed"+str(NOISE_SEED)
        input_path = ROOT/"inputs"/(run+".npz")
        evaluation_path = ROOT/"evaluation"/(run+".npz")
        parameter_path = ROOT/"parameters"/(run+".npz")
        np.savez_compressed(input_path, Y=clean+sigma*common_noise, s=positions, t=times)
        np.savez_compressed(evaluation_path, clean=clean, theta=phases, alpha=alpha,
                            sigma=np.asarray(sigma))
        manifest.append({"run": run, "case": name, "seed": NOISE_SEED,
            "input": str(input_path), "evaluation": str(evaluation_path),
            "parameters": str(parameter_path), "input_sha256": digest(input_path)})
    return manifest


def fit_public_inputs(entry):
    """This API neither reads evaluation files nor receives generator values."""
    started = time.perf_counter()
    if digest(entry["input"]) != entry["input_sha256"]:
        raise ValueError("public input changed before fitting")
    with np.load(entry["input"]) as public:
        if set(public.files) != {"Y", "s", "t"}:
            raise ValueError("unexpected public input keys")
        values, positions, times = public["Y"], public["s"], public["t"]
    if values.shape[1] != TRAIN+VALIDATION+FUTURE or not np.array_equal(times, np.arange(len(times))):
        raise ValueError("the predeclared common sample counts/times are required")
    spatial = NodalSpatialModel.from_positions(positions)
    noise = noise_from_training_high_band(values[:, :TRAIN])
    fitted = fit_factor_ssa(values[:, :TRAIN], values[:, TRAIN:TRAIN+VALIDATION], SETTINGS,
        sampling_model=spatial, noise_estimate=noise, discovery_module=cover)
    model = fitted["clock_models"][-1]
    if model.fit is None or fitted["audit"]["final_lattice"] is None:
        raise ValueError("no observed-supported phase-factor field model was returned")
    fit = model.fit
    parameters = {"fit_labels": fit.labels, "fit_coefficients": fit.coefficients,
        "fit_frequencies": fit.frequencies, "spatial_mass_sqrt": spatial.mass_sqrt,
        "mass_to_sensor": np.linalg.solve(spatial.mass_sqrt.T, spatial.design.T).T,
        "physical_mass": spatial.physical_mass,
        "phase_products": np.asarray(fitted["audit"]["final_lattice"]["products"], dtype=int),
        "acquired_frequencies": np.asarray(fitted["acquired_frequencies"]),
        "selected_temporal_ridge": np.asarray(fit.selected["ridge_scale"])}
    if not all(np.isfinite(value).all() for value in parameters.values()):
        raise FloatingPointError("nonfinite fitted parameter array")
    path = Path(entry["parameters"])
    np.savez_compressed(path, **parameters)
    return {"run": entry["run"], "case": entry["case"], "seed": entry["seed"],
        "run_status": "fitted_and_frozen", "inference": fitted["audit"],
        "noise_estimate_train_only": noise,
        "selected_temporal_ridge": float(fit.selected["ridge_scale"]),
        "input": entry["input"], "public_input_sha256": digest(entry["input"]),
        "source_model": str(path), "source_model_sha256": digest(path),
        "frozen_parameters": str(path), "frozen_parameters_sha256": digest(path),
        "parameter_keys": list(parameters), "all_frozen_parameters_finite": True,
        "fit_inputs": "Y_train, Y_validation, known sensor coordinates; noise estimated from TRAIN only",
        "generator_alpha_rank_theta_or_equation_given_to_fit": False,
        "future_observations_used_in_fit": False, "future_projection_or_forecast_created": False,
        "evaluation_file_read": False, "elapsed_seconds": time.perf_counter()-started}


def main():
    if OUTPUT.exists() or ROOT.exists():
        raise FileExistsError("preserve any existing control inputs and fits")
    started = time.perf_counter()
    for directory in ("inputs", "evaluation", "parameters"):
        (ROOT/directory).mkdir(parents=True)
    source_hashes = fingerprints()
    manifest = generate_inputs()
    payload = {"status": "fitting", "manifest": manifest, "rows": [],
        "source_fingerprints": source_hashes, "settings": asdict(SETTINGS),
        "input_scope": "public NPZ contains exactly Y,s,t; separate evaluation files never read here",
        "control_generation": {"position_seed": POSITION_SEED, "noise_seed": NOISE_SEED,
            "noise_ratio_to_train_clean_sensor_RMS": NOISE_RATIO,
            "same_standard_normal_draw_across_three_controls": True,
            "sensor_count": 8, "spatial_dimension": 2,
            "field": "sum of four smooth spatially modulated cosines at theta1, theta2, theta1+theta2, 2*theta1",
            "construction_parameters_not_passed_to_fitting_API": True},
        "evaluation_files_opened": False,
        "future_predictions_or_phase_projections_created": False,
        "scoring_performed": False, "case_dependent_settings": False}
    save(payload)
    for entry in manifest:
        print("fitting public control "+entry["run"], flush=True)
        try:
            row = fit_public_inputs(entry)
        except Exception as error:
            row = {"run": entry["run"], "case": entry["case"], "seed": entry["seed"],
                "run_status": "exception", "exception": repr(error), "traceback": traceback.format_exc(),
                "evaluation_file_read": False}
        payload["rows"].append(row)
        save(payload)
        print(json.dumps({"run": entry["run"], "status": row["run_status"],
            "phase_rank": row.get("inference", {}).get("r_candidate_conditional"),
            "selected_temporal_ridge": row.get("selected_temporal_ridge"),
            "exception": row.get("exception")}), flush=True)
    payload["source_fingerprints_unchanged"] = fingerprints() == source_hashes
    payload["public_inputs_unchanged"] = all(digest(entry["input"])==entry["input_sha256"] for entry in manifest)
    payload["all_three_fixed_attempts_completed_before_any_truth_read"] = len(payload["rows"])==3
    payload["all_parameters_frozen"] = all(row["run_status"]=="fitted_and_frozen" for row in payload["rows"])
    payload["status"] = "prepared_and_fitted"
    payload["total_elapsed_seconds"] = time.perf_counter()-started
    save(payload)
    print(json.dumps({"status": payload["status"], "manifest": str(OUTPUT),
        "all_parameters_frozen": payload["all_parameters_frozen"],
        "evaluation_files_opened": payload["evaluation_files_opened"],
        "elapsed_seconds": payload["total_elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
