"""Fixed 18-run law-blind generality audit of the frozen phase-factor method.

Six common scalar fields vary physical phase rank, support, travelling spatial
phase, sensor geometry and visible weak energy. Three iid noise realizations
are fixed in advance. The inference function opens only the Y,s,t files.
All field forecasts/readouts and three predeclared capacity MSSA forecasts are
saved before the evaluation files are first read. This is a boundary audit,
not a calibrated finite-sample rank guarantee or a spatial interpolation test.
"""

from __future__ import annotations

from project_paths import RESULTS_DIR

from dataclasses import asdict
import hashlib
from itertools import combinations
import json
from math import gcd
from pathlib import Path
import time
import traceback

import numpy as np
from sympy import Matrix
from sympy.matrices.normalforms import hermite_normal_form

from final_phase_factor_assay import fixed_settings
from iterative_ssa import Settings, fit_iterative_ssa, predict_field
from nodal_spatial import NodalSpatialModel
from run_actual_wave_iterative import fingerprints as core_fingerprints, noise_from_training_high_band, serializable
from run_mssa_capacity_check import fit_capacity_mssa
from ssa_character import aligned_character_rows, apply_field_character


HERE = Path(__file__).resolve().parent
ROOT = RESULTS_DIR / 'general_robustness'
OUTPUT = RESULTS_DIR / 'general_robustness_suite.json'
TRAIN, VALIDATION, FUTURE = 2400, 1200, 1200
NOISE_SEEDS = (202610081, 202610082, 202610083)
NOISE_RATIO = .08
ALPHA = np.asarray([np.sqrt(2)/11, np.sqrt(3)/7, np.sqrt(5)/5])
ORIGIN = np.asarray([.31, 1.07, 1.83])
CASES = (
    {"name": "r1_travelling_infinite", "rank": 1, "q": 8, "layout": "spread", "kind": "single_poisson"},
    {"name": "r2_travelling_weak08", "rank": 2, "q": 12, "layout": "spread", "kind": "weak08"},
    {"name": "r2_clustered_boundary_weak02", "rank": 2, "q": 8, "layout": "weak_cluster", "kind": "weak02_cluster"},
    {"name": "r2_primitive_no_fundamentals", "rank": 2, "q": 8, "layout": "spread", "kind": "primitive_finite"},
    {"name": "r3_travelling_infinite_weak", "rank": 3, "q": 12, "layout": "spread", "kind": "three_phases"},
    {"name": "r3_random_smooth_no_fundamentals", "rank": 3, "q": 12, "layout": "shifted", "kind": "random_finite"},
)
BASELINE_CASES = frozenset((CASES[2]["name"], CASES[4]["name"], CASES[5]["name"]))
SETTINGS = fixed_settings(Settings(n_train=TRAIN, n_validation=VALIDATION, n_future=FUTURE,
    lag_window=128, frontend_mode="dynamic", dynamic_ranks=(4, 8, 16),
    joint_frequency_mode="harmonic_nuisance", phase_metric_mode="field",
    temporal_gate_mode="iid_reference", k_candidates=(16, 32)))


def hashes():
    result = core_fingerprints()
    for name in ("run_general_robustness_suite.py", "final_phase_factor_assay.py", "run_mssa_capacity_check.py"):
        result[name] = hashlib.sha256((HERE/name).read_bytes()).hexdigest()
    return result


def save(payload):
    tmp = OUTPUT.with_suffix(".tmp")
    tmp.write_text(json.dumps(serializable(payload), ensure_ascii=False, indent=2, allow_nan=False))
    tmp.replace(OUTPUT)


def signature(rows, physical_rank):
    rows = np.asarray(rows, dtype=int).reshape(-1, physical_rank)
    if not len(rows):
        return {"rank": 0, "basis": [], "index": None}
    integer = Matrix(rows)
    rank = int(integer.rank())
    hnf = hermite_normal_form(integer.T)
    index = None
    if rank == physical_rank:
        index = 0
        for selected in combinations(rows.tolist(), physical_rank):
            index = gcd(index, abs(int(Matrix(selected).det())))
    return {"rank": rank, "basis": [[int(x) for x in row] for row in hnf.T.tolist()], "index": index}


def sensor_positions(spec, index):
    rng = np.random.default_rng(71100+index)
    if spec["layout"] == "weak_cluster":
        return np.column_stack((rng.uniform(.015, .12, spec["q"]), rng.uniform(.1, .9, spec["q"])))
    if spec["layout"] == "shifted":
        return np.column_stack((rng.uniform(.32, .93, spec["q"]), rng.uniform(.18, .82, spec["q"])))
    corners = np.asarray([[.04, .04], [.96, .04], [.04, .96], [.96, .96]])
    return np.vstack((corners, rng.uniform(.08, .92, (spec["q"]-4, 2))))


def terms_for_generator(spec):
    kind = spec["kind"]
    if kind == "single_poisson":
        return [("poisson", (1,), 1., .8)]
    if kind in ("weak08", "weak02_cluster"):
        return [("poisson", (1, 0), 1., .75), ("cos", (0, 1), .08 if kind == "weak08" else .02, None)]
    if kind == "primitive_finite":
        return [("cos", mode, amplitude, None) for mode, amplitude in
                (((2, 0), 1.), ((0, 3), .8), ((1, 1), .3), ((1, -1), .25))]
    if kind == "three_phases":
        return [("poisson", (1, 0, 0), 1., .7), ("cos", (0, 1, 0), .35, None),
                ("cos", (0, 0, 1), .08, None), ("cos", (1, -1, 1), .12, None)]
    if kind == "random_finite":
        initial = [("cos", mode, amplitude, None) for mode, amplitude in
                   (((1, 1, 0), 1.), ((0, 1, 1), .75), ((1, 0, 1), .45), ((1, 1, 1), .22))]
        rng = np.random.default_rng(91913)
        modes = {term[1] for term in initial}
        while len(initial) < 12:
            mode = rng.integers(-3, 4, 3)
            if not np.any(mode) or np.sum(abs(mode)) > 4 or np.sum(abs(mode)) == 1:
                continue
            if next(value for value in mode if value) < 0:
                mode = -mode
            value = tuple(map(int, mode))
            if value in modes:
                continue
            modes.add(value)
            initial.append(("cos", value, float(.15*rng.uniform(.7, 1.3)/(1+np.sum(abs(mode)))), None))
        return initial
    raise ValueError(kind)


def generate_cache():
    """Generator produces separate input/evaluation files before any inference."""
    times = np.arange(TRAIN+VALIDATION+FUTURE)
    manifest = []
    for index, spec in enumerate(CASES):
        positions = sensor_positions(spec, index)
        rank = spec["rank"]
        theta = ORIGIN[:rank, None] + ALPHA[:rank, None]*times[None, :]
        clean = np.zeros((spec["q"], len(times)))
        catalog_modes, catalog_coefficients, catalog_groups = [], [], []
        group_modes = []
        for group, (family, mode, amplitude, rho) in enumerate(terms_for_generator(spec)):
            mode = np.asarray(mode, dtype=int)
            group_modes.append(mode.tolist())
            x, y = positions.T
            profile = .85+.20*np.cos(np.pi*(x+.21*group))+.10*np.sin(np.pi*(y+.37*group))
            if spec["kind"] == "weak02_cluster" and group == 1:
                profile = .12+.88*np.sin(np.pi*x)**2
            spatial_phase = 2*np.pi*((1.2+.31*group)*x+(.7-.19*group)*y
                +.18*np.sin(2*np.pi*x)*np.cos(np.pi*y))+.41*group
            angles = np.einsum("i,it->t", mode, theta, optimize=False)[None, :]-spatial_phase[:, None]
            if family == "poisson":
                value = ((1-rho*rho)/(1-2*rho*np.cos(angles)+rho*rho)-1)/np.sqrt(2*rho*rho/(1-rho*rho))
                clean += amplitude*profile[:, None]*value
                for harmonic in range(1, 65):
                    catalog_modes.append((harmonic*mode).tolist())
                    catalog_coefficients.append(amplitude*profile*np.sqrt((1-rho*rho)/2)*rho**(harmonic-1)
                                                *np.exp(-1j*harmonic*spatial_phase))
                    catalog_groups.append(group)
            else:
                clean += amplitude*profile[:, None]*np.cos(angles)
                catalog_modes.append(mode.tolist())
                catalog_coefficients.append(.5*amplitude*profile*np.exp(-1j*spatial_phase))
                catalog_groups.append(group)
        sigma = NOISE_RATIO*np.sqrt(np.mean(clean[:, :TRAIN]**2))
        for seed in NOISE_SEEDS:
            run = spec["name"]+"_seed"+str(seed)
            input_path, evaluation_path = ROOT/"inputs"/(run+".npz"), ROOT/"evaluation"/(run+".npz")
            y = clean+sigma*np.random.default_rng(seed).standard_normal(clean.shape)
            np.savez_compressed(input_path, Y=y, s=positions, t=times)
            np.savez_compressed(evaluation_path, clean=clean, theta=theta, alpha=ALPHA[:rank],
                sigma=np.asarray(sigma), rank=np.asarray(rank), group_modes=np.asarray(group_modes),
                catalog_modes=np.asarray(catalog_modes), catalog_coefficients=np.asarray(catalog_coefficients),
                catalog_groups=np.asarray(catalog_groups))
            manifest.append({"run": run, "case": spec["name"], "seed": seed,
                             "input": str(input_path), "evaluation": str(evaluation_path),
                             "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest()})
    return manifest


def fit_public_inputs(entry):
    """Inference has access only to the three public array keys."""
    started = time.perf_counter()
    with np.load(entry["input"]) as public:
        if set(public.files) != {"Y", "s", "t"}:
            raise ValueError("unexpected input key")
        observed, positions, times = public["Y"], public["s"], public["t"]
    spatial = NodalSpatialModel.from_positions(positions)
    noise = noise_from_training_high_band(observed[:, :TRAIN])
    fitted = fit_iterative_ssa(observed[:, :TRAIN], observed[:, TRAIN:TRAIN+VALIDATION], SETTINGS,
                               sampling_model=spatial, noise_estimate=noise)
    prediction = predict_field(fitted["clock_models"][-1], times[-FUTURE:], positions, spatial)
    mass_future = spatial.to_mass_coordinates(observed[:, -FUTURE:])
    train_phase, future_phase = [], []
    for index, front in enumerate(fitted["frozen_frontends"]):
        prefix = fitted["clock_models"][index].predict(times[-FUTURE:])
        readout = apply_field_character(front, mass_future, prefix)
        train_phase.append(np.asarray(front["phase"]))
        future_phase.append(np.asarray(readout["phase"]))
    train_phase = aligned_character_rows(train_phase)
    future_phase = aligned_character_rows(future_phase)
    arrays = {"prediction": prediction, "train_phase": train_phase, "future_phase": future_phase,
              "physical_mass": spatial.physical_mass}
    baseline = None
    if entry["case"] in BASELINE_CASES and entry["seed"] == NOISE_SEEDS[0]:
        baseline = fit_capacity_mssa(observed[:, :TRAIN], observed[:, TRAIN:TRAIN+VALIDATION], FUTURE, entry["case"])
        arrays["capacity_mssa_prediction"] = baseline["prediction"]
    if not all(np.isfinite(values).all() for values in arrays.values()):
        raise FloatingPointError("nonfinite frozen output")
    frozen_path = ROOT/"frozen"/(entry["run"]+".npz")
    np.savez_compressed(frozen_path, **arrays)
    return {"run": entry["run"], "case": entry["case"], "seed": entry["seed"], "run_status": "fitted_and_frozen",
            "inference": fitted["audit"], "input_sha256": entry["input_sha256"],
            "frozen_arrays": str(frozen_path), "frozen_arrays_sha256": hashlib.sha256(frozen_path.read_bytes()).hexdigest(),
            "all_frozen_arrays_finite": True, "sensor_count": len(positions),
            "noise_estimate_train_only": noise,
            "capacity_mssa": ({key: value for key, value in baseline.items() if key not in
                ("prediction", "validation_prediction", "training_reconstruction", "descending_covariance_eigenvalues")}
                if baseline is not None else None),
            "elapsed_seconds": time.perf_counter()-started}


def score_frozen_row(row, entry):
    with np.load(entry["evaluation"]) as evaluation:
        clean, theta = evaluation["clean"], evaluation["theta"]
        alpha, sigma = evaluation["alpha"], float(evaluation["sigma"])
        physical_rank = int(evaluation["rank"])
        group_modes = evaluation["group_modes"]
        catalog = evaluation["catalog_modes"]
        coefficients = evaluation["catalog_coefficients"]
        groups = evaluation["catalog_groups"]
    with np.load(row["frozen_arrays"]) as output:
        prediction, mass = output["prediction"], output["physical_mass"]
        train_phase, future_phase = output["train_phase"], output["future_phase"]
        capacity = output["capacity_mssa_prediction"] if "capacity_mssa_prediction" in output.files else None
    error = prediction-clean[:, -FUTURE:]
    clean_future = clean[:, -FUTURE:]
    field_score = {"future_sensor_nrmse": float(np.linalg.norm(error)/np.linalg.norm(clean_future)),
        "future_mass_nrmse": float(np.sqrt(np.einsum("it,ij,jt->", error, mass, error, optimize=False)
            /np.einsum("it,ij,jt->", clean_future, mass, clean_future, optimize=False))),
        "evaluation_scope": "untouched future times at the same observed sensors; not new spatial points"}
    if capacity is not None:
        field_score["capacity_MSSA_L512_future_sensor_nrmse"] = float(np.linalg.norm(capacity-clean_future)/np.linalg.norm(clean_future))
    energies = 2*np.real(np.einsum("mi,ij,mj->m", np.conj(coefficients), mass, coefficients, optimize=False))
    raw_energies = 2*np.sum(abs(coefficients)**2, axis=1)
    catalog_frequency = np.angle(np.exp(1j*np.einsum("ij,j->i", catalog, alpha, optimize=False)))
    tolerance = .25*2*np.pi/TRAIN
    labels, matches = [], []
    for frequency in row["inference"]["acquired_frequencies"]:
        near = []
        for index, target in enumerate(catalog_frequency):
            for sign in (1, -1):
                gap = float(abs(np.angle(np.exp(1j*(frequency-sign*target)))))
                if gap <= tolerance:
                    near.append({"label": (sign*catalog[index]).tolist(), "gap": gap,
                                 "catalog_mass_energy": float(energies[index])})
        unique = {tuple(item["label"]) for item in near}
        label = next(iter(unique)) if len(unique) == 1 else None
        labels.append(label)
        matches.append({"frequency": frequency, "matches": near,
                        "status": "unique" if len(unique) == 1 else "ambiguous" if unique else "unmatched_bounded_catalog"})
    expected = signature(group_modes, physical_rank)
    lattice = row["inference"].get("final_lattice")
    module = {"status": "unscored_ambiguous_or_unmatched_labels", "frequency_matching": matches,
              "truth_signature": expected, "matching_tolerance": tolerance,
              "catalog_scope": "exact finite support, or first64 Poisson harmonics; ambiguity is not scored as failure"}
    if lattice is not None and labels and all(label is not None for label in labels):
        physical_labels = np.asarray(labels, dtype=int)
        acquired_signature = signature(physical_labels, physical_rank)
        products = np.asarray(lattice["products"], dtype=int)
        physical_basis = np.einsum("ij,jk->ik", products, physical_labels, optimize=False)
        inferred_signature = signature(physical_basis, physical_rank)
        relation_valid = bool(np.array_equal(np.asarray(lattice["peak_labels"], dtype=int) @ physical_basis, physical_labels))
        complete = inferred_signature["basis"] == expected["basis"]
        module.update(status="scored_unique_bounded_catalog", acquired_signature=acquired_signature,
            inferred_signature=inferred_signature, relation_valid=relation_valid,
            observable_module_complete=complete,
            inferred_rank_correct=row["inference"]["r_candidate_conditional"]==expected["rank"],
            correct_module_rank_relations=bool(complete and relation_valid
                and row["inference"]["r_candidate_conditional"]==expected["rank"]),
            phase_basis_in_physical_coordinates=physical_basis.tolist())
        lost = np.asarray([signature(np.vstack((physical_basis, mode)), physical_rank)["basis"]
                           != inferred_signature["basis"] for mode in catalog])
        module["oracle_factor_missing_visible_character_energy_fraction"] = float(np.sum(energies[lost])/np.sum(energies))
        module["oracle_factor_missing_visible_character_RMS_fraction"] = float(np.sqrt(np.sum(energies[lost])/np.sum(energies)))
        module["missing_energy_is_posthoc_factor_expressivity_score_not_detection_certificate"] = True
        count_train, count_future = train_phase.shape[1], future_phase.shape[1]
        mapped_train = np.einsum("ij,jt->it", products, train_phase, optimize=False)
        mapped_future = np.einsum("ij,jt->it", products, future_phase, optimize=False)
        truth_train = np.einsum("ij,jt->it", physical_basis, theta[:, :count_train], optimize=False)
        truth_future = np.einsum("ij,jt->it", physical_basis, theta[:, TRAIN+VALIDATION:TRAIN+VALIDATION+count_future], optimize=False)
        origin = np.angle(np.mean(np.exp(1j*(mapped_train-truth_train)), axis=1))
        module["future_train_origin_phase_mae_rad"] = float(np.mean(abs(np.angle(np.exp(1j*(mapped_future-origin[:, None]-truth_future))))))
    group_diagnostics = []
    noise_energy = sigma*sigma*float(np.trace(mass))
    for group, mode in enumerate(group_modes):
        indices = groups==group
        mass_energy, raw_energy = float(np.sum(energies[indices])), float(np.sum(raw_energies[indices]))
        group_diagnostics.append({"generator_mode_truth_only": mode.tolist(), "mass_Haar_energy": mass_energy,
            "visible_group_RMS_over_iid_field_noise_RMS": float(np.sqrt(mass_energy/noise_energy)),
            "ideal_known_frequency_coherent_train_SNR_diagnostic": float(np.sqrt(TRAIN*raw_energy/(sigma*sigma)))})
    row.update(run_status="finished", future_field=field_score, module_truth_only=module,
        visible_energy_noise_diagnostics_truth_only=group_diagnostics,
        noise_sigma_truth_only=sigma, physical_phase_rank_truth_only=physical_rank,
        phase_completeness_certified=False, finite_catalog_mass_energy=float(np.sum(energies)))


def main():
    if OUTPUT.exists():
        raise FileExistsError("preserve the existing audit and use a new explicit destination")
    for directory in (ROOT/"inputs", ROOT/"evaluation", ROOT/"frozen"):
        directory.mkdir(parents=True, exist_ok=True)
    initial_hashes = hashes()
    started = time.perf_counter()
    manifest = generate_cache()
    rows = []
    payload = {"status": "running", "rows": rows, "predeclared_cases": list(CASES),
        "noise_seeds": list(NOISE_SEEDS), "noise_ratio": NOISE_RATIO,
        "settings": asdict(SETTINGS), "source_fingerprints": initial_hashes, "manifest": manifest,
        "capacity_baseline_rule": {"cases": sorted(BASELINE_CASES), "only_noise_seed": NOISE_SEEDS[0], "window": 512,
            "rank_selection": "fixed grid through384, observed validation sensor MSE"},
        "input_scope": "Y,s,t only; geometry-only nodal model; no source equation/spatial profile/true rank/frequency",
        "truth_opening_scope": "after all18 inferences and predeclared baselines have finished and forecasts are frozen",
        "settings_change_from_truth_between_runs": False,
        "evaluation_files_opened": False, "exceptions_preserved_without_case_substitution": True}
    save(payload)
    for entry in manifest:
        print("starting generality "+entry["run"], flush=True)
        try:
            row = fit_public_inputs(entry)
        except Exception as error:
            row = {"run": entry["run"], "case": entry["case"], "seed": entry["seed"], "run_status": "exception",
                   "exception": str(error), "traceback": traceback.format_exc()}
        rows.append(row)
        save(payload)
        print(json.dumps({"run": entry["run"], "status": row.get("inference", {}).get("status", row["run_status"]),
                          "rank": row.get("inference", {}).get("r_candidate_conditional"),
                          "elapsed": row.get("elapsed_seconds"), "exception": row.get("exception")}), flush=True)
    payload["all_fits_finished_before_truth_opening"] = True
    payload["source_fingerprints_unchanged_before_scoring"] = hashes()==initial_hashes
    save(payload)
    for row, entry in zip(rows, manifest):
        if row["run_status"] == "exception":
            continue
        score_frozen_row(row, entry)
    payload.update(status="finished", invocation_finished=True, evaluation_files_opened=True,
        source_fingerprints_unchanged=hashes()==initial_hashes, total_elapsed_seconds=time.perf_counter()-started)
    finished = [row for row in rows if row["run_status"]=="finished"]
    scored = [row for row in finished if row["module_truth_only"]["status"]=="scored_unique_bounded_catalog"]
    payload["summary"] = {"requested_runs": 18, "finished_runs": len(finished), "exceptions": len(rows)-len(finished),
        "module_scored_unique_catalog_runs": len(scored),
        "correct_observed_modules_ranks_relations_among_scored": sum(row["module_truth_only"]["correct_module_rank_relations"] for row in scored),
        "median_future_sensor_nrmse": float(np.median([row["future_field"]["future_sensor_nrmse"] for row in finished])) if finished else None,
        "inferred_rank_correct_all_finished": sum(row["inference"]["r_candidate_conditional"]==row["physical_phase_rank_truth_only"] for row in finished)}
    save(payload)
    print(json.dumps(payload["summary"]), flush=True)


if __name__ == "__main__":
    main()
