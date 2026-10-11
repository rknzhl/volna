"""Predeclared fresh6 audit of the complete, frozen phase-field pipeline.

Discovery runs anew from Y,s,t; no cached factor parameters are loaded.
Existing compact, dictionary and matched-MSSA inference functions are reused
unchanged with process-local output directories. The unpruned comparator
uses the SAME discovered factor/group. All6 attempts and outputs finish and
freeze before this runner first opens any separate evaluation archive.
"""

from project_paths import RESULTS_DIR

from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
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
import run_actual_wave_iterative as wave
import run_compact_group_hankel as compact
import compact_group_mssa_comparison as comparison
import run_phase_preserving_dictionary as sparse
from run_general_robustness_suite import signature
from generate_fresh_phase_fields import generate, PROTOCOL, TRAIN, VALIDATION, FUTURE


HERE = Path(__file__).resolve().parent
ROOT = RESULTS_DIR / 'fresh_phase_pipeline'
OUTPUT = ROOT.with_suffix(".json")
SETTINGS = fixed_settings(Settings(n_train=TRAIN, n_validation=VALIDATION, n_future=FUTURE,
    lag_window=128, frontend_mode="dynamic", dynamic_ranks=(4, 8, 16),
    joint_frequency_mode="harmonic_nuisance", phase_metric_mode="field",
    temporal_gate_mode="iid_reference", k_candidates=(16, 32)))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprints():
    result = wave.fingerprints()
    result.update({name:digest(HERE/name) for name in ("run_fresh_phase_pipeline.py",
        "generate_fresh_phase_fields.py", "factor_ssa.py", "cover_selection_ssa.py",
        "final_phase_factor_assay.py", "phase_factor_regression.py", "nodal_spatial.py",
        "run_compact_group_hankel.py", "compact_group_hankel.py", "observed_phase_group.py",
        "hankel_torus_projection.py", "compact_group_mssa_comparison.py", "mssa_column_encoding.py",
        "run_phase_preserving_dictionary.py", "phase_preserving_dictionary.py",
        "group_witness_conditioning.py", "run_original_hankel_torus.py", "run_general_robustness_suite.py")})
    return result


def save(payload):
    temporary = OUTPUT.with_suffix(".tmp")
    temporary.write_text(json.dumps(wave.serializable(payload), indent=2, allow_nan=False)+"\n")
    temporary.replace(OUTPUT)


@contextmanager
def output_scope(module, root):
    """Only the current process output ROOT changes; all source files frozen."""
    previous = module.ROOT
    module.ROOT = Path(root)
    try:
        yield
    finally:
        module.ROOT = previous


def fit_discovery(public_entry):
    """Input-only API: no generation spec, rank, frequency or catalogue."""
    started = time.perf_counter()
    if digest(public_entry["input"]) != public_entry["input_sha256"]:
        raise ValueError("changed public observations")
    with np.load(public_entry["input"], allow_pickle=False) as public:
        if set(public.files) != {"Y", "s", "t"}:
            raise ValueError("exactly Y,s,t required")
        y, s, t = public["Y"], public["s"], public["t"]
    if y.shape[1] != TRAIN+VALIDATION+FUTURE or not np.array_equal(t, np.arange(len(t))):
        raise ValueError("predeclared common uniform sample counts required")
    spatial = NodalSpatialModel.from_positions(s)
    noise = wave.noise_from_training_high_band(y[:, :TRAIN])
    fitted = fit_factor_ssa(y[:, :TRAIN], y[:, TRAIN:TRAIN+VALIDATION], SETTINGS,
        sampling_model=spatial, noise_estimate=noise, discovery_module=cover)
    model = fitted["clock_models"][-1]
    if model.fit is None or fitted["audit"]["final_lattice"] is None:
        raise ValueError("no observation-supported phase-factor field returned")
    fit = model.fit
    lattice = fitted["audit"]["final_lattice"]
    arrays = dict(fit_labels=fit.labels, fit_coefficients=fit.coefficients,
        fit_frequencies=fit.frequencies, spatial_mass_sqrt=spatial.mass_sqrt,
        mass_to_sensor=np.linalg.solve(spatial.mass_sqrt.T, spatial.design.T).T,
        physical_mass=spatial.physical_mass, acquired_labels=np.asarray(lattice["peak_labels"], dtype=int),
        phase_products=np.asarray(lattice["products"], dtype=int),
        acquired_frequencies=np.asarray(fitted["acquired_frequencies"]),
        selected_temporal_ridge=np.asarray(fit.selected["ridge_scale"]))
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise FloatingPointError("nonfinite fresh discovery parameters")
    path = ROOT/"discovery_parameters"/(public_entry["run"]+".npz")
    np.savez_compressed(path, **arrays)
    return dict(run=public_entry["run"], run_status="fitted_and_frozen", inference=fitted["audit"],
        source_model=str(path), source_model_sha256=digest(path), selected_temporal_ridge=float(fit.selected["ridge_scale"]),
        noise_estimate_train_only=noise, sensor_count=len(s), spatial_dimension=s.shape[1],
        fitted_phase_count=len(fit.frequencies), original_dictionary_count=len(fit.labels),
        observation_validated_acquired=dict(acquired_labels=lattice["peak_labels"], phase_products=lattice["products"]),
        fit_inputs="Y_train,Y_validation,known sensor coordinates only; TRAIN-only noise estimate",
        cached_old_model_used=False, evaluation_file_opened=False,
        elapsed_seconds=time.perf_counter()-started)


def fit_public_entry(public_entry):
    """Complete inference: receives no evaluator path or generator parameter."""
    started = time.perf_counter()
    discovery = fit_discovery(public_entry)
    entry = {**public_entry, "case": public_entry["run"], "kind":"fresh", "previously_examined_benchmark":False,
             "source_model":discovery["source_model"], "source_model_sha256":discovery["source_model_sha256"],
             "temporal_ridge":discovery["selected_temporal_ridge"]}
    with output_scope(compact, ROOT/"compact"):
        group = compact.fit_and_freeze(entry)
    with output_scope(comparison, ROOT/"matched_mssa"):
        classical = comparison.fit_and_freeze(entry, group)
    group_metadata = {key:group[key] for key in ("selected_model", "group_proposal", "frozen_parameters",
        "frozen_parameters_sha256", "frozen_output", "frozen_output_sha256")}
    comparator_metadata = {key:classical[key] for key in ("noise_estimate_train_only", "frozen_output", "frozen_output_sha256")}
    with output_scope(sparse, ROOT/"dictionary"):
        dictionary = sparse.fit_and_freeze(entry, group_metadata, comparator_metadata,
                                          discovery["observation_validated_acquired"])
    return dict(run=public_entry["run"], case=public_entry["run"], run_status="fitted_and_frozen",
        discovery=discovery, compact=group, matched_mssa=classical, dictionary=dictionary,
        all_inference_inputs="Y,s,t only; independent evaluator path excluded from this API",
        unpruned_baseline_same_discovery_and_selected_group=True,
        elapsed_seconds=time.perf_counter()-started)


def module_evaluation(row, evaluation, mass):
    """Post-all-freeze catalogue matching; never fed back to discovery/selection."""
    inference = row["discovery"]["inference"]
    physical_rank = int(evaluation["rank"])
    catalog = evaluation["catalog_modes"]
    catalog_coefficient = evaluation["catalog_coefficients"]
    physical_alpha = evaluation["alpha"]
    catalog_rotation = np.angle(np.exp(1j*np.einsum("ij,j->i", catalog, physical_alpha, optimize=False)))
    energy = 2*np.real(np.einsum("mi,ij,mj->m", np.conj(catalog_coefficient), mass, catalog_coefficient, optimize=False))
    tolerance = .25*2*np.pi/TRAIN
    acquired = [];matches = []
    for frequency in inference["acquired_frequencies"]:
        near = []
        for index, rotation in enumerate(catalog_rotation):
            for sign in (1, -1):
                gap = float(abs(np.angle(np.exp(1j*(frequency-sign*rotation)))))
                if gap <= tolerance:
                    near.append(dict(label=(sign*catalog[index]).tolist(), gap=gap,
                                     catalog_mass_energy=float(energy[index])))
        unique = {tuple(item["label"]) for item in near}
        acquired.append(next(iter(unique)) if len(unique) == 1 else None)
        matches.append(dict(frequency=frequency, matches=near,
            status="unique" if len(unique) == 1 else "ambiguous" if unique else "unmatched"))
    target = signature(evaluation["group_modes"], physical_rank)
    result = dict(status="unscored_ambiguous_or_unmatched_bounded_catalog", frequency_matching=matches,
        matching_tolerance=tolerance, truth_signature=target, inferred_rank=inference["r_candidate_conditional"],
        physical_rank=physical_rank, bounded_catalog_scope="all finite modes or first64 Poisson harmonics",
        inference_received_true_catalog=False)
    if acquired and all(item is not None for item in acquired):
        lattice = inference["final_lattice"]
        physical_labels = np.asarray(acquired, dtype=int)
        physical_basis = np.einsum("ij,jk->ik", np.asarray(lattice["products"], dtype=int), physical_labels, optimize=False)
        implied = np.einsum("ij,jk->ik", np.asarray(lattice["peak_labels"], dtype=int), physical_basis, optimize=False)
        recovered_signature = signature(physical_basis, physical_rank)
        complete = recovered_signature["basis"] == target["basis"]
        correct = bool(complete and np.array_equal(implied, physical_labels)
                       and inference["r_candidate_conditional"] == target["rank"])
        result.update(status="scored_unique_bounded_catalog", physical_basis=physical_basis.tolist(),
            acquired_signature=signature(physical_labels, physical_rank), recovered_signature=recovered_signature,
            relations_valid=bool(np.array_equal(implied, physical_labels)), character_basis_complete=correct,
            character_basis_completeness_is_group_correctness=False)
        missing = np.asarray([signature(np.vstack((physical_basis, mode)), physical_rank)["basis"] != recovered_signature["basis"] for mode in catalog])
        result["missing_visible_character_mass_energy_fraction_truth_only"] = float(np.sum(energy[missing])/np.sum(energy))
    noise_energy = float(evaluation["sigma"])**2*float(np.trace(mass))
    groups = evaluation["catalog_groups"]
    result["visible_group_rms_over_noise_rms_truth_only"] = [
        dict(group_mode=mode.tolist(), ratio=float(np.sqrt(np.sum(energy[groups == index])/noise_energy)))
        for index, mode in enumerate(evaluation["group_modes"])]
    return result


def score(row, entry):
    dictionary = row["dictionary"]
    if digest(dictionary["frozen_output"]) != dictionary["frozen_output_sha256"]:
        raise ValueError("fresh output changed before evaluation")
    if digest(entry["evaluation"]) != entry["evaluation_sha256"]:
        raise ValueError("fresh evaluation archive changed")
    with np.load(entry["evaluation"], allow_pickle=False) as archive:
        evaluation = {key:archive[key] for key in archive.files}
    with np.load(dictionary["frozen_parameters"], allow_pickle=False) as archive:
        parameters = {key:archive[key] for key in archive.files}
    with np.load(dictionary["frozen_output"], allow_pickle=False) as archive:
        arrays = {key:archive[key] for key in archive.files}
    clean = evaluation["clean"][:, TRAIN+VALIDATION:]
    mass = parameters["physical_mass"]
    endpoint = {name:wave.field_score(arrays[key], clean, mass) for name, key in
                (("pruned", "prediction_endpoint"), ("unpruned", "old_prediction_endpoint"), ("MSSA", "mssa_prediction_endpoint"))}
    diagonal = {name:wave.field_score(arrays[key], clean, mass) for name, key in
                (("pruned", "prediction_diagonal"), ("unpruned", "old_prediction_diagonal"), ("MSSA", "mssa_prediction_diagonal"))}
    module = module_evaluation(row, evaluation, mass)
    phase_scores = dict(status="unscored_without_unique_catalog_basis")
    group_correct = False
    if "physical_basis" in module:
        basis = np.asarray(module["physical_basis"], dtype=int)
        alpha = parameters["frequencies"]
        theta = evaluation["theta"]
        true_train = np.einsum("ij,jt->it", basis, theta[:, :TRAIN], optimize=False)
        true_future = np.einsum("ij,jt->it", basis, theta[:, TRAIN+VALIDATION:], optimize=False)
        origin = np.angle(np.mean(np.exp(1j*(alpha[:, None]*np.arange(TRAIN)-true_train)), axis=1))
        scores = {}
        for name, key in (("pruned", "phase_at_window_end"), ("unpruned", "old_phase_at_window_end")):
            error = np.angle(np.exp(1j*(arrays[key]-origin[:, None]-true_future)))
            scores[name] = dict(mae_rad=float(np.mean(abs(error))), rmse_rad=float(np.sqrt(np.mean(error**2))))
        phase_scores = dict(status="scored", recovered_factor_coordinates=scores,
            calibration="shared decoder clock gauge against TRAIN truth only; no future origin fit",
            future_truth_used_to_choose_basis_or_origin=False,
            incomplete_factor_not_physical_full_state=not module["character_basis_complete"])
        if module["character_basis_complete"] and basis.shape == (int(evaluation["rank"]),)*2:
            inverse = np.linalg.inv(basis)
            if not np.allclose(inverse, np.rint(inverse), atol=1e-10):
                raise ArithmeticError("complete physical character basis needs integer inverse")
            inverse = np.rint(inverse).astype(int)
            canonical_clock = np.einsum("ij,jt->it", inverse, alpha[:, None]*np.arange(TRAIN), optimize=False)
            canonical_origin = np.angle(np.mean(np.exp(1j*(canonical_clock-theta[:, :TRAIN])), axis=1))
            canonical = {}
            for name, key in (("pruned", "phase_at_window_end"), ("unpruned", "old_phase_at_window_end")):
                mapped = np.einsum("ij,jt->it", inverse, arrays[key], optimize=False)
                error = np.angle(np.exp(1j*(mapped-canonical_origin[:, None]-theta[:, TRAIN+VALIDATION:])))
                canonical[name] = dict(mae_rad=float(np.mean(abs(error))), rmse_rad=float(np.sqrt(np.mean(error**2))))
            phase_scores["canonical_physical_generator_coordinates"] = canonical
        relation = parameters["relation"]
        residue = (float(abs(np.angle(np.exp(1j*np.dot(relation, np.einsum("ij,j->i", basis, evaluation["alpha"], optimize=False))))))
                   if len(relation) else 0.)
        # All six intended generators have a full nonresonant physical torus;
        # any imposed nonzero relation is false, even if finite loss is small.
        group_correct = bool(module["character_basis_complete"] and not len(relation))
        module.update(imposed_group_relation=relation.tolist(), physical_rotation_relation_residue_rad_per_sample=residue,
            imposed_relation_physically_valid=not bool(len(relation)), physical_latent_group_correct=group_correct,
            exact_group_certified_from_observations=False,
            intended_generator_group="full nonresonant torus under predeclared algebraic rotation")
    else:
        module["physical_latent_group_correct"] = False
    row.update(run_status="finished", endpoint_field=endpoint, diagonal_field=diagonal,
        module_truth_only=module, phase_truth_only=phase_scores,
        full_physical_module_and_group_correct_truth_only=group_correct,
        noise_sigma_truth_only=float(evaluation["sigma"]),
        phase_rank_or_group_completeness_certified_by_inference=False)


def main():
    if OUTPUT.exists() or ROOT.exists():
        raise FileExistsError("preserve existing fresh6 inputs and fixed protocol results")
    initial = fingerprints();started = time.perf_counter()
    for directory in ("discovery_parameters", "compact/parameters", "compact/projected",
                      "matched_mssa/parameters", "matched_mssa/projected", "dictionary/parameters", "dictionary/projected"):
        (ROOT/directory).mkdir(parents=True, exist_ok=False)
    manifest = generate(ROOT)
    rows = []
    payload = dict(status="running", created_at=datetime.now(timezone.utc).isoformat(), manifest=manifest, rows=rows,
        source_fingerprints=initial, settings=asdict(SETTINGS), generation_protocol=PROTOCOL,
        compact_group_options=compact.GROUP_OPTIONS, projection_settings=asdict(compact.PROJECTION),
        protected_dictionary_thresholds=list(sparse.THRESHOLDS), conditioning_preservation_factor=sparse.CONDITIONING_FACTOR,
        mssa_rank_grid=list(comparison.RANK_GRID), evaluation_files_opened=False,
        complete_unknown_rank_discovery_from_fresh_public_inputs=True, old_cached_parameters_loaded=False,
        inference_input_scope="Y,s,t only; generator/evaluation paths removed before fit API",
        unpruned_comparator="same newly discovered model and selected compact group, before protected pruning",
        raw_window_context="same trailing128 future column block; D uses all its columns for both methods",
        selection_rules_changed_by_new_data=False, independent_from_previous_benchmarks=True,
        paired_weak_cases_not_statistically_independent=True, six_cases_not_a_statistical_guarantee=True,
        conditional_validation_reuse_disclaimer="discovery, group and dictionary reuse VAL; MSSA ranks use plug-in estimated-noise SURE; no calibrated risk guarantee",
        exact_group_or_completeness_certified=False, novelty_claimed=False)
    save(payload)
    for entry in manifest:
        print("starting complete fresh phase pipeline "+entry["run"], flush=True)
        # Evaluator path/rank/alpha/catalogue deliberately absent from fit API.
        public_entry = {key:entry[key] for key in ("run", "input", "input_sha256", "n_train", "n_validation", "n_future")}
        try:
            row = fit_public_entry(public_entry)
        except Exception as error:
            row = dict(run=entry["run"], case=entry["case"], run_status="exception",
                       exception=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        rows.append(row);save(payload)
        print(json.dumps(dict(run=row["run"], status=row["run_status"],
            rank=row.get("discovery", {}).get("fitted_phase_count"),
            group=row.get("compact", {}).get("selected_model"),
            threshold=row.get("dictionary", {}).get("selected_threshold"),
            count=row.get("dictionary", {}).get("selected_count"),
            seconds=row.get("elapsed_seconds"), exception=row.get("exception"))), flush=True)
    if fingerprints() != initial:
        raise ValueError("frozen pipeline source changed during the fresh run")
    if any(digest(entry["input"]) != entry["input_sha256"] for entry in manifest):
        raise ValueError("public fresh observations changed")
    payload.update(all_six_attempts_and_outputs_frozen_before_truth_open=True,
        inference_finished_at=datetime.now(timezone.utc).isoformat(), source_fingerprints_unchanged_before_scoring=True)
    save(payload)
    for row, entry in zip(rows, manifest):
        if row["run_status"] == "fitted_and_frozen":
            score(row, entry)
            print(json.dumps(dict(run=row["run"], module_correct=row["full_physical_module_and_group_correct_truth_only"],
                endpoint={key:value["sensor_nrmse"] for key,value in row["endpoint_field"].items()},
                diagonal={key:value["sensor_nrmse"] for key,value in row["diagonal_field"].items()})), flush=True)
    finished = [row for row in rows if row["run_status"] == "finished"]
    payload.update(status="finished", evaluation_files_opened=True, source_fingerprints_unchanged=fingerprints() == initial,
        total_elapsed_seconds=time.perf_counter()-started,
        summary=dict(requested_runs=6, finished_runs=len(finished), exceptions=len(rows)-len(finished),
            correct_full_physical_module_and_group=sum(row["full_physical_module_and_group_correct_truth_only"] for row in finished),
            better_endpoint_than_MSSA=sum(row["endpoint_field"]["pruned"]["sensor_nrmse"] < row["endpoint_field"]["MSSA"]["sensor_nrmse"] for row in finished),
            better_diagonal_than_MSSA=sum(row["diagonal_field"]["pruned"]["sensor_nrmse"] < row["diagonal_field"]["MSSA"]["sensor_nrmse"] for row in finished),
            better_endpoint_than_same_discovery_unpruned=sum(row["endpoint_field"]["pruned"]["sensor_nrmse"] < row["endpoint_field"]["unpruned"]["sensor_nrmse"] for row in finished)))
    save(payload);print(json.dumps(payload["summary"]), flush=True)


if __name__ == "__main__":
    main()
