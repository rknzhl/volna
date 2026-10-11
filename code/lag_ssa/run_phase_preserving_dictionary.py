"""Fixed20-case, anchor-preserving decoder pruning; no discovery retuning.

All candidate coefficients come from raw TRAIN. Complementary masked-lag
VAL prediction chooses among thresholds(0,1,2,4) satisfying a TRAIN-only
conditioning guard. Every selected future output freezes before evaluation.
These are previously inspected benchmarks, not a fresh external test set.
"""

from project_paths import RESULTS_DIR

from datetime import datetime, timezone
from itertools import product
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np

from group_witness_conditioning import conditioning
from phase_preserving_dictionary import prepare_dictionary_candidates, THRESHOLDS
from run_compact_group_hankel import observer, lag_cv
from run_original_hankel_torus import evaluate_field
from compact_group_mssa_comparison import group_columns, diagonal_average
import run_actual_wave_iterative as wave


HERE = Path(__file__).resolve().parent
SOURCE = RESULTS_DIR / 'compact_group_hankel.json'
COMPARISON = RESULTS_DIR / 'compact_group_mssa_comparison.json'
ROOT = RESULTS_DIR / 'phase_preserving_dictionary'
OUTPUT = ROOT.with_suffix(".json")
WINDOW = 128
CONDITIONING_FACTOR = .5


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprints():
    return {name: digest(HERE/name) for name in ("run_phase_preserving_dictionary.py",
        "phase_preserving_dictionary.py", "group_witness_conditioning.py",
        "run_compact_group_hankel.py", "compact_group_hankel.py",
        "hankel_torus_projection.py", "compact_group_mssa_comparison.py",
        "run_original_hankel_torus.py", "run_actual_wave_iterative.py")}


def save(payload):
    temporary = OUTPUT.with_suffix(".tmp")
    temporary.write_text(json.dumps(wave.serializable(payload), indent=2,
                                   allow_nan=False)+"\n")
    temporary.replace(OUTPUT)


def filtered_acquired_metadata():
    """Filter source JSONs to inference A/B only; scores never enter fitting."""
    result = {};sources = {}
    for name in ("original_all_waves_robust_hankel.json", "general_robustness_factor_cover.json",
                 "compact_group_controls.json"):
        path = RESULTS_DIR/name
        sources[str(path)] = digest(path)
        payload = json.loads(path.read_text())
        for row in payload["rows"]:
            lattice = row["inference"]["final_lattice"]
            run = row.get("run", row["case"])
            result[run] = dict(acquired_labels=lattice["peak_labels"],
                phase_products=lattice["products"], source=str(path), source_sha256=sources[str(path)])
    return result, sources


def _conditioning(specification, acquired):
    try:
        _, info = conditioning(specification["labels"], specification["coefficients_mass"],
            specification["frequencies"], acquired["acquired_labels"], acquired["phase_products"],
            relation=specification.get("relation"), lag_count=WINDOW, rcond=1e-10)
        return info
    except ValueError as error:
        return dict(conditioning_status="unresolved", guard_sigma_available=False,
                    sigma0_numerical=0., reason=str(error))


def fit_and_freeze(entry, source_row, comparison_row, acquired):
    started = time.perf_counter()
    for path, expected in ((entry["input"], entry["input_sha256"]),
                           (source_row["frozen_parameters"], source_row["frozen_parameters_sha256"]),
                           (source_row["frozen_output"], source_row["frozen_output_sha256"]),
                           (comparison_row["frozen_output"], comparison_row["frozen_output_sha256"])):
        if digest(path) != expected:
            raise ValueError("changed frozen source: "+str(path))
    with np.load(entry["input"], allow_pickle=False) as public:
        if set(public.files) != {"Y", "s", "t"}:
            raise ValueError("public observations must contain only Y,s,t")
        y = public["Y"]
    with np.load(source_row["frozen_parameters"], allow_pickle=False) as archive:
        parameters = {key: archive[key] for key in archive.files}
    family = source_row["selected_model"]
    labels = parameters[family+"__labels"]
    alpha = parameters[family+"__frequencies"]
    coefficient = parameters[family+"__coefficients_mass"]
    relation = parameters["relation"] if family == "compact" else None
    alias = source_row["group_proposal"].get("refined_template", {}) if family != "continuous" else {}
    nyquist = np.asarray([row.tolist() in alias.get("nyquist_cosine_only_labels", []) for row in labels])
    n_train, n_validation = entry["n_train"], entry["n_validation"]
    cutoff = n_train+n_validation
    prepared = prepare_dictionary_candidates(y[:, :n_train], np.arange(n_train), alpha, labels,
        coefficient, entry["temporal_ridge"], parameters["spatial_mass_sqrt"],
        comparison_row["noise_estimate_train_only"]["sigma"], acquired["acquired_labels"],
        acquired["phase_products"], relation=relation,
        multiplicities=alias.get("quotient_multiplicities"), nyquist=nyquist)
    candidates = prepared["candidates"]
    baseline_condition = _conditioning(candidates[0], acquired)
    baseline_sigma = baseline_condition.get("sigma0_numerical", 0.)
    guard_resolved = bool(baseline_condition.get("guard_sigma_available", False) and baseline_sigma > 0)
    mass = np.einsum("ij,jt->it", parameters["spatial_mass_sqrt"], y, optimize=False)
    validation = mass[:, n_train:cutoff]
    validation_window = min(WINDOW, n_validation-2)
    records = [];cache = {}
    for index, specification in enumerate(candidates):
        key = tuple(specification["retained_indices"])
        if key in cache:
            prior = cache[key]
            record = {**prior, "threshold": specification["threshold"], "candidate_index": index,
                      "exact_dictionary_duplicate_of_threshold": prior["threshold"]}
            records.append(record);continue
        condition = baseline_condition if index == 0 else _conditioning(specification, acquired)
        guarded = bool(guard_resolved and condition.get("guard_sigma_available", False)
                       and condition.get("sigma0_numerical", 0.) >= CONDITIONING_FACTOR*baseline_sigma)
        eligible = bool(index == 0 or (specification["anchor_preserving_pruning_eligible"] and guarded))
        record = dict(candidate_index=index, threshold=specification["threshold"],
            retained_count=specification["retained_count"], retained_indices=specification["retained_indices"],
            character_lattice_index=specification["character_lattice_index"], conditioning=condition,
            conditioning_guard_passed=guarded, eligible_for_validation_selection=eligible,
            all_original_observed_anchors_preserved=specification["all_original_observed_anchors_preserved"],
            temporal_gram_condition=specification["temporal_gram_condition"],
            ridge_hessian_condition=specification["ridge_hessian_condition"])
        if eligible:
            record["validation_cv"] = lag_cv(specification, validation, validation_window)
        else:
            record["rejection"] = ("unresolved_baseline_conditioning" if not guard_resolved else
                "anchor_lattice_or_conditioning_guard_failed")
        cache[key] = record;records.append(record)
    eligible = [row for row in records if row["eligible_for_validation_selection"]]
    selected = min(eligible, key=lambda row: (row["validation_cv"]["mass_mse_per_field_snapshot"], row["threshold"]))
    chosen = candidates[selected["candidate_index"]]
    observed = mass[:, cutoff-WINDOW+1:]
    projected = observer(chosen, WINDOW).project(observed)
    start_phase = projected["phase"]
    endpoint_phase = np.angle(np.exp(1j*(start_phase+alpha[:, None]*(WINDOW-1))))
    endpoint_mass = evaluate_field(chosen["labels"], chosen["coefficients_mass"], endpoint_phase)
    endpoint = np.einsum("ij,jt->it", parameters["mass_to_sensor"], endpoint_mass, optimize=False)
    columns = group_columns(chosen["labels"], chosen["coefficients_mass"], alpha, start_phase, WINDOW)
    composition_defect = float(np.max(abs(columns[-1]-endpoint_mass)))
    averaged_mass, counts = diagonal_average(columns)
    diagonal = np.einsum("ij,jt->it", parameters["mass_to_sensor"], averaged_mass[:, WINDOW-1:], optimize=False)
    # Reuse the old observer and classical predictions as comparators, never as fit inputs.
    with np.load(comparison_row["frozen_output"], allow_pickle=False) as comparison:
        old_endpoint = comparison["group_endpoint"];old_diagonal = comparison["group_diagonal"]
        classical_endpoint = comparison["mssa_endpoint"];classical_diagonal = comparison["mssa_diagonal"]
        old_counts = comparison["diagonal_available_counts"]
        old_times = comparison["time_targets"]
    with np.load(source_row["frozen_output"], allow_pickle=False) as previous:
        old_phase = previous[family+"__phase_at_window_end"]
    if not np.array_equal(counts[WINDOW-1:], old_counts):
        raise ArithmeticError("new and frozen comparator D use different available columns")
    if not np.array_equal(old_times, np.arange(cutoff, cutoff+endpoint.shape[1])):
        raise ArithmeticError("new and frozen comparators target different timestamps")
    if any(value.shape != endpoint.shape for value in (diagonal, old_endpoint, old_diagonal,
                                                        classical_endpoint, classical_diagonal)):
        raise ArithmeticError("new and frozen comparator sensor/time arrays differ in shape")
    arrays = dict(phase_at_window_start=start_phase, phase_at_window_end=endpoint_phase,
        prediction_endpoint=endpoint, prediction_diagonal=diagonal,
        old_prediction_endpoint=old_endpoint, old_prediction_diagonal=old_diagonal,
        mssa_prediction_endpoint=classical_endpoint, mssa_prediction_diagonal=classical_diagonal,
        old_phase_at_window_end=old_phase, diagonal_available_counts=counts[WINDOW-1:],
        window_cost=np.asarray([row["final_cost"] for row in projected["diagnostics"]]))
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise FloatingPointError("nonfinite frozen dictionary-pilot output")
    output_path = ROOT/"projected"/(entry["run"]+".npz")
    parameter_path = ROOT/"parameters"/(entry["run"]+".npz")
    np.savez_compressed(output_path, **arrays)
    parameter_arrays = dict(labels=chosen["labels"], frequencies=alpha,
        coefficients_mass=chosen["coefficients_mass"], physical_mass=parameters["physical_mass"],
        mass_to_sensor=parameters["mass_to_sensor"], spatial_mass_sqrt=parameters["spatial_mass_sqrt"],
        relation=np.asarray([] if relation is None else relation),
        acquired_labels=np.asarray(acquired["acquired_labels"]), phase_products=np.asarray(acquired["phase_products"]))
    for number, specification in enumerate(candidates):
        parameter_arrays.update({str(number)+"__labels": specification["labels"],
            str(number)+"__coefficients_mass": specification["coefficients_mass"]})
    np.savez_compressed(parameter_path, **parameter_arrays)
    diagnostics = projected["diagnostics"]
    return dict(run=entry["run"], case=entry["case"], kind=entry["kind"], run_status="fitted_and_frozen",
        selected_group_family_frozen=family, phase_count=len(alpha), rotation_group_or_phase_rank_retuned=False,
        dictionary_preparation=prepared["metadata"], candidate_validation=records,
        baseline_conditioning=baseline_condition, baseline_conditioning_resolved=guard_resolved,
        selected_threshold=selected["threshold"], selected_count=chosen["retained_count"], original_count=len(labels),
        selected_validation_cv=selected["validation_cv"], conditioning_preservation_factor=CONDITIONING_FACTOR,
        noise_estimate_train_only=comparison_row["noise_estimate_train_only"],
        source_group_parameters=source_row["frozen_parameters"], source_group_parameters_sha256=source_row["frozen_parameters_sha256"],
        frozen_classical_and_old_group_comparison=comparison_row["frozen_output"],
        frozen_classical_and_old_group_comparison_sha256=comparison_row["frozen_output_sha256"],
        frozen_parameters=str(parameter_path), frozen_parameters_sha256=digest(parameter_path),
        frozen_output=str(output_path), frozen_output_sha256=digest(output_path),
        future_start=cutoff, future_count=endpoint.shape[1], same_window_and_column_block_as_old_comparison=True,
        phase_composition_mass_endpoint_max_absolute_difference=composition_defect,
        optimizer=dict(gradient_converged_count=sum(row["converged_gradient"] for row in diagnostics),
            window_count=len(diagnostics), final_gradient_max=max(row["final_gradient_max"] for row in diagnostics),
            multistart=projected.get("multistart", {}), global_optimum_certified=False),
        baseline0_endpoint_max_difference_from_old=float(np.max(abs(endpoint-old_endpoint))) if selected["threshold"] == 0 else None,
        baseline0_diagonal_max_difference_from_old=float(np.max(abs(diagonal-old_diagonal))) if selected["threshold"] == 0 else None,
        elapsed_seconds=time.perf_counter()-started)


def phase_score(row, entry, theta, parameters, physical_basis, complete):
    basis = np.asarray(physical_basis, dtype=int)
    alpha = parameters["frequencies"]
    n_train = entry["n_train"]
    train_target = np.einsum("ij,jt->it", basis, theta[:, :n_train], optimize=False)
    origin = np.angle(np.mean(np.exp(1j*(alpha[:, None]*np.arange(n_train)-train_target)), axis=1))
    target = np.einsum("ij,jt->it", basis, theta[:, row["future_start"]:row["future_start"]+row["future_count"]], optimize=False)
    with np.load(row["frozen_output"]) as arrays:
        scores = {}
        for name, key in (("new", "phase_at_window_end"), ("old", "old_phase_at_window_end")):
            error = np.angle(np.exp(1j*(arrays[key]-origin[:, None]-target)))
            scores[name] = dict(mae_rad=float(np.mean(abs(error))), rmse_rad=float(np.sqrt(np.mean(error**2))))
    return dict(physical_module_complete=bool(complete), phase_errors_in_recovered_factor_coordinates=scores,
        physical_basis_truth_only=basis.tolist(), origin_calibration="shared fixed decoder clock gauge against TRAIN truth only",
        future_truth_used_to_choose_basis_or_origin=False, known_incomplete_factor_claimed_complete=False)


def unique_eval_basis(frequencies, physical_alpha, n_train):
    """Post-freeze, bounded source-rotation matching, with explicit ambiguity."""
    rank = len(physical_alpha)
    catalog = np.asarray([row for row in product(range(-6, 7), repeat=rank)
                          if 0 < sum(abs(x) for x in row) <= 6], dtype=int)
    rotation = np.einsum("ij,j->i", catalog, physical_alpha, optimize=False)
    result = []
    for frequency in frequencies:
        gap = abs(np.angle(np.exp(1j*(rotation-frequency))))
        matches = np.flatnonzero(gap <= .25*2*np.pi/n_train)
        if len(matches) != 1:
            return None
        result.append(catalog[int(matches[0])])
    return np.asarray(result, dtype=int)


def score(row, entry):
    if digest(row["frozen_output"]) != row["frozen_output_sha256"]:
        raise ValueError("pilot output changed before scoring")
    with np.load(entry["evaluation"], allow_pickle=False) as evaluation:
        clean = evaluation["clean_Y"] if entry["kind"] == "wave" else evaluation["clean"]
        theta = evaluation["theta"] if "theta" in evaluation.files else None
        physical_alpha = evaluation["alpha"] if "alpha" in evaluation.files else None
        source_hz = evaluation["source_frequencies_hz"] if "source_frequencies_hz" in evaluation.files else None
    with np.load(row["frozen_parameters"]) as archive:
        parameters = {key: archive[key] for key in archive.files}
    target = clean[:, row["future_start"]:row["future_start"]+row["future_count"]]
    with np.load(row["frozen_output"]) as arrays:
        endpoint = {name: wave.field_score(arrays[key], target, parameters["physical_mass"])
            for name, key in (("new", "prediction_endpoint"), ("old", "old_prediction_endpoint"), ("MSSA", "mssa_prediction_endpoint"))}
        diagonal = {name: wave.field_score(arrays[key], target, parameters["physical_mass"])
            for name, key in (("new", "prediction_diagonal"), ("old", "old_prediction_diagonal"), ("MSSA", "mssa_prediction_diagonal"))}
    row.update(run_status="finished", endpoint_field=endpoint, diagonal_field=diagonal,
               pure_phase_class_control=entry["case"] != "wave_1d_source_noise")
    if entry["kind"] == "generic":
        evaluation_source = json.loads((RESULTS_DIR / 'general_robustness_factor_cover.json').read_text())
        source_row = next(item for item in evaluation_source["rows"] if item["run"] == entry["run"])
        metadata = source_row["module_truth_only"]
        basis = metadata.get("phase_basis_in_physical_coordinates")
        if basis is not None:
            row["phase_truth_only"] = phase_score(row, entry, theta, parameters, basis,
                metadata.get("correct_module_rank_relations", False))
    else:
        if entry["kind"] == "wave":
            with np.load(entry["input"]) as public:
                times = public["t"]
            theta = 2*np.pi*source_hz[:, None]*times[None, :]
            physical_alpha = 2*np.pi*source_hz*float(times[1]-times[0])
            row["sampling_qualification_truth_only"] = "original tonal sampled closure Z50 x S1; source-noise control outside pure phase-field class"
        basis = unique_eval_basis(parameters["frequencies"], physical_alpha, entry["n_train"])
        if basis is not None:
            row["phase_truth_only"] = phase_score(row, entry, theta, parameters, basis,
                abs(round(np.linalg.det(basis))) == 1 if basis.shape[0] == basis.shape[1] else False)
        else:
            row["phase_truth_only"] = dict(status="unscored_nonunique_bounded_rotation_match")
    if entry["case"] == "wave_1d_source_noise" and "phase_truth_only" in row:
        row["phase_truth_only"].update(carrier_coordinates_only=True,
            physical_module_complete=False, completeness_claim_for_full_field=False,
            source_noise_control_outside_pure_phase_field_class=True)


def main():
    if OUTPUT.exists():
        raise FileExistsError("preserve the existing fixed20 dictionary pilot")
    for path in (ROOT/"parameters", ROOT/"projected"):
        path.mkdir(parents=True, exist_ok=True)
    source = json.loads(SOURCE.read_text());comparison = json.loads(COMPARISON.read_text())
    if source["status"] != "finished" or comparison["status"] != "finished" or len(source["manifest"]) != 20:
        raise ValueError("requires the completed, frozen compact20 and MSSA20 comparators")
    acquired, inference_sources = filtered_acquired_metadata()
    comparison_rows = {row["run"]: {key: row[key] for key in ("noise_estimate_train_only",
        "frozen_output", "frozen_output_sha256")} for row in comparison["rows"]}
    initial = fingerprints();started = time.perf_counter();rows = []
    payload = dict(status="running", rows=rows, manifest=source["manifest"], source_fingerprints=initial,
        inference_metadata_sources=inference_sources, compact_source_sha256=digest(SOURCE),
        classical_comparison_source_sha256=digest(COMPARISON), evaluation_files_opened=False,
        rules_declared_before_new_scoring=dict(thresholds=list(THRESHOLDS), conditioning_factor=CONDITIONING_FACTOR,
            window=WINDOW, validation_window="min(128,Nval-2)", validation_masks="even/odd complementary relative lags",
            phase_seed="weighted free-span coefficients on fitting lag mask only",
            rotation_group_and_phase_rank="original selected compact20 fixed",
            ridge="same frozen temporal ridge; quotient multiplicities and cosine-only order2 lines retained",
            noise="old matched comparison TRAIN-only high-band raw noise estimate",
            anchor_protection="ALL observation-validated acquired characters up to sign and UNSATURATED ZR"),
        all20_benchmarks_previously_inspected=True, fresh_external_validation_claimed=False,
        conditional_coefficient_noise_proxy_is_calibrated_test=False,
        validation_criterion_conditional_on_earlier_validation_selected_dictionary=True,
        exact_physical_phase_rank_or_group_certified=False, novelty_claimed=False)
    save(payload)
    for entry, source_row in zip(source["manifest"], source["rows"]):
        print("starting phase-preserving dictionary "+entry["run"], flush=True)
        try:
            source_fit = {key: source_row[key] for key in ("selected_model", "group_proposal",
                "frozen_parameters", "frozen_parameters_sha256", "frozen_output", "frozen_output_sha256")}
            row = fit_and_freeze(entry, source_fit, comparison_rows[entry["run"]], acquired[entry["run"]])
        except Exception as error:
            row = dict(run=entry["run"], case=entry["case"], kind=entry["kind"], run_status="exception",
                exception=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        rows.append(row);save(payload)
        print(json.dumps(dict(run=row["run"], status=row["run_status"], threshold=row.get("selected_threshold"),
            old_count=row.get("original_count"), selected_count=row.get("selected_count"),
            seconds=row.get("elapsed_seconds"), exception=row.get("exception"))), flush=True)
    if fingerprints() != initial:
        raise ValueError("new pilot source changed during inference")
    if any(digest(path) != expected for path, expected in inference_sources.items()):
        raise ValueError("filtered acquired-metadata source changed during inference")
    if any(digest(entry["input"]) != entry["input_sha256"] for entry in source["manifest"]):
        raise ValueError("public Y,s,t changed during inference")
    payload.update(all20_attempts_outputs_and_parameters_frozen_before_truth_open=True,
        inference_finished_at=datetime.now(timezone.utc).isoformat(), source_fingerprints_unchanged_before_scoring=True)
    save(payload)
    for entry, row in zip(source["manifest"], rows):
        if row["run_status"] == "fitted_and_frozen":
            score(row, entry)
            print(json.dumps(dict(run=row["run"], endpoint={k:v["sensor_nrmse"] for k,v in row["endpoint_field"].items()},
                diagonal={k:v["sensor_nrmse"] for k,v in row["diagonal_field"].items()})), flush=True)
    finished = [row for row in rows if row["run_status"] == "finished"]
    payload.update(status="finished", evaluation_files_opened=True, source_fingerprints_unchanged=fingerprints() == initial,
        total_elapsed_seconds=time.perf_counter()-started,
        summary=dict(requested_runs=20, finished_runs=len(finished), exceptions=len(rows)-len(finished),
            dictionaries_pruned=sum(row["selected_count"] < row["original_count"] for row in finished),
            better_endpoint_than_MSSA=sum(row["endpoint_field"]["new"]["sensor_nrmse"] < row["endpoint_field"]["MSSA"]["sensor_nrmse"] for row in finished),
            better_diagonal_than_MSSA=sum(row["diagonal_field"]["new"]["sensor_nrmse"] < row["diagonal_field"]["MSSA"]["sensor_nrmse"] for row in finished)))
    save(payload);print(json.dumps(payload["summary"]), flush=True)


if __name__ == "__main__":
    main()
