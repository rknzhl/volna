"""Validation-selected one-relation compact-group Hankel reduction.

Uses already frozen, public-data temporal discoveries (11 original waves,
six generic first-noise-seed fields and three new controls).  TRAIN/VAL raw
window encodings propose relations before constraints are imposed.  Two
complementary lag masks compare continuous, snapped continuous and compact
models on VAL.  Only the fitting mask supplies phase seeds and optimization.
This CV is conditional on the earlier learned dictionary, which itself used
validation for discovery; it is not an unbiased risk certificate.  All new
future outputs are saved before evaluation files are read by this runner.
The cached17 benchmarks were previously examined; controls are new.
"""

from project_paths import RESULTS_DIR

from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np

from compact_group_hankel import WeightedHankelProjector, refit_snapped_template
from hankel_torus_projection import ProjectionSettings
from observed_phase_group import estimate_observed_group
import run_actual_wave_iterative as wave
from run_original_hankel_torus import evaluate_field


HERE = Path(__file__).resolve().parent
ROOT = RESULTS_DIR / 'compact_group_hankel'
OUTPUT = ROOT.with_suffix(".json")
PROJECTION = ProjectionSettings()
GROUP_OPTIONS = dict(max_integer_bound=100, mixed_coefficient_bound=6,
    max_mixed_support=None, max_primitive_directions=4096,
    power_proposals_per_mixed_direction=3, coherence_threshold=.95)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(payload):
    temporary = OUTPUT.with_suffix(".tmp")
    temporary.write_text(json.dumps(wave.serializable(payload), ensure_ascii=False,
                                    indent=2, allow_nan=False) + "\n")
    temporary.replace(OUTPUT)


def fingerprints():
    return {name: sha(HERE / name) for name in ("run_compact_group_hankel.py",
        "compact_group_hankel.py", "observed_phase_group.py", "hankel_torus_projection.py",
        "prepare_compact_group_controls.py", "phase_factor_regression.py")}


def manifest():
    result = []
    originals = json.loads((RESULTS_DIR / 'original_all_waves_robust_hankel.json').read_text())
    for entry, row in zip(originals["manifest"], originals["rows"]):
        result.append(dict(run=entry["case"], case=entry["case"], kind="wave",
            input=entry["input"], input_sha256=entry["input_sha256"],
            evaluation=str(wave.DATA / "evaluation" / (entry["case"] + ".npz")),
            source_model=row["frozen_parameters"], source_model_sha256=row["frozen_parameters_sha256"],
            n_train=row["settings"]["n_train"], n_validation=row["settings"]["n_validation"],
            temporal_ridge=row["inference"]["final_clock_model"]["global_dictionary"]["selected_dictionary"]["ridge_scale"],
            previously_examined_benchmark=True))
    generic = json.loads((RESULTS_DIR / 'general_robustness_factor_cover.json').read_text())
    for entry, row in zip(generic["manifest"], generic["rows"]):
        if entry["seed"] != 202610081:
            continue
        result.append(dict(run=entry["run"], case=entry["case"], kind="generic",
            input=entry["input"], input_sha256=entry["input_sha256"], evaluation=entry["evaluation"],
            source_model=row["frozen_arrays"], source_model_sha256=row["frozen_arrays_sha256"],
            n_train=2400, n_validation=1200,
            temporal_ridge=row["inference"]["final_clock_model"]["global_dictionary"]["selected_dictionary"]["ridge_scale"],
            previously_examined_benchmark=True))
    controls = json.loads((RESULTS_DIR / 'compact_group_controls.json').read_text())
    for entry, row in zip(controls["manifest"], controls["rows"]):
        if row["run_status"] != "fitted_and_frozen":
            raise ValueError("all new controls must have completed frozen discovery before this run")
        result.append(dict(run=entry["run"], case=entry["case"], kind="control",
            input=entry["input"], input_sha256=entry["input_sha256"], evaluation=entry["evaluation"],
            source_model=row["source_model"], source_model_sha256=row["source_model_sha256"],
            n_train=2400, n_validation=1200, temporal_ridge=row["selected_temporal_ridge"],
            previously_examined_benchmark=False))
    if len(result) != 20:
        raise ValueError("the declared protocol requires exactly20 cases")
    return result


def observer(specification, window, weights=None):
    return WeightedHankelProjector(specification["labels"], specification["coefficients_mass"],
        specification["frequencies"], weights=weights, relation=specification.get("relation"),
        settings=replace(PROJECTION, window=window), max_branch_order=100)


def lag_cv(specification, validation_mass, window):
    """Fit each window's phase on one parity; score the unseen other parity."""
    windows = np.lib.stride_tricks.sliding_window_view(validation_mass, window, axis=1)
    folds = []
    total = 0.
    for parity in (0, 1):
        fit_lags = np.arange(window) % 2 == parity
        held_lags = ~fit_lags
        weights = fit_lags.astype(float) / fit_lags.sum()
        projector = observer(specification, window, weights)
        projected = projector.project(validation_mass)
        phase = projected["phase"]
        error = 0.
        for lag in np.flatnonzero(held_lags):
            predicted = evaluate_field(specification["labels"], specification["coefficients_mass"],
                                       phase + specification["frequencies"][:, None] * lag)
            error += float(np.sum((predicted - windows[:, :, lag]) ** 2))
        target_count = int(phase.shape[1] * held_lags.sum())
        mse = error / target_count
        total += error
        folds.append(dict(fitting_parity=parity, fitting_lags=int(fit_lags.sum()),
            scored_lags=int(held_lags.sum()), starts=phase.shape[1], squared_mass_error=error,
            mass_mse_per_field_snapshot=mse, held_lag_observations_in_phase_seed=False))
    denominator = windows.shape[1] * window
    return dict(mass_mse_per_field_snapshot=total / denominator, folds=folds,
        validation_window=window, no_masked_target_in_its_phase_encoding=True,
        parameter_selection_conditional_on_earlier_validation_selected_dictionary=True,
        calibrated_risk_estimate_claimed=False)


def fit_and_freeze(entry):
    started = time.perf_counter()
    for path_key, hash_key in (("input", "input_sha256"), ("source_model", "source_model_sha256")):
        if sha(entry[path_key]) != entry[hash_key]:
            raise ValueError("changed frozen source " + path_key)
    with np.load(entry["input"], allow_pickle=False) as public:
        if set(public.files) != {"Y", "s", "t"}:
            raise ValueError("onlyY,s,t allowed in public observations")
        y = public["Y"]
    # Never import old train/future phase arrays or old scored predictions.
    with np.load(entry["source_model"], allow_pickle=False) as parameters:
        labels = parameters["fit_labels"]
        alpha = parameters["fit_frequencies"]
        coefficients = parameters["fit_coefficients"]
        mass_sqrt = parameters["spatial_mass_sqrt"]
        mapping = parameters["mass_to_sensor"]
        physical_mass = parameters["physical_mass"]
    n_train, n_validation = entry["n_train"], entry["n_validation"]
    cutoff = n_train + n_validation
    mass = np.einsum("ij,jt->it", mass_sqrt, y, optimize=False)
    baseline = dict(name="continuous", labels=labels, frequencies=alpha,
                    coefficients_mass=coefficients, relation=None)
    validation = mass[:, n_train:cutoff]
    validation_window = min(PROJECTION.window, n_validation - 2)
    train_observed = observer(baseline, PROJECTION.window).project(mass[:, :n_train])
    val_observed = observer(baseline, validation_window).project(validation)
    group = estimate_observed_group(np.exp(1j * train_observed["phase"]),
                                   np.exp(1j * val_observed["phase"]), **GROUP_OPTIONS)
    specifications = {"continuous": baseline}
    proposal = dict(status="no_supported_relation")
    presentation = group["presentation"]
    relations = presentation["integer_relation_row_basis_HNF"]
    if presentation["candidate_basis_empirically_compatible"] and len(relations) == 1:
        relation = np.asarray(relations[0], dtype=int)
        try:
            refined = refit_snapped_template(y[:, :n_train], np.arange(n_train), alpha, labels,
                                            entry["temporal_ridge"], relation)
            refined_mass = np.einsum("nq,iq->ni", refined["coefficients_sensor"], mass_sqrt, optimize=False)
            snapped = dict(name="snapped_continuous", labels=refined["labels"],
                frequencies=refined["frequencies"], coefficients_mass=refined_mass, relation=None)
            compact = {**snapped, "name": "compact", "relation": relation}
            # Explicit constructor checks; unresolved quotient observer cannot
            # be silently treated as a supported group reconstruction.
            observer(compact, validation_window)
            specifications["compact"] = compact
            try:
                observer(snapped, validation_window)
                specifications["snapped_continuous"] = snapped
            except (ValueError, ArithmeticError, np.linalg.LinAlgError) as error:
                proposal["snapped_continuous_ablation_unavailable"] = str(error)
            proposal.update(status="single_relation_candidate_fitted", relation=relation.tolist(),
                            refined_template=refined["metadata"], coset_origin="TRAIN-clock refit gauge0")
        except (ValueError, ArithmeticError, np.linalg.LinAlgError) as error:
            proposal.update(status="unsupported_candidate", reason=str(error), relation=relation.tolist())
    elif len(relations) > 1:
        proposal.update(status="unsupported_relation_rank", relation_rank=len(relations),
                        reason="this prototype implements one independent relation; baseline retained")
    cvs = {}
    accepted = {}
    for name in ("continuous", "snapped_continuous", "compact"):
        if name not in specifications:
            continue
        try:
            cvs[name] = lag_cv(specifications[name], validation, validation_window)
            accepted[name] = specifications[name]
        except (ValueError, ArithmeticError, np.linalg.LinAlgError) as error:
            if name == "continuous":
                raise
            proposal[name + "_cv_unavailable"] = str(error)
    # Ties preserve the continuous baseline; no test error chooses the family.
    selected = min(accepted, key=lambda name: (cvs[name]["mass_mse_per_field_snapshot"],
                    ("continuous", "snapped_continuous", "compact").index(name)))
    output_arrays = dict(training_unconstrained_phase=train_observed["phase"],
                         validation_unconstrained_phase=val_observed["phase"])
    parameters_arrays = dict(physical_mass=physical_mass, mass_to_sensor=mapping,
        spatial_mass_sqrt=mass_sqrt, selected_model=np.asarray(selected),
        relation=np.asarray(proposal.get("relation", []), dtype=int))
    future_mass = mass[:, cutoff - PROJECTION.window + 1:]
    observer_metadata = {}
    for name, specification in accepted.items():
        projector = observer(specification, PROJECTION.window)
        projected = projector.project(future_mass)
        start_phase = projected["phase"]
        endpoint_phase = np.angle(np.exp(1j * (start_phase +
                         specification["frequencies"][:, None] * (PROJECTION.window - 1))))
        predicted_mass = evaluate_field(specification["labels"], specification["coefficients_mass"], endpoint_phase)
        prediction = np.einsum("ij,jt->it", mapping, predicted_mass, optimize=False)
        output_arrays.update({name + "__phase_at_window_start": start_phase,
            name + "__phase_at_window_end": endpoint_phase, name + "__prediction": prediction,
            name + "__window_cost": np.asarray([item["final_cost"] for item in projected["diagnostics"]])})
        parameters_arrays.update({name + "__labels": specification["labels"],
            name + "__frequencies": specification["frequencies"],
            name + "__coefficients_mass": specification["coefficients_mass"]})
        if specification["relation"] is not None:
            phase_residue = np.angle(np.exp(1j * np.einsum("i,it->t", specification["relation"], endpoint_phase, optimize=False)))
            output_arrays[name + "__relation_residue"] = phase_residue
        observer_metadata[name] = dict(multistart=projected.get("multistart", {}),
            window_count=start_phase.shape[1],
            gradient_converged_count=sum(item.get("converged_gradient", False) for item in projected["diagnostics"]),
            final_gradient_max=max(item.get("final_gradient_max", 0.) for item in projected["diagnostics"]),
            absolute_future_time_used_in_seed=False, global_optimum_certified=False)
    if not all(np.isfinite(value).all() for value in output_arrays.values()):
        raise FloatingPointError("nonfinite frozen phase or prediction")
    output_arrays["selected_prediction"] = output_arrays[selected + "__prediction"]
    output_path = ROOT / "projected" / (entry["run"] + ".npz")
    parameter_path = ROOT / "parameters" / (entry["run"] + ".npz")
    np.savez_compressed(output_path, **output_arrays)
    np.savez_compressed(parameter_path, **parameters_arrays)
    return dict(run=entry["run"], case=entry["case"], kind=entry["kind"], run_status="fitted_and_frozen",
        source_model=entry["source_model"], source_model_sha256=entry["source_model_sha256"],
        original_phase_count=len(alpha), group_estimate=group, group_proposal=proposal,
        selected_model=selected, validation_cv=cvs, observers=observer_metadata,
        future_start=cutoff, future_count=y.shape[1] - cutoff,
        frozen_parameters=str(parameter_path), frozen_parameters_sha256=sha(parameter_path),
        frozen_output=str(output_path), frozen_output_sha256=sha(output_path),
        exact_relation_or_orbit_closure_certified=False,
        relation_proposed_from_unrestricted_observed_phase_before_snap=True,
        previously_examined_benchmark=entry["previously_examined_benchmark"],
        elapsed_seconds=time.perf_counter() - started)


def score(row, entry):
    if sha(row["frozen_output"]) != row["frozen_output_sha256"]:
        raise ValueError("new predictions changed before scoring")
    with np.load(entry["evaluation"], allow_pickle=False) as truth:
        clean = truth["clean_Y"] if entry["kind"] == "wave" else truth["clean"]
    target = clean[:, row["future_start"]:row["future_start"] + row["future_count"]]
    with np.load(row["frozen_parameters"]) as parameters:
        physical_mass = parameters["physical_mass"]
    with np.load(row["frozen_output"]) as output:
        scores = {name: wave.field_score(output[name + "__prediction"], target, physical_mass)
                  for name in row["observers"]}
    row.update(run_status="finished", future_field=scores, selected_future_field=scores[row["selected_model"]])


def main():
    if OUTPUT.exists():
        raise FileExistsError("preserve this group's declared protocol and outputs")
    for path in (ROOT / "parameters", ROOT / "projected"):
        path.mkdir(parents=True, exist_ok=True)
    sources = fingerprints()
    cases = manifest()
    rows = []
    payload = dict(status="running", created_at=datetime.now(timezone.utc).isoformat(),
        manifest=cases, rows=rows, source_fingerprints=sources, group_options=GROUP_OPTIONS,
        projection_settings=asdict(PROJECTION), evaluation_files_opened=False,
        branch_selection_scope="conditional VAL even/odd masked-lag prediction; no held lag in its own phase seed",
        supported_group_relation_rank=1, group_component_limit=100,
        candidate_group_not_a_certificate_of_rationality=True,
        previous_benchmarks_not_fresh_external_test=True, new_control_cases=3,
        future_prediction_sources="actual same trailing128 raw-window encodings only",
        spatial_extension_claimed=False, novelty_claimed=False)
    save(payload)
    for entry in cases:
        print("starting compact-group comparison " + entry["run"], flush=True)
        try:
            row = fit_and_freeze(entry)
        except Exception as error:
            row = dict(run=entry["run"], case=entry["case"], kind=entry["kind"], run_status="exception",
                       exception=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        rows.append(row)
        save(payload)
        print(json.dumps(dict(run=row["run"], status=row["run_status"], selected=row.get("selected_model"),
                              seconds=row.get("elapsed_seconds"), error=row.get("exception"))), flush=True)
    if sources != fingerprints():
        raise ValueError("prototype sources changed during the declared run")
    payload.update(status="all_inference_attempts_finished_before_new_scoring",
                   inference_finished_at=datetime.now(timezone.utc).isoformat())
    save(payload)
    for row, entry in zip(rows, cases):
        if row["run_status"] == "fitted_and_frozen":
            score(row, entry)
    payload.update(status="finished", evaluation_files_opened=True,
                   finished_at=datetime.now(timezone.utc).isoformat())
    save(payload)
    print(json.dumps(dict(status="finished", successful=sum(row["run_status"] == "finished" for row in rows),
                          failed=sum(row["run_status"] == "exception" for row in rows))), flush=True)


if __name__ == "__main__":
    main()
