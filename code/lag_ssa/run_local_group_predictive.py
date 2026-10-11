"""Local linear-Gaussian predictive comparison of frozen beam models.

Default writes a metadata-only immutable protocol. --run must be authorized
after source/math review. No new NONLINEAR temporal optimization, relation
discovery or physical evaluator is called. A local linear TRAIN posterior
mean update is included. The old FUTURE block becomes a posthoc selection block;
it is not an independent test of the newly selected model.
"""

from project_paths import RESULTS_DIR

from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.linalg import qr, svdvals

from local_gaussian_predictive import score_local_predictive


HERE = Path(__file__).resolve().parent
SOURCE = RESULTS_DIR / 'ledger_beam_group_selection.json'
ROOT = RESULTS_DIR / 'local_group_predictive'
OUTPUT = ROOT.with_suffix(".json")
PROTOCOL = RESULTS_DIR / 'local_group_predictive_protocol.json'
SOURCES = ("local_gaussian_predictive.py", "run_local_group_predictive.py")
SETTINGS = dict(orthonormal_basis_rank_rcond=1e-12,
                orthonormal_basis_check_tolerance=1e-10,
                numerical_covariance_relative_tolerance=1e-10,
                candidate_scope="ALL 237 frozen archives, including two R=0 point controls per case",
                posterior_scope="local RAW-sensor linear Gaussian model at frozen TRAIN parameters",
                posterior_data="TRAIN [0,2400) only",
                scoring_data="old FUTURE [3600,4800) raw observations only",
                frequency_cells="old one-bin cells around unchanged RAW ledger frequencies, recorded but NOT truncating the Gaussian",
                selection="strict minimum of each of three declared scores; ties keep earlier model",
                nonlinear_pushforward_mean="diagnostic only; no additional score, clipping, threshold or selection")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(item) for item in value]
    return value


def save_json(path, payload):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(serializable(payload), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_archive(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def declaration():
    """Hashes and existing inference metadata only: never load public Y here."""
    source = json.loads(SOURCE.read_text())
    if (source["status"] != "finished_inference_only" or len(source["rows"]) != 6
            or not source["all_six_attempts_and_outputs_frozen_before_truth"]
            or source["evaluation_files_opened"] or not source["source_and_input_hashes_unchanged"]):
        raise ValueError("six frozen public-only beam results required")
    manifest, protected = [], []
    for row, public in zip(source["rows"], source["manifest"]):
        if row["run"] != public["run"] or row["run_status"] != "fitted_and_frozen":
            raise ValueError("all six fixed beam rows required")
        attempts = {attempt["stored_model_index"]: attempt for attempt in row["inference"]["all_model_attempts"]
                    if "stored_model_index" in attempt}
        models = []
        for model in row["model_files"]:
            attempt = attempts[model["model_index"]]
            relation = model["presentation"]["relation_rows_HNF"]
            winding = attempt.get("integer_windings", [])
            if len(winding) != len(relation):
                raise ValueError("exact frozen winding missing")
            models.append(dict(model_index=model["model_index"], model_kind=model["model_kind"],
                beam_state_id=model["beam_state_id"], parameters=model["parameters"],
                parameters_sha256=model["parameters_sha256"], relation_rows_HNF=relation,
                integer_windings=winding, continuous_dimension=model["presentation"]["continuous_dimension"]))
        entry = {key: public[key] for key in ("run", "input", "input_sha256", "n_train", "n_validation", "n_future",
                                             "ledger_parameters", "ledger_parameters_sha256")}
        if (entry["n_train"], entry["n_validation"], entry["n_future"]) != (2400, 1200, 1200):
            raise ValueError("the frozen interval declaration changed")
        entry.update(noise_estimate_train_only=row["noise_estimate_train_only"], models=models)
        manifest.append(entry)
    for name, expected in source["source_fingerprints"].items():
        protected.append(dict(path=str(HERE / name), sha256=expected))
    protected.extend(dict(path=item["parameters"], sha256=item["parameters_sha256"])
                     for item in source["protected_original_group_models"])
    protected.extend(dict(path=source[name], sha256=source[name + "_sha256"])
                     for name in ("source_alternatives", "source_group"))
    for item in protected:
        if digest(item["path"]) != item["sha256"]:
            raise ValueError("old source/model changed")
    for entry in manifest:
        for path, expected in ((entry["input"], entry["input_sha256"]),
                               (entry["ledger_parameters"], entry["ledger_parameters_sha256"])):
            if digest(path) != expected:
                raise ValueError("old public/ledger input changed")
        for model in entry["models"]:
            if digest(model["parameters"]) != model["parameters_sha256"]:
                raise ValueError("old frozen beam archive changed")
    if sum(len(entry["models"]) for entry in manifest) != 237:
        raise ValueError("all 237 frozen model archives required")
    return dict(status="prepared_not_run", created_at=datetime.now(timezone.utc).isoformat(),
        source_beam=str(SOURCE), source_beam_sha256=digest(SOURCE), settings=SETTINGS,
        source_fingerprints={name: digest(HERE / name) for name in SOURCES},
        manifest=manifest, protected_original_artifacts=protected, model_count=237,
        local_gaussian_approximation=True, proper_nonlinear_Bayesian_evidence_claimed=False,
        proper_prior_model_probabilities_or_model_averaging_used=False,
        baseline_and_free_refit_are_duplicate_statistical_families=True,
        duplicates_retained_only_as_original_point_controls=True,
        original_catalogue_and_models_selected_using_TRAIN_and_VAL=True,
        raw_VAL_observations_reused_in_this_score=False,
        old_FUTURE_observations_now_used_for_model_selection=True,
        independent_test_after_new_selection=False, previously_examined_benchmark=True,
        physical_evaluation_files_opened=False, physical_labels_or_true_noise_supplied=False,
        only_public_Y_s_t_given_to_comparison=True, new_relations_or_nonlinear_temporal_refits=False,
        local_linear_TRAIN_posterior_mean_update_included=True,
        undefined_local_posteriors_do_not_remove_original_point_forecast_controls=True,
        frequency_Gaussian_is_untruncated_and_local=True,
        original_one_bin_margins_and_frequency_phase_uncertainty_recorded=True,
        phase_SD_scope="frequency uncertainty times elapsed time from TRAIN center, not full initial-phase uncertainty",
        nonlinear_diagnostic_scope="exact field MEAN pushforward of local Gaussian parameter reference, not an exact nonlinear predictive density",
        no_nonlinear_diagnostic_acceptance_gate=True,
        physical_group_rank_or_phase_encoding_certified=False,
        observed_window_or_MSSA_comparison_claimed=False, novelty_claimed=False)


def validate(prepared):
    current = declaration()
    for key in ("source_beam_sha256", "settings", "source_fingerprints", "manifest", "protected_original_artifacts", "model_count"):
        if current[key] != prepared[key]:
            raise ValueError("frozen predictive protocol changed: " + key)


def exact_matrix(values):
    array = np.asarray(values)
    result = np.empty(array.shape, dtype=object)
    for index in np.ndindex(array.shape):
        result[index] = int(array[index])
    return result


def frequency_basis(parameters, metadata):
    nu = parameters["frequencies"]
    relation = exact_matrix(parameters["relation_rows_HNF_integer_text"])
    transform = exact_matrix(parameters["V_integer_text"])
    rank, dimension = relation.shape[0], len(nu)
    free = dimension - rank
    if (relation.tolist() != metadata["relation_rows_HNF"] or free != metadata["continuous_dimension"]
            or transform.shape != (dimension, dimension)):
        raise ValueError("exact frozen relation/gauge differs from metadata")
    integer_free = transform[:, rank:]
    if np.any(relation @ integer_free != 0):
        raise ArithmeticError("exact R V_free is nonzero")
    if free:
        floating = np.asarray(integer_free, dtype=float)
        if any(int(first) != int(second) for first, second in zip(floating.flat, integer_free.flat)):
            raise ArithmeticError("free gauge integers cannot be represented exactly by doubles")
        basis, triangular = qr(floating, mode="economic")
        singular = svdvals(triangular)
        if singular[-1] <= SETTINGS["orthonormal_basis_rank_rcond"] * singular[0]:
            raise np.linalg.LinAlgError("free frequency tangent numerically unresolved")
    else:
        basis, singular = np.empty((dimension, 0)), np.empty(0)
    floating_relation = np.asarray(relation, dtype=float)
    null_error = float(np.linalg.norm(np.einsum("ij,jr->ir", floating_relation, basis, optimize=False)))
    null_relative = null_error / max(1., float(np.linalg.norm(floating_relation)))
    orthogonal_error = float(np.max(abs(np.einsum("jr,js->rs", basis, basis, optimize=False) - np.eye(free)), initial=0.))
    if (not np.isfinite(null_relative) or null_relative > SETTINGS["orthonormal_basis_check_tolerance"]
            or orthogonal_error > SETTINGS["orthonormal_basis_check_tolerance"]):
        raise np.linalg.LinAlgError("orthonormal null basis numerically unresolved")
    equality = np.einsum("ij,j->i", floating_relation, nu, optimize=False) - 2 * np.pi * np.asarray(metadata["integer_windings"])
    return basis, dict(relation_rank=rank, free_frequency_dimension=free,
        exact_R_V_free_zero=True, free_gauge_singular_values=singular.tolist(),
        orthonormality_max_defect=orthogonal_error, R_Q_absolute_norm=null_error,
        R_Q_relative_norm=null_relative, frozen_R_nu_minus_2pi_w_max=float(np.max(abs(equality), initial=0.)),
        orthonormal_coordinates_not_raw_Smith_parameter_volume=True)


def temporal_arrays(parameters, times, basis):
    nu, coefficient = parameters["frequencies"], parameters["coefficients_sensor_center"]
    count, tau = len(nu), np.asarray(times, dtype=float) - float(parameters["time_center"])
    angles = tau[:, None] * nu[None, :]
    design = np.column_stack((np.ones(len(tau)), np.cos(angles), np.sin(angles)))
    point = np.einsum("np,pq->nq", design, coefficient, optimize=False)
    derivative = tau[:, None, None] * (-np.sin(angles)[:, :, None] * coefficient[1:1 + count][None, :, :]
                                       + np.cos(angles)[:, :, None] * coefficient[1 + count:][None, :, :])
    derivative = np.einsum("njq,jr->nqr", derivative, basis, optimize=False)
    return design, derivative, point, tau


def marginal_frequency_diagnostics(nu, basis, core, tau, original_nu, n_train, minimum_frequency):
    delta = np.einsum("jr,r->j", basis, core["posterior_frequency_mean_shift"], optimize=False)
    covariance = np.einsum("jr,rs,ks->jk", basis, core["posterior_frequency_covariance"], basis, optimize=False)
    variance = np.diag(covariance)
    scale = max(float(np.max(abs(covariance), initial=0.)), 1e-30)
    if np.any(variance < -SETTINGS["numerical_covariance_relative_tolerance"] * scale):
        raise np.linalg.LinAlgError("frequency posterior covariance has unresolved negative diagonal")
    standard_deviation = np.sqrt(np.maximum(variance, 0.))
    half_width = 2 * np.pi / n_train
    lower, upper = np.maximum(minimum_frequency, original_nu - half_width), np.minimum(np.pi - minimum_frequency, original_nu + half_width)
    ratios = []
    for point, shift, std, lo, hi in zip(nu, delta, standard_deviation, lower, upper):
        ratios.append(dict(frequency=float(point), local_mean_frequency=float(point + shift),
            inherited_lower=float(lo), inherited_upper=float(hi), posterior_sd=float(std),
            frozen_lower_margin_over_sd=float((point - lo) / std) if std else None,
            frozen_upper_margin_over_sd=float((hi - point) / std) if std else None,
            local_mean_lower_margin_over_sd=float((point + shift - lo) / std) if std else None,
            local_mean_upper_margin_over_sd=float((hi - point - shift) / std) if std else None,
            fixed_frequency_at_reference=bool(std == 0)))
    phase_sd = float(np.max(abs(tau), initial=0.)) * standard_deviation
    return delta, covariance, dict(mean_rotation_displacement=delta.tolist(),
        frequency_posterior_marginal_standard_deviation=standard_deviation.tolist(),
        maximum_elapsed_samples_from_TRAIN_center=float(np.max(abs(tau), initial=0.)),
        frequency_only_phase_SD_max_by_line=phase_sd.tolist(),
        frequency_only_phase_SD_max=float(np.max(phase_sd, initial=0.)), inherited_one_bin_margins=ratios,
        one_bin_Gaussian_truncation_applied=False, local_mean_clipped_to_old_cells=False)


def nonlinear_mean(parameters, basis, core, tau, delta_nu, covariance_nu):
    coefficient = parameters["coefficients_sensor_center"] + core["posterior_coefficient_mean_shift"]
    count = len(parameters["frequencies"])
    mean_nu = parameters["frequencies"] + delta_nu
    cross = np.einsum("pqr,jr->pqj", core["posterior_coefficient_frequency_covariance"], basis, optimize=False)
    amplitude = coefficient[1:1 + count] - 1j * coefficient[1 + count:]
    amplitude_cross = np.asarray([cross[1 + j, :, j] - 1j * cross[1 + count + j, :, j] for j in range(count)])
    factor = np.exp(1j * tau[:, None] * mean_nu[None, :] - .5 * tau[:, None] ** 2 * np.diag(covariance_nu)[None, :])
    values = (amplitude[None, :, :] + 1j * tau[:, None, None] * amplitude_cross[None, :, :]) * factor[:, :, None]
    return coefficient[0][None, :] + np.real(np.sum(values, axis=1))


def score_model(entry, metadata, raw_train, raw_future, times_train, times_future, original_nu, minimum_frequency):
    parameters = load_archive(metadata["parameters"])
    if not np.array_equal(parameters["sensor_positions"], entry["positions"]):
        raise ValueError("frozen model/public sensor positions differ")
    if float(parameters["time_center"]) != float(np.mean(times_train)):
        raise ValueError("frozen TRAIN time gauge changed")
    basis, basis_audit = frequency_basis(parameters, metadata)
    x_train, d_train, point_train, _ = temporal_arrays(parameters, times_train, basis)
    x_future, d_future, point_future, tau = temporal_arrays(parameters, times_future, basis)
    e_train, e_future = raw_train.T - point_train, raw_future.T - point_future
    sigma = float(entry["noise_estimate_train_only"]["sigma"])
    core = score_local_predictive(x_train, d_train, e_train, x_future, d_future, e_future, sigma)
    linear_mean = point_future + core["predicted_shift"]
    delta_nu, covariance_nu, frequency_audit = marginal_frequency_diagnostics(parameters["frequencies"], basis, core, tau,
        original_nu, entry["n_train"], minimum_frequency)
    nonlinear = nonlinear_mean(parameters, basis, core, tau, delta_nu, covariance_nu)
    discrepancy = nonlinear - linear_mean
    point_mse = float(np.mean(e_future ** 2))
    shifted_mse = float(np.mean((raw_future.T - linear_mean) ** 2))
    arrays = dict(point_prediction_future=point_future.T, local_linear_prediction_mean_future=linear_mean.T,
        local_Gaussian_nonlinear_pushforward_mean_diagnostic_future=nonlinear.T,
        future_times=times_future, orthonormal_frequency_basis=basis,
        posterior_coefficient_mean_shift=core["posterior_coefficient_mean_shift"],
        posterior_frequency_mean_shift=core["posterior_frequency_mean_shift"],
        posterior_frequency_covariance=core["posterior_frequency_covariance"],
        posterior_coefficient_frequency_covariance=core["posterior_coefficient_frequency_covariance"],
        frequency_mean_displacement=delta_nu, ambient_frequency_covariance=covariance_nu,
        frequencies_frozen=parameters["frequencies"], coefficients_frozen=parameters["coefficients_sensor_center"],
        time_center=parameters["time_center"])
    if any(not np.isfinite(value).all() for value in arrays.values()):
        raise FloatingPointError("nonfinite local predictive output; no clipping or new gate")
    path = ROOT / "parameters" / (entry["run"] + f"__model{metadata['model_index']:03d}.npz")
    np.savez_compressed(path, **arrays)
    return dict(model_index=metadata["model_index"], model_kind=metadata["model_kind"],
        beam_state_id=metadata["beam_state_id"], status="scored_and_frozen", source_parameters=metadata["parameters"],
        source_parameters_sha256=metadata["parameters_sha256"], archive=str(path), archive_sha256=digest(path),
        negative_log_predictive_density=float(core["negative_log_density"]),
        negative_log_predictive_density_per_observation=float(core["negative_log_density_per_observation"]),
        logdet_inflation=float(core["logdet_inflation"]), standardized_quadratic=float(core["standardized_quadratic"]),
        point_FUTURE_raw_sensor_mse=point_mse, local_linear_mean_FUTURE_raw_sensor_mse=shifted_mse,
        core_point_MSE_control_difference=abs(point_mse - float(core["point_mse_original"])),
        core_shifted_MSE_control_difference=abs(shifted_mse - float(core["point_mse_shifted"])),
        raw_TRAIN_sensor_mse=float(np.mean(e_train ** 2)), raw_TRAIN_sensor_rss=float(np.sum(e_train ** 2)),
        posterior_coefficient_mean_shift_norm=float(np.linalg.norm(core["posterior_coefficient_mean_shift"])),
        posterior_frequency_mean_shift=core["posterior_frequency_mean_shift"].tolist(),
        rank_diagnostics=core["rank_diagnostics"], mean_stationarity=core["mean_stationarity"],
        frequency_basis=basis_audit, frequency_uncertainty=frequency_audit,
        nonlinear_mean_diagnostic=dict(rms_discrepancy_over_sigma=float(np.sqrt(np.mean(discrepancy ** 2)) / sigma),
            maximum_discrepancy_over_sigma=float(np.max(abs(discrepancy), initial=0.) / sigma),
            used_for_score_or_selection=False, exact_Gaussian_pushforward_mean_of_frozen_local_reference=True,
            not_exact_nonlinear_predictive_density=True),
        no_new_nonlinear_temporal_optimization_or_relation=True,
        local_linear_TRAIN_posterior_mean_update_included=True)


def run_entry(entry, minimum_frequency):
    with np.load(entry["input"], allow_pickle=False) as public:
        if set(public.files) != {"Y", "s", "t"}:
            raise ValueError("only public Y,s,t archive required")
        times, positions = public["t"], public["s"]
        observed = public["Y"]
        first = entry["n_train"] + entry["n_validation"]
        last = first + entry["n_future"]
        # VAL values are never supplied to any numerical comparison.
        train, future = observed[:, :entry["n_train"]], observed[:, first:last]
    if (not np.array_equal(times, np.arange(last)) or train.shape[0] != len(positions)
            or future.shape[1] != entry["n_future"]):
        raise ValueError("unchanged public sample frame required")
    with np.load(entry["ledger_parameters"], allow_pickle=False) as old:
        original_nu = old["frequencies"]
        if not np.array_equal(old["sensor_positions"], positions):
            raise ValueError("raw ledger geometry changed")
    rows = []
    state = dict(entry, positions=positions)
    for metadata in entry["models"]:
        # The frozen nonlinear POINT forecast remains a valid control even if
        # this model's local tangent posterior is numerically unresolved.
        parameters = load_archive(metadata["parameters"])
        if (not np.array_equal(parameters["sensor_positions"], positions)
                or float(parameters["time_center"]) != float(np.mean(times[:entry["n_train"]]))):
            raise ValueError("frozen point geometry/time gauge changed")
        point_tau = times[first:last] - float(parameters["time_center"])
        point_angles = point_tau[:, None] * parameters["frequencies"][None, :]
        point_design = np.column_stack((np.ones(len(point_tau)), np.cos(point_angles), np.sin(point_angles)))
        point = np.einsum("np,pq->nq", point_design, parameters["coefficients_sensor_center"], optimize=False)
        if not np.isfinite(point).all():
            raise FloatingPointError("original frozen point forecast nonfinite")
        point_mse = float(np.mean((future.T - point) ** 2))
        try:
            rows.append(score_model(state, metadata, train, future, times[:entry["n_train"]], times[first:last], original_nu, minimum_frequency))
        except (ValueError, np.linalg.LinAlgError, ArithmeticError, RuntimeError) as error:
            path = ROOT / "parameters" / (entry["run"] + f"__model{metadata['model_index']:03d}.npz")
            np.savez_compressed(path, point_prediction_future=point.T, future_times=times[first:last],
                frequencies_frozen=parameters["frequencies"], coefficients_frozen=parameters["coefficients_sensor_center"],
                time_center=parameters["time_center"])
            rows.append(dict(model_index=metadata["model_index"], model_kind=metadata["model_kind"],
                beam_state_id=metadata["beam_state_id"], status="numerically_undefined_local_model",
                exception=f"{type(error).__name__}: {error}", new_parameter_or_threshold_retry=False,
                point_FUTURE_raw_sensor_mse=point_mse, original_point_forecast_control_retained=True,
                archive=str(path), archive_sha256=digest(path),
                source_parameters=metadata["parameters"], source_parameters_sha256=metadata["parameters_sha256"]))
    available = [row for row in rows if row["status"] == "scored_and_frozen"]
    selections = {}
    for score in ("negative_log_predictive_density", "point_FUTURE_raw_sensor_mse", "local_linear_mean_FUTURE_raw_sensor_mse"):
        candidates = rows if score == "point_FUTURE_raw_sensor_mse" else available
        best = min(candidates, key=lambda item: item[score]) if candidates else None
        selections[score] = dict(model_index=best["model_index"], value=best[score]) if best else None
    return dict(run=entry["run"], status="all_original_models_attempted", rows=rows,
        requested_models=len(entry["models"]), scored_models=len(available), undefined_models=len(rows) - len(available),
        original_point_forecast_controls=len(rows),
        undefined_local_posteriors_excluded_only_from_density_and_shifted_mean=True,
        selection_block=[first, last], TRAIN_block=[0, entry["n_train"]], sensor_count=len(positions),
        spatial_dimension=positions.shape[1], declared_selection=selections,
        no_test_performance_or_physical_rank_scored=True,
        old_FUTURE_is_now_a_selection_block=True, raw_VAL_observations_used=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    arguments = parser.parse_args()
    if not arguments.run:
        if PROTOCOL.exists() or OUTPUT.exists() or ROOT.exists():
            raise FileExistsError("preserve prior predictive protocol/results")
        prepared = declaration()
        save_json(PROTOCOL, prepared)
        print(json.dumps(dict(status=prepared["status"], models=prepared["model_count"], protocol=str(PROTOCOL))), flush=True)
        return
    prepared = json.loads(PROTOCOL.read_text())
    validate(prepared)
    if OUTPUT.exists() or ROOT.exists():
        raise FileExistsError("preserve original predictive outputs")
    (ROOT / "parameters").mkdir(parents=True)
    # Original minimum-frequency setting is metadata, not a new tuning rule.
    beam = json.loads(SOURCE.read_text())
    minimum_frequency = float(beam["settings"]["minimum_interior_frequency"])
    output = dict(prepared, status="running_public_only_predictive_comparison", rows=[],
        protocol_sha256=digest(PROTOCOL), all_six_outputs_frozen_before_any_physical_evaluation=False)
    save_json(OUTPUT, output)
    tick = time.perf_counter()
    for entry in prepared["manifest"]:
        row = run_entry(entry, minimum_frequency)
        output["rows"].append(row)
        save_json(OUTPUT, output)
        print(json.dumps(dict(run=row["run"], scored=row["scored_models"], undefined=row["undefined_models"],
                              selections=row["declared_selection"])), flush=True)
    validate(prepared)
    if any(digest(row["archive"]) != row["archive_sha256"] for case in output["rows"] for row in case["rows"]):
        raise ValueError("new frozen predictive archive changed")
    output.update(status="finished_public_only_predictive_comparison", elapsed_seconds=time.perf_counter() - tick,
        all_six_outputs_frozen_before_any_physical_evaluation=True,
        source_input_and_all_original_model_hashes_unchanged=True,
        summary=dict(requested_models=237, scored_models=sum(row["scored_models"] for row in output["rows"]),
            undefined_models=sum(row["undefined_models"] for row in output["rows"]), requested_cases=6,
            original_point_forecast_controls=sum(row["original_point_forecast_controls"] for row in output["rows"]),
            physical_evaluation_files_opened=False, independent_test_of_selected_models=False))
    save_json(OUTPUT, output)
    print(json.dumps(dict(status=output["status"], summary=output["summary"])), flush=True)


if __name__ == "__main__":
    main()
