"""Observed free-line discovery on raw sensor fields; no harmonic synthesis.

The original SSA/DMD proposal equations and heuristic gates supply candidates.
Each proposed line has an independent frequency and free sensor profile. All
frequencies/profiles are jointly refitted on RAW TRAIN by unpenalized variable
projection. Sensor iid likelihood is distinct from the physical mass metric
used for SSA proposals. No integer labels, products or relation lattice enter.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import time

import numpy as np
from scipy.linalg import lstsq, qr, solve_triangular
from scipy.optimize import minimize

from alternative_cover_selection_ssa import character_frontend
from iterative_ssa import Settings
from nodal_spatial import NodalSpatialModel
from ssa_character import apply_ssa_character


@dataclass(frozen=True)
class LedgerSettings:
    n_train: int = 2400
    n_validation: int = 1200
    n_future: int = 1200
    lag_window: int = 128
    dynamic_ranks: tuple[int, ...] = (4, 8, 16)
    maximum_lines: int = 48
    maximum_frequency_iterations: int = 100
    frequency_half_width_bins: float = 1.0
    initial_duplicate_bins: float = .25
    minimum_interior_frequency: float = 1e-8
    optimizer_ftol: float = 1e-13
    optimizer_gtol: float = 1e-10
    design_rcond: float = 1e-12
    information_rcond: float = 1e-12


SETTINGS = LedgerSettings()


def _design(times, frequencies):
    angles = times[:, None] * frequencies[None, :]
    return np.column_stack((np.ones(len(times)), np.cos(angles), np.sin(angles)))


def merge_initial_frequencies(frequencies, bin_width, settings=SETTINGS):
    """Stable observation-only merge; keep the first initializer in a cluster."""
    retained, source_to_retained, decisions = [], [], []
    tolerance = settings.initial_duplicate_bins * bin_width
    for index, frequency in enumerate(np.asarray(frequencies, dtype=float)):
        gaps = abs(np.asarray(retained)-frequency)
        match = int(np.argmin(gaps)) if len(gaps) else None
        if match is not None and gaps[match] <= tolerance:
            source_to_retained.append(match)
            decisions.append(dict(source_index=index, merged_into_retained_index=match,
                frequency=float(frequency), retained_frequency=float(retained[match]),
                gap=float(gaps[match]), tolerance=float(tolerance)))
        else:
            source_to_retained.append(len(retained))
            retained.append(float(frequency))
    return np.asarray(retained), dict(initial_count=len(frequencies), retained_count=len(retained),
        source_to_retained=source_to_retained, merged_initializers=decisions,
        rule="stable first observed initializer; no synthetic line or integer relation")


@dataclass(frozen=True)
class FreeLineModel:
    frequencies: np.ndarray
    coefficients_sensor: np.ndarray
    time_center: float
    frequency_covariance: np.ndarray
    coefficient_covariance_per_sensor: np.ndarray
    temporal_gram: np.ndarray
    audit: dict

    def predict(self, times):
        design = _design(np.asarray(times, dtype=float)-self.time_center, self.frequencies)
        return np.einsum("nj,jq->qn", design, self.coefficients_sensor, optimize=False)

    def complex_profiles(self, origin_gauge=False):
        count = len(self.frequencies)
        profile = .5*(self.coefficients_sensor[1:1+count]
                      -1j*self.coefficients_sensor[1+count:])
        if origin_gauge:
            profile = profile*np.exp(-1j*self.frequencies*self.time_center)[:, None]
        return profile


def fit_free_lines(raw_train, times, initial_frequencies, noise_sigma, settings=SETTINGS):
    """Joint RAW iid-sensor ML fit with analytic variable-projection gradient.

    Frequency displacements are scaled by N. Coefficients are unpenalized LS,
    so the envelope derivative of RSS is -2 <residual, dX/dnu * coefficients>.
    Fisher/score outputs are local conditional diagnostics, not confidence sets.
    """
    y = np.asarray(raw_train, dtype=float)
    t = np.asarray(times, dtype=float)
    initial = np.asarray(initial_frequencies, dtype=float).reshape(-1)
    if (y.ndim != 2 or t.shape != (y.shape[1],) or not np.isfinite(y).all()
            or not np.isfinite(t).all() or not np.isfinite(initial).all()
            or not np.array_equal(t-t[0], np.arange(len(t)))
            or not np.isfinite(noise_sigma) or noise_sigma <= 0
            or np.any((initial <= 0) | (initial >= np.pi))):
        raise ValueError("finite raw Y, unit-spaced times, interior frequencies and iid sigma required")
    q, n = y.shape
    if len(initial) > settings.maximum_lines or n <= 2*len(initial)+2:
        raise ValueError("free-line budget or temporal design degrees exhausted")
    bin_width = 2*np.pi/n
    frequencies0, merge = merge_initial_frequencies(initial, bin_width, settings)
    count = len(frequencies0)
    center = float(t.mean())
    tau, target = t-center, y.T

    def profile(frequencies, derivatives=True):
        design = _design(tau, frequencies)
        coefficient, _, rank, _ = lstsq(design, target,
            cond=settings.design_rcond, lapack_driver="gelsy")
        if rank != design.shape[1]:
            raise np.linalg.LinAlgError("free observed-line amplitude design is rank deficient")
        residual = target-np.einsum("nj,jq->nq", design, coefficient, optimize=False)
        if not derivatives:
            return design, coefficient, residual
        angles = tau[:, None]*frequencies[None, :]
        derivative = tau[:, None, None]*(
            -np.sin(angles)[:, :, None]*coefficient[1:1+count][None, :, :]
            +np.cos(angles)[:, :, None]*coefficient[1+count:][None, :, :])
        return design, coefficient, residual, derivative

    half_width = 2*np.pi*settings.frequency_half_width_bins
    lower = np.maximum(-half_width, (settings.minimum_interior_frequency-frequencies0)*n)
    upper = np.minimum(half_width, (np.pi-settings.minimum_interior_frequency-frequencies0)*n)

    def objective(displacement):
        _, coefficient, residual, derivative = profile(frequencies0+displacement/n)
        rss = float(np.sum(residual**2))
        gradient = -2*np.einsum("nq,njq->j", residual, derivative, optimize=False)/n
        return rss/(q*n), gradient/(q*n)

    if count:
        result = minimize(objective, np.zeros(count), jac=True, method="L-BFGS-B",
            bounds=list(zip(lower, upper)), options=dict(ftol=settings.optimizer_ftol,
                gtol=settings.optimizer_gtol, maxiter=settings.maximum_frequency_iterations))
        frequency = frequencies0+result.x/n
        optimizer = dict(success=bool(result.success), message=str(result.message),
            iterations=int(result.nit), function_evaluations=int(result.nfev),
            frequency_scaled_displacement=result.x.tolist(),
            scaled_objective_gradient=np.asarray(result.jac).tolist(),
            at_lower_bound=np.isclose(result.x, lower, rtol=0, atol=1e-7).tolist(),
            at_upper_bound=np.isclose(result.x, upper, rtol=0, atol=1e-7).tolist())
    else:
        frequency = frequencies0
        optimizer = dict(success=True, message="constant-only LS", iterations=0,
            function_evaluations=0, frequency_scaled_displacement=[],
            scaled_objective_gradient=[], at_lower_bound=[], at_upper_bound=[])
    design, coefficient, residual, derivative = profile(frequency)
    gram = np.einsum("ni,nj->ij", design, design, optimize=False)
    # QR supplies the full fixed-design covariance without squaring the
    # numerical conditioning through inversion of the normal-equation Gram.
    _, triangular = qr(design, mode="economic")
    inverse_triangular = solve_triangular(triangular, np.eye(len(triangular)))
    coefficient_covariance = noise_sigma**2*np.einsum(
        "ij,kj->ik", inverse_triangular, inverse_triangular, optimize=False)
    coefficient_covariance = .5*(coefficient_covariance+coefficient_covariance.T)
    covariance = np.empty((0, 0))
    information = np.empty((0, 0))
    frequency_se = []
    condition = None
    score_distance = None
    information_status = "no frequency parameters"
    if count:
        flattened = derivative.reshape(n, count*q)
        nuisance = np.einsum("ni,ij->nj", design,
            lstsq(design, flattened, cond=settings.design_rcond, lapack_driver="gelsy")[0],
            optimize=False)
        efficient = (flattened-nuisance).reshape(n, count, q)
        information = np.einsum("njq,nkq->jk", efficient, efficient, optimize=False)
        information = .5*(information+information.T)
        eigenvalues = np.linalg.eigvalsh(information)
        condition = float(eigenvalues[-1]/eigenvalues[0]) if eigenvalues[0] > 0 else None
        if eigenvalues[-1] > 0 and eigenvalues[0] > settings.information_rcond*eigenvalues[-1]:
            covariance = noise_sigma**2*np.linalg.inv(information)
            covariance = .5*(covariance+covariance.T)
            frequency_se = np.sqrt(np.diag(covariance)).tolist()
            score = np.einsum("nq,njq->j", residual, derivative, optimize=False)
            local_step = np.linalg.solve(information, score)
            score_distance = float(local_step@np.linalg.solve(covariance, local_step))
            information_status = "local iid conditional efficient information only"
        else:
            covariance = np.empty((0, 0))
            information_status = "unresolved local frequency information; no inverse covariance returned"
    gaps = abs(frequency[:, None]-frequency[None, :]) if count else np.empty((0, 0))
    upper_pairs = np.triu_indices(count, 1)
    unresolved_pairs = [dict(first=int(i), second=int(j), gap=float(gaps[i, j]))
        for i, j in zip(*upper_pairs) if gaps[i, j] <= settings.initial_duplicate_bins*bin_width]
    rss = float(np.sum(residual**2))
    gram_condition = float(np.linalg.cond(gram))
    audit = dict(frequencies=frequency.tolist(), initial_frequencies=frequencies0.tolist(),
        initial_merge=merge, frequency_search_half_width=half_width/n,
        frequency_search_cells=[[float(f+lo/n), float(f+hi/n)]
                                for f, lo, hi in zip(frequencies0, lower, upper)],
        line_count=count, time_center=center, optimizer=optimizer,
        raw_sensor_rss=rss, raw_sensor_mse=rss/(q*n),
        temporal_gram_condition=gram_condition if np.isfinite(gram_condition) else None,
        amplitude_design_rank=design.shape[1], postfit_unresolved_frequency_pairs=unresolved_pairs,
        conditional_frequency_standard_errors=frequency_se,
        conditional_information_condition=condition, conditional_information_status=information_status,
        conditional_score_distance_squared=score_distance,
        frequency_covariance_is_calibrated=False, supplied_noise_sigma_raw=float(noise_sigma),
        sensor_noise_model="iid sigma_raw^2 I_q; not iid physical-mass coordinates",
        coefficient_covariance_scope="fixed frequency/design only; frequency/model selection and tail bias omitted",
        frequency_covariance_scope="full-raw free-line local tangent; omitted lines/tail/selection not calibrated",
        objective="unpenalized RAW sensor RSS/(q*N); full temporal Gram",
        frequency_units="radians per uniform sample", synthesized_harmonics=False,
        integer_labels_or_relations_used=False)
    for array in (frequency, coefficient, covariance, coefficient_covariance, gram):
        if not np.isfinite(array).all():
            raise FloatingPointError("nonfinite free-line parameter or diagnostic")
        array.setflags(write=False)
    return FreeLineModel(frequency, coefficient, center, covariance,
                         coefficient_covariance, gram, audit)


def fit_raw_spectral_ledger(y_train, y_validation, t_train, t_validation,
                            sensor_positions, noise_estimate, settings=SETTINGS):
    """Collect observed lines with bounded same-gate SSA proposal retries.

    No derived harmonic profile is ever fitted or removed. After rejection,
    accepted models remain untouched: fits have no shared mutable caches.
    """
    train, validation = np.asarray(y_train, dtype=float), np.asarray(y_validation, dtype=float)
    t_train, t_validation = np.asarray(t_train), np.asarray(t_validation)
    if (train.ndim != 2 or validation.shape != (train.shape[0], settings.n_validation)
            or train.shape[1] != settings.n_train
            or not np.array_equal(t_train, np.arange(settings.n_train))
            or not np.array_equal(t_validation, np.arange(settings.n_train,
                                             settings.n_train+settings.n_validation))):
        raise ValueError("predeclared TRAIN/VAL unit-time blocks required")
    spatial = NodalSpatialModel.from_positions(sensor_positions)
    if len(spatial.positions) != train.shape[0]:
        raise ValueError("sensor geometry and raw observations disagree")
    sigma = float(noise_estimate["sigma"])
    frontend_settings = Settings(n_train=settings.n_train, n_validation=settings.n_validation,
        n_future=settings.n_future, lag_window=settings.lag_window,
        frontend_mode="dynamic", dynamic_ranks=settings.dynamic_ranks,
        phase_metric_mode="field", temporal_gate_mode="iid_reference")
    model = fit_free_lines(train, t_train, [], sigma, settings)
    snapshots = [model]
    accepted_frontends, accepted_lines, stages = [], [], []
    status = "fixed_line_budget_reached"
    bin_width = 2*np.pi/settings.n_train
    for step in range(settings.maximum_lines):
        residual_train = train-model.predict(t_train)
        residual_validation = validation-model.predict(t_validation)
        before = float(np.mean(residual_validation**2))
        residual_mass = spatial.to_mass_coordinates(spatial.recover(residual_train))
        try:
            frontend = character_frontend(residual_mass, sigma, frontend_settings, spatial)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError, RuntimeError) as error:
            stages.append(dict(step=step+1, status="frontend_exception", exception=str(error)))
            status = "unresolved_frontend_exception"
            break
        if "candidate_frequency" not in frontend:
            stages.append(dict(step=step+1, status="no_supported_ssa_candidate", frontend=frontend,
                               validation_raw_sensor_mse_before=before))
            status = "no_supported_residual_ssa_candidate"
            break
        bank = frontend.pop("candidate_alternatives", [frontend])
        accepted = False
        for index, proposal in enumerate(bank):
            started = time.perf_counter()
            initializer = float(proposal["candidate_frequency"])
            stage = dict(step=step+1, alternative_index=index, supported_alternatives=len(bank),
                initializer_frequency=initializer, original_dynamic_rank=proposal.get("dynamic_rank"),
                original_dynamic_mode=proposal.get("alternative_dynamic_mode"),
                original_quality=proposal.get("metrics"), original_gates=proposal.get("gates"),
                proposal_source="original supported SSA/DMD bank on physical-mass free-line residual",
                accepted_frequency_count_before=len(model.frequencies),
                accepted_frequencies_before=model.frequencies.tolist(),
                validation_raw_sensor_mse_before=before)
            tentative, merge = merge_initial_frequencies(
                np.r_[model.frequencies, initializer], bin_width, settings)
            stage["initial_observed_frequency_merge"] = merge
            if len(tentative) <= len(model.frequencies):
                stage.update(status="merged_observed_duplicate_no_new_line",
                             accepted_state_unchanged=True, unresolved_close_distinct_lines_not_excluded=True)
                stages.append(stage)
                continue
            try:
                trial = fit_free_lines(train, t_train, tentative, sigma, settings)
                stage["joint_raw_fit"] = trial.audit
                after = float(np.mean((validation-trial.predict(t_validation))**2))
                stage.update(validation_raw_sensor_mse_after=after, validation_gain=before-after)
                reasons = []
                if not trial.audit["optimizer"]["success"]:
                    reasons.append("frequency_optimizer_not_converged")
                if trial.audit["postfit_unresolved_frequency_pairs"]:
                    reasons.append("postfit_line_collision_unresolved")
                if not before-after > 0:
                    reasons.append("nonpositive_validation_raw_field_gain")
                if reasons:
                    stage.update(status="rejected_candidate", rejection_reasons=reasons,
                                 accepted_state_unchanged=True)
                    stages.append(stage)
                    continue
                validation_mass = spatial.to_mass_coordinates(spatial.recover(residual_validation))
                measured_validation = apply_ssa_character(proposal, validation_mass)
                line = dict(line_id=len(accepted_lines), step=step+1,
                    initializer_frequency=initializer, fitted_frequency_at_acceptance=float(trial.frequencies[-1]),
                    prefix_snapshot_index=len(snapshots)-1,
                    train_phase_source="observed residual SSA filter; not fitted clocks",
                    validation_phase_source="same TRAIN-frozen filter on VAL free-line residual")
                measured = dict(proposal)
                measured["validation_observed_character"] = measured_validation["normalized_character"]
                accepted_lines.append(line)
                accepted_frontends.append(measured)
                model = trial
                snapshots.append(model)
                stage.update(status="accepted_observed_free_line", accepted_line_id=line["line_id"],
                    accepted_frequencies_after=model.frequencies.tolist(),
                    accepted_frequency_count_after=len(model.frequencies), elapsed_seconds=time.perf_counter()-started)
                stages.append(stage)
                accepted = True
                break
            except (ValueError, np.linalg.LinAlgError, FloatingPointError, RuntimeError) as error:
                stage.update(status="rejected_fit_exception", exception=f"{type(error).__name__}: {error}",
                             accepted_state_unchanged=True, elapsed_seconds=time.perf_counter()-started)
                stages.append(stage)
        if not accepted:
            status = "supported_alternatives_exhausted"
            break
    audit = dict(status=status, settings=asdict(settings), observed_line_count=len(model.frequencies),
        accepted_lines=accepted_lines, stages=stages, final_fit=model.audit,
        proposal_order="descending original represented_energy; stable rank/mode ties",
        inputs="Y_train,Y_validation,s,t and TRAIN-only raw iid-noise estimate",
        group_rank_or_integer_relations_inferred=False, integer_labels_or_synthetic_harmonics_used=False,
        automatic_full_spectral_support_or_phase_completeness_certified=False,
        selection="first viable original supported proposal with strictly positive RAW sensor VAL gain",
        validation_reused_adaptively=True, selection_or_noise_reference_statistically_calibrated=False,
        geometry_role="nodal P1 physical mass in SSA proposals; no law/profile supplied",
        raw_fit_noise_role="iid raw sensor likelihood; mass covariance equals sigma_raw^2 M_s",
        no_future_observations_or_truth_given_to_fit=True,
        candidate_new_line_collisions_reported_as_unresolved=True,
        no_expensive_peak_split_or_exhaustive_frequency_search=True,
        limitations=["local one-bin variable projection may miss another basin",
            "frequency uncertainty is conditional, not calibrated after discovery",
            "initial or fitted sub-quarter-bin pairs remain unresolved",
            "weak lines may fail unchanged short-window SSA gates",
            "finite48-line budget gives no infinite-harmonic tail certificate",
            "spectral lines are not independent dynamical phases"])
    return dict(model=model, snapshots=snapshots, frontends=accepted_frontends,
                spatial_model=spatial, audit=audit)


def frozen_parameter_arrays(fitted, t_train, t_validation):
    model, spatial = fitted["model"], fitted["spatial_model"]
    coefficient_mass = np.einsum("jq,kq->jk", model.coefficients_sensor,
                                 spatial.mass_sqrt, optimize=False)
    arrays = dict(frequencies=model.frequencies, coefficients_sensor_center=model.coefficients_sensor,
        coefficients_mass_center=coefficient_mass, time_center=np.asarray(model.time_center),
        complex_profiles_sensor_center=model.complex_profiles(),
        complex_profiles_sensor_origin=model.complex_profiles(origin_gauge=True),
        complex_profiles_mass_center=np.einsum("jq,kq->jk", model.complex_profiles(),
                                               spatial.mass_sqrt, optimize=False),
        complex_profiles_mass_origin=np.einsum("jq,kq->jk", model.complex_profiles(origin_gauge=True),
                                               spatial.mass_sqrt, optimize=False),
        physical_mass=spatial.physical_mass, spatial_mass_sqrt=spatial.mass_sqrt,
        sensor_positions=spatial.positions, temporal_gram=model.temporal_gram,
        conditional_frequency_covariance=model.frequency_covariance,
        conditional_coefficient_covariance_per_sensor=model.coefficient_covariance_per_sensor,
        predicted_train=model.predict(t_train), predicted_validation=model.predict(t_validation))
    for index, snapshot in enumerate(fitted["snapshots"]):
        arrays[f"snapshot_{index:03d}_frequencies"] = snapshot.frequencies
        arrays[f"snapshot_{index:03d}_coefficients_sensor_center"] = snapshot.coefficients_sensor
    if fitted["frontends"]:
        for key, output in (("lag_filter", "observed_lag_filters_mass"),
                ("signal_mean", "observed_signal_means_mass"),
                ("normalized_character", "observed_train_characters"),
                ("complex_coordinate", "observed_train_complex_coordinates"),
                ("validation_observed_character", "observed_validation_characters")):
            arrays[output] = np.stack([frontend[key] for frontend in fitted["frontends"]])
    for value in arrays.values():
        if not np.isfinite(value).all():
            raise FloatingPointError("nonfinite frozen ledger output")
    return arrays
