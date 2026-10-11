"""Prototype iterative conditional-mean SSA with an explicit spatial model.

This is an exploratory algorithm with predeclared finite budgets and
uncalibrated residual-model diagnostics. No physical phase labels or true
frequencies enter fitting. The default known polynomial spatial basis is an
explicit assumption; a geometry-only nodal model can be injected instead.
A finite accepted character set is never a completeness proof.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import pi

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.spatial import cKDTree
from scipy.stats import chi2

from frequency_fit import fit_joint_frequencies
from phase_lattice import infer_character_basis_global
from spatial_field import SPATIAL_MODEL, symmetric_sqrt
from ssa_character import extract_ssa_character, apply_ssa_character, apply_field_character, aligned_character_rows
from ssa_dynamic_character import extract_ssa_dynamic_character


@dataclass(frozen=True)
class Settings:
    n_train: int = 3200
    n_validation: int = 1600
    n_future: int = 1600
    lag_window: int = 32
    max_steps: int = 6
    max_relation_order: int = 6
    max_phase_rank: int = 4
    association_p_gate: float = 1e-8
    k_candidates: tuple[int, ...] = (32, 64)
    frontend_mode: str = "pair"
    joint_frequency_mode: str = "raw"
    dynamic_ranks: tuple[int, ...] = (4, 8, 16)
    phase_metric_mode: str = "field"
    temporal_gate_mode: str = "legacy"
    conditional_model_mode: str = "legacy_knn"
    enforce_validation_improvement: bool = False
    phase_relation_mode: str = "clock_diagnostic"
    phase_readout_refinement: str = "none"
    observer_windows: tuple[int, ...] = (128, 256, 512)
    fourier_sieve_mode: str = "legacy"


def noise_from_spatial_orthogonal_component(observed: np.ndarray,
                                          sampling_model=None) -> dict:
    """Exact iid-noise variance estimator when the field lies in known V."""
    y = np.asarray(observed, dtype=float)
    spatial = SPATIAL_MODEL if sampling_model is None else sampling_model
    design = spatial.design
    degrees = (y.shape[0]-design.shape[1])*y.shape[1]
    if degrees <= 0:
        raise ValueError("no spatial orthogonal noise degrees of freedom; supply noise_estimate explicitly")
    projector = np.einsum("ij,jk->ik", design, np.linalg.pinv(design),
                          optimize=False)
    orthogonal = y-np.einsum("ij,jt->it", projector, y, optimize=False)
    sigma_squared = float(np.sum(orthogonal**2)/degrees)
    return {"sigma": float(np.sqrt(sigma_squared)),
            "sigma_squared": sigma_squared,
            "degrees_of_freedom": degrees,
            "assumption": "raw iid Gaussian sensor noise and field exactly in known V"}


def _phase_embedding(frequencies: np.ndarray, times: np.ndarray) -> np.ndarray:
    angles = np.asarray(times)[:, None]*np.asarray(frequencies)[None, :]
    return np.concatenate((np.cos(angles), np.sin(angles)), axis=1)


class ClockModel:
    def __init__(self, times: np.ndarray, values: np.ndarray,
                 frequencies: np.ndarray, selected_k: int | None,
                 selection_errors: dict):
        self.times = np.asarray(times, dtype=float)
        self.values = np.asarray(values, dtype=float)
        self.frequencies = np.asarray(frequencies, dtype=float)
        self.selected_k = selected_k
        self.selection_errors = selection_errors
        self.mean = self.values.mean(axis=1)
        self.tree = (cKDTree(_phase_embedding(self.frequencies, self.times))
                     if len(self.frequencies) else None)

    def predict(self, times: np.ndarray, exclude_time_radius: int | None = None) -> np.ndarray:
        t = np.asarray(times, dtype=float)
        if self.tree is None:
            return np.broadcast_to(self.mean[:, None], (len(self.mean), len(t))).copy()
        k = min(self.selected_k, len(self.times))
        query_count = min(len(self.times), k+(2*exclude_time_radius+1
                                             if exclude_time_radius is not None else 0))
        distances, neighbours = self.tree.query(_phase_embedding(self.frequencies, t),
                                                k=query_count)
        if query_count == 1:
            neighbours = neighbours[:, None]
            distances = distances[:, None]
        if exclude_time_radius is not None:
            valid = abs(self.times[neighbours]-t[:, None]) > exclude_time_radius
            if np.any(np.sum(valid, axis=1) < k):
                raise ValueError("not enough phase neighbours outside the temporal exclusion interval")
            order = np.argsort(np.where(valid, distances, np.inf), axis=1)
            neighbours = np.take_along_axis(neighbours, order[:, :k], axis=1)
        return self.values[:, neighbours].mean(axis=2)

    def metadata(self) -> dict:
        return {"frequencies": self.frequencies.tolist(),
                "k": self.selected_k, "train_selection_mse": self.selection_errors,
        "same_neighbours_and_uniform_weights_for_all_mass_coordinates": True}


class HierarchicalClockModel:
    """Frozen sum of lower-dimensional factor fits and residual corrections.

    Empirical components are not asserted orthogonal. Every correction is
    fitted on the whole enlarged factor, using the residual of earlier fits.
    """

    def __init__(self, components: list[ClockModel],
                 phase_products: list[np.ndarray], exclusion_radius: int,
                 model_peak_frequencies: np.ndarray,
                 conditional_model_mode: str = "legacy_knn",
                 factor_extra_labels=None, fourier_sieve_mode: str = "legacy"):
        self.components = tuple(components)
        self.phase_products = tuple(np.asarray(b, dtype=int).copy() for b in phase_products)
        self.exclusion_radius = exclusion_radius
        self.model_peak_frequencies = np.asarray(model_peak_frequencies, dtype=float).copy()
        self.selected_k = self.components[-1].selected_k
        self.frequencies = self.components[-1].frequencies
        self.conditional_model_mode = conditional_model_mode
        self.factor_extra_labels = factor_extra_labels or [[] for _ in phase_products]
        self.fourier_sieve_mode = fourier_sieve_mode

    def predict(self, times: np.ndarray, exclude_time_radius: int | None = None) -> np.ndarray:
        predictions = [component.predict(times, exclude_time_radius)
                       for component in self.components]
        return np.sum(predictions, axis=0)

    def predict_training(self, times: np.ndarray) -> np.ndarray:
        return self.predict(times, exclude_time_radius=self.exclusion_radius
                            if self.conditional_model_mode == "legacy_knn" else None)

    def metadata(self) -> dict:
        return {"structure": "mean plus nested-factor residual corrections",
                "conditional_model_mode": self.conditional_model_mode,
                "fourier_sieve_mode": self.fourier_sieve_mode,
                "frequencies": self.frequencies.tolist(),
                "k": self.selected_k,
                "components": [component.metadata() for component in self.components],
                "factor_products_in_original_acquired_coordinates": [b.tolist() for b in self.phase_products],
                "discovered_extra_labels_by_factor": self.factor_extra_labels,
                "model_peak_frequencies": self.model_peak_frequencies.tolist(),
                "training_temporal_exclusion_radius": self.exclusion_radius
                    if self.conditional_model_mode == "legacy_knn" else None,
                "empirical_orthogonality_not_claimed": True}


def fit_hierarchical_model(times: np.ndarray, values: np.ndarray,
                           phase_products: list[np.ndarray],
                           acquired_frequencies: np.ndarray,
                           k_candidates=(32, 64), exclusion_radius: int = 128,
                           conditional_model_mode: str = "legacy_knn",
                           factor_extra_labels=None, fourier_sieve_mode: str = "legacy"
                           ) -> HierarchicalClockModel:
    # Refit every older factor at the jointly updated frequencies, keeping
    # its original dimension. Residual targets are recomputed coherently.
    components = [fit_clock_model(times, values, np.empty(0), k_candidates)]
    partial = components[0].predict(times)
    factor_extra_labels = factor_extra_labels or [[] for _ in phase_products]
    for factor_index, products_matrix in enumerate(phase_products):
        count = products_matrix.shape[1]
        frequencies = np.angle(np.exp(1j*np.einsum(
            "ij,j->i", products_matrix, acquired_frequencies[:count], optimize=False)))
        if conditional_model_mode == "legacy_knn":
            correction = fit_clock_model(times, values-partial, frequencies, k_candidates)
        elif conditional_model_mode == "fourier_sieve":
            from phase_factor_regression import fit_fourier_factor
            correction = fit_fourier_factor(times, values-partial, frequencies,
                                             extra_labels=factor_extra_labels[factor_index], sieve_mode=fourier_sieve_mode)
        else:
            raise ValueError("unknown conditional model mode")
        components.append(correction)
        partial += correction.predict(times, exclude_time_radius=exclusion_radius
                                      if conditional_model_mode == "legacy_knn" else None)
    return HierarchicalClockModel(components, phase_products, exclusion_radius,
                                  acquired_frequencies, conditional_model_mode, factor_extra_labels, fourier_sieve_mode)


def fit_clock_model(times: np.ndarray, values: np.ndarray,
                    frequencies: np.ndarray, k_candidates=(32, 64)) -> ClockModel:
    t = np.asarray(times, dtype=float)
    y = np.asarray(values, dtype=float)
    frequencies = np.asarray(frequencies, dtype=float)
    if not len(frequencies):
        return ClockModel(t, y, frequencies, None, {})
    split = max(100, int(.72*len(t)))
    tree = cKDTree(_phase_embedding(frequencies, t[:split]))
    query = _phase_embedding(frequencies, t[split:])
    errors = {}
    for desired in k_candidates:
        k = min(desired, split)
        _, indices = tree.query(query, k=k)
        if k == 1:
            indices = indices[:, None]
        predicted = y[:, :split][:, indices].mean(axis=2)
        errors[str(desired)] = float(np.mean(np.sum((y[:, split:]-predicted)**2,
                                                 axis=0)))
    selected = min(errors, key=errors.get)
    return ClockModel(t, y, frequencies, int(selected), errors)


def _field_from_mass(mass: np.ndarray, positions: np.ndarray,
                     sampling_model=None) -> np.ndarray:
    spatial = SPATIAL_MODEL if sampling_model is None else sampling_model
    coefficients = np.linalg.solve(spatial.mass_sqrt, mass)
    return spatial.evaluate(coefficients, positions)


def _circular_gap(first: float, second: float) -> float:
    return float(abs(np.angle(np.exp(1j*(first-second)))))


def family_membership(candidate: float, frequencies: np.ndarray,
                      order: int, tolerance: float) -> dict:
    """Meet-in-the-middle closest bounded integer combination of old lines."""
    old = np.asarray(frequencies, dtype=float)
    if not len(old):
        return {"within_old_module": False, "gap": None, "coefficients": []}
    midpoint = len(old)//2
    left = np.asarray(list(product(range(-order, order+1), repeat=midpoint)), dtype=int)
    right = np.asarray(list(product(range(-order, order+1), repeat=len(old)-midpoint)), dtype=int)
    if midpoint == 0:
        left = np.empty((1, 0), dtype=int)
    left_angle = np.mod(left@old[:midpoint], 2*pi)
    right_angle = np.mod(right@old[midpoint:], 2*pi)
    permutation = np.argsort(right_angle)
    sorted_right = right_angle[permutation]
    best = (float("inf"), None)
    for left_coefficients, angle in zip(left, left_angle):
        target = (candidate-angle)%(2*pi)
        insertion = int(np.searchsorted(sorted_right, target))
        for index in (insertion-1, insertion, 0, len(sorted_right)-1):
            selected = index%len(sorted_right)
            coefficients = np.r_[left_coefficients, right[permutation[selected]]]
            gap = _circular_gap(candidate, float(np.dot(coefficients, old)))
            if gap < best[0]:
                best = (gap, coefficients.copy())
    return {"within_old_module": bool(best[0] <= tolerance),
            "gap": float(best[0]), "coefficients": best[1].tolist(),
            "tolerance": tolerance, "max_coefficient": order}


def refine_candidate_frequency(residual_mass: np.ndarray, frontend: dict,
                               n_train: int) -> dict:
    metrics = frontend["metrics"]
    slope = float(frontend["candidate_frequency"])
    dmd = float(metrics["frequency_temporal_dmd"])
    if (not 0 < slope < pi or _circular_gap(slope, dmd) > .015):
        center, source = dmd, "temporal_dmd_due_to_unwrap_or_slope_disagreement"
    else:
        center, source = slope, "ssa_phase_slope"
    if not 0 < center < pi:
        raise ValueError("SSA character has no interior positive frequency")
    n = residual_mass.shape[1]
    centered = (residual_mass-residual_mass.mean(axis=1, keepdims=True))*np.hanning(n)
    times = np.arange(n)

    def negative_power(frequency: float) -> float:
        wave = np.exp(-1j*frequency*times)
        coefficient = np.einsum("it,t->i", centered, wave, optimize=False)
        return -float(np.sum(abs(coefficient)**2))

    width = 2*pi/n_train
    low, high = max(1e-8, center-width), min(pi-1e-8, center+width)
    grid = np.linspace(low, high, 11)
    values = [negative_power(float(value)) for value in grid]
    best = int(np.argmin(values))
    interval = (grid[max(0, best-1)], grid[min(len(grid)-1, best+1)])
    result = minimize_scalar(negative_power, bounds=interval,
                             method="bounded", options={"xatol": 1e-11})
    frequency = float(result.x)
    return {"frequency": frequency, "center": center, "center_source": source,
            "ssa_slope": slope, "ssa_dmd": dmd,
            "frequency_refinement_interval": [low, high],
            "refined_hann_power": -float(result.fun)}


def fixed_sinusoid_association(raw_validation_residual: np.ndarray,
                               times: np.ndarray, frequency: float,
                               noise_sigma: float) -> dict:
    """Held-out raw-sensor fixed-line diagnostic; selection is on train only."""
    t = np.asarray(times, dtype=float)
    y = np.asarray(raw_validation_residual, dtype=float).T
    angles = frequency*t
    design = np.column_stack((np.ones(len(t)), np.cos(angles), np.sin(angles)))
    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    null_residual = y-y.mean(axis=0)
    alternative_residual = y-np.einsum("ij,jk->ik", design, coefficients,
                                       optimize=False)
    improvement = float(np.sum(null_residual**2)-np.sum(alternative_residual**2))
    statistic = improvement/noise_sigma**2
    degrees = 2*y.shape[1]
    return {"heldout_raw_sensor_rss_improvement": improvement,
            "statistic_using_spatial_orthogonal_noise": statistic,
            "chi_square_reference_df": degrees,
            "chi_square_reference_p": float(chi2.sf(max(statistic, 0), degrees)),
            "status": "diagnostic_only_conditional_mean_bias_and_selection_not_calibrated"}


def compact_frontend(frontend: dict) -> dict:
    return {key: frontend.get(key) for key in
            ("status", "reason", "candidate_frequency", "metrics", "gates",
             "gate_status", "window", "hankel_columns", "dynamic_rank",
             "dynamic_proposals", "raw_lag1_correlation", "phase_metric_mode",
             "filter_input_coordinates", "temporal_reference", "readout_source",
             "phase_readout_method", "observer_refinement")}


def character_frontend(residual: np.ndarray, noise_sigma: float,
                       settings: Settings, sampling_model=None) -> dict:
    spatial = SPATIAL_MODEL if sampling_model is None else sampling_model
    weights = np.ones(len(residual))/len(residual)
    if settings.phase_metric_mode == "field":
        inference_values = residual
        whitener = None
        inference_noise_covariance = spatial.noise_covmass(noise_sigma)
    elif settings.phase_metric_mode == "noise":
        covariance_at_unit_sigma = spatial.noise_covmass(1.)
        whitener = symmetric_sqrt(covariance_at_unit_sigma, inverse=True)
        inference_values = np.einsum("ij,jt->it", whitener, residual, optimize=False)
        inference_noise_covariance = noise_sigma**2*np.eye(len(residual))
    else:
        raise ValueError("unknown phase metric mode")
    if settings.frontend_mode == "pair":
        fitted = extract_ssa_character(inference_values, weights, settings.lag_window,
                                       noise_sigma=noise_sigma)
    elif settings.frontend_mode == "dynamic":
        fitted = extract_ssa_dynamic_character(
            inference_values, weights, settings.lag_window, noise_sigma=noise_sigma,
            ranks=settings.dynamic_ranks, temporal_gate_mode=settings.temporal_gate_mode,
            noise_covariance=inference_noise_covariance)
    else:
        raise ValueError("unknown SSA frontend mode")
    if whitener is not None and "lag_filter" in fitted:
        fitted["lag_filter"] = np.einsum("li,ij->lj", fitted["lag_filter"],
                                         whitener, optimize=False)
        fitted["signal_mean"] = residual.mean(axis=1)
    fitted["phase_metric_mode"] = settings.phase_metric_mode
    fitted["filter_input_coordinates"] = "physical mass coordinates; whitening pulled back"
    return fitted


def _joint_lattice(raw_train: np.ndarray, proposed: np.ndarray,
                   settings: Settings, noise_sigma: float | None = None) -> dict:
    try:
        if settings.joint_frequency_mode == "raw":
            fitted = fit_joint_frequencies(raw_train, proposed, half_width_cycles=.35)
        elif settings.joint_frequency_mode == "harmonic_nuisance":
            from nuisance_frequency_fit import fit_joint_frequencies_with_harmonic_nuisance
            fitted = fit_joint_frequencies_with_harmonic_nuisance(
                raw_train, proposed, half_width_cycles=.35, noise_sigma=noise_sigma)
        else:
            raise ValueError("unknown joint frequency fit mode")
        radius = float(np.sqrt(chi2.ppf(.99, len(proposed))))
        joint = {key: fitted[key] for key in
                 ("optimizer_success", "optimizer_message", "stationary_score_distance_squared",
                  "stationary_in_frequency_uncertainty", "information_condition",
                  "noise_variance_estimate", "covariance_status")}
        joint.update({"frequencies": fitted["frequencies"].tolist(),
                      "covariance": fitted["covariance"].tolist(),
                      "standard_errors": fitted["frequency_standard_errors"].tolist(),
                      "confidence_radius": radius})
        if settings.joint_frequency_mode == "harmonic_nuisance":
            joint["nuisance_audit"] = {key: value for key, value in fitted.items()
                if key not in ("frequencies", "covariance", "coefficients", "residual",
                               "frequency_standard_errors") and key not in joint}
        lattice = infer_character_basis_global(
            fitted["frequencies"], tolerance=1e-12,
            max_order=settings.max_relation_order, max_rank=settings.max_phase_rank,
            frequency_covariance=fitted["covariance"], confidence_radius=radius)
        return {"joint_fit": joint, "lattice": lattice,
                "fitted_frequencies": fitted["frequencies"],
                "usable_lattice": bool(lattice.get("rank") is not None
                                       and fitted["stationary_in_frequency_uncertainty"])}
    except (FloatingPointError, np.linalg.LinAlgError, ValueError,
            ArithmeticError, RuntimeError) as error:
        return {"joint_fit": {"failure": f"{type(error).__name__}: {error}"},
                "lattice": {"rank": None, "reason": "joint frequency fit failed"},
                "fitted_frequencies": proposed,
                "usable_lattice": False}


def _observed_acquired_characters(frontends: list, prefix_models: list,
                                  mass_validation: np.ndarray,
                                  validation_times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply each acquired filter to its own pre-acquisition residual model."""
    train = aligned_character_rows([frontend["normalized_character"] for frontend in frontends])
    validation = aligned_character_rows([
        apply_field_character(frontend, mass_validation,
            prefix_models[index].predict(validation_times))["normalized_character"]
        for index, frontend in enumerate(frontends)])
    return train, validation


def _refine_relation_readouts(candidate_frontend: dict, candidate_frequency: float,
                              old_frontends: list, old_frequencies: np.ndarray,
                              mass_train: np.ndarray, mass_validation: np.ndarray,
                              residual_train: np.ndarray, residual_validation: np.ndarray,
                              relations_candidate_first: np.ndarray,
                              settings: Settings) -> dict:
    """Train-inner selected convolution observer; outer validation is a gate."""
    from phase_readout_refinement import fit_field_character_observer
    from phase_factor_regression import observed_relation_check
    split = max(100, int(.72 * settings.n_train))
    sources_train = [residual_train] + [mass_train] * len(old_frontends)
    sources_validation = [residual_validation] + [mass_validation] * len(old_frontends)
    frequencies = np.r_[candidate_frequency, old_frequencies]
    relations = np.asarray(relations_candidate_first, dtype=int)
    grid = []
    for window in settings.observer_windows:
        if window >= min(split, settings.n_train-split) or window > settings.n_validation:
            grid.append({"window": window, "failure": "unavailable inner-training or validation window"})
            continue
        try:
            observers = [fit_field_character_observer(values[:, :split], frequency, window)
                         for values, frequency in zip(sources_train, frequencies)]
            train_rows = aligned_character_rows([observer.apply(values[:, :split])["normalized_character"]
                                                  for observer, values in zip(observers, sources_train)])
            holdout_rows = aligned_character_rows([observer.apply(values[:, split:])["normalized_character"]
                                                    for observer, values in zip(observers, sources_train)])
            check = observed_relation_check(train_rows, holdout_rows, relations)
            score = min(min(row["train_coherence"], row["validation_coherence"],
                            row["validation_frozen_origin_agreement"]) for row in check["relations"])
            grid.append({"window": window, "minimum_inner_relation_agreement": score,
                         "inner_training_relation_check": check})
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
            grid.append({"window": window, "failure": f"{type(error).__name__}: {error}"})
    choices = [row for row in grid if "minimum_inner_relation_agreement" in row]
    audit = {"selection_grid": grid, "selection_data": "inner-training only",
             "frequencies_and_residual_model_estimated_on_full_training": True,
             "validation_or_future_used_in_window_selection": False,
             "refined_filter_returned_as_phase_output_after_atomic_commit": True,
             "method": "Hann matched field-profile convolution at data-learned frequency"}
    if not choices:
        return {"supported": False, "audit": dict(audit, status="no_feasible_observer_window")}
    selected = max(choices, key=lambda row: (row["minimum_inner_relation_agreement"], -row["window"]))
    window = selected["window"]
    observers = [fit_field_character_observer(values, frequency, window)
                 for values, frequency in zip(sources_train, frequencies)]
    refined = []
    for index, (observer, values) in enumerate(zip(observers, sources_train)):
        original = candidate_frontend if index == 0 else old_frontends[index-1]
        frontend = dict(original)
        output = observer.apply(values)
        frontend.update(lag_filter=observer.weights[:, None] *
            np.exp(-1j*observer.frequency*np.arange(window))[:, None] *
            np.conj(observer.spatial_profile)[None, :], signal_mean=observer.signal_mean,
            window=window, hankel_columns=len(output["phase"]), phase=output["phase"],
            normalized_character=output["normalized_character"],
            candidate_frequency=observer.frequency,
            readout_source="residual" if index == 0 else "raw_field",
            phase_readout_method="Hann matched field-profile convolution",
            observer_refinement=observer.metadata())
        refined.append(frontend)
    train_rows = aligned_character_rows([frontend["normalized_character"] for frontend in refined])
    validation_rows = aligned_character_rows([apply_ssa_character(frontend, values)["normalized_character"]
                                               for frontend, values in zip(refined, sources_validation)])
    check = observed_relation_check(train_rows, validation_rows, relations)
    audit.update(selected_window=window, validation_observed_relation_check=check,
                 status="supported" if check["all_relations_supported"] else "unresolved_near_relation",
                 old_generator_readout_source="raw_field", candidate_readout_source="current_residual")
    return {"supported": check["all_relations_supported"], "audit": audit,
            "candidate_frontend": refined[0], "old_frontends": refined[1:],
            "candidate_validation_character": validation_rows[0],
            "old_train_characters": train_rows[1:], "old_validation_characters": validation_rows[1:]}


def fit_iterative_ssa(raw_train: np.ndarray, raw_validation: np.ndarray,
                      settings: Settings, sampling_model=None,
                      noise_estimate: dict | None = None) -> dict:
    """Fit without physical truth; return serializable audit and frozen models."""
    y_train = np.asarray(raw_train, dtype=float)
    y_validation = np.asarray(raw_validation, dtype=float)
    if settings.phase_relation_mode not in ("clock_diagnostic", "observed"):
        raise ValueError("phase_relation_mode must be clock_diagnostic or observed")
    if settings.phase_readout_refinement not in ("none", "hann_inner"):
        raise ValueError("phase_readout_refinement must be none or hann_inner")
    spatial = SPATIAL_MODEL if sampling_model is None else sampling_model
    if (y_train.shape != (len(spatial.positions), settings.n_train)
            or y_validation.shape != (len(spatial.positions), settings.n_validation)):
        raise ValueError("record sizes disagree with settings")
    time_train = np.arange(settings.n_train)
    time_validation = np.arange(settings.n_train,
                                settings.n_train+settings.n_validation)
    if noise_estimate is None:
        noise = noise_from_spatial_orthogonal_component(y_train, spatial)
    else:
        noise = dict(noise_estimate)
        sigma = float(noise.get("sigma", float("nan")))
        if not np.isfinite(sigma) or sigma <= 0:
            raise ValueError("injected noise_estimate requires a positive finite sigma")
        noise.setdefault("sigma_squared", sigma**2)
        noise.setdefault("source", "injected external sensor-noise estimate")
    mass_train = spatial.to_mass_coordinates(spatial.recover(y_train))
    mass_validation = spatial.to_mass_coordinates(spatial.recover(y_validation))
    acquired = np.empty(0)
    models = []
    factor_products = []
    factor_extra_labels = []
    frozen_frontends = []
    family_readouts = []
    stages = []
    model = fit_hierarchical_model(time_train, mass_train, factor_products,
                                  acquired, settings.k_candidates, settings.lag_window,
                                  settings.conditional_model_mode,
                                  fourier_sieve_mode=settings.fourier_sieve_mode)
    models.append(model)
    final_lattice = None
    final_status = "max_steps_reached"
    for step in range(settings.max_steps):
        candidate_old_model = model
        pending_frontends = frozen_frontends
        residual_train = mass_train-model.predict_training(time_train)
        residual_validation = mass_validation-model.predict(time_validation)
        try:
            frontend = character_frontend(residual_train, noise["sigma"], settings, spatial)
        except (FloatingPointError, np.linalg.LinAlgError, ValueError,
                RuntimeError) as error:
            stages.append({"step": step+1, "status": "ssa_frontend_exception",
                           "exception": f"{type(error).__name__}: {error}"})
            final_status = "no_new_validated_character"
            break
        stage = {"step": step+1, "frontend": compact_frontend(frontend),
                 "old_model": model.metadata(),
                 "validation_residual_mass_mse_before": float(np.mean(np.sum(residual_validation**2, axis=0)))}
        if "candidate_frequency" not in frontend:
            stage["status"] = "no_ssa_character_candidate"
            stages.append(stage)
            final_status = "no_new_validated_character"
            break
        proposed = refine_candidate_frequency(residual_train, frontend, settings.n_train)
        candidate = proposed["frequency"]
        stage["candidate"] = proposed
        membership = family_membership(candidate, acquired, settings.max_relation_order,
                                       2*pi/settings.n_train)
        candidate_validation_character = None
        old_train_characters = old_validation_characters = None
        if settings.phase_relation_mode == "observed":
            candidate_validation_character = apply_ssa_character(
                frontend, residual_validation)["normalized_character"]
            if len(frozen_frontends):
                old_train_characters, old_validation_characters = _observed_acquired_characters(
                    frozen_frontends, models, mass_validation, time_validation)
        if final_lattice and (settings.conditional_model_mode == "fourier_sieve"
                              or settings.phase_relation_mode == "observed"):
            from phase_factor_regression import associate_observed_character_with_factor, phase_product
            association_arguments = {}
            association_extra_labels = list(factor_extra_labels[-1])
            if settings.phase_relation_mode == "observed":
                factor_labels = np.asarray(final_lattice["products"], dtype=int)
                if membership["within_old_module"]:
                    proposed_factor_label = np.einsum("i,ij->j",
                        np.asarray(membership["coefficients"], dtype=int),
                        np.asarray(final_lattice["peak_labels"], dtype=int), optimize=False)
                    association_extra_labels.append(proposed_factor_label.tolist())
                    stage["frequency_proposed_factor_label_for_observed_check"] = proposed_factor_label.tolist()
                association_arguments = {
                    "factor_characters": phase_product(old_train_characters, factor_labels),
                    "validation_factor_characters": phase_product(old_validation_characters, factor_labels),
                    "validation_character": candidate_validation_character}
            polynomial_association = associate_observed_character_with_factor(
                frontend["normalized_character"], model.frequencies,
                association_extra_labels, **association_arguments)
            stage["old_factor_phase_polynomial_association"] = polynomial_association
            if (settings.phase_readout_refinement == "hann_inner" and settings.phase_relation_mode == "observed"
                    and membership["within_old_module"] and not polynomial_association["within_old_factor"]):
                relations = [np.r_[1, -np.asarray(membership["coefficients"], dtype=int)]]
                relations.extend(np.r_[0, relation] for relation in final_lattice.get("relations", []))
                refinement = _refine_relation_readouts(frontend, candidate, frozen_frontends, acquired,
                    mass_train, mass_validation, residual_train, residual_validation,
                    np.asarray(relations), settings)
                stage["phase_readout_refinement_attempt"] = refinement["audit"]
                if refinement["supported"]:
                    pending_frontends = refinement["old_frontends"]
                    frontend = refinement["candidate_frontend"]
                    candidate_validation_character = refinement["candidate_validation_character"]
                    old_train_characters = refinement["old_train_characters"]
                    old_validation_characters = refinement["old_validation_characters"]
                    relation_check = refinement["audit"]["validation_observed_relation_check"]
                    verified = relation_check["relations"][0]
                    factor_label = np.einsum("i,ij->j", np.asarray(membership["coefficients"], dtype=int),
                        np.asarray(final_lattice["peak_labels"], dtype=int), optimize=False)
                    stage["old_factor_association_before_refinement"] = polynomial_association
                    polynomial_association = {"within_old_factor": True,
                        "label": factor_label.tolist(), "coherence": verified["train_coherence"],
                        "observed_relation_check": relation_check, "coherence_threshold": .95,
                        "association_mode": "observed_frozen_refined_field_phase_products",
                        "status": "bounded frequency-proposed relation supported by refined observed phase products; heuristic"}
                    stage["old_factor_phase_polynomial_association"] = polynomial_association
                    stage["frontend"] = compact_frontend(frontend)
            if polynomial_association["within_old_factor"]:
                old_frequency_membership = membership
                coefficient = np.asarray(polynomial_association["label"], dtype=int) @ np.asarray(final_lattice["products"], dtype=int)
                membership = {"within_old_module": True,
                              "coefficients": coefficient.tolist(),
                              "source": polynomial_association["association_mode"],
                              "phase_polynomial_coherence": polynomial_association["coherence"],
                              "bounded_frequency_membership": old_frequency_membership,
                              "exact_module_membership_certified": False}
            elif settings.phase_relation_mode == "observed" and membership["within_old_module"]:
                stage["old_module_membership"] = membership
                stage["status"] = "unresolved_near_relation"
                stage["near_relation_reason"] = "bounded frequency relation lacks observed train/validation phase-product support"
                stage["accepted_state_rolled_back"] = True
                stages.append(stage)
                final_status = "unresolved_near_relation"
                break
        stage["old_module_membership"] = membership
        if membership["within_old_module"]:
            if settings.conditional_model_mode == "fourier_sieve" and final_lattice:
                extra = np.asarray(membership["coefficients"], dtype=int) @ np.asarray(final_lattice["peak_labels"], dtype=int)
                tentative_extra = [list(map(list, labels)) for labels in factor_extra_labels]
                tentative_extra[-1] = tentative_extra[-1] + [extra.tolist()]
                stage["family_decoder_enrichment_label"] = extra.tolist()
                try:
                    enriched_model = fit_hierarchical_model(
                        time_train, mass_train, factor_products, model.model_peak_frequencies,
                        settings.k_candidates, settings.lag_window,
                        settings.conditional_model_mode, tentative_extra, settings.fourier_sieve_mode)
                    enriched_residual = mass_validation-enriched_model.predict(time_validation)
                    enriched_mse = float(np.mean(np.sum(enriched_residual**2, axis=0)))
                    stage["validation_residual_mass_mse_after"] = enriched_mse
                    stage["validation_mse_improvement"] = stage["validation_residual_mass_mse_before"]-enriched_mse
                    stage["validation_gain_positive"] = bool(stage["validation_mse_improvement"] > 0)
                    stage["tentative_clock_model_after"] = enriched_model.metadata()
                    if stage["validation_gain_positive"]:
                        model = enriched_model
                        models[-1] = enriched_model
                        factor_extra_labels = tentative_extra
                        frozen_frontends = pending_frontends
                        if "phase_readout_refinement_attempt" in stage:
                            family_readouts.append({"frontend": frontend,
                                "prefix_model": candidate_old_model, "factor_label": extra.copy(),
                                "step": step+1})
                        stage["status"] = "accepted_family_character_decoder_enrichment"
                        stage["accepted_character_count_after_enrichment"] = len(frozen_frontends)
                        stage["clock_model_after"] = model.metadata()
                        stages.append(stage)
                        continue
                    stage["status"] = "unresolved_family_tail_no_decoder_improvement"
                    stage["accepted_state_rolled_back"] = True
                except (ValueError, np.linalg.LinAlgError, FloatingPointError, RuntimeError) as error:
                    stage["status"] = "unresolved_family_decoder_failure"
                    stage["conditional_model_failure"] = f"{type(error).__name__}: {error}"
                    stage["accepted_state_rolled_back"] = True
                stages.append(stage)
                final_status = "unresolved_family_tail"
                break
            if (settings.conditional_model_mode == "legacy_knn"
                    and model.selected_k != 16 and len(acquired)):
                retry_model = fit_hierarchical_model(
                    time_train, mass_train, factor_products, model.model_peak_frequencies,
                    k_candidates=(16,), exclusion_radius=settings.lag_window)
                retry_residual = mass_train-retry_model.predict_training(time_train)
                retry_frontend = character_frontend(retry_residual, noise["sigma"], settings, spatial)
                if "candidate_frequency" in retry_frontend:
                    retry_proposed = refine_candidate_frequency(retry_residual,
                                                                retry_frontend,
                                                                settings.n_train)
                    retry_membership = family_membership(
                        retry_proposed["frequency"], acquired,
                        settings.max_relation_order, 2*pi/settings.n_train)
                    stage["family_tail_retry"] = {
                        "model": retry_model.metadata(),
                        "frontend": compact_frontend(retry_frontend),
                        "candidate": retry_proposed,
                        "membership": retry_membership}
                    if not retry_membership["within_old_module"]:
                        candidate_old_model = retry_model
                        if not settings.enforce_validation_improvement:
                            model = retry_model
                            models[-1] = retry_model
                        frontend = retry_frontend
                        stage["initial_frontend_before_retry"] = stage["frontend"]
                        stage["frontend"] = compact_frontend(retry_frontend)
                        candidate = retry_proposed["frequency"]
                        stage["candidate"] = retry_proposed
                        stage["old_module_membership"] = retry_membership
                        residual_validation = mass_validation-candidate_old_model.predict(time_validation)
                        stage["old_model"] = candidate_old_model.metadata()
                        stage["validation_residual_mass_mse_before"] = float(
                            np.mean(np.sum(residual_validation**2, axis=0)))
                else:
                    stage["family_tail_retry"] = {"frontend": compact_frontend(retry_frontend),
                                                  "status": "no_candidate_on_retry"}
            if stage["old_module_membership"]["within_old_module"]:
                stage["status"] = "unresolved_family_tail"
                stages.append(stage)
                final_status = "unresolved_family_tail"
                break
        raw_validation_prediction = _field_from_mass(candidate_old_model.predict(time_validation),
                                                     spatial.positions, spatial)
        association = fixed_sinusoid_association(
            y_validation-raw_validation_prediction, time_validation,
            candidate, noise["sigma"])
        stage["heldout_association"] = association
        if association["chi_square_reference_p"] >= settings.association_p_gate:
            stage["status"] = "no_new_validated_character"
            stages.append(stage)
            final_status = "no_new_validated_character"
            break
        tentative = np.r_[acquired, candidate]
        joint = _joint_lattice(y_train, tentative, settings, noise_sigma=noise["sigma"])
        stage["joint_frequency_fit"] = joint["joint_fit"]
        stage["lattice"] = joint["lattice"]
        if settings.phase_relation_mode == "observed" and joint["usable_lattice"]:
            from phase_factor_regression import observed_relation_check
            candidate_train = np.asarray(frontend["normalized_character"])
            train_rows = aligned_character_rows(list(old_train_characters) + [candidate_train]
                            if len(frozen_frontends) else [candidate_train])
            validation_rows = aligned_character_rows(list(old_validation_characters) + [candidate_validation_character]
                                 if len(frozen_frontends) else [candidate_validation_character])
            relation_check = observed_relation_check(
                train_rows, validation_rows, np.asarray(joint["lattice"].get("relations", []), dtype=int))
            stage["proposed_lattice_observed_phase_relation_check"] = relation_check
            if not relation_check["all_relations_supported"] and settings.phase_readout_refinement == "hann_inner":
                relations = np.asarray(joint["lattice"].get("relations", []), dtype=int)
                candidate_first = relations[:, np.r_[len(frozen_frontends), np.arange(len(frozen_frontends))]]
                refinement = _refine_relation_readouts(frontend, float(joint["fitted_frequencies"][-1]),
                    frozen_frontends, np.asarray(joint["fitted_frequencies"][:-1]), mass_train, mass_validation,
                    residual_train, residual_validation, candidate_first, settings)
                stage["phase_readout_refinement_attempt"] = refinement["audit"]
                if refinement["supported"]:
                    pending_frontends = refinement["old_frontends"]
                    frontend = refinement["candidate_frontend"]
                    candidate_validation_character = refinement["candidate_validation_character"]
                    old_train_characters = refinement["old_train_characters"]
                    old_validation_characters = refinement["old_validation_characters"]
                    relation_check = observed_relation_check(
                        aligned_character_rows(list(old_train_characters) + [frontend["normalized_character"]]),
                        aligned_character_rows(list(old_validation_characters) + [candidate_validation_character]), relations)
                    stage["proposed_lattice_observed_phase_relation_check"] = relation_check
                    stage["frontend"] = compact_frontend(frontend)
            if not relation_check["all_relations_supported"]:
                stage["status"] = "unresolved_near_relation"
                stage["near_relation_reason"] = "proposed lattice relation lacks observed train/validation phase-product support"
                stage["accepted_state_rolled_back"] = True
                stages.append(stage)
                final_status = "unresolved_near_relation"
                break
        candidate_acquired = joint["fitted_frequencies"] if joint["usable_lattice"] else tentative
        if joint["usable_lattice"]:
            candidate_lattice = joint["lattice"]
            products_matrix = np.asarray(candidate_lattice["products"], dtype=int)
            model_frequencies = np.angle(np.exp(1j*np.einsum(
                "ij,j->i", np.asarray(candidate_lattice["peak_labels"], dtype=int),
                np.asarray(candidate_lattice["basis"], dtype=float), optimize=False)))
        else:
            candidate_lattice = None
            products_matrix = np.eye(len(candidate_acquired), dtype=int)
            model_frequencies = candidate_acquired
        tentative_factor_products = factor_products + [products_matrix]
        tentative_extra_labels = factor_extra_labels + [[]]
        try:
            candidate_model = fit_hierarchical_model(
                time_train, mass_train, tentative_factor_products,
                model_frequencies, settings.k_candidates, settings.lag_window,
                settings.conditional_model_mode, tentative_extra_labels, settings.fourier_sieve_mode)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError, RuntimeError) as error:
            stage["status"] = "rejected_conditional_model_failure"
            stage["conditional_model_failure"] = f"{type(error).__name__}: {error}"
            stage["accepted_state_rolled_back"] = True
            stages.append(stage)
            final_status = "unresolved_conditional_model_failure"
            break
        new_validation_residual = mass_validation-candidate_model.predict(time_validation)
        new_mse = float(np.mean(np.sum(new_validation_residual**2, axis=0)))
        stage["validation_residual_mass_mse_after"] = new_mse
        stage["validation_mse_improvement"] = (
            stage["validation_residual_mass_mse_before"]-new_mse)
        stage["validation_gain_positive"] = bool(stage["validation_mse_improvement"] > 0)
        stage["maximum_joint_constraint_frequency_shift"] = float(np.max(abs(
            np.angle(np.exp(1j*(model_frequencies-candidate_acquired))))))
        stage["tentative_clock_model_after"] = candidate_model.metadata()
        if settings.enforce_validation_improvement:
            rejection_reasons = []
            if not stage["validation_gain_positive"]:
                rejection_reasons.append("nonpositive_validation_gain")
            if not joint["usable_lattice"]:
                rejection_reasons.append("unsupported_frequency_relation")
            if rejection_reasons:
                stage["status"] = "rejected_candidate_" + rejection_reasons[0]
                stage["candidate_rejection_reasons"] = rejection_reasons
                stage["accepted_state_rolled_back"] = True
                stage["accepted_character_count_after_rejection"] = len(frozen_frontends)
                stages.append(stage)
                final_status = "unresolved_rejected_candidate"
                break
        # Commit the whole tentative state atomically after strict gates pass.
        acquired = candidate_acquired
        final_lattice = candidate_lattice
        factor_products = tentative_factor_products
        factor_extra_labels = tentative_extra_labels
        models[-1] = candidate_old_model
        model = candidate_model
        models.append(model)
        frozen_frontends = list(pending_frontends) + [frontend]
        stage["status"] = ("accepted_character_conditional_lattice"
                           if joint["usable_lattice"] else
                           "accepted_character_but_unsupported_frequency_relation")
        stage["clock_model_after"] = model.metadata()
        stages.append(stage)
        if not joint["usable_lattice"]:
            final_lattice = None
            final_status = "unsupported_frequency_relation"
            break
    final_relation_check = None
    if settings.phase_relation_mode == "observed" and final_lattice and frozen_frontends:
        from phase_factor_regression import observed_relation_check
        final_train_characters, final_validation_characters = _observed_acquired_characters(
            frozen_frontends, models, mass_validation, time_validation)
        final_relation_check = observed_relation_check(final_train_characters, final_validation_characters,
            np.asarray(final_lattice.get("relations", []), dtype=int))
    audit = {
        "status": final_status,
        "frontend_mode": settings.frontend_mode,
        "joint_frequency_mode": settings.joint_frequency_mode,
        "phase_metric_mode": settings.phase_metric_mode,
        "temporal_gate_mode": settings.temporal_gate_mode,
        "conditional_model_mode": settings.conditional_model_mode,
        "enforce_validation_improvement": settings.enforce_validation_improvement,
        "phase_relation_mode": settings.phase_relation_mode,
        "phase_relation_support": "observed training and independently frozen validation phase products"
            if settings.phase_relation_mode == "observed" else "legacy frequency/clock association diagnostics",
        "observed_relation_threshold_is_heuristic": settings.phase_relation_mode == "observed",
        "phase_readout_refinement": settings.phase_readout_refinement,
        "fourier_sieve_mode": settings.fourier_sieve_mode,
        "maximum_phase_readout_window": max((frontend["window"] for frontend in frozen_frontends), default=settings.lag_window),
        "phase_readout_sources": [frontend.get("readout_source", "residual") for frontend in frozen_frontends],
        "final_observed_relation_check_of_returned_filters": final_relation_check,
        "strict_mode_requires_usable_frequency_lattice": settings.enforce_validation_improvement,
        "noise_estimate": noise,
        "spatial_model_class": type(spatial).__name__,
        "spatial_model": spatial.metadata() if hasattr(spatial, "metadata") else
                         {"model": "known stable polynomial spatial V"},
        "acquired_frequencies": acquired.tolist(),
        "accepted_character_count": len(frozen_frontends),
        "r_candidate_conditional": final_lattice.get("rank") if final_lattice else None,
        "completeness_status": "unresolved_no_finite_sample_certificate",
        "search_termination_is_phase_completeness_proof": False,
        "final_lattice": final_lattice,
        "final_clock_model": models[-1].metadata(),
        "stages": stages,
        "assumptions": ["explicit spatial reconstruction and mass metric",
                        "iid raw sensor noise for sigma estimator or injected noise scale",
                        "finite model Fisher covariance omits Poisson-tail bias",
                        "frequency relation gap and complete visible generators not certified",
                        "heldout association chi-square reference is diagnostic, not calibrated for model bias"],
        "conditional_mean_structure": "nested-factor residual corrections; earlier factor dimensions retained",
        "training_residual_predictions_exclude_temporal_neighbours": settings.lag_window
            if settings.conditional_model_mode == "legacy_knn" else None,
    }
    return {"audit": audit, "clock_models": models,
            "frozen_frontends": frozen_frontends,
            "acquired_frequencies": acquired,
            "mass_train": mass_train, "mass_validation": mass_validation,
            "sampling_model": spatial, "family_readouts": family_readouts}


def predict_mass(model: ClockModel, times: np.ndarray) -> np.ndarray:
    return model.predict(times)


def predict_field(model: ClockModel, times: np.ndarray,
                  positions: np.ndarray, sampling_model=None) -> np.ndarray:
    return _field_from_mass(model.predict(times), positions, sampling_model)
