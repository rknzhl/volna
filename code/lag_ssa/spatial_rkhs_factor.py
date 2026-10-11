"""Spatial RKHS profiles for an already learned finite temporal dictionary.

For fixed real X=[1,cos(K alpha t),sin(K alpha t)], fit raw-sensor coefficients
C by G C + lambda_s C K_s^{-1}=R, where G=X.T X/N+lambda_t D and
R=X.T Y.T/N. The complete temporal Gram, including nonorthogonal lines and
the mean, is retained. Matérn3/2 spatial regularization is standard.

Kernel/ridge selection uses raw-training leave-one-retained-sensor-out error.
An optional train-only guard preserves half the baseline numerical primitive
witness singular value. This preserves a learned-template conditioning
criterion; it is neither physical phase confidence nor a novelty claim.
"""

from dataclasses import dataclass, asdict

import numpy as np
from scipy.linalg import eigh, solve
from scipy.spatial.distance import cdist

from hankel_torus_projection import HankelTorusProjector, ProjectionSettings
from nodal_spatial import NodalSpatialModel
from phase_witness_conditioning import witness_conditioning, canonical_witnesses


@dataclass(frozen=True)
class SpatialRKHSSettings:
    length_multipliers: tuple = (.5, 1., 2., 4.)
    spatial_ridges: tuple = (0., 1e-5, 1e-3, .1, 1.)
    minimum_witness_sigma_ratio: float = .5
    hankel_window: int = 128


def _points(positions):
    positions = np.asarray(positions, dtype=float)
    if positions.ndim == 1:
        positions = positions[:, None]
    if positions.ndim != 2 or not np.isfinite(positions).all():
        raise ValueError("finite sensor-coordinate rows required")
    return positions


def matern32(first, second, length):
    first, second = _points(first), _points(second)
    if first.shape[1] != second.shape[1] or length <= 0:
        raise ValueError("matching spatial dimensions and positive kernel length required")
    distance = np.sqrt(3.)*cdist(first, second)/length
    return (1+distance)*np.exp(-distance)


def temporal_design(times, frequencies, labels):
    frequency = np.einsum("ij,j->i", labels, frequencies, optimize=False)
    angle = np.asarray(times, dtype=float)[:, None]*frequency[None, :]
    return np.column_stack((np.ones(len(times)), np.cos(angle), np.sin(angle)))


def _kernel_eigensystem(kernel):
    eigenvalues, vectors = eigh(kernel, check_finite=True)
    if eigenvalues[0] <= 0:
        raise ValueError("numerically positive definite spatial kernel required; no jitter is added")
    return eigenvalues, vectors


def solve_profile_coefficients(gram, right, kernel, spatial_ridge, temporal_eigensystem=None):
    """Solve the full Sylvester equation in the two symmetric eigenbases."""
    if spatial_ridge < 0:
        raise ValueError("nonnegative spatial ridge required")
    if spatial_ridge == 0:
        return solve(gram, right, assume_a="pos", check_finite=True)
    g, u = eigh(gram) if temporal_eigensystem is None else temporal_eigensystem
    k, v = _kernel_eigensystem(kernel)
    denominator = g[:, None]+spatial_ridge/k[None, :]
    if np.any(denominator <= 0):
        raise ValueError("positive Sylvester spectral denominators required")
    transformed = np.einsum("ji,jq->iq", u, right, optimize=False)
    transformed = np.einsum("iq,qj->ij", transformed, v, optimize=False)
    temporary = np.einsum("ij,jq->iq", u, transformed/denominator, optimize=False)
    return np.einsum("iq,jq->ij", temporary, v, optimize=False)


class SpatialRKHSFactorModel:
    def __init__(self, coefficients, positions, frequencies, labels, length, spatial_ridge,
                 temporal_ridge, mass_sqrt, information):
        self.coefficients_sensor = np.asarray(coefficients, dtype=float).copy()
        self.positions = _points(positions).copy()
        self.frequencies = np.asarray(frequencies, dtype=float).copy()
        self.labels = np.asarray(labels, dtype=int).copy()
        self.length = float(length)
        self.spatial_ridge = float(spatial_ridge)
        self.temporal_ridge = float(temporal_ridge)
        self.kernel = matern32(self.positions, self.positions, self.length)
        self.coefficients_mass = (None if mass_sqrt is None else
            np.einsum("nq,iq->ni", self.coefficients_sensor, mass_sqrt, optimize=False))
        self.information = dict(information)

    def query_coefficients(self, query_positions):
        query = _points(query_positions)
        cross = matern32(self.positions, query, self.length)
        weights = solve(self.kernel, cross, assume_a="pos", check_finite=True)
        return np.einsum("nq,qj->nj", self.coefficients_sensor, weights, optimize=False)

    def predict(self, times, query_positions=None):
        coefficients = self.coefficients_sensor if query_positions is None else self.query_coefficients(query_positions)
        design = temporal_design(times, self.frequencies, self.labels)
        return np.einsum("tn,nq->qt", design, coefficients, optimize=False)

    def predict_on_phase(self, phase, query_positions=None):
        phase = np.asarray(phase, dtype=float)
        if phase.ndim != 2 or phase.shape[0] != len(self.frequencies):
            raise ValueError("one phase row per learned generator required")
        angles = np.einsum("ij,jt->it", self.labels, phase, optimize=False)
        design = np.column_stack((np.ones(phase.shape[1]), np.cos(angles.T), np.sin(angles.T)))
        coefficients = self.coefficients_sensor if query_positions is None else self.query_coefficients(query_positions)
        return np.einsum("tn,nq->qt", design, coefficients, optimize=False)

    def metadata(self):
        return {**self.information, "kernel": "Matérn3/2, unit marginal amplitude",
            "kernel_length": self.length, "spatial_ridge": self.spatial_ridge,
            "temporal_ridge": self.temporal_ridge,
            "query_rule": "C_query=C_sensor K_sensor^{-1} K_sensor,query",
            "spatial_domain": "kernel extension to arbitrary query coordinates; no out-of-node accuracy certificate",
            "full_temporal_gram_preserved": True, "kernel_jitter_added": False}


def _conditioning(coefficients, mass_sqrt, frequencies, labels, acquired_labels, phase_products, window):
    if mass_sqrt is None:
        return {"status": "unresolved_mass_geometry", "sigma0_numerical": None}
    if acquired_labels is None or phase_products is None:
        return {"status": "unresolved_missing_primitive_witness", "sigma0_numerical": None}
    if len(frequencies) == 0 or len(labels) == 0:
        return {"status": "unresolved_empty_phase_dictionary", "sigma0_numerical": None}
    mass_coefficients = np.einsum("nq,iq->ni", coefficients, mass_sqrt, optimize=False)
    try:
        indices, _, _ = canonical_witnesses(labels, acquired_labels, phase_products)
        if mass_sqrt.shape[0]*window < 2*len(indices):
            return {"status": "unresolved_witness_columns_exceed_window_dimension", "sigma0_numerical": None}
        projector = HankelTorusProjector(labels, mass_coefficients, frequencies,
            ProjectionSettings(window=window))
        _, information = witness_conditioning(projector, acquired_labels, phase_products)
    except (ValueError, ArithmeticError, np.linalg.LinAlgError) as error:
        return {"status": "unresolved_witness", "sigma0_numerical": None, "reason": str(error)}
    if not information["witness_full_column_rank_at_declared_tolerance"] or information["sigma0_numerical"] <= 0:
        return {**information, "status": "unresolved_zero_or_degenerate_witness"}
    return {**information, "status": "resolved_numerical_learned_template_conditioning"}


def fit_spatial_rkhs_factor(raw_train, positions, times, frequencies, labels, temporal_ridge,
                            acquired_labels=None, phase_products=None, sampling_model=None,
                            settings=SpatialRKHSSettings(), baseline_coefficients_sensor=None):
    """Return CV-minimum and conditioning-guarded models, without outer data.

The dictionary and frequencies are frozen inputs learned by the public-data
pipeline. They are not reselected inside sensor folds; this is conditional
profile CV. Missing/degenerate witness information leaves the guarded model
explicitly unresolved while preserving the unconstrained spatial estimate.
"""
    values = np.asarray(raw_train, dtype=float)
    points = _points(positions)
    frequencies = np.asarray(frequencies, dtype=float)
    original_labels = np.asarray(labels)
    labels = np.asarray(labels, dtype=int)
    times = np.asarray(times, dtype=float)
    if (values.ndim != 2 or values.shape != (len(points), len(times)) or len(points) < 2
            or not np.isfinite(values).all() or not np.isfinite(times).all()
            or frequencies.ndim != 1 or not np.isfinite(frequencies).all()
            or labels.ndim != 2 or labels.shape[1] != len(frequencies)
            or temporal_ridge < 0 or not np.isfinite(temporal_ridge) or len(times) == 0
            or not np.array_equal(original_labels, labels)):
        raise ValueError("finite matching training observations, geometry and frozen temporal dictionary required")
    if len(times) > 1 and not np.allclose(np.diff(times), 1., atol=1e-12, rtol=0):
        raise ValueError("frequencies are radians per sample; pass unit-spaced sample indices as times")
    if settings.minimum_witness_sigma_ratio <= 0:
        raise ValueError("positive declared witness-preservation ratio required")
    distances = cdist(points, points)
    np.fill_diagonal(distances, np.inf)
    nearest = distances.min(axis=1)
    if np.any(nearest <= 0):
        raise ValueError("distinct retained sensor positions required")
    spacing = float(np.median(nearest))
    design = temporal_design(times, frequencies, labels)
    samples = len(times)
    raw_gram = np.einsum("tn,tm->nm", design, design, optimize=False)/samples
    penalty = np.eye(design.shape[1]);penalty[0, 0] = 0.
    gram = raw_gram+temporal_ridge*penalty
    right = np.einsum("tn,qt->nq", design, values, optimize=False)/samples
    temporal_eigensystem = eigh(gram)
    baseline = solve(gram, right, assume_a="pos", check_finite=True)
    baseline_difference = None
    if baseline_coefficients_sensor is not None:
        provided = np.asarray(baseline_coefficients_sensor, dtype=float)
        if provided.shape != baseline.shape:
            raise ValueError("baseline raw coefficient shape does not match the fixed temporal design")
        baseline_difference = float(np.max(abs(baseline-provided)))
    geometry_status = "provided_same_retained_nodes"
    mass_sqrt = None
    if sampling_model is None:
        try:
            sampling_model = NodalSpatialModel.from_positions(points)
            geometry_status = "geometry_only_nodal_P1_on_retained_nodes"
        except ValueError as error:
            geometry_status = "unresolved_nodal_mass: "+str(error)
    if sampling_model is not None:
        if (hasattr(sampling_model, "positions") and
                (np.asarray(sampling_model.positions).shape != points.shape or
                 not np.allclose(sampling_model.positions, points, atol=1e-12, rtol=0))):
            raise ValueError("the supplied mass geometry must use these same retained sensors in the same order")
        mass_sqrt = np.asarray(sampling_model.mass_sqrt, dtype=float)
        if mass_sqrt.shape != (len(points), len(points)) or not np.isfinite(mass_sqrt).all():
            raise ValueError("matching finite retained-sensor mass square root required")
    baseline_conditioning = _conditioning(baseline, mass_sqrt, frequencies, labels,
        acquired_labels, phase_products, settings.hankel_window)
    baseline_sigma = baseline_conditioning.get("sigma0_numerical")
    guard_available = baseline_conditioning["status"] == "resolved_numerical_learned_template_conditioning"
    candidates, usable = [], []
    for multiplier in settings.length_multipliers:
        length = float(multiplier)*spacing
        kernel = matern32(points, points, length)
        for spatial_ridge in settings.spatial_ridges:
            record = {"length_multiplier": float(multiplier), "kernel_length": length,
                "spatial_ridge": float(spatial_ridge), "folds": []}
            try:
                kernel_values, _ = _kernel_eigensystem(kernel)
                record["full_kernel_condition"] = float(kernel_values[-1]/kernel_values[0])
                total_squared_error = 0.
                for held in range(len(points)):
                    retained = np.delete(np.arange(len(points)), held)
                    fold_kernel = kernel[np.ix_(retained, retained)]
                    coefficients = solve_profile_coefficients(gram, right[:, retained], fold_kernel,
                        float(spatial_ridge), temporal_eigensystem)
                    weights = solve(fold_kernel, kernel[retained, held], assume_a="pos")
                    query_coefficient = np.einsum("nq,q->n", coefficients, weights, optimize=False)
                    prediction = np.einsum("tn,n->t", design, query_coefficient, optimize=False)
                    squared_error = float(np.sum((prediction-values[held])**2))
                    if not np.isfinite(squared_error):
                        raise FloatingPointError("nonfinite retained-sensor CV error")
                    total_squared_error += squared_error
                    record["folds"].append({"held_retained_sensor_index": held, "raw_train_squared_error": squared_error})
                record["cv_raw_train_sensor_sse"] = total_squared_error
                record["cv_raw_train_sensor_mse"] = total_squared_error/(samples*len(points))
                coefficients = (baseline.copy() if spatial_ridge == 0 else
                    solve_profile_coefficients(gram, right, kernel, float(spatial_ridge), temporal_eigensystem))
                conditioning = _conditioning(coefficients, mass_sqrt, frequencies, labels,
                    acquired_labels, phase_products, settings.hankel_window)
                sigma = conditioning.get("sigma0_numerical")
                feasible = bool(guard_available and conditioning["status"] == "resolved_numerical_learned_template_conditioning"
                    and sigma >= settings.minimum_witness_sigma_ratio*baseline_sigma)
                record.update(status="fitted", conditioning=conditioning,
                    witness_sigma_ratio_to_baseline=(None if not guard_available or sigma is None else float(sigma/baseline_sigma)),
                    conditioning_guard_feasible=feasible,
                    conditioning_guard_rejection_reason=(None if feasible else
                        "baseline_primitive_witness_unresolved" if not guard_available else
                        "candidate_primitive_witness_unresolved" if conditioning["status"] != "resolved_numerical_learned_template_conditioning" else
                        "sigma_below_declared_baseline_ratio"))
                if spatial_ridge > 0:
                    inverse_kernel = solve(kernel, np.eye(len(points)), assume_a="pos")
                    stationarity = np.einsum("nm,mq->nq", gram, coefficients, optimize=False)+spatial_ridge*np.einsum(
                        "nq,qj->nj", coefficients, inverse_kernel, optimize=False)-right
                    record["sylvester_relative_stationarity_residual"] = float(np.linalg.norm(stationarity)/max(np.linalg.norm(right), 1e-30))
                else:
                    record["sylvester_relative_stationarity_residual"] = float(np.linalg.norm(
                        np.einsum("nm,mq->nq", gram, coefficients, optimize=False)-right)/max(np.linalg.norm(right), 1e-30))
                usable.append((len(candidates), record["cv_raw_train_sensor_sse"], coefficients, length, float(spatial_ridge)))
            except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
                record.update(status="unusable", reason=str(error), conditioning_guard_feasible=False)
            candidates.append(record)
    if not usable:
        raise ValueError("no usable spatial kernel/ridge candidate")
    minimum = min(usable, key=lambda item: (item[1], item[0]))
    zero_ridge = min((item for item in usable if item[4] == 0.), key=lambda item: (item[1], item[0]), default=None)
    feasible = [item for item in usable if candidates[item[0]]["conditioning_guard_feasible"]]
    guarded = min(feasible, key=lambda item: (item[1], item[0])) if feasible else None

    def model(item):
        if item is None:
            return None
        index, _, coefficients, length, ridge = item
        return SpatialRKHSFactorModel(coefficients, points, frequencies, labels, length, ridge,
            temporal_ridge, mass_sqrt, {"candidate_index": index,
                "selection": candidates[index], "conditioning_guard_ratio": settings.minimum_witness_sigma_ratio})

    information = {"settings": asdict(settings), "median_retained_sensor_nearest_neighbor_spacing": spacing,
        "candidate_grid_declared_before_scoring": True, "candidates": candidates,
        "unconstrained_selected_candidate_index": minimum[0],
        "zero_spatial_ridge_selected_candidate_index": None if zero_ridge is None else zero_ridge[0],
        "guarded_selected_candidate_index": None if guarded is None else guarded[0],
        "guard_status": "resolved_numerical_conditioning_preservation" if guarded is not None else "unresolved_primitive_witness_or_mass_geometry",
        "guard_rejected_fitted_candidate_count": sum(item["status"] == "fitted" and not item["conditioning_guard_feasible"] for item in candidates),
        "raw_training_samples": samples, "retained_sensor_count": len(points), "geometry_status": geometry_status,
        "temporal_ridge_frozen_from_original_global_fit": float(temporal_ridge),
        "lambda_s_zero_reproduces_frozen_temporal_ridge_regression": True,
        "provided_baseline_coefficients_max_absolute_difference": baseline_difference,
        "temporal_gram_smallest_eigenvalue": float(temporal_eigensystem[0][0]),
        "full_nonorthogonal_temporal_gram_used": True,
        "cv_scope": "raw TRAIN leave-one-retained-sensor-out; frozen frequencies/dictionary; kernel length uses full retained training geometry",
        "outer_sensor_observations_used": False, "physical_law_or_generator_information_used": False,
        "guard_interpretation": "heuristic train-only preservation of the learned finite-template global inverse conditioning; not physical phase confidence",
        "novelty_claimed": False}
    return {"unconstrained_model": model(minimum), "guarded_model": model(guarded),
        "zero_spatial_ridge_model": model(zero_ridge),
        "baseline_coefficients_sensor": baseline, "baseline_conditioning": baseline_conditioning,
        "metadata": information}
