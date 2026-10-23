"""Diffusion-basis regularized Koopman generator from Giannakis (2019).

Reference: https://arxiv.org/html/1507.02338v4, Sections 4.2--4.3 and 5,
Appendix A, Algorithms 1--4.  Inputs are time-ordered embedding vectors;
delay construction and observation scaling belong to the caller.  No function
accepts a true phase, true frequency, or prescribed phase dimension.

The implementation uses the L2 diffusion basis.  Its ordinary eigenproblem
``(V - epsilon * diag(eta)) a = gamma a`` is equivalent to the Sobolev-basis
generalized problem (21), (32), (33) under a_i=c_i/sqrt(eta_i), i>0.
The constant is retained with zero diffusion penalty.  Empirical advection is
skew-symmetrized by default, preserving the generator's theoretical adjoint
symmetry despite quadrature/boundary errors in a finite trajectory.

Two printed inconsistencies in the algorithm listing are resolved using the
main-text equations: (21) uses sqrt(eta), and the exact symmetric conjugate of
the Markov matrix (24) uses stationary weights proportional to d/q.
"""

from __future__ import annotations

from itertools import product
import time

import numpy as np
from scipy import linalg
from scipy.spatial.distance import cdist


REFERENCE = "https://arxiv.org/html/1507.02338v4"
_TINY = np.finfo(float).tiny


def _matmul(left, right):
    """Ignore stale macOS BLAS flags, but reject actual nonfinite results."""
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        result = np.matmul(left, right)
    if not np.isfinite(result).all():
        raise FloatingPointError("matrix multiplication produced nonfinite values")
    return result


def _matrix(value, *, name="data", minimum_rows=1):
    result = np.asarray(value, dtype=float)
    if result.ndim != 2 or len(result) < minimum_rows or not result.shape[1]:
        raise ValueError(f"{name} must be a finite samples-by-features matrix")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains nonfinite values")
    return result


def _radius_squared(distance_squared, neighbors):
    """Algorithm 1's mean distances to neighbors 2,...,k_nn.

    The same rule, relative to the frozen training set, defines the bandwidth
    function at query points.  It reproduces training radii exactly.
    """
    count = min(int(neighbors), distance_squared.shape[1])
    if count < 2:
        raise ValueError("at least two density neighbors are required")
    nearest = np.partition(distance_squared, count - 1, axis=1)[:, :count]
    radius = (nearest.sum(axis=1) - nearest.min(axis=1)) / (count - 1)
    positive = radius[radius > 0]
    floor = max(float(np.median(positive)) * 1e-12, _TINY) if len(positive) else _TINY
    return np.maximum(radius, floor)


def _bandwidth(distance_ratio, *, maximum_pairs=32768):
    """Algorithm 1 scale-function maximization, using a fixed pair sample.

    Small matrices use the exact kernel sum.  Large matrices estimate only
    this scalar scale function; the final diffusion matrix is never sampled.
    Keeping the diagonal contribution exact prevents false small-scale peaks.
    """
    count = len(distance_ratio)
    exponent = np.arange(-30., 10.00001, .1)
    candidates = np.exp2(exponent)
    if distance_ratio.size <= maximum_pairs:
        values = distance_ratio.ravel()
        diagonal_mass = 0.
        mode = "exact_all_pairs"
    else:
        rng = np.random.default_rng(917)
        left = rng.integers(0, count, maximum_pairs)
        right = rng.integers(0, count - 1, maximum_pairs)
        right += right >= left
        values = distance_ratio[left, right]
        diagonal_mass = 1. / count
        mode = "deterministic_uniform_offdiagonal_pair_sample"
    logs = np.empty(len(candidates))
    for index, epsilon in enumerate(candidates):
        value = np.mean(np.exp(-values / epsilon))
        logs[index] = np.log(max(diagonal_mass + (1. - diagonal_mass) * value, _TINY))
    slopes = np.diff(logs) / np.diff(np.log(candidates))
    selected = int(np.argmax(slopes))
    return float(candidates[selected]), float(2. * slopes[selected]), dict(
        rule="Algorithm 1 maximum logarithmic kernel-sum slope",
        scale_function_mode=mode, scale_function_pair_count=int(len(values)),
        candidate_log2_min=-30., candidate_log2_max=10., candidate_log2_step=.1,
        selected_log2=float(exponent[selected]), maximum_log_slope=float(slopes[selected]))


def fit_diffusion_basis(x, n_basis=96, bandwidth_scale=1., neighbors=8):
    """Fit Algorithm 1's variable-bandwidth diffusion basis on training only.

    ``x`` has shape (time, embedding dimension).  The density scale function
    also supplies a continuous dimension estimate, without a known torus rank.
    ``bandwidth_scale`` multiplies the automatically selected *main* bandwidth.
    """
    started = time.monotonic()
    x = _matrix(x, minimum_rows=8)
    n_basis = min(int(n_basis), len(x) - 1)
    if n_basis < 3 or bandwidth_scale <= 0:
        raise ValueError("n_basis >= 3 and bandwidth_scale > 0 are required")
    distance_squared = cdist(x, x, metric="sqeuclidean")
    radius_squared = _radius_squared(distance_squared, neighbors)
    radius = np.sqrt(radius_squared)
    density_ratio = distance_squared / np.outer(radius, radius)
    density_epsilon, dimension, density_selection = _bandwidth(density_ratio)
    # Bounds only enforce a possible dimension in the ambient embedding.
    dimension = float(np.clip(dimension, .25, x.shape[1]))
    density_sum = np.exp(-density_ratio / density_epsilon).sum(axis=1)
    log_density = (np.log(np.maximum(density_sum / len(x), _TINY))
                   - .5 * dimension * np.log(np.pi * density_epsilon * radius_squared))
    density_center = float(log_density.mean())
    # A common density factor only rescales epsilon.  Removing it prevents
    # overflow in high-dimensional delay embeddings without changing geometry.
    rho = np.exp(np.clip(-(log_density - density_center) / dimension, -300., 300.))
    main_ratio = distance_squared / np.outer(rho, rho)
    automatic_epsilon, _, main_selection = _bandwidth(main_ratio)
    epsilon = automatic_epsilon * float(bandwidth_scale)
    kernel = np.exp(-main_ratio / epsilon)
    q = np.maximum(kernel.sum(axis=1), _TINY)
    d = np.maximum((kernel / q[None, :]).sum(axis=1), _TINY)
    stationary = d / q
    weights = stationary / stationary.sum()
    normalization = np.sqrt(d * q)
    symmetric = kernel / np.outer(normalization, normalization)
    symmetric = (symmetric + symmetric.T) * .5
    kappa, eigenvectors = linalg.eigh(symmetric,
        subset_by_index=(len(x) - n_basis, len(x) - 1), driver="evr",
        check_finite=False)
    kappa, eigenvectors = kappa[::-1], eigenvectors[:, ::-1]
    keep = kappa > 1e-12
    kappa, eigenvectors = kappa[keep], eigenvectors[:, keep]
    if len(kappa) < 3 or 1. - kappa[1] < 1e-10:
        raise ValueError("diffusion kernel is disconnected or lacks a positive spectral gap")
    if abs(kappa[0] - 1.) > 1e-7:
        raise FloatingPointError("diffusion normalization failed to preserve constants")
    kappa[0] = 1.
    phi = eigenvectors / np.sqrt(weights[:, None])
    for column in range(phi.shape[1]):
        pivot = int(np.argmax(np.abs(phi[:, column])))
        if phi[pivot, column] < 0:
            phi[:, column] *= -1.
    phi[:, 0] = 1.
    eta = np.log(kappa) / np.log(kappa[1])
    eta[0] = 0.
    gram = _matmul(phi.T, weights[:, None] * phi)
    return dict(train_x=x.copy(), train_basis=phi, weights=weights, eta=eta,
        kappa=kappa, q=q, d=d, rho=rho, neighbors=int(neighbors),
        radius=radius,
        density_epsilon=density_epsilon, density_log_center=density_center,
        density_dimension=dimension, epsilon=epsilon,
        dimension_estimate=dimension,
        metadata=dict(reference=REFERENCE, algorithm="1; equations 23--27",
            basis_kind="variable-bandwidth conformal diffusion maps",
            n_train=len(x), embedding_dimension=x.shape[1], n_basis=len(eta),
            dimension_estimate=dimension, density_neighbors=int(neighbors),
            density_bandwidth=density_epsilon, main_bandwidth=epsilon,
            bandwidth_scale=float(bandwidth_scale), density_selection=density_selection,
            kernel_selection=main_selection,
            stationary_weight_rule="w proportional to d/q; exact reversible normalization of equation 24",
            weighted_orthogonality_max_error=float(np.max(abs(gram - np.eye(len(eta))))),
            first_markov_spectral_gap=float(1. - kappa[1]),
            elapsed_fit_seconds=time.monotonic() - started, query_refit=False))


def extend_diffusion_basis(basis, x_query, n_basis=None, batch_size=256):
    """Frozen Nyström extension P(query,train) Phi / kappa.

    Query points never change training radii, density normalization, kernel
    bandwidth, stationary weights, or eigenfunctions.  No query-query distances
    are formed, so this is invariant under query batching.
    """
    x_query = _matrix(x_query, name="query")
    if x_query.shape[1] != basis["train_x"].shape[1]:
        raise ValueError("training and query embedding dimensions differ")
    count = len(basis["eta"]) if n_basis is None else min(int(n_basis), len(basis["eta"]))
    result = np.empty((len(x_query), count))
    dimension = basis["density_dimension"]
    for start in range(0, len(x_query), int(batch_size)):
        stop = min(start + int(batch_size), len(x_query))
        distance_squared = cdist(x_query[start:stop], basis["train_x"], metric="sqeuclidean")
        radius_squared = _radius_squared(distance_squared, basis["neighbors"])
        radius = np.sqrt(radius_squared)
        train_radius = basis["radius"]
        density_sum = np.exp(-distance_squared / (
            basis["density_epsilon"] * radius[:, None] * train_radius[None, :])).sum(axis=1)
        log_density = (np.log(np.maximum(density_sum / len(basis["train_x"]), _TINY))
            - .5 * dimension * np.log(np.pi * basis["density_epsilon"] * radius_squared))
        rho = np.exp(np.clip(-(log_density - basis["density_log_center"]) / dimension, -300., 300.))
        kernel = np.exp(-distance_squared / (
            basis["epsilon"] * rho[:, None] * basis["rho"][None, :]))
        row = kernel / basis["q"][None, :]
        total = row.sum(axis=1)
        if np.any(total <= _TINY):
            raise ValueError("query has no numerically resolved diffusion-kernel support")
        row /= total[:, None]
        result[start:stop] = _matmul(row, basis["train_basis"][:, :count]) / basis["kappa"][None, :count]
    result[:, 0] = 1.
    return result


def fit_generator(basis, dt=1., regularization=.001, n_basis=None, skew_adjoint=True):
    """Fit Algorithm 2; retained modes are ordered by Dirichlet energy.

    Truncations reuse a previously fitted diffusion basis.  Frequencies here
    are Im(gamma), including the central-difference finite-sampling bias of
    equation (32c); no oracle frequency correction is made.
    """
    started = time.monotonic()
    if dt <= 0 or regularization <= 0:
        raise ValueError("dt and diffusion regularization must be positive")
    count = len(basis["eta"]) if n_basis is None else min(int(n_basis), len(basis["eta"]))
    if count < 3:
        raise ValueError("at least three diffusion basis functions are required")
    phi, eta, weights = basis["train_basis"][:, :count], basis["eta"][:count], basis["weights"]
    derivative = (phi[2:] - phi[:-2]) / (2. * dt)
    advection_raw = _matmul(phi[1:-1].T, weights[1:-1, None] * derivative)
    symmetric_error = float(linalg.norm(advection_raw + advection_raw.T)
                            / max(linalg.norm(advection_raw), 1e-15))
    advection = ((advection_raw - advection_raw.T) / 2. if skew_adjoint
                 else advection_raw.copy())
    advection[0, :] = 0.
    advection[:, 0] = 0.
    operator = advection - float(regularization) * np.diag(eta)
    eigenvalues, coefficients = linalg.eig(operator, check_finite=False)
    eigenfunctions = _matmul(phi, coefficients)
    norms = np.sqrt(np.sum(weights[:, None] * abs(eigenfunctions) ** 2, axis=0))
    coefficients /= np.maximum(norms, 1e-15)[None, :]
    eigenfunctions /= np.maximum(norms, 1e-15)[None, :]
    energy = np.sum(eta[:, None] * abs(coefficients) ** 2, axis=0)
    order = np.lexsort((eigenvalues.imag, energy))
    coefficients, eigenfunctions = coefficients[:, order], eigenfunctions[:, order]
    eigenvalues, energy = eigenvalues[order], energy[order]
    # Fix one harmless complex gauge deterministically using training only.
    for column in range(count):
        pivot = int(np.argmax(abs(eigenfunctions[:, column])))
        gauge = np.exp(-1j * np.angle(eigenfunctions[pivot, column]))
        coefficients[:, column] *= gauge
        eigenfunctions[:, column] *= gauge
    radius = abs(eigenfunctions)
    radius_cv = np.sqrt(np.sum(weights[:, None] * (radius - np.sum(
        weights[:, None] * radius, axis=0)) ** 2, axis=0)) / np.maximum(
            np.sum(weights[:, None] * radius, axis=0), 1e-15)
    unit = eigenfunctions / np.maximum(radius, 1e-15)
    transition_weights = .5 * (weights[:-1] + weights[1:])
    transition_weights /= transition_weights.sum()
    phase_step = np.sum(transition_weights[:, None] * np.conj(unit[:-1]) * unit[1:], axis=0)
    phase_frequency = np.angle(phase_step) / dt
    phase_step_coherence = abs(phase_step)
    temporal_derivative = (eigenfunctions[2:] - eigenfunctions[:-2]) / (2. * dt)
    residual = temporal_derivative - eigenfunctions[1:-1] * (1j * eigenvalues.imag[None, :])
    temporal_residual = np.sqrt(np.sum(weights[1:-1, None] * abs(residual) ** 2, axis=0))
    return dict(basis=basis, n_basis=count, dt=float(dt),
        regularization=float(regularization), coefficients=coefficients,
        eigenvalues=eigenvalues, dirichlet_energy=energy,
        train_eigenfunctions=eigenfunctions, radius_cv=radius_cv,
        train_phase_frequency=phase_frequency,
        phase_step_coherence=phase_step_coherence,
        temporal_residual=temporal_residual,
        metadata=dict(reference=REFERENCE, algorithm="2; equations 21,32,33",
            operator="Galerkin central-time-derivative minus diffusion regularization",
            formulation="L2 basis equivalent to Sobolev generalized eigenproblem",
            n_basis=count, regularization=float(regularization), dt=float(dt),
            finite_difference="second-order central; endpoints omitted in quadrature",
            skew_adjoint_stabilization=bool(skew_adjoint),
            raw_advection_relative_symmetry_error=symmetric_error,
            constant_exact_nullspace=True,
            eigenfunction_order="increasing normalized Dirichlet energy",
            phase_frequency_refinement="equation 5: weighted circular mean of one-sample unit-eigenfunction increments; raw gamma retained",
            eigenvector_condition=float(np.linalg.cond(coefficients)),
            elapsed_fit_seconds=time.monotonic() - started, query_refit=False))


def transform_generator(model, x_query, batch_size=256):
    """Encode observations with the frozen diffusion/Galerkin maps."""
    phi = extend_diffusion_basis(model["basis"], x_query, n_basis=model["n_basis"],
                                 batch_size=batch_size)
    eigenfunctions = _matmul(phi, model["coefficients"])
    return dict(basis=phi, eigenfunctions=eigenfunctions,
                phases=np.angle(eigenfunctions), query_refit=False)


def _integer_relation(frequency, chosen, tolerance, max_order, *, dt,
                      new_phase, chosen_phases, weights, minimum_coherence):
    """Require a sampled frequency relation and the eigenfunction group law.

    The group test (Definition 2) avoids rejecting distinct characters whose
    frequencies happen to fall within one finite-record resolution interval.
    Frequencies are compared modulo the sampling frequency; the product must
    additionally be almost constant over the training states.
    """
    if not len(chosen):
        return None
    # q_new*omega_new + q dot omega_old = 0; q_new>0 removes duplicates.
    # The caller caps dimension to avoid an exponential search on noisy data.
    labels = np.asarray(list(product(range(-max_order, max_order + 1), repeat=len(chosen))))
    combinations = _matmul(labels, np.asarray(chosen))
    for multiplier in range(1, max_order + 1):
        relations = multiplier * frequency + combinations
        errors = abs(np.angle(np.exp(1j * dt * relations))) / dt
        possible = np.flatnonzero(errors <= tolerance)
        for index in possible[np.argsort(errors[possible])]:
            phase_product = np.exp(1j * (multiplier * new_phase
                + _matmul(chosen_phases, labels[index])))
            coherence = float(abs(np.sum(weights * phase_product)))
            if coherence >= minimum_coherence:
                return dict(new_multiplier=multiplier,
                    previous_coefficients=labels[index].tolist(),
                    sampled_frequency_error=float(errors[index]),
                    unwrapped_frequency_error=float(abs(relations[index])),
                    training_phase_product_coherence=coherence)
    return None


def select_independent_phases(model, max_phases=None, max_order=4,
                             frequency_tolerance=None, max_radius_cv=.35,
                             max_relative_residual=.5,
                             minimum_relation_coherence=.9):
    """Train-only numerical version of Algorithm 2(f).

    At finite sampling interval, Im(gamma) has central-difference sine bias.
    Relation selection therefore refines frequencies from the eigenfunctions'
    one-step phase increments (equation 5), and requires the corresponding
    normalized eigenfunction product to be nearly constant (Definition 2).
    Both additions use frozen training functions only; raw gamma is retained.

    This is a bounded-resolution diagnostic, not a proof of torus dimension or
    a primitive integer lattice.  Without a supplied cap, the estimated
    diffusion dimension is rounded and clipped to 1,...,4 for finite search.
    A nonempty result can still be incomplete/nonprimitive; metadata says so.
    """
    count = len(model["basis"]["train_x"])
    duration = max((count - 1) * model["dt"], model["dt"])
    if frequency_tolerance is None:
        frequency_tolerance = .5 * 2. * np.pi / duration
    cap = (int(np.clip(np.rint(model["basis"]["dimension_estimate"]), 1, 4))
           if max_phases is None else min(int(max_phases), 4))
    selected, frequencies, decisions = [], [], []
    all_phases = np.angle(model["train_eigenfunctions"])
    for index, value in enumerate(model["eigenvalues"]):
        raw_frequency = float(value.imag)
        frequency = float(model["train_phase_frequency"][index])
        if raw_frequency <= frequency_tolerance or frequency <= frequency_tolerance:
            continue  # constants and conjugate duplicates
        relative_residual = float(model["temporal_residual"][index] / max(abs(raw_frequency), 1e-15))
        info = dict(index=index, frequency=frequency,
            raw_generator_frequency=raw_frequency,
            training_phase_step_coherence=float(model["phase_step_coherence"][index]),
            dirichlet_energy=float(model["dirichlet_energy"][index]),
            radius_cv=float(model["radius_cv"][index]),
            relative_temporal_residual=relative_residual)
        if info["radius_cv"] > max_radius_cv or relative_residual > max_relative_residual:
            info["decision"] = "rejected_training_eigenfunction_quality"
        else:
            relation = _integer_relation(frequency, frequencies,
                frequency_tolerance, int(max_order), dt=model["dt"],
                new_phase=all_phases[:, index], chosen_phases=all_phases[:, selected],
                weights=model["basis"]["weights"], minimum_coherence=float(minimum_relation_coherence))
            if relation is not None:
                info.update(decision="rejected_bounded_integer_relation", relation=relation)
            elif len(selected) >= cap:
                info["decision"] = "beyond_density_dimension_cap"
            else:
                selected.append(index)
                frequencies.append(frequency)
                info["decision"] = "selected"
        decisions.append(info)
    phases = np.angle(model["train_eigenfunctions"][:, selected])
    return dict(indices=selected, frequencies=frequencies, train_phases=phases,
        phase_dimension=len(selected), status="selected_numerically" if selected else "unresolved_no_reliable_generators",
        metadata=dict(selection="Algorithm 2(f), energy order; finite-sampling adaptation using equation 5 frequencies and Definition 2 phase-product relations",
            frequency_source="training weighted circular mean of unit-eigenfunction one-step increments",
            raw_generator_eigenvalues_retained=True,
            frequency_tolerance=float(frequency_tolerance), integer_max_order=int(max_order),
            sampled_frequency_relations_modulo=2. * np.pi / model["dt"],
            minimum_relation_phase_product_coherence=float(minimum_relation_coherence),
            estimated_dimension_cap=cap, max_radius_cv=float(max_radius_cv),
            max_relative_temporal_residual=float(max_relative_residual),
            truth_used=False, phase_dimension_prescribed=False,
            primitive_lattice_guaranteed=False, decisions=decisions))


def fit_observable(model, y_train, mode_indices=None, ridge=0., rcond=1e-10):
    """Weighted Gram reconstruction of an observable, Section 4.3 / Alg. 3.

    The full Galerkin eigenfunction basis is the default.  If the caller
    truncates it, selection must use training information only.  A weighted
    least-squares solve implements G^(-1)<z,f> without squaring its condition.
    """
    y = _matrix(y_train, name="observable")
    if len(y) != len(model["train_eigenfunctions"]):
        raise ValueError("observable must align with training embedding vectors")
    indices = (np.arange(model["n_basis"], dtype=int) if mode_indices is None
               else np.asarray(mode_indices, dtype=int))
    z = model["train_eigenfunctions"][:, indices]
    root = np.sqrt(model["basis"]["weights"])
    design, target = root[:, None] * z, root[:, None] * y
    if ridge > 0:
        design = np.vstack((design, np.sqrt(ridge) * np.eye(len(indices))))
        target = np.vstack((target, np.zeros((len(indices), y.shape[1]))))
    coefficients, _, rank, singular = linalg.lstsq(design, target, cond=rcond,
                                                 lapack_driver="gelsd", check_finite=False)
    return dict(mode_indices=indices, coefficients=coefficients,
        metadata=dict(readout="weighted spectral Gram least squares",
            mode_count=len(indices), numerical_rank=int(rank), ridge=float(ridge),
            condition=float(singular[0] / max(singular[-1], 1e-15)),
            train_only=True))


def reconstruct_observable(readout, transformed):
    """Reconstruct from frozen-encoded query observations."""
    return np.real(_matmul(transformed["eigenfunctions"][:, readout["mode_indices"]],
                           readout["coefficients"]))


def forecast_observable(model, readout, initial_eigenfunctions, lead_times,
                        use_damped_generator=False):
    """Point forecast via spectral evolution, the observable dual of Alg. 4.

    Pure-point forecasting uses exp(i Im(gamma)t), as in the paper.  The
    optional damped approximation uses exp(gamma t) and is labeled explicitly.
    ``initial_eigenfunctions`` is one frozen-encoded state, not future data.
    """
    initial = np.asarray(initial_eigenfunctions, dtype=complex).reshape(-1)
    indices = readout["mode_indices"]
    values = model["eigenvalues"][indices]
    rates = values if use_damped_generator else 1j * values.imag
    evolved = np.exp(np.asarray(lead_times, dtype=float).reshape(-1, 1) * rates[None, :])
    evolved *= initial[indices][None, :]
    return np.real(_matmul(evolved, readout["coefficients"]))


def product_basis(phases, spectral_order=2):
    """Algorithm 3's tensor-product characters and integer labels."""
    phases = _matrix(phases, name="phases")
    order = int(spectral_order)
    if order < 0 or (2 * order + 1) ** phases.shape[1] > 4096:
        raise ValueError("spectral product dictionary is too large")
    labels = np.asarray(list(product(range(-order, order + 1), repeat=phases.shape[1])), dtype=int)
    return np.exp(1j * _matmul(phases, labels.T)), labels


def fit_phase_observable(model, phase_selection, y_train, spectral_order=2, rcond=1e-10):
    """Algorithm 3/4 product dictionary, Gram fit, and vector-field readout."""
    if not phase_selection["indices"]:
        raise ValueError("no selected generating eigenfunctions")
    y = _matrix(y_train, name="observable")
    phase = phase_selection["train_phases"]
    z, labels = product_basis(phase, spectral_order)
    root = np.sqrt(model["basis"]["weights"])
    coefficients, _, rank, singular = linalg.lstsq(root[:, None] * z,
        root[:, None] * y, cond=rcond, lapack_driver="gelsd", check_finite=False)
    return dict(indices=list(phase_selection["indices"]), labels=labels,
        frequencies=np.asarray(phase_selection["frequencies"]), coefficients=coefficients,
        metadata=dict(reference=REFERENCE, algorithm="3--4; product-basis Gram reconstruction",
            spectral_order=int(spectral_order), product_basis_size=len(labels),
            numerical_rank=int(rank), condition=float(singular[0] / max(singular[-1], 1e-15)),
            train_only=True))


def reconstruct_phase_observable(readout, transformed, return_fields=False):
    """Reconstruct sensors and optionally Algorithm 3's vector components.

    Fields are evaluated through the spectral observation map.  Their finite
    reconstruction is approximate; commutation is a phase-space property,
    not asserted as an exact identity of arbitrary ambient extensions.
    """
    phase = np.angle(transformed["eigenfunctions"][:, readout["indices"]])
    z = np.exp(1j * _matmul(phase, readout["labels"].T))
    result = dict(reconstruction=np.real(_matmul(z, readout["coefficients"])), phases=phase)
    if return_fields:
        fields = []
        for dimension, frequency in enumerate(readout["frequencies"]):
            rates = 1j * frequency * readout["labels"][:, dimension]
            fields.append(np.real(_matmul(z * rates[None, :], readout["coefficients"])))
        result["vector_fields"] = np.stack(fields, axis=1)
    return result
