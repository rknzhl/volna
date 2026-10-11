"""Blind dominant-character readout from a weighted SSA trajectory matrix.

This module uses only the observed field, its supplied spatial weights, and
the lag window. It does not infer a full phase lattice. The acceptance gates
are diagnostic heuristics, not calibrated confidence statements.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import svds


def _complex_pair(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    values, vectors = np.linalg.eig(matrix)
    positive = np.flatnonzero(np.imag(values) > 1e-5)
    negative = np.flatnonzero(np.imag(values) < -1e-5)
    if len(positive) != 1 or len(negative) != 1:
        return None
    if abs(values[positive[0]] - np.conj(values[negative[0]])) > 1e-5:
        return None
    return values, vectors


def _complex_pairs(values: np.ndarray) -> list[list[float]]:
    return [[float(value.real), float(value.imag)] for value in values]


def extract_ssa_character(
    signals: np.ndarray,
    spatial_weights: np.ndarray,
    window: int,
    noise_sigma: float | None = None,
) -> dict:
    """Read one positive-frequency character from the dominant SSA pair.

    `phase` and `normalized_character` are indexed by Hankel column starts.
    Their zero is arbitrary. A returned `unresolved` status is substantive:
    a top-two subspace exists for noise too, but need not represent a phase.
    """
    y = np.asarray(signals, dtype=float)
    weights = np.asarray(spatial_weights, dtype=float).reshape(-1)
    if (y.ndim != 2 or y.shape[0] != len(weights) or y.shape[1] <= window + 3
            or window < 4 or np.any(weights <= 0) or not np.isfinite(y).all()):
        raise ValueError("finite (sensor,time) field, positive weights and valid window required")
    weights = weights / np.sum(weights)
    q, n = y.shape
    centered = y-y.mean(axis=1, keepdims=True)
    k = n-window+1
    root_weights = np.sqrt(weights/window)[:, None]
    hankel = np.vstack([root_weights*centered[:, lag:lag+k]
                        for lag in range(window)])
    # The left singular vectors are eigenvectors of the lag covariance HH*.
    count = min(4, min(hankel.shape)-1)
    initial_vector = np.random.default_rng(471).standard_normal(min(hankel.shape))
    u, singular, _ = svds(hankel, k=count, which="LM", tol=1e-8,
                          v0=initial_vector)
    order = np.argsort(singular)[::-1]
    u = u[:, order]
    singular = singular[order]
    if len(singular) < 3 or singular[1] <= 0:
        return {"status": "unresolved", "reason": "insufficient numerical pair rank"}
    subspace = u[:, :2]
    lag_zero = subspace[:-q]
    lag_one = subspace[q:]
    esprit_operator = np.linalg.lstsq(lag_zero, lag_one, rcond=None)[0]
    esprit_pair = _complex_pair(esprit_operator)
    espace_error = float(np.linalg.norm(lag_one-lag_zero@esprit_operator)
                         / max(np.linalg.norm(lag_one), 1e-15))
    projected = np.einsum("ij,jk->ik", subspace.T, hankel, optimize=False)
    previous, following = projected[:, :-1], projected[:, 1:]
    cross = np.einsum("ij,kj->ik", following, previous, optimize=False)
    gram = np.einsum("ij,kj->ik", previous, previous, optimize=False)
    temporal_operator = np.einsum("ij,jk->ik", cross, np.linalg.inv(gram), optimize=False)
    temporal_pair = _complex_pair(temporal_operator)
    raw_lag1 = float(np.sum(weights[:, None]*centered[:, 1:]*centered[:, :-1])
                     / max(np.sum(weights[:, None]*centered[:, :-1]**2), 1e-15))
    metrics = {
        "singular_values": singular.tolist(),
        "first_to_second_ratio": float(singular[0]/singular[1]),
        "pair_to_next_gap": float(singular[1]/singular[2]),
        "esprit_subspace_residual": espace_error,
        "raw_lag1_correlation": raw_lag1,
        "esprit_eigenvalues": _complex_pairs(np.linalg.eigvals(esprit_operator)),
        "temporal_dmd_eigenvalues": _complex_pairs(np.linalg.eigvals(temporal_operator)),
        "noise_sigma_supplied": noise_sigma is not None,
        "noise_sigma": float(noise_sigma) if noise_sigma is not None else None,
    }
    if esprit_pair is None or temporal_pair is None:
        return {"status": "unresolved", "reason": "top pair has no simple complex rotation",
                "metrics": metrics}
    esprit_eigenvalues = esprit_pair[0]
    temporal_eigenvalues = temporal_pair[0]
    esprit_positive = esprit_eigenvalues[np.imag(esprit_eigenvalues) > 0][0]
    temporal_positive = temporal_eigenvalues[np.imag(temporal_eigenvalues) > 0][0]
    nu_esprit = float(np.angle(esprit_positive))
    nu_dmd = float(np.angle(temporal_positive))
    # A left eigenvector of the temporal 2x2 shift yields a scalar coordinate.
    left_values, left_vectors = np.linalg.eig(temporal_operator.T)
    selected = int(np.argmin(abs(left_values-temporal_positive)))
    coordinate = np.einsum("i,ij->j", left_vectors[:, selected], projected,
                           optimize=False)
    lag_filter = np.einsum("i,ji->j", left_vectors[:, selected], subspace,
                           optimize=False).reshape(window, q)*root_weights.T
    radii = abs(coordinate)
    if np.median(radii) <= 1e-12:
        return {"status": "unresolved", "reason": "complex coordinate has zero radius",
                "metrics": metrics}
    z = coordinate / np.maximum(radii, 1e-15)
    phase = np.unwrap(np.angle(z))
    times = np.arange(k, dtype=float)
    time_centered = times-times.mean()
    nu_slope = float(np.dot(time_centered, phase-phase.mean())
                     / np.dot(time_centered, time_centered))
    phase_fit = phase.mean()+nu_slope*time_centered
    phase_residual = float(np.sqrt(np.mean((phase-phase_fit)**2)))
    coherence = float(abs(np.mean(z[1:]*np.conj(z[:-1]))))
    radius_cv = float(np.std(radii)/np.mean(radii))
    frequency_disagreement = float(abs(np.angle(np.exp(1j*(nu_esprit-nu_dmd)))))
    slope_disagreement = float(abs(np.angle(np.exp(1j*(nu_esprit-nu_slope)))))
    metrics.update({
        "frequency_esprit": nu_esprit,
        "frequency_temporal_dmd": nu_dmd,
        "frequency_phase_slope": nu_slope,
        "frequency_disagreement": frequency_disagreement,
        "slope_esprit_disagreement": slope_disagreement,
        "esprit_modulus": float(abs(esprit_positive)),
        "temporal_dmd_modulus": float(abs(temporal_positive)),
        "phase_linear_fit_rms": phase_residual,
        "phase_step_coherence": coherence,
        "coordinate_radius_cv": radius_cv,
        "coordinate_radius_median": float(np.median(radii)),
    })
    gates = {
        "balanced_pair": metrics["first_to_second_ratio"] <= 1.5,
        "separated_pair": metrics["pair_to_next_gap"] >= 1.2,
        "lag_shift_consistent": espace_error <= .35,
        "esprit_unit_modulus": abs(abs(esprit_positive)-1) <= .15,
        "temporal_unit_modulus": abs(abs(temporal_positive)-1) <= .15,
        "frequency_agreement": frequency_disagreement <= .04,
        "slope_agreement": slope_disagreement <= .015,
        "phase_coherent": coherence >= .8,
        "radius_stable": radius_cv <= .8,
        "raw_temporal_dependence": abs(raw_lag1) >= .1,
    }
    gates = {name: bool(value) for name, value in gates.items()}
    return {
        "status": "resolved" if all(gates.values()) else "unresolved",
        "reason": None if all(gates.values()) else "diagnostic purity gate failed",
        "candidate_frequency": nu_slope,
        "complex_coordinate": coordinate,
        "lag_filter": lag_filter,
        "signal_mean": y.mean(axis=1),
        "normalized_character": z,
        "phase": np.angle(z),
        "phase_unwrapped": phase,
        "metrics": metrics,
        "gates": gates,
        "gate_status": "heuristic_uncalibrated",
        "window": window,
        "hankel_columns": k,
    }


def apply_ssa_character(model: dict, signals: np.ndarray) -> dict:
    """Apply a frozen lag filter on a new record; no new SVD is fitted."""
    y = np.asarray(signals, dtype=float)
    lag_filter = np.asarray(model["lag_filter"])
    center = np.asarray(model["signal_mean"])
    window, q = lag_filter.shape
    if y.ndim != 2 or y.shape[0] != q or y.shape[1] < window:
        raise ValueError("record and frozen lag-filter dimensions disagree")
    columns = y.shape[1]-window+1
    coordinate = np.zeros(columns, dtype=complex)
    centered = y-center[:, None]
    for lag in range(window):
        coordinate += np.einsum("i,in->n", lag_filter[lag],
                                 centered[:, lag:lag+columns], optimize=False)
    radius = abs(coordinate)
    return {"complex_coordinate": coordinate, "phase": np.angle(coordinate),
            "normalized_character": coordinate/np.maximum(radius, 1e-15),
            "radii": radius}


def apply_field_character(model: dict, mass_values: np.ndarray,
                           prefix_prediction=None) -> dict:
    """Apply the saved filter to its declared raw-field or residual input."""
    source = model.get("readout_source", "residual")
    if source == "raw_field":
        values = mass_values
    elif source == "residual":
        if prefix_prediction is None:
            raise ValueError("residual character requires its saved prefix prediction")
        values = mass_values-prefix_prediction
    else:
        raise ValueError("unknown saved character input source")
    return apply_ssa_character(model, values)


def aligned_character_rows(rows) -> np.ndarray:
    """All window starts share time zero; trim their different end lengths."""
    rows = [np.asarray(row) for row in rows]
    if not rows:
        return np.empty((0, 0))
    count = min(len(row) for row in rows)
    return np.asarray([row[:count] for row in rows])
