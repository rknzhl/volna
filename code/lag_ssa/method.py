"""Finite positive-lag SSA for a spatial field.

The module separates an exact geometric construction (positive moment
quadrature) from empirical phase extraction (truncated DMD on the SSA matrix).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product

import numpy as np
from scipy.linalg import qr
from scipy.optimize import linprog, minimize
from scipy.signal import find_peaks
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class LagDesign:
    lags: np.ndarray
    weights: np.ndarray
    moment_residual: float
    candidate_count: int


def voronoi_weights_1d(positions: np.ndarray, bounds: tuple[float, float]) -> np.ndarray:
    """Lengths of clipped 1D Voronoi cells, in the input sensor order."""
    x = np.asarray(positions, dtype=float).reshape(-1)
    lo, hi = bounds
    if not (lo < hi and np.all((lo <= x) & (x <= hi))):
        raise ValueError("positions must lie inside ordered bounds")
    order = np.argsort(x)
    xs = x[order]
    if np.any(np.diff(xs) <= 0):
        raise ValueError("sensor positions must be distinct")
    edges = np.r_[lo, (xs[:-1] + xs[1:]) / 2, hi]
    result = np.empty_like(xs)
    result[order] = np.diff(edges)
    return result / result.sum()


def voronoi_weights_grid(sensors: np.ndarray, resolution: int = 80) -> np.ndarray:
    """Midpoint-grid approximation of Voronoi cell volumes in [0,1]^d."""
    points = np.asarray(sensors, dtype=float)
    if points.ndim != 2 or points.shape[1] not in (1, 2, 3):
        raise ValueError("sensors must be (q,d), d=1,2,3")
    axes = [(np.arange(resolution) + .5) / resolution] * points.shape[1]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, points.shape[1])
    nearest = cKDTree(points).query(grid, workers=1)[1]
    result = np.bincount(nearest, minlength=len(points)).astype(float)
    if np.any(result == 0):
        raise ValueError("increase resolution; a sensor has no sampled Voronoi cell")
    return result / result.sum()


def moment_matrix(frequencies: np.ndarray, candidate_lags: np.ndarray) -> np.ndarray:
    """Real linear constraints for all distinct signed-frequency differences."""
    nu = np.asarray(frequencies, dtype=float).reshape(-1)
    lag = np.asarray(candidate_lags, dtype=int).reshape(-1)
    differences = []
    for a in range(len(nu)):
        for b in range(a):
            diff = abs(np.angle(np.exp(1j * (nu[a] - nu[b]))))
            if diff < 1e-9:
                raise ValueError("selected frequencies alias after sampling")
            if not any(abs(diff - old) < 1e-8 for old in differences):
                differences.append(diff)
    rows = [np.ones(len(lag))]
    for diff in differences:
        rows.extend([np.cos(diff * lag), np.sin(diff * lag)])
    return np.asarray(rows)


def design_positive_lags(
    frequencies: np.ndarray,
    max_lag: int = 256,
    first_count: int = 16,
    tolerance: float = 1e-8,
) -> LagDesign:
    """Find a sparse nonnegative quadrature on integer delays.

    `frequencies` are angular frequencies in radians per sample, including
    conjugate negative frequencies for a real signal. A linear program favors
    shorter delays. Zero weights are omitted from the returned support.
    """
    count = first_count
    while True:
        count = min(max_lag + 1, count)
        lag = np.arange(count)
        mat = moment_matrix(frequencies, lag)
        rhs = np.r_[1.0, np.zeros(mat.shape[0] - 1)]
        result = linprog(
            c=lag.astype(float), A_eq=mat, b_eq=rhs,
            bounds=(0, None), method="highs",
        )
        if result.success:
            keep = result.x > 1e-10
            selected_lags = lag[keep]
            weights = result.x[keep]
            weights /= weights.sum()
            residual = float(np.max(np.abs(
                np.einsum("ij,j->i", mat[:, keep], weights, optimize=False) - rhs)))
            if residual > tolerance:
                raise RuntimeError(f"moment residual {residual:g} exceeds tolerance")
            return LagDesign(selected_lags, weights, residual, count)
        if count == max_lag + 1:
            raise RuntimeError(f"no positive quadrature through lag {max_lag}")
        count = max(count + 1, 2 * count)


def design_min_variance_lags(frequencies: np.ndarray, horizon: int) -> LagDesign:
    """Exact positive moments with minimum squared weights on a fixed horizon.

    For independent equal-variance measurement noise, sum(w_j**2) controls
    the noise variance of a projected harmonic coefficient. This design uses
    the same candidate horizon as a classical consecutive-lag MSSA baseline.
    """
    if horizon < 2:
        raise ValueError("horizon must contain at least two candidate lags")
    candidate = np.arange(horizon)
    mat = moment_matrix(frequencies, candidate)
    rhs = np.r_[1.0, np.zeros(mat.shape[0] - 1)]
    _, upper, pivot = qr(mat.T, pivoting=True, mode="economic")
    independent = np.sum(abs(np.diag(upper)) > 1e-9)
    constraints = mat[pivot[:independent]]
    target = rhs[pivot[:independent]]
    initial = design_positive_lags(frequencies, max_lag=horizon - 1)
    start = np.zeros(horizon)
    start[initial.lags] = initial.weights
    uniform = np.ones(horizon) / horizon
    result = minimize(
        fun=lambda w: .5 * np.sum((w - uniform)**2),
        x0=start, jac=lambda w: w - uniform,
        bounds=[(0, None)] * horizon,
        constraints=[{"type": "eq", "fun": lambda w: np.einsum("ij,j->i", constraints, w, optimize=False) - target,
                      "jac": lambda w: constraints}],
        method="SLSQP", options={"ftol": 1e-13, "maxiter": 2000},
    )
    if not result.success:
        raise RuntimeError(f"minimum-variance positive quadrature failed: {result.message}")
    keep = result.x > 1e-9
    lags = candidate[keep]
    weights = result.x[keep]
    weights /= weights.sum()
    residual = float(np.max(abs(np.einsum("ij,j->i", mat[:, keep], weights, optimize=False) - rhs)))
    if residual > 1e-8:
        raise RuntimeError(f"minimum-variance moment residual {residual:g}")
    return LagDesign(lags, weights, residual, horizon)


def trajectory_matrix(
    signals: np.ndarray,
    spatial_weights: np.ndarray,
    design: LagDesign,
) -> np.ndarray:
    """Rows are weighted (lag, sensor) pairs; columns are start times."""
    y = np.asarray(signals, dtype=float)
    area = np.asarray(spatial_weights, dtype=float)
    if y.ndim != 2 or y.shape[0] != len(area) or np.any(area <= 0):
        raise ValueError("signals must be (sensors, times) and weights positive")
    if len(design.lags) != len(design.weights) or np.any(design.weights <= 0):
        raise ValueError("lag design is invalid")
    columns = y.shape[1] - int(np.max(design.lags))
    if columns < 2:
        raise ValueError("record shorter than the selected lag horizon")
    root_area = np.sqrt(area)[:, None]
    return np.vstack([
        np.sqrt(weight) * root_area * y[:, lag:lag + columns]
        for lag, weight in zip(design.lags, design.weights)
    ])


def estimate_peaks(
    signals: np.ndarray,
    spatial_weights: np.ndarray,
    count: int = 2,
    pad_factor: int = 8,
    separation_cycles: float = 6.0,
    suppress_harmonics: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Preliminary spatially aggregated Fourier peaks, in radians/sample.

    This is a screening step, not a certified estimator of primitive phases.
    """
    y = np.asarray(signals, dtype=float)
    y = y - y.mean(axis=1, keepdims=True)
    n = y.shape[1]
    nfft = 1 << int(np.ceil(np.log2(pad_factor * n)))
    window = np.hanning(n)
    power = np.sum(np.asarray(spatial_weights)[:, None]
                   * abs(np.fft.rfft(y * window, n=nfft, axis=1)) ** 2, axis=0)
    power[:max(2, int(nfft / n))] = 0
    peaks, _ = find_peaks(power, distance=max(2, int(separation_cycles * nfft / n)))
    peaks = peaks[np.argsort(power[peaks])[::-1]]
    chosen = []
    strengths = []
    resolution = 2 * np.pi / n
    for peak in peaks:
        frequency = 2 * np.pi * peak / nfft
        if suppress_harmonics and any(abs(frequency - k * old) < 2 * resolution
                                      for old in chosen for k in range(1, 6)):
            continue
        chosen.append(frequency)
        strengths.append(power[peak])
        if len(chosen) == count:
            break
    if len(chosen) < count:
        raise RuntimeError("fewer than requested spectral peaks were resolved")
    return np.asarray(chosen), np.asarray(strengths)


def infer_frequency_lattice(
    peaks: np.ndarray,
    tolerance: float,
    max_rank: int = 3,
    max_order: int = 6,
) -> dict:
    """Find the lowest-rank bounded integer lattice fitting observed peaks.

    This is conditional on the supplied peak list and harmonic order bound;
    it cannot certify that arbitrarily weak phases are absent.
    """
    nu = np.asarray(peaks, dtype=float).reshape(-1)
    if len(nu) == 0:
        raise ValueError("at least one peak is needed")
    best = None
    for rank in range(1, min(max_rank, len(nu)) + 1):
        integers = np.array(list(product(range(-max_order, max_order + 1), repeat=rank)))
        integers = integers[np.any(integers != 0, axis=1)]
        for chosen in combinations(range(len(nu)), rank):
            basis = nu[list(chosen)]
            predictions = np.einsum("ij,j->i", integers, basis, optimize=False)
            distance = abs(np.angle(np.exp(1j * (nu[:, None] - predictions[None, :]))))
            assignment = np.argmin(distance, axis=1)
            error = distance[np.arange(len(nu)), assignment]
            if np.max(error) > tolerance:
                continue
            representation = integers[assignment]
            score = (np.sum(abs(representation)), np.sum(basis), np.sum(error))
            if best is None or score < best["score"]:
                best = {"rank": rank, "basis": basis.tolist(),
                        "basis_peak_indices": list(chosen),
                        "coefficients": representation.tolist(),
                        "max_relation_error": float(np.max(error)),
                        "score": score}
        if best is not None:
            break
    if best is None:
        return {"rank": None, "basis": [], "basis_peak_indices": [],
                "coefficients": [], "max_relation_error": None,
                "reason": "no bounded integer lattice fits the supplied peaks"}
    best.pop("score")
    return best


def dmd_phase_coordinates(
    trajectory: np.ndarray,
    target_frequencies: np.ndarray,
    rank: int = 8,
) -> dict:
    """Estimate complex SSA/DMD coordinates near target positive frequencies.

    The procedure is empirical: it does not establish the torus guarantee.
    """
    h = np.asarray(trajectory, dtype=float)
    centered = h - h.mean(axis=1, keepdims=True)
    x0, x1 = centered[:, :-1], centered[:, 1:]
    if min(x0.shape) <= 256:
        u, singular, vh = np.linalg.svd(x0, full_matrices=False)
    else:
        # Deterministic randomized SVD keeps a long classical MSSA window
        # computationally comparable to the sparse lag design.
        sketch_rank = min(min(x0.shape), rank + 16)
        rng = np.random.default_rng(101)
        omega = rng.standard_normal((x0.shape[1], sketch_rank))
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            q, _ = np.linalg.qr(x0 @ omega, mode="reduced")
            for _ in range(2):
                q, _ = np.linalg.qr(x0 @ (x0.T @ q), mode="reduced")
            small = q.T @ x0
        if not (np.isfinite(q).all() and np.isfinite(small).all()):
            raise FloatingPointError("randomized SVD produced nonfinite values")
        small_u, singular, vh = np.linalg.svd(small, full_matrices=False)
        u = np.einsum("ij,jk->ik", q, small_u, optimize=False)
    usable = int(np.sum(singular > singular[0] * 1e-10))
    rank = min(rank, usable)
    u, singular, vh = u[:, :rank], singular[:rank], vh[:rank]
    ux1 = np.einsum("ij,jk->ik", u.T, x1, optimize=False)
    reduced_shift = np.einsum("ij,jk->ik", ux1, vh.T, optimize=False) / singular[None, :]
    eigenvalues, eigenvectors = np.linalg.eig(reduced_shift)
    projection = np.einsum("ij,jk->ik", u.T, centered, optimize=False)
    coordinates = np.linalg.solve(eigenvectors, projection)
    used = set()
    indices = []
    for target in np.asarray(target_frequencies):
        order = np.argsort(abs(eigenvalues - np.exp(1j * target)))
        choice = next(int(k) for k in order if int(k) not in used)
        indices.append(choice)
        used.add(choice)
    selected = coordinates[indices]
    return {
        "phases": np.angle(selected),
        "radii": np.abs(selected),
        "eigenvalues": eigenvalues[indices],
        "singular_values": singular,
        "all_eigenvalues": eigenvalues,
    }


def circular_mae(estimate: np.ndarray, truth: np.ndarray) -> float:
    """Mean phase error, allowing one constant origin shift per coordinate."""
    delta = np.asarray(estimate) - np.asarray(truth)
    offset = np.angle(np.mean(np.exp(1j * delta), axis=1, keepdims=True))
    return float(np.mean(abs(np.angle(np.exp(1j * (delta - offset))))))


def torus_xyz(phases: np.ndarray, major: float = 2.0, minor: float = 0.7) -> np.ndarray:
    """Standard 3D torus chart from two recovered phase coordinates."""
    theta, phi = phases
    radius = major + minor * np.cos(phi)
    return np.stack([radius * np.cos(theta), radius * np.sin(theta),
                     minor * np.sin(phi)], axis=-1)
