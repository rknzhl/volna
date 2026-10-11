"""Stable spatial decoding for the known degree-two field space.

The sampling Gram of iterative_fields.spatial_basis is whitened at its 12
training sensors. That does not make it orthonormal in physical L2([0,1]^2).
This module keeps the physical mass matrix separate from the sampling Gram.
The finite-dimensional sampling result is a spatial assumption, not a new
property of SSA or a guarantee for fields outside this polynomial space.
"""

from __future__ import annotations

from project_paths import RESULTS_DIR

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable

import numpy as np

from iterative_fields import AREA, SENSORS, generate_field, spatial_basis


OUT = RESULTS_DIR
CHECKS_FILE = OUT / "spatial_frame_checks.json"

# Exact L2 Gram on [0,1]^2 of [1,u,v,u^2,uv,v^2], u=2x-1, v=2y-1.
# All odd moments vanish; E[u^2]=1/3, E[u^4]=1/5, E[u^2v^2]=1/9.
RAW_MONOMIAL_MASS = np.array([
    [1, 0, 0, 1/3, 0, 1/3],
    [0, 1/3, 0, 0, 0, 0],
    [0, 0, 1/3, 0, 0, 0],
    [1/3, 0, 0, 1/5, 0, 1/9],
    [0, 0, 0, 0, 1/9, 0],
    [1/3, 0, 0, 1/9, 0, 1/5],
], dtype=float)


def raw_monomials(positions: np.ndarray) -> np.ndarray:
    points = np.asarray(positions, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("positions must have shape (points, 2)")
    u, v = (2 * points - 1).T
    return np.stack((np.ones(len(points)), u, v, u*u, u*v, v*v), axis=1)


def exact_degree_two_mass(basis_evaluator: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """Tensor Gauss-3 integrates products of degree-two polynomials exactly."""
    nodes, weights = np.polynomial.legendre.leggauss(3)
    x, y = np.meshgrid((nodes + 1)/2, (nodes + 1)/2, indexing="ij")
    positions = np.column_stack((x.ravel(), y.ravel()))
    quadrature = (weights[:, None] * weights[None, :] / 4).ravel()
    values = np.asarray(basis_evaluator(positions), dtype=float)
    if values.shape[0] != 9 or not np.isfinite(values).all():
        raise ValueError("basis evaluator returned invalid Gauss-grid values")
    return values.T @ (quadrature[:, None] * values)


def symmetric_sqrt(matrix: np.ndarray, inverse: bool = False) -> np.ndarray:
    eigenvalues, vectors = np.linalg.eigh(matrix)
    if eigenvalues[0] <= 0:
        raise ValueError("physical or sampling Gram is not positive definite")
    powers = eigenvalues**(-.5 if inverse else .5)
    return (vectors * powers) @ vectors.T


@dataclass(frozen=True)
class SamplingModel:
    positions: np.ndarray
    weights: np.ndarray
    basis_evaluator: Callable[[np.ndarray], np.ndarray]
    design: np.ndarray
    sampling_gram: np.ndarray
    physical_mass: np.ndarray
    decoder: np.ndarray
    mass_sqrt: np.ndarray
    frame_eigenvalues: np.ndarray

    @classmethod
    def from_positions(cls, positions: np.ndarray, weights: np.ndarray,
                       basis_evaluator: Callable[[np.ndarray], np.ndarray] = spatial_basis,
                       physical_mass: np.ndarray | None = None) -> "SamplingModel":
        """Construct T=(S*WS)^-1 S*W and frame bounds relative to L2(Omega).

        If physical_mass is omitted, the supplied basis must consist of
        degree-two polynomials on [0,1]^2 for Gauss-3 exactness.
        """
        points = np.asarray(positions, dtype=float)
        w = np.asarray(weights, dtype=float).reshape(-1)
        if (points.ndim != 2 or points.shape[1] != 2 or len(points) != len(w)
                or not np.isfinite(points).all() or not np.isfinite(w).all()
                or np.any(w <= 0)):
            raise ValueError("finite positions and positive matching weights required")
        design = np.asarray(basis_evaluator(points), dtype=float)
        if design.ndim != 2 or design.shape[0] != len(points):
            raise ValueError("basis evaluator returned wrong sample dimensions")
        if np.linalg.matrix_rank(design) < design.shape[1]:
            raise ValueError("sensor positions do not identify the supplied spatial field space")
        gram = design.T @ (w[:, None] * design)
        mass = (exact_degree_two_mass(basis_evaluator) if physical_mass is None
                else np.asarray(physical_mass, dtype=float))
        if mass.shape != gram.shape:
            raise ValueError("physical mass and sampling Gram dimensions disagree")
        mass = (mass + mass.T)/2
        gram = (gram + gram.T)/2
        mass_sqrt = symmetric_sqrt(mass)
        mass_inv_sqrt = symmetric_sqrt(mass, inverse=True)
        frame = mass_inv_sqrt @ gram @ mass_inv_sqrt
        frame_eigenvalues = np.linalg.eigvalsh((frame + frame.T)/2)
        if frame_eigenvalues[0] <= 0:
            raise ValueError("sensor positions do not stably sample this space")
        decoder = np.linalg.solve(gram, design.T * w[None, :])
        return cls(points, w, basis_evaluator, design, gram, mass, decoder,
                   mass_sqrt, frame_eigenvalues)

    def recover(self, signals: np.ndarray) -> np.ndarray:
        """Decode q sensor values into coefficients; time columns are optional."""
        data = np.asarray(signals, dtype=float)
        if data.ndim not in (1, 2) or data.shape[0] != len(self.positions):
            raise ValueError("signals must have sensor index as the first axis")
        if data.ndim == 1:
            return np.einsum("ij,j->i", self.decoder, data, optimize=False)
        return np.einsum("ij,jt->it", self.decoder, data, optimize=False)

    def evaluate(self, coefficients: np.ndarray, positions: np.ndarray) -> np.ndarray:
        """Evaluate reconstructed field in the same physical basis."""
        values = np.asarray(coefficients, dtype=float)
        if values.ndim not in (1, 2) or values.shape[0] != self.design.shape[1]:
            raise ValueError("coefficient dimension mismatch")
        basis = self.basis_evaluator(np.asarray(positions, dtype=float))
        if values.ndim == 1:
            return np.einsum("ij,j->i", basis, values, optimize=False)
        return np.einsum("ij,jt->it", basis, values, optimize=False)

    def field_normcoeff(self, coefficients: np.ndarray) -> float | np.ndarray:
        """Physical L2([0,1]^2) norm, scalar or one value per time column."""
        values = np.asarray(coefficients, dtype=float)
        if values.ndim == 1 and len(values) == self.design.shape[1]:
            return float(np.sqrt(values @ self.physical_mass @ values))
        if values.ndim == 2 and values.shape[0] == self.design.shape[1]:
            squared = np.einsum("it,ij,jt->t", values, self.physical_mass, values,
                                optimize=False)
            return np.sqrt(np.maximum(squared, 0))
        raise ValueError("coefficient dimension mismatch")

    def to_mass_coordinates(self, coefficients: np.ndarray) -> np.ndarray:
        """Euclidean coordinates with norm equal to the physical field norm."""
        values = np.asarray(coefficients, dtype=float)
        if values.ndim not in (1, 2) or values.shape[0] != self.design.shape[1]:
            raise ValueError("coefficient dimension mismatch")
        if values.ndim == 1:
            return np.einsum("ij,j->i", self.mass_sqrt, values, optimize=False)
        return np.einsum("ij,jt->it", self.mass_sqrt, values, optimize=False)

    def noise_covcoeff(self, sigma: float) -> np.ndarray:
        """Covariance of decoded coefficients for raw iid sensor noise."""
        if not np.isfinite(sigma) or sigma < 0:
            raise ValueError("nonnegative finite noise standard deviation required")
        return sigma**2 * (self.decoder @ self.decoder.T)

    def noise_covmass(self, sigma: float) -> np.ndarray:
        return self.mass_sqrt @ self.noise_covcoeff(sigma) @ self.mass_sqrt


SPATIAL_MODEL = SamplingModel.from_positions(SENSORS, AREA)


def frame_checks() -> dict:
    model = SPATIAL_MODEL
    rng = np.random.default_rng(1742)
    coefficients = rng.standard_normal((model.design.shape[1], 24))
    sampled = np.einsum("ij,jt->it", model.design, coefficients, optimize=False)
    recovered = model.recover(sampled)
    heldout = rng.uniform(.05, .95, (80, 2))
    heldout_roundtrip = model.evaluate(recovered, heldout)
    heldout_exact = np.einsum("ij,jt->it", spatial_basis(heldout), coefficients,
                             optimize=False)
    raw_at_sensors = raw_monomials(SENSORS)
    whitener = np.linalg.lstsq(raw_at_sensors, model.design, rcond=None)[0]
    analytic_mass = whitener.T @ RAW_MONOMIAL_MASS @ whitener
    test_times = np.arange(256)

    valid_sensors = generate_field("r2_partial_cover", test_times, SENSORS)["clean"]
    valid_heldout = generate_field("r2_partial_cover", test_times, heldout)["clean"]
    valid_decoded = model.evaluate(model.recover(valid_sensors), heldout)
    hidden_sensors = generate_field("spatial_hidden_weak_negative", test_times, SENSORS)["clean"]
    hidden_heldout = generate_field("spatial_hidden_weak_negative", test_times, heldout)["clean"]
    hidden_coefficients = model.recover(hidden_sensors)
    hidden_training_residual = np.einsum("ij,jt->it", model.design, hidden_coefficients,
                                         optimize=False) - hidden_sensors
    hidden_heldout_residual = model.evaluate(hidden_coefficients, heldout) - hidden_heldout

    checks = {
        "domain": "[0,1]^2 with physical Lebesgue L2 norm",
        "sensor_count": len(SENSORS), "basis_dimension": model.design.shape[1],
        "basis": "known degree-two polynomials whitened only in sensor quadrature",
        "physical_mass": model.physical_mass.tolist(),
        "sampling_gram": model.sampling_gram.tolist(),
        "frame_eigenvalues_physical_L2": model.frame_eigenvalues.tolist(),
        "frame_lower_bound": float(model.frame_eigenvalues[0]),
        "frame_upper_bound": float(model.frame_eigenvalues[-1]),
        "physical_mass_condition": float(np.linalg.cond(model.physical_mass)),
        "sampling_gram_condition": float(np.linalg.cond(model.sampling_gram)),
        "mass_gauss3_vs_analytic_max_error": float(np.max(abs(model.physical_mass-analytic_mass))),
        "decoder_times_sampling_identity_max_error": float(np.max(abs(
            model.decoder @ model.design - np.eye(model.design.shape[1])))),
        "coefficient_roundtrip_max_error": float(np.max(abs(recovered-coefficients))),
        "heldout_roundtrip_max_error": float(np.max(abs(heldout_roundtrip-heldout_exact))),
        "mass_coordinate_norm_max_error": float(np.max(abs(
            np.linalg.norm(model.to_mass_coordinates(coefficients), axis=0)
            - model.field_normcoeff(coefficients)))),
        "known_space_infinite_temporal_field_heldout_rmse": float(np.sqrt(np.mean(
            (valid_decoded-valid_heldout)**2))),
        "outside_known_space_hidden_sensor_residual_max": float(np.max(abs(
            hidden_training_residual))),
        "outside_known_space_hidden_heldout_rmse": float(np.sqrt(np.mean(
            hidden_heldout_residual**2))),
        "unit_sigma_coefficient_noise_covariance": model.noise_covcoeff(1).tolist(),
        "unit_sigma_mass_coordinate_noise_eigenvalues": np.linalg.eigvalsh(
            model.noise_covmass(1)).tolist(),
        "meaning": "Stable decoding holds on the specified polynomial space; the hidden negative lies outside it.",
    }
    if (checks["frame_lower_bound"] <= 0
            or checks["coefficient_roundtrip_max_error"] > 1e-10
            or checks["heldout_roundtrip_max_error"] > 1e-10
            or checks["known_space_infinite_temporal_field_heldout_rmse"] > 1e-10
            or checks["outside_known_space_hidden_sensor_residual_max"] > 1e-10
            or checks["outside_known_space_hidden_heldout_rmse"] < 1e-3):
        raise AssertionError("spatial frame or hidden-field audit failed")
    return checks


if __name__ == "__main__":
    result = frame_checks()
    CHECKS_FILE.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in (
        "frame_lower_bound", "frame_upper_bound", "physical_mass_condition",
        "sampling_gram_condition", "mass_gauss3_vs_analytic_max_error",
        "coefficient_roundtrip_max_error", "known_space_infinite_temporal_field_heldout_rmse",
        "outside_known_space_hidden_sensor_residual_max",
        "outside_known_space_hidden_heldout_rmse")}, indent=2))
