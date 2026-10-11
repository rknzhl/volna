"""Oracle pilot of spatial-Gram-balanced delay weights.

The finite Fourier set below omits an analytic infinite tail.  Its cross-error
is a truncated diagnostic, not a certificate for the full field.
"""

from __future__ import annotations

from project_paths import RESULTS_DIR

import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from method import LagDesign, dmd_phase_coordinates, trajectory_matrix
from run_controlled import AREA, PROFILES, case_data, field, phase_error_permutation


OUT = RESULTS_DIR
OUT.mkdir(exist_ok=True)


def finite_modes(frequency: np.ndarray, nonlinear: float, weak: float) -> tuple[np.ndarray, np.ndarray]:
    a, b, c = PROFILES
    positive = [
        ((1, 0), weak * a / 2),
        ((2, 0), a / 2),
        ((0, 1), b / 2),
    ] + [((k, k), nonlinear * c * .6**k) for k in range(1, 6)]
    modes, profiles = [], []
    for m, profile in positive:
        for sign in (1, -1):
            modes.append([sign * z for z in m])
            profiles.append(profile)
    return np.asarray(modes), np.asarray(profiles)


def cross_terms(modes: np.ndarray, profiles: np.ndarray,
                frequency: np.ndarray, area: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    delays = np.arange(horizon)
    rows, strengths = [], []
    for i in range(len(modes)):
        for j in range(i):
            spatial_gram = abs(np.sum(area * profiles[i] * profiles[j]))
            size = float(np.linalg.norm(modes[i]) * np.linalg.norm(modes[j]) * spatial_gram)
            if size < 1e-12:
                continue
            angular = (modes[i] - modes[j]) @ frequency
            rows.append(np.exp(1j * angular * delays))
            strengths.append(size)
    return np.asarray(rows), np.asarray(strengths)


def balanced_weights(rows: np.ndarray, strengths: np.ndarray,
                     horizon: int, relative_penalty: float) -> np.ndarray:
    cosine, sine = rows.real, rows.imag
    penalty = relative_penalty * strengths.sum()
    uniform = np.ones(horizon) / horizon

    def objective(weights: np.ndarray) -> float:
        real = np.einsum("ij,j->i", cosine, weights, optimize=False)
        imag = np.einsum("ij,j->i", sine, weights, optimize=False)
        radius = np.sqrt(real**2 + imag**2 + 1e-10)
        return float(np.dot(strengths, radius) + penalty * np.sum(weights**2))

    def gradient(weights: np.ndarray) -> np.ndarray:
        real = np.einsum("ij,j->i", cosine, weights, optimize=False)
        imag = np.einsum("ij,j->i", sine, weights, optimize=False)
        radius = np.sqrt(real**2 + imag**2 + 1e-10)
        return (np.einsum("ij,i->j", cosine, strengths * real / radius, optimize=False)
                + np.einsum("ij,i->j", sine, strengths * imag / radius, optimize=False)
                + 2 * penalty * weights)

    result = minimize(objective, uniform, jac=gradient,
                      bounds=[(0, None)] * horizon,
                      constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1,
                                    "jac": lambda w: np.ones(horizon)}],
                      method="SLSQP", options={"maxiter": 1000, "ftol": 1e-10})
    if not result.success:
        raise RuntimeError(result.message)
    return result.x / result.x.sum()


def main() -> None:
    truth, frequency, nonlinear, weak = case_data("weak_primitive")
    clean = field(truth, nonlinear, weak)
    scale = np.sqrt(np.mean(np.sum(AREA[:, None] * clean**2, axis=0)))
    observed = clean + .08 * scale * np.random.default_rng(20261007).standard_normal(clean.shape)
    horizon = 32
    modes, profiles = finite_modes(frequency, nonlinear, weak)
    rows, strengths = cross_terms(modes, profiles, frequency, AREA, horizon)
    uniform = np.ones(horizon) / horizon
    variants = [("spatial_uniform", uniform, AREA),
                ("classical_mssa", uniform, np.ones(len(AREA))/len(AREA))]
    for penalty in (0.0, .1, 1.0, 10.0):
        variants.append((f"gram_balanced_penalty_{penalty:g}",
                         balanced_weights(rows, strengths, horizon, penalty), AREA))
    results = []
    for label, weights, spatial in variants:
        keep = weights > 1e-8
        design = LagDesign(np.arange(horizon)[keep], weights[keep]/weights[keep].sum(),
                           float("nan"), horizon)
        h = trajectory_matrix(observed, spatial, design)
        dmd = dmd_phase_coordinates(h, frequency, rank=8)
        moment = np.einsum("ij,j->i", rows, weights, optimize=False)
        results.append({
            "design": label, "lag_count": len(design.lags),
            "effective_lag_count": float(1 / np.sum(weights**2)),
            "truncated_cross_error": float(np.dot(strengths, abs(moment))),
            "phase_mae_rad": phase_error_permutation(dmd["phases"], truth[:, :h.shape[1]]),
        })
        print(label, round(results[-1]["truncated_cross_error"], 4),
              round(results[-1]["phase_mae_rad"], 4), flush=True)
    payload = {"case": "weak_primitive", "horizon": horizon, "noise_relative_rms": .08,
               "oracle_coefficients": True, "mixed_harmonics_included_through_order": 5,
               "full_tail_certified": False, "seed": 20261007, "results": results}
    (OUT / "gram_balance.json").write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
