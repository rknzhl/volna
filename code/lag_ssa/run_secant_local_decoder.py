"""Local inversion of learned delay maps, separate from SVD/DMD readout.

The Fourier model and lag design use only the noisy training half. Each
held-out test column is matched independently to a 64x64 phase grid.
This is a conditional model-based decoder, not a new SSA decomposition.
"""

from __future__ import annotations

from project_paths import RESULTS_DIR

import json
from pathlib import Path

import numpy as np

from method import trajectory_matrix
from run_controlled import AREA, case_data, field, phase_error_permutation
from run_secant_data_driven import J, L, NOISE, TRAIN_LENGTH, MODES, basis, fitted_model
from run_secant_phase import lag_design


OUT = RESULTS_DIR
GRID_SIZE = 64


def templates(grid: np.ndarray, frequency: np.ndarray,
              coefficients: np.ndarray, design) -> np.ndarray:
    return np.vstack([
        np.sqrt(weight*AREA)[:, None]
        * np.einsum("ij,jk->ik", basis(grid + frequency[:, None]*lag),
                    coefficients, optimize=False).T
        for lag, weight in zip(design.lags, design.weights)
    ])


def model_and_jacobian(phases: np.ndarray, coefficients: np.ndarray):
    angles = np.einsum("ij,jk->ik", MODES, phases, optimize=False)
    cosine, sine = np.cos(angles), np.sin(angles)
    features = np.empty((phases.shape[1], 1+2*len(MODES)))
    derivatives = np.zeros((2, phases.shape[1], features.shape[1]))
    features[:, 0] = 1.0
    features[:, 1::2] = cosine.T
    features[:, 2::2] = sine.T
    for coordinate in range(2):
        derivatives[coordinate, :, 1::2] = (-MODES[:, coordinate, None]*sine).T
        derivatives[coordinate, :, 2::2] = (MODES[:, coordinate, None]*cosine).T
    value = np.einsum("ij,jk->ik", features, coefficients, optimize=False).T
    jacobian = np.stack([np.einsum("ij,jk->ik", part, coefficients,
                                  optimize=False).T for part in derivatives])
    return value, jacobian


def refine(initial: np.ndarray, h: np.ndarray, frequency: np.ndarray,
           coefficients: np.ndarray, design, steps: int = 6) -> np.ndarray:
    phase = initial.copy()
    for _ in range(steps):
        a00 = np.full(h.shape[1], 1e-4)
        a11 = np.full(h.shape[1], 1e-4)
        a01 = np.zeros(h.shape[1])
        b0 = np.zeros(h.shape[1])
        b1 = np.zeros(h.shape[1])
        for block, (lag, weight) in enumerate(zip(design.lags, design.weights)):
            value, jacobian = model_and_jacobian(phase+frequency[:, None]*lag, coefficients)
            scaling = np.sqrt(weight*AREA)[:, None]
            residual = h[block*len(AREA):(block+1)*len(AREA)] - scaling*value
            d0, d1 = scaling*jacobian[0], scaling*jacobian[1]
            a00 += np.sum(d0*d0, axis=0)
            a11 += np.sum(d1*d1, axis=0)
            a01 += np.sum(d0*d1, axis=0)
            b0 += np.sum(d0*residual, axis=0)
            b1 += np.sum(d1*residual, axis=0)
        determinant = a00*a11-a01*a01
        delta = np.stack([(a11*b0-a01*b1)/determinant,
                          (a00*b1-a01*b0)/determinant])
        phase += np.clip(delta, -.2, .2)
    return np.mod(phase, 2*np.pi)


def decode(observed: np.ndarray, grid: np.ndarray,
           frequency: np.ndarray, coefficients: np.ndarray, design, k: int) -> tuple[np.ndarray, np.ndarray]:
    template = templates(grid, frequency, coefficients, design)
    h = trajectory_matrix(observed, AREA, design)[:, :k]
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        squared = np.sum(template*template, axis=0)[:, None] - 2*(template.T @ h)
    if not np.isfinite(squared).all():
        raise FloatingPointError("nonfinite template distances")
    coarse = grid[:, np.argmin(squared, axis=0)]
    return coarse, refine(coarse, h, frequency, coefficients, design)


def main() -> None:
    training_result = json.loads((OUT / "secant_data_driven.json").read_text())
    axis = np.linspace(0, 2*np.pi, GRID_SIZE, endpoint=False)
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    grid = np.stack([xx.ravel(), yy.ravel()])
    k = 1200-L+1
    rows = []
    for record in training_result["rows"]:
        case, seed = record["case"], record["seed"]
        phases, _, nonlinear, weak = case_data(case)
        clean_train = field(phases[:, :TRAIN_LENGTH], nonlinear, weak)
        train_scale = np.sqrt(np.mean(np.sum(AREA[:, None]*clean_train*clean_train, axis=0)))
        train = clean_train + NOISE*train_scale*np.random.default_rng(6100+seed).standard_normal(clean_train.shape)
        frequency, coefficients, _ = fitted_model(train)
        truth = phases[:, TRAIN_LENGTH:]
        clean_test = field(truth, nonlinear, weak)
        test_scale = np.sqrt(np.mean(np.sum(AREA[:, None]*clean_test*clean_test, axis=0)))
        test = clean_test + NOISE*test_scale*np.random.default_rng(8100+seed).standard_normal(clean_test.shape)
        variants = {
            "data_secant_9": lag_design(record["selected_lags"], record["selected_weights"]),
            "consecutive_9": lag_design(list(range(J)), [1/J]*J),
            "uniform_full_64": lag_design(list(range(L)), [1/L]*L),
        }
        errors = {}
        coarse_errors = {}
        for name, design in variants.items():
            coarse, estimate = decode(test, grid, frequency, coefficients, design, k)
            errors[name] = phase_error_permutation(estimate, truth[:, :k])
            coarse_errors[name] = phase_error_permutation(coarse, truth[:, :k])
        errors["direct_frequency_train"] = phase_error_permutation(
            frequency[:, None]*np.arange(k)[None, :], truth[:, :k])
        test_frequency, _, _ = fitted_model(test)
        errors["direct_frequency_test"] = phase_error_permutation(
            test_frequency[:, None]*np.arange(k)[None, :], truth[:, :k])
        rows.append({"case": case, "seed": seed, "phase_mae_rad": errors,
                     "coarse_grid_mae_rad": coarse_errors})
    summary = []
    names = ["data_secant_9", "consecutive_9", "uniform_full_64",
             "direct_frequency_train", "direct_frequency_test"]
    for case in ["two_modes", "infinite_weak", "infinite_strong", "weak_primitive"]:
        selected = [r for r in rows if r["case"] == case]
        median = {name: float(np.median([r["phase_mae_rad"][name] for r in selected]))
                  for name in names}
        failures = {name: sum(r["phase_mae_rad"][name] > .3 for r in selected)
                    for name in names}
        summary.append({"case": case, "median_phase_mae_rad": median,
                        "failures_over_0_3rad": failures})
        print(case, {key: round(value, 4) for key, value in median.items()}, flush=True)
    payload = {"description": "independent per-column 64x64 phase-grid inversion plus six damped Gauss-Newton steps",
               "training_samples": TRAIN_LENGTH, "test_samples_scored": k,
               "noise_rms_fraction": NOISE, "grid_size_per_phase": GRID_SIZE,
               "lag_budget": J, "horizon": L, "rows": rows, "summary": summary}
    (OUT / "secant_local_decoder.json").write_text(json.dumps(payload, indent=2)+"\n")


if __name__ == "__main__":
    main()
