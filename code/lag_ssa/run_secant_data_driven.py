"""Data-trained secant lag design, checked on an independent true field.

The design uses noisy first-half observations and an explicit conditional
rank-two Fourier model. The true generator is used only for evaluation.
This tests geometry, not end-to-end phase recovery or a rigorous certificate.
"""

from __future__ import annotations

from project_paths import RESULTS_DIR

import json
from pathlib import Path

import numpy as np

from method import estimate_peaks
from run_controlled import AREA, ORIGIN, case_data, field
from run_secant_pilot import J, L, greedy_support, phase_pairs, score, secant_matrix
from run_spectral_baseline import refine_frequency


OUT = RESULTS_DIR
TRAIN_LENGTH = 1200
NOISE = .08
TRAIN_PAIRS = 2500
VALIDATION_PAIRS = 20000
SEEDS = list(range(10))


def harmonics(max_order: int = 4) -> np.ndarray:
    return np.asarray([(a, b) for a in range(-max_order, max_order + 1)
                       for b in range(-max_order, max_order + 1)
                       if 0 < abs(a) + abs(b) <= max_order
                       and (a > 0 or (a == 0 and b > 0))], dtype=int)


MODES = harmonics()


def basis(phases: np.ndarray) -> np.ndarray:
    angles = np.einsum("ij,jk->ik", MODES, phases, optimize=False)
    out = np.empty((phases.shape[1], 1 + 2 * len(MODES)))
    out[:, 0] = 1.0
    out[:, 1::2] = np.cos(angles).T
    out[:, 2::2] = np.sin(angles).T
    return out


def fitted_model(observed: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    peaks, powers = estimate_peaks(observed, AREA, count=8, suppress_harmonics=False)
    visible = peaks[powers / powers[0] > .0005]
    if len(visible) < 2:
        raise RuntimeError("fewer than two visible peaks")
    frequencies = np.sort([refine_frequency(observed, x) for x in np.sort(visible)[:2]])
    phases = frequencies[:, None] * np.arange(observed.shape[1])[None, :]
    design = basis(phases)
    ridge = .1
    normal = np.einsum("ij,ik->jk", design, design, optimize=False)
    normal += ridge * np.diag(np.r_[0.0, np.ones(design.shape[1]-1)])
    rhs = np.einsum("ij,ik->jk", design, observed.T, optimize=False)
    coefficients = np.linalg.solve(normal, rhs)
    prediction = np.einsum("ij,jk->ik", design, coefficients, optimize=False)
    residual = np.sqrt(np.mean((prediction - observed.T)**2))
    info = {"frequencies": frequencies.tolist(), "visible_peak_count": len(visible),
            "training_rmse": float(residual), "fourier_terms": len(MODES), "ridge": ridge}
    return frequencies, coefficients, info


def fitted_secants(
    first: np.ndarray, second: np.ndarray,
    frequencies: np.ndarray, coefficients: np.ndarray,
) -> np.ndarray:
    delta = np.angle(np.exp(1j * (first - second)))
    denominator = np.sum(delta*delta, axis=0)
    out = np.empty((first.shape[1], L))
    for lag in range(L):
        shift = frequencies[:, None] * lag
        a = np.einsum("ij,jk->ik", basis(first + shift), coefficients, optimize=False)
        b = np.einsum("ij,jk->ik", basis(second + shift), coefficients, optimize=False)
        difference = (a - b).T
        out[:, lag] = np.sum(AREA[:, None] * difference*difference, axis=0) / denominator
    return out


def main() -> None:
    training_pairs = phase_pairs(np.random.default_rng(4201), TRAIN_PAIRS)
    validation_pairs = phase_pairs(np.random.default_rng(9202), VALIDATION_PAIRS)
    rows = []
    for case in ["two_modes", "infinite_weak", "infinite_strong", "weak_primitive"]:
        phases, frequency, nonlinear, weak = case_data(case)
        clean = field(phases[:, :TRAIN_LENGTH], nonlinear, weak)
        scale = np.sqrt(np.mean(np.sum(AREA[:, None]*clean*clean, axis=0)))
        true_validation = secant_matrix(validation_pairs[0]+ORIGIN[:, None],
                                        validation_pairs[1]+ORIGIN[:, None],
                                        frequency, nonlinear, weak)
        reference_train = secant_matrix(training_pairs[0]+ORIGIN[:, None],
                                        training_pairs[1]+ORIGIN[:, None],
                                        frequency, nonlinear, weak)
        oracle_weights, _, oracle_lags = greedy_support(reference_train, J)
        consecutive = np.zeros(L); consecutive[:J] = 1/J
        full = np.ones(L)/L
        for seed in SEEDS:
            observed = clean + NOISE*scale*np.random.default_rng(6100+seed).standard_normal(clean.shape)
            try:
                estimated_frequency, coefficients, fit_info = fitted_model(observed)
                surrogate = fitted_secants(*training_pairs, estimated_frequency, coefficients)
                weights, training_objective, lags = greedy_support(surrogate, J)
                scores = {"data_design": score(true_validation, weights),
                          "oracle_design": score(true_validation, oracle_weights),
                          "consecutive_9": score(true_validation, consecutive),
                          "uniform_full_64": score(true_validation, full)}
                row = {"case": case, "seed": seed, "computed": True,
                       "fit": fit_info, "selected_lags": lags,
                       "selected_weights": [float(weights[lag]) for lag in lags],
                       "sampled_training_objective": training_objective,
                       "true_geometry_independent_pairs": scores,
                       "frequency_abs_error": np.abs(np.sort(estimated_frequency)-np.sort(frequency)).tolist(),
                       "oracle_lags_for_evaluation_only": oracle_lags,
                       "oracle_weights_for_evaluation_only": [float(oracle_weights[lag]) for lag in oracle_lags]}
            except (RuntimeError, np.linalg.LinAlgError) as exc:
                row = {"case": case, "seed": seed, "computed": False, "reason": str(exc)}
            rows.append(row)
        valid = [r for r in rows if r["case"] == case and r["computed"]]
        print(case, len(valid), {name: round(float(np.median([
            r["true_geometry_independent_pairs"][name]["sample_min"] for r in valid])), 4)
            for name in ["data_design", "oracle_design", "consecutive_9", "uniform_full_64"]},
            flush=True)
    payload = {"description": "conditional rank-two data-trained secant design; validation sample minima are upper bounds on true gamma",
               "training_samples": TRAIN_LENGTH, "noise_rms_fraction": NOISE,
               "training_pairs": TRAIN_PAIRS, "validation_pairs": VALIDATION_PAIRS,
               "lag_budget": J, "horizon": L, "seeds": SEEDS, "rows": rows}
    (OUT / "secant_data_driven.json").write_text(json.dumps(payload, indent=2)+"\n")


if __name__ == "__main__":
    main()
