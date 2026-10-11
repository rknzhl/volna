"""Numerical checks of nontrivial identities and one identifiability counterexample."""

from __future__ import annotations

from project_paths import RESULTS_DIR

import json
from pathlib import Path

import numpy as np

from method import design_positive_lags, trajectory_matrix
from run_controlled import AREA, FREQUENCY, ORIGIN, PROFILES, SENSORS, field, theoretical_tail


def main() -> None:
    frequency = FREQUENCY
    design = design_positive_lags(np.r_[frequency, -frequency])
    a, b, _ = PROFILES
    rng = np.random.default_rng(77)
    theta = rng.uniform(-np.pi, np.pi, (2, 100))
    phi = rng.uniform(-np.pi, np.pi, (2, 100))

    def embedded(phases: np.ndarray) -> np.ndarray:
        result = []
        for lag, weight in zip(design.lags, design.weights):
            shifted = phases + frequency[:, None] * lag
            result.append(np.sqrt(weight * AREA)[:, None] * field(shifted, 0.0))
        return np.vstack(result)

    observed_squared = np.sum((embedded(theta) - embedded(phi))**2, axis=0)
    coef_a = a / 2
    coef_b = b / 2
    expected_squared = (
        2 * np.sum(AREA * coef_a**2) * abs(np.exp(1j*theta[0])-np.exp(1j*phi[0]))**2
        + 2 * np.sum(AREA * coef_b**2) * abs(np.exp(1j*theta[1])-np.exp(1j*phi[1]))**2
    )
    identity_error = float(np.max(abs(observed_squared - expected_squared)))
    assert identity_error < 1e-12

    samples = 200
    times = np.arange(samples)
    phases = ORIGIN[:, None] + frequency[:, None] * times[None, :]
    clean = field(phases, .05)
    noise = .1 * rng.standard_normal(clean.shape)
    clean_h = trajectory_matrix(clean, AREA, design)
    noisy_h = trajectory_matrix(clean + noise, AREA, design)
    per_time_bound = max(np.linalg.norm(np.sqrt(AREA)[:, None] * noise, axis=0))
    column_error = float(max(np.linalg.norm(noisy_h-clean_h, axis=0)))
    assert column_error <= per_time_bound + 1e-12

    # At 6 Hz sampled at 100 Hz, the first phase occupies exactly 50 values.
    n = np.arange(1000)
    sampled_first_phase = ORIGIN[0] + 2*np.pi*.06*n
    invisible_perturbation = np.sin(50 * (sampled_first_phase - ORIGIN[0]))
    alias_error = float(max(abs(invisible_perturbation)))
    assert alias_error < 1e-11

    bound = theoretical_tail(.05, None)
    assert bound["certified"]

    # Circular delay copies only rotate each Fourier coefficient.  Summing
    # their weighted spectral energies gives exactly the original spectrum.
    circular_samples = 256
    random_signal = rng.standard_normal((len(SENSORS), circular_samples))
    original_fft = np.fft.fft(random_signal, axis=1)
    lagged_fft = np.vstack([
        np.fft.fft(np.sqrt(weight * AREA)[:, None]
                   * np.roll(random_signal, -int(lag), axis=1), axis=1)
        for lag, weight in zip(design.lags, design.weights)
    ])
    spectral_original = np.sum(AREA[:, None] * abs(original_fft)**2, axis=0)
    spectral_lagged = np.sum(abs(lagged_fft)**2, axis=0)
    spectral_identity_relative_error = float(
        np.max(abs(spectral_original - spectral_lagged)) / np.max(spectral_original))
    assert spectral_identity_relative_error < 1e-12

    payload = {
        "moment_residual": design.moment_residual,
        "finite_mode_metric_identity_max_error": identity_error,
        "max_noisy_column_error": column_error,
        "max_single_snapshot_noise_norm": float(per_time_bound),
        "alias_invisible_perturbation_max_on_samples": alias_error,
        "analytic_infinite_tail_certificate": bound,
        "circular_ssa_fourier_energy_identity_relative_error": spectral_identity_relative_error,
    }
    destination = RESULTS_DIR / "method_checks.json"
    destination.write_text(json.dumps(payload, indent=2) + "\n")
    print(destination)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
