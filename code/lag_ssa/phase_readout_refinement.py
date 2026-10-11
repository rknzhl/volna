"""Frozen convolution readout of one data-estimated field character.

Input values are already spatial mass coordinates. The frequency is estimated
elsewhere from observations. No oscillator is used as the output phase:
each output is a linear functional of one observed field delay window.
This is a tapered harmonic projection, not a new Fourier transform.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import lstsq


@dataclass(frozen=True)
class FieldCharacterObserver:
    frequency: float
    window: int
    spatial_profile: np.ndarray
    signal_mean: np.ndarray
    weights: np.ndarray
    training_samples: int

    def apply(self, mass_values: np.ndarray) -> dict:
        values = np.asarray(mass_values, dtype=float)
        if values.ndim != 2 or values.shape[0] != len(self.signal_mean) or values.shape[1] < self.window:
            raise ValueError("matching field coordinates and at least one full window required")
        projected = np.einsum("q,qt->t", np.conj(self.spatial_profile),
                              values - self.signal_mean[:, None], optimize=False)
        windows = np.lib.stride_tricks.sliding_window_view(projected, self.window)
        lag_filter = self.weights * np.exp(-1j * self.frequency * np.arange(self.window))
        coordinate = np.einsum("nl,l->n", windows, lag_filter, optimize=False)
        radius = abs(coordinate)
        character = coordinate / np.maximum(radius, 1e-15)
        return {"coordinate": coordinate, "normalized_character": character,
                "radii": radius, "phase": np.angle(character)}

    def metadata(self) -> dict:
        return {"frequency_rad_per_sample": self.frequency, "window": self.window,
                "window_shape": "nonnegative Hann weights, normalized to unit sum",
                "profile_training_samples": self.training_samples,
                "spatial_profile_real": self.spatial_profile.real.tolist(),
                "spatial_profile_imag": self.spatial_profile.imag.tolist(),
                "phase_source": "linear convolution of the observed field, not inferred clocks",
                "profile_estimator": "Hann-weighted constant/cos/sin regression at learned frequency"}


def fit_field_character_observer(mass_values: np.ndarray, frequency: float,
                                 window: int) -> FieldCharacterObserver:
    values = np.asarray(mass_values, dtype=float)
    if values.ndim != 2 or values.shape[1] <= window or window < 4 or not np.isfinite(values).all():
        raise ValueError("finite training field longer than the valid window required")
    times = np.arange(values.shape[1])
    design = np.column_stack((np.ones(len(times)), np.cos(frequency * times),
                              np.sin(frequency * times)))
    root = np.sqrt(np.hanning(len(times)))
    coefficients = lstsq(root[:, None] * design, root[:, None] * values.T,
                         lapack_driver="gelsy")[0]
    profile = .5 * (coefficients[1] - 1j * coefficients[2])
    if np.linalg.norm(profile) <= 1e-12:
        raise ValueError("learned character has zero spatial profile")
    weights = np.hanning(window)
    weights /= weights.sum()
    return FieldCharacterObserver(float(frequency), int(window), profile.copy(),
                                  coefficients[0].copy(), weights, values.shape[1])


def observer_quality(readout: dict, frequency: float, noise_covariance: np.ndarray,
                     observer: FieldCharacterObserver) -> dict:
    z = np.asarray(readout["normalized_character"])
    times = np.arange(len(z))
    radius = np.asarray(readout["radii"])
    projected_variance = float(np.real(np.einsum("i,ij,j->", np.conj(observer.spatial_profile),
                                                noise_covariance, observer.spatial_profile,
                                                optimize=False)))
    noise_rms = np.sqrt(max(0., projected_variance) * np.sum(observer.weights ** 2))
    return {"columns": len(z), "radius_cv": float(radius.std() / max(radius.mean(), 1e-15)),
            "phase_step_coherence": float(abs(np.mean(z[1:] * np.conj(z[:-1])))),
            "inferred_clock_coherence_diagnostic": float(abs(np.mean(z * np.exp(-1j * frequency * times)))),
            "median_coordinate_radius": float(np.median(radius)),
            "noise_only_coordinate_rms_diagnostic": float(noise_rms),
            "median_coordinate_radius_over_noise_rms_diagnostic": float(np.median(radius) / max(noise_rms, 1e-15))}
