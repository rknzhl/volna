"""Frozen joint-MSSA column encoder with a lag-zero field decoder.

The spatial metric and delay window match the torus encoder. Validation rank
selection uses plug-in SURE, not reconstruction against the same noisy input.
For iid raw sensor noise, the divergence of the lag0 decoder with respect to
the measured lag0 block equals trace(P_k[lag0,lag0]) after any invertible
spatial mass transformation. Overlapping windows affect variance, not this
pointwise expectation. A training-only estimated sigma makes this diagnostic
SURE; no finite-sample calibration or noise-band validity is asserted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh, solve
from scipy.linalg.blas import dgemm


RANK_GRID = (2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512)


def trajectory(values, mean, mass_sqrt, window):
    values = np.asarray(values, dtype=float)
    if values.shape[1] < window:
        raise ValueError("one complete delay window is required")
    mass = np.einsum("ij,jt->it", mass_sqrt, values-mean[:, None], optimize=False)
    count = values.shape[1]-window+1
    return np.asfortranarray(np.vstack([mass[:, lag:lag+count] for lag in range(window)])/np.sqrt(window))


@dataclass
class MSSAColumnEncoder:
    mean: np.ndarray
    mass_sqrt: np.ndarray
    window: int
    basis: np.ndarray
    selected_rank: int
    sigma_train: float
    selection: list
    numeric_rank: int
    numeric_rank_threshold: float
    singular_values: np.ndarray

    def encode(self, values):
        hankel = trajectory(values, self.mean, self.mass_sqrt, self.window)
        u = np.asfortranarray(self.basis[:, :self.selected_rank])
        coordinate = dgemm(1., u, hankel, trans_a=True)
        mass_lag0 = np.sqrt(self.window)*dgemm(1., np.asfortranarray(u[:len(self.mean)]), coordinate)
        sensor_lag0 = solve(self.mass_sqrt, mass_lag0, assume_a="pos", check_finite=True)+self.mean[:, None]
        if not np.isfinite(sensor_lag0).all():
            raise FloatingPointError("nonfinite MSSA column decoder")
        return {"prediction": sensor_lag0, "column_coordinate": coordinate}

    def metadata(self):
        return {"method": "classical joint weighted-MSSA column encoder and lag0 decoder",
                "window": self.window, "selected_rank": self.selected_rank,
                "declared_rank_grid": list(RANK_GRID), "numeric_training_hankel_rank": self.numeric_rank,
                "numeric_rank_singular_threshold": self.numeric_rank_threshold,
                "rank_selection": "minimum validation plug-in SURE on RAW sensor lag0 error; exact ties choose lower rank",
                "SURE_formula": "mean(||prediction_lag0-observed_lag0||^2)+2*sigma_train^2*trace(P00)-q*sigma_train^2",
                "SURE_reference_assumption": "iid additive raw sensor noise, equal variance sigma^2; independently trained frozen mean and basis",
                "noise_variance_known_to_inference": False,
                "noise_scale_source": "training-only high-band estimate; finite-band dominance assumption",
                "SURE_finite_sample_noise_calibration_verified": False,
                "overlapping_windows_are_independent": False,
                "validation_selection_curve": self.selection,
                "decoding_information": "the same observed future window as the torus encoder, no future clocks or recurrence",
                "diagonal_averaging_used": False,
                "reason_for_lag0_decode": "preserve identical per-window input context; full diagonal averaging uses wider overlapping context"}


def fit_mssa_column_encoder(training, validation, mass_sqrt, sigma_train, window=128):
    training, validation = np.asarray(training), np.asarray(validation)
    q, n = training.shape
    if not np.isfinite(sigma_train) or sigma_train <= 0:
        raise ValueError("positive finite training noise estimate required")
    mean = training.mean(axis=1)
    hankel = trajectory(training, mean, mass_sqrt, window)
    # All audit cases have ambient dimension below the number of columns.
    # Retain a Gram fallback so the declared rank grid stays feasible.
    if hankel.shape[0] <= hankel.shape[1]:
        covariance = dgemm(1., hankel, hankel, trans_b=True)
        eigenvalues, eigenvectors = eigh(covariance, check_finite=True, driver="evr")
        order = np.argsort(eigenvalues)[::-1]
        singular = np.sqrt(np.maximum(0., eigenvalues[order]))
        threshold = float(singular[0]*max(hankel.shape)*np.finfo(float).eps)
        numeric_rank = int(np.sum(singular>threshold))
        maximum = min(max(RANK_GRID), numeric_rank)
        basis = np.asfortranarray(eigenvectors[:, order[:maximum]])
    else:
        gram = dgemm(1., hankel, hankel, trans_a=True)
        eigenvalues, eigenvectors = eigh(gram, check_finite=True, driver="evr")
        order = np.argsort(eigenvalues)[::-1]
        singular = np.sqrt(np.maximum(0., eigenvalues[order]))
        threshold = float(singular[0]*max(hankel.shape)*np.finfo(float).eps)
        numeric_rank = int(np.sum(singular>threshold))
        maximum = min(max(RANK_GRID), numeric_rank)
        basis = dgemm(1., hankel, np.asfortranarray(eigenvectors[:,order[:maximum]]))/singular[:maximum][None,:]
        basis = np.asfortranarray(basis)
    if maximum <= 0:
        raise ValueError("no nonzero numerical Hankel direction")
    validation_hankel = trajectory(validation, mean, mass_sqrt, window)
    coordinates = dgemm(1., basis, validation_hankel, trans_a=True)
    grid = sorted({min(rank, maximum) for rank in RANK_GRID})
    target = validation[:, :validation_hankel.shape[1]]
    selection = []
    for rank in grid:
        mass_prediction = np.sqrt(window)*dgemm(1., np.asfortranarray(basis[:q,:rank]),
                                               np.asfortranarray(coordinates[:rank]))
        prediction = solve(mass_sqrt, mass_prediction, assume_a="pos", check_finite=True)+mean[:,None]
        measured_loss = float(np.mean(np.sum((prediction-target)**2,axis=0)))
        trace_p00 = float(np.sum(basis[:q,:rank]**2))
        sure = measured_loss+2*sigma_train*sigma_train*trace_p00-q*sigma_train*sigma_train
        selection.append({"rank":rank,"validation_observed_raw_lag0_mse_sum_over_sensors":measured_loss,
                          "trace_P00":trace_p00,"validation_SURE_sum_over_sensors":sure,
                          "validation_columns":validation_hankel.shape[1],
                          "rank_grid_value_clipped_to_feasible_max":rank not in RANK_GRID})
    selected = min(selection,key=lambda row:(row["validation_SURE_sum_over_sensors"],row["rank"]))
    return MSSAColumnEncoder(mean,np.asarray(mass_sqrt).copy(),int(window),basis,
        selected["rank"],float(sigma_train),selection,numeric_rank,threshold,singular)
