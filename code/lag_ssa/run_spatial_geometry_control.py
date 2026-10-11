"""Controls for the distinction between spatial field and labelled channels.

The known spatial space is a declared modelling assumption. Coordinates
couple sensor observations through S; a diagonal sensor norm alone does
not impose that constraint. Noise calibration uses raw iid Gaussian noise.
"""

from __future__ import annotations

from project_paths import RESULTS_DIR

import json
from pathlib import Path

import numpy as np
from scipy.stats import chi2

from iterative_fields import AREA, SENSORS, generate_field, spatial_basis
from spatial_field import SPATIAL_MODEL, SamplingModel


def main():
    model = SPATIAL_MODEL
    rng = np.random.default_rng(55000)
    heldout = rng.uniform(.03, .97, (100, 2))
    times = np.arange(2400)
    sensor_field = generate_field("r2_partial_cover", times)["clean"]
    heldout_field = generate_field("r2_partial_cover", times, heldout)["clean"]
    coefficients = model.recover(sensor_field)
    physical_metric = np.einsum("ai,ab,bj->ij", model.decoder,
                                model.physical_mass, model.decoder,
                                optimize=False)
    field_norm2 = np.asarray(model.field_normcoeff(coefficients))**2
    sensor_norm2 = np.einsum("it,ij,jt->t", sensor_field, physical_metric,
                              sensor_field, optimize=False)
    permutation = rng.permutation(len(SENSORS))
    permuted_model = SamplingModel.from_positions(SENSORS[permutation], AREA[permutation])
    reordered_coefficients = permuted_model.recover(sensor_field[permutation])
    wrong_coefficients = model.recover(sensor_field[permutation])
    wrong_sensor_fit = model.evaluate(wrong_coefficients, SENSORS)
    wrong_spatial_prediction = model.evaluate(wrong_coefficients, heldout)
    correct_spatial_prediction = model.evaluate(coefficients, heldout)

    line_positions = np.column_stack((np.linspace(.05, .95, 12), np.full(12, .5)))
    try:
        SamplingModel.from_positions(line_positions, np.full(12, 1/12))
        line_rejected = False
    except ValueError:
        line_rejected = True

    # Euclidean orthogonal residual has (q-M)*N independent Gaussian degrees
    # of freedom, regardless of the temporal complexity inside the space.
    sensor_basis, _ = np.linalg.qr(model.design, mode="reduced")
    sigma = .08
    dof = (len(SENSORS)-model.design.shape[1])*len(times)
    sigma_variance_ratios = []
    for seed in range(55000, 55030):
        observed = sensor_field+sigma*np.random.default_rng(seed).standard_normal(sensor_field.shape)
        projection = np.einsum("ij,kj,kt->it", sensor_basis, sensor_basis,
                                observed, optimize=False)
        residual = observed-projection
        sigma_variance_ratios.append(float(np.sum(residual**2)/(dof*sigma**2)))
    lower, upper = chi2.ppf((.005, .995), dof)/dof

    raw_noise = sigma*rng.standard_normal((len(SENSORS), 40000))
    mass_noise = model.to_mass_coordinates(model.recover(raw_noise))
    empirical_covariance = np.cov(mass_noise)
    predicted_covariance = model.noise_covmass(sigma)
    covariance_relative_error = float(np.linalg.norm(empirical_covariance-predicted_covariance, 2)
                                      /np.linalg.norm(predicted_covariance, 2))

    # A visible degree-three profile lies outside V_6 and produces a coherent
    # sensor residual. A hidden smooth profile can violate V_6 invisibly.
    extra_profile = (2*SENSORS[:, 0]-1)**3
    outside_signal = extra_profile[:, None]*np.cos(.43*times)[None, :]
    outside_residual = outside_signal-model.evaluate(model.recover(outside_signal), SENSORS)
    hidden_sensor = generate_field("spatial_hidden_weak_negative", times)["clean"]
    hidden_true_field = generate_field("spatial_hidden_weak_negative", times, heldout)["clean"]
    hidden_coefficients = model.recover(hidden_sensor)
    hidden_sensor_residual = hidden_sensor-model.evaluate(hidden_coefficients, SENSORS)
    hidden_field_prediction = model.evaluate(hidden_coefficients, heldout)

    answer = {
        "spatial_model": "known degree-two polynomial field on [0,1]^2",
        "sensor_count": len(SENSORS), "spatial_dimension": model.design.shape[1],
        "spatial_sampling_matrix": model.design.tolist(),
        "sensor_field_metric_T_transpose_mass_T": physical_metric.tolist(),
        "metric_rank": int(np.linalg.matrix_rank(physical_metric)),
        "off_diagonal_metric_fraction": float(np.linalg.norm(
            physical_metric-np.diag(np.diag(physical_metric)))/np.linalg.norm(physical_metric)),
        "norm_identity_max_error": float(np.max(abs(sensor_norm2-field_norm2))),
        "matched_sensor_permutation_coefficient_max_error": float(np.max(abs(
            reordered_coefficients-coefficients))),
        "correct_heldout_field_rmse": float(np.sqrt(np.mean(
            (correct_spatial_prediction-heldout_field)**2))),
        "mismatched_sensor_coordinates_residual_rms": float(np.sqrt(np.mean(
            (wrong_sensor_fit-sensor_field[permutation])**2))),
        "mismatched_sensor_coordinates_heldout_field_rmse": float(np.sqrt(np.mean(
            (wrong_spatial_prediction-heldout_field)**2))),
        "collinear_sensor_layout_rejected": line_rejected,
        "noise_variance_ratios": sigma_variance_ratios,
        "noise_variance_ratio_mean": float(np.mean(sigma_variance_ratios)),
        "noise_variance_ratio_empirical_sd": float(np.std(sigma_variance_ratios, ddof=1)),
        "noise_variance_ratio_exact_sd": float(np.sqrt(2/dof)),
        "noise_variance_ratio_99pct_interval": [float(lower), float(upper)],
        "noise_variance_interval_coverage_out_of_30": int(sum(lower <= value <= upper
                                                               for value in sigma_variance_ratios)),
        "mass_noise_covariance_relative_operator_error": covariance_relative_error,
        "mass_noise_covariance_eigenvalues": np.linalg.eigvalsh(predicted_covariance).tolist(),
        "visible_outside_model_sensor_residual_rms": float(np.sqrt(np.mean(outside_residual**2))),
        "hidden_outside_model_sensor_residual_max": float(np.max(abs(hidden_sensor_residual))),
        "hidden_outside_model_heldout_field_rmse": float(np.sqrt(np.mean(
            (hidden_field_prediction-hidden_true_field)**2))),
        "scope": "Spatial consistency and noise controls; they do not certify phase rank or new SSA novelty.",
    }
    assert line_rejected
    assert answer["norm_identity_max_error"] < 1e-10
    assert answer["matched_sensor_permutation_coefficient_max_error"] < 1e-10
    assert answer["correct_heldout_field_rmse"] < 1e-10
    assert covariance_relative_error < .05
    assert answer["mismatched_sensor_coordinates_heldout_field_rmse"] > .05
    assert answer["hidden_outside_model_sensor_residual_max"] < 1e-10
    destination = RESULTS_DIR/"spatial_geometry_control.json"
    destination.write_text(json.dumps(answer, indent=2, allow_nan=False)+"\n")
    print(json.dumps({key: value for key, value in answer.items()
                      if not isinstance(value, list)}, indent=2))


if __name__ == "__main__":
    main()
