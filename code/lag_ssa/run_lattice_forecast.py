"""Out-of-sample predictive check for saved conditional lattice decisions.

Both forecasts use the same selected finite set of sinusoidal lines. Their
frequencies are frozen from frequency_covariance.json; only sensor-specific
linear coefficients are fitted on the regenerated training record. Truth is
used solely to generate the field and to score clean future error.

The reported z compares *future predictive losses* conditional on the two
frozen forecasts. It is not a confidence statement about lattice equality or
phase rank: training selection, forecast-parameter uncertainty, and omitted
infinite harmonics are not covered by the conditional Gaussian calculation.
"""

from __future__ import annotations

from project_paths import RESULTS_DIR

import json
from pathlib import Path

import numpy as np

from frequency_fit import sinusoid_design
from run_character_ssa import ALPHA, AREA, CASES, N, Q, RHO, poisson, spatial_profiles


OUT = RESULTS_DIR
DESTINATION = OUT / "lattice_forecast.json"
FUTURE_N = N
FUTURE_SEED_OFFSET = 100_000
ONE_SIDED_99_PERCENT_Z = 2.3263478740408408
CASE_BY_NAME = {case.name: case for case in CASES}


def clean_field_at_times(case, times: np.ndarray) -> np.ndarray:
    """Extend the same spatial field and phase orbit beyond the training N."""
    origin = np.array([.23, 1.08])[:case.dimension]
    phase = origin[:, None] + ALPHA[:case.dimension, None] * times[None, :]
    profiles = spatial_profiles(len(case.modes))
    field = np.zeros((Q, len(times)), dtype=float)
    for j, (mode, amplitude) in enumerate(zip(case.modes, case.amplitudes)):
        angle = np.einsum("i,in->n", np.asarray(mode, dtype=float), phase,
                          optimize=False) + .21 * j
        temporal = poisson(angle) if case.infinite else np.cos(angle)
        field += amplitude * profiles[:, j, None] * temporal[None, :]
    return field


def fixed_frequency_forecast(observed_train: np.ndarray, frequencies: np.ndarray,
                             train_times: np.ndarray,
                             future_times: np.ndarray) -> tuple[np.ndarray, float]:
    """Fit only intercept/cos/sin coefficients, independently for each sensor."""
    x_train = sinusoid_design(frequencies, train_times)
    coefficients, _, rank, _ = np.linalg.lstsq(x_train, observed_train.T, rcond=None)
    if rank != x_train.shape[1]:
        raise np.linalg.LinAlgError("fixed-frequency sinusoid design is rank deficient")
    fitted_train = np.einsum("nj,ji->ni", x_train, coefficients, optimize=False)
    residual = observed_train.T - fitted_train
    forecast = np.einsum("nj,ji->ni", sinusoid_design(frequencies, future_times),
                         coefficients, optimize=False)
    return forecast.T, float(np.sum(residual * residual))


def constrained_frequencies(lattice: dict, free: np.ndarray) -> tuple[np.ndarray, float, bool]:
    labels = np.asarray(lattice["peak_labels"], dtype=np.int64)
    basis = np.asarray(lattice["basis"], dtype=float)
    if labels.shape != (len(free), len(basis)):
        raise ValueError("lattice label and frequency dimensions disagree")
    signed = np.angle(np.exp(1j * (labels @ basis)))
    discrepancy = float(np.max(abs(np.angle(np.exp(1j * (signed - free))))))
    if discrepancy > .005 or np.any((signed <= 0) | (signed >= np.pi)):
        raise ValueError("lattice reconstruction is not near positive joint frequencies")
    # p=r leaves no frequency constraint. In exact arithmetic M*B*free=free
    # modulo 2*pi; use the identical numerical vector as an equivalence control.
    no_constraint = len(free) == len(basis)
    if no_constraint:
        if discrepancy > 1e-8:
            raise ValueError("unconstrained lattice control changed a frequency")
        return free.copy(), discrepancy, True
    return signed, discrepancy, False


def evaluate(row: dict, saved: dict) -> dict:
    case = CASE_BY_NAME[row["case"]]
    noise = float(row["noise"])
    seed = int(row["seed"])
    free = np.asarray(row["joint_fit"]["frequencies"], dtype=float)
    lattice = row["new_lattice"]
    constrained, frequency_discrepancy, no_constraint = constrained_frequencies(lattice, free)
    if len(saved["screen"]["refined"]) != len(free):
        raise ValueError("saved peak count disagrees with the joint fit")

    train_index = np.arange(N, dtype=float)
    future_index = np.arange(N, N + FUTURE_N, dtype=float)
    clean_train = clean_field_at_times(case, train_index)
    clean_future = clean_field_at_times(case, future_index)
    scale = float(np.sqrt(np.mean(np.sum(AREA[:, None] * clean_train**2, axis=0))))
    actual_sigma = noise * scale
    observed_train = clean_train + actual_sigma * np.random.default_rng(seed).standard_normal(clean_train.shape)
    observed_future = clean_future + actual_sigma * np.random.default_rng(
        seed + FUTURE_SEED_OFFSET).standard_normal(clean_future.shape)

    center = (N - 1) / 2
    train_times = train_index - center
    future_times = future_index - center
    pred_free, train_rss_free = fixed_frequency_forecast(
        observed_train, free, train_times, future_times)
    if no_constraint:
        pred_constrained, train_rss_constrained = pred_free, train_rss_free
    else:
        pred_constrained, train_rss_constrained = fixed_frequency_forecast(
            observed_train, constrained, train_times, future_times)

    saved_train_rss = float(row["joint_fit"]["rss"])
    reproduction_error = abs(train_rss_free - saved_train_rss) / max(1.0, saved_train_rss)
    if reproduction_error > 1e-6:
        raise ArithmeticError("regenerated training record does not reproduce saved fit")
    sigma_hat = float(np.sqrt(row["joint_fit"]["noise_variance_estimate"]))
    delta = pred_free - pred_constrained
    delta_norm = float(np.linalg.norm(delta))
    rss_free = float(np.sum((observed_future - pred_free)**2))
    rss_constrained = float(np.sum((observed_future - pred_constrained)**2))
    difference = rss_constrained - rss_free
    # Conditional on frozen forecasts, independent Gaussian future noise gives
    # Var(D) = 4*sigma^2*||pred_free-pred_constrained||^2. Under equal expected
    # future loss, z is standard normal if sigma is known; plug-in sigma_hat
    # makes the threshold an approximate diagnostic.
    z = (difference / (2 * sigma_hat * delta_norm)
         if delta_norm > 1e-10 else None)
    clean_rmse_free = float(np.sqrt(np.mean((clean_future - pred_free)**2)))
    clean_rmse_constrained = float(np.sqrt(np.mean((clean_future - pred_constrained)**2)))

    return {
        "case": case.name, "noise": noise, "seed": seed,
        "finite_signal": not case.infinite,
        "new_module_complete": bool(row["new_module_complete"]),
        "selected_peak_count": len(free), "lattice_rank": int(lattice["rank"]),
        "no_frequency_constraint_control": no_constraint,
        "max_lattice_frequency_discrepancy": frequency_discrepancy,
        "train_rss_reproduction_relative_error": reproduction_error,
        "train_rss_free": train_rss_free,
        "train_rss_constrained": train_rss_constrained,
        "sigma_hat_from_joint_train_fit": sigma_hat,
        "actual_noise_sigma_evaluation_only": actual_sigma,
        "future_rss_free": rss_free,
        "future_rss_constrained": rss_constrained,
        "future_loss_difference_constrained_minus_free": difference,
        "future_forecast_difference_norm": delta_norm,
        "predictive_loss_z": z,
        "reject_equal_predictive_loss_in_favor_of_free_at_one_sided_99_percent":
            bool(z is not None and z > ONE_SIDED_99_PERCENT_Z),
        "clean_future_rmse_free_evaluation_only": clean_rmse_free,
        "clean_future_rmse_constrained_evaluation_only": clean_rmse_constrained,
        "clean_future_loss_difference_evaluation_only": float(
            np.sum((clean_future - pred_constrained)**2)
            - np.sum((clean_future - pred_free)**2)),
    }


def summarize(rows: list[dict]) -> list[dict]:
    summary = []
    for finite_signal in (True, False):
        for complete in (True, False):
            group = [row for row in rows if row["finite_signal"] == finite_signal
                     and row["new_module_complete"] == complete]
            if not group:
                continue
            effective = [row for row in group if row["predictive_loss_z"] is not None]
            summary.append({
                "finite_signal": finite_signal,
                "new_module_complete": complete,
                "accepted_lattices": len(group),
                "no_constraint_controls": sum(row["no_frequency_constraint_control"]
                                              for row in group),
                "positive_forecast_difference": sum(
                    row["future_loss_difference_constrained_minus_free"] > 0
                    for row in effective),
                "predictive_loss_rejections_at_one_sided_99_percent": sum(
                    row["reject_equal_predictive_loss_in_favor_of_free_at_one_sided_99_percent"]
                    for row in group),
                "clean_future_free_better_evaluation_only": sum(
                    row["clean_future_loss_difference_evaluation_only"] > 0
                    for row in group),
                "median_sigma_hat_over_actual_noise_evaluation_only": float(np.median([
                    row["sigma_hat_from_joint_train_fit"]
                    / row["actual_noise_sigma_evaluation_only"] for row in group])),
                "median_predictive_loss_z_with_constraint":
                    float(np.median([row["predictive_loss_z"] for row in effective]))
                    if effective else None,
                "median_clean_rmse_difference_constrained_minus_free":
                    float(np.median([row["clean_future_rmse_constrained_evaluation_only"]
                                     - row["clean_future_rmse_free_evaluation_only"]
                                     for row in group])),
            })
    return summary


def main() -> None:
    frequency_source = json.loads((OUT / "frequency_covariance.json").read_text())
    character_source = json.loads((OUT / "character_ssa.json").read_text())
    if len(frequency_source["rows"]) != 240 or not frequency_source["complete"]:
        raise ValueError("expected the complete saved 240-row frequency experiment")
    saved_by_key = {(item["case"], float(item["noise"]), int(item["seed"])): item
                    for item in character_source["rows"]}
    rows = []
    for source_row in frequency_source["rows"]:
        lattice = source_row.get("new_lattice", {})
        if (lattice.get("rank") is None or
                not source_row.get("joint_fit", {}).get("stationary_in_frequency_uncertainty")):
            continue
        key = (source_row["case"], float(source_row["noise"]), int(source_row["seed"]))
        rows.append(evaluate(source_row, saved_by_key[key]))
    payload = {
        "description": "Independent future-block predictive-loss check of saved lattice decisions",
        "source": ["frequency_covariance.json", "character_ssa.json"],
        "train_samples": N, "future_samples": FUTURE_N,
        "future_noise_seed_offset": FUTURE_SEED_OFFSET,
        "same_selected_finite_harmonic_lines_both_forecasts": True,
        "frequency_estimates_frozen_before_future_block": True,
        "only_linear_sensor_coefficients_refit_on_train": True,
        "one_sided_99_percent_z_threshold": ONE_SIDED_99_PERCENT_Z,
        "conditional_gaussian_forecast_loss_diagnostic_only": True,
        "does_not_certify_lattice_rank": True,
        "training_selection_and_forecast_parameter_uncertainty_not_covered": True,
        "infinite_harmonic_tail_bias_not_bounded": True,
        "eligible_rows": len(rows), "summary": summarize(rows), "rows": rows,
    }
    DESTINATION.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"eligible_rows": len(rows), "summary": payload["summary"]}, indent=2))


if __name__ == "__main__":
    main()
