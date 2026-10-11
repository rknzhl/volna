"""Targeted omitted-harmonic audit of the dominant/weak iterative SSA case.

Starts come from saved train-selected SSA candidates, never physical truth.
A finite nuisance-harmonic model is selected by training BIC. The untouched
future block and physical frequencies are used only for scoring. This assay
does not replace conditional SSA or certify an infinite Fourier tail.
"""

from __future__ import annotations

from project_paths import RESULTS_DIR

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from scipy.linalg import lstsq
from scipy.optimize import minimize

from frequency_fit import fit_joint_frequencies
from iterative_fields import AREA, generate_field
from iterative_ssa import fit_hierarchical_model
from spatial_field import SPATIAL_MODEL


OUT = RESULTS_DIR
SOURCE = OUT / "iterative_frequency_bias_starts.json"
DESTINATION = OUT / "iterative_frequency_bias.json"
CASE = "r2_poisson_dominant_weak"
ORDER_GRID = (1, 2, 4, 8, 12, 16, 24)


def design_and_labels(times: np.ndarray, frequencies: np.ndarray,
                      order: int) -> tuple[np.ndarray, np.ndarray]:
    labels = np.zeros((order+1, 2), dtype=float)
    labels[:order, 0] = np.arange(1, order+1)
    labels[-1, 1] = 1
    angles = times[:, None]*(labels@frequencies)[None, :]
    return np.column_stack((np.ones(len(times)), np.cos(angles), np.sin(angles))), labels


def fit_nuisance_harmonics(observed: np.ndarray, starting: np.ndarray,
                          order: int, half_width_cycles: float = .35) -> dict:
    """Fit bases nu1,nu2; nuisance lines h*nu1 have free sensor amplitudes."""
    q, n = observed.shape
    times = np.arange(n, dtype=float)-(n-1)/2
    target = observed.T
    initial = np.asarray(starting, dtype=float)

    def profile(frequencies):
        design, labels = design_and_labels(times, frequencies, order)
        coefficients, _, rank, _ = lstsq(design, target, lapack_driver="gelsy")
        if rank != design.shape[1]:
            raise np.linalg.LinAlgError("nuisance harmonic design is rank deficient")
        fitted = np.einsum("nj,ji->ni", design, coefficients, optimize=False)
        residual = target-fitted
        count = len(labels)
        angles = times[:, None]*(labels@frequencies)[None, :]
        line_derivatives = times[:, None, None]*(
            -np.sin(angles)[:, :, None]*coefficients[1:1+count][None, :, :]
            +np.cos(angles)[:, :, None]*coefficients[1+count:][None, :, :])
        derivatives = np.einsum("nhq,hj->njq", line_derivatives, labels,
                                optimize=False)
        return design, coefficients, residual, derivatives

    def objective(displacement):
        _, _, residual, derivatives = profile(initial+displacement/n)
        rss = float(np.sum(residual**2))
        gradient = -2*np.einsum("ni,nji->j", residual, derivatives,
                                optimize=False)/n
        return rss/(q*n), gradient/(q*n)

    width = 2*np.pi*half_width_cycles
    result = minimize(objective, np.zeros(2), jac=True, method="L-BFGS-B",
                      bounds=[(-width, width)]*2,
                      options={"ftol": 1e-13, "gtol": 1e-10, "maxiter": 200})
    frequencies = initial+result.x/n
    _, coefficients, residual, _ = profile(frequencies)
    rss = float(np.sum(residual**2))
    parameters = q*(2*(order+1)+1)+2
    bic = q*n*np.log(rss/(q*n))+parameters*np.log(q*n)
    return {"frequencies": frequencies, "coefficients": coefficients,
            "order": order, "rss": rss, "bic": float(bic),
            "parameters": parameters, "times_center": (n-1)/2,
            "optimizer_success": bool(result.success),
            "optimizer_message": str(result.message),
            "iterations": int(result.nit)}


def fit_fixed_amplitudes(observed: np.ndarray, frequencies: np.ndarray,
                         order: int) -> dict:
    n = observed.shape[1]
    center = (n-1)/2
    design, _ = design_and_labels(np.arange(n)-center, frequencies, order)
    coefficients = lstsq(design, observed.T, lapack_driver="gelsy")[0]
    return {"frequencies": frequencies, "coefficients": coefficients,
            "order": order, "times_center": center}


def predict_harmonic(model: dict, times: np.ndarray) -> np.ndarray:
    design, _ = design_and_labels(times-model["times_center"],
                                  model["frequencies"], model["order"])
    return np.einsum("nj,ji->in", design, model["coefficients"], optimize=False)


def score_decomposition(observed: np.ndarray, nuisance: dict) -> dict:
    """Train-only omitted-tail score for a misspecified two-line model.

    Linearized frequency displacement is I^{-1} J*P residual, where P removes
    the fitted constant/sinusoid amplitudes. Nuisance harmonics provide a
    data-fitted decomposition, without using physical component labels.
    """
    n, q = observed.shape[1], observed.shape[0]
    times = np.arange(n)-nuisance["times_center"]
    frequencies = nuisance["frequencies"]
    raw_design, _ = design_and_labels(times, frequencies, 1)
    coefficients = lstsq(raw_design, observed.T, lapack_driver="gelsy")[0]
    angles = times[:, None]*frequencies[None, :]
    derivatives = times[:, None, None]*(
        -np.sin(angles)[:, :, None]*coefficients[1:3][None, :, :]
        +np.cos(angles)[:, :, None]*coefficients[3:][None, :, :])
    flat = derivatives.reshape(n, 2*q)
    projected = np.einsum("nj,ji->ni", raw_design,
        lstsq(raw_design, flat, lapack_driver="gelsy")[0], optimize=False)
    if not np.isfinite(projected).all():
        raise FloatingPointError("nonfinite efficient-score projection")
    efficient = (flat-projected).reshape(n, 2, q)
    information = np.einsum("nji,nki->jk", efficient, efficient, optimize=False)
    fitted = np.einsum("nj,ji->ni", raw_design, coefficients, optimize=False)
    residual = observed.T-fitted
    total_score = np.einsum("ni,nji->j", residual, efficient, optimize=False)
    full_design, _ = design_and_labels(times, frequencies, nuisance["order"])
    count = nuisance["order"]+1
    rows = []
    tail_score = np.zeros(2)
    for harmonic in range(2, nuisance["order"]+1):
        index = harmonic-1
        field = (full_design[:, 1+index, None]*nuisance["coefficients"][1+index]
                 +full_design[:, 1+count+index, None]
                 *nuisance["coefficients"][1+count+index])
        score = np.einsum("ni,nji->j", field, efficient, optimize=False)
        tail_score += score
        rows.append({"harmonic": harmonic, "efficient_score": score.tolist()})
    return {"reference_frequencies_train_fitted": frequencies.tolist(),
            "total_efficient_score": total_score.tolist(),
            "estimated_nuisance_tail_efficient_score": tail_score.tolist(),
            "remainder_efficient_score": (total_score-tail_score).tolist(),
            "linearized_total_frequency_shift": np.linalg.solve(information, total_score).tolist(),
            "linearized_nuisance_tail_frequency_shift": np.linalg.solve(information, tail_score).tolist(),
            "by_omitted_harmonic": rows,
            "meaning": "local score diagnosis; finite fitted tail, not an infinite-tail bound"}


def summary(model: dict) -> dict:
    return {key: (value.tolist() if isinstance(value, np.ndarray) else value)
            for key, value in model.items() if key != "coefficients" and key != "residual"}


def main() -> None:
    # Snapshot the original training-selected starts so later pilot reruns do
    # not silently change this regression-bias assay.
    saved = json.loads(SOURCE.read_text())
    if saved["case"] != CASE or saved["physical_truth_supplied"]:
        raise ValueError("requires the declared train-only candidate snapshot")
    settings = saved["settings"]
    n_train, n_validation, n_future = (settings[key] for key in
                                      ("n_train", "n_validation", "n_future"))
    # These are saved candidates selected from training observations only.
    starting = np.asarray(saved["starting_frequencies_train_selected"], dtype=float)
    times = np.arange(n_train+n_validation+n_future)
    generated = generate_field(CASE, times, n_train=n_train)
    clean = generated["clean"]
    sigma = saved["relative_sensor_noise"]*float(np.sqrt(np.mean(np.sum(
        AREA[:, None]*clean[:, :n_train]**2, axis=0))))
    observed = clean+sigma*np.random.default_rng(saved["seed"]).standard_normal(clean.shape)
    train = observed[:, :n_train]
    future_times = times[n_train+n_validation:]
    future_clean = clean[:, n_train+n_validation:]
    raw_joint = fit_joint_frequencies(train, starting, half_width_cycles=.35)
    fits = [fit_nuisance_harmonics(train, starting, order) for order in ORDER_GRID]
    selected = min(fits, key=lambda model: model["bic"])
    strategies = {"freeze_residual_candidate": starting,
                  "raw_two_line_joint": raw_joint["frequencies"],
                  "train_selected_nuisance_harmonics": selected["frequencies"]}
    mass_train = SPATIAL_MODEL.to_mass_coordinates(SPATIAL_MODEL.recover(train))
    evaluation = {}
    # No physical frequency or clean future value enters these fitted models.
    for name, frequencies in strategies.items():
        harmonic = fit_fixed_amplitudes(train, frequencies, selected["order"])
        hierarchy = fit_hierarchical_model(np.arange(n_train), mass_train,
            [np.ones((1, 1), dtype=int), np.eye(2, dtype=int)], frequencies,
            tuple(settings["k_candidates"]), settings["lag_window"])
        harmonic_future = predict_harmonic(harmonic, future_times)
        hierarchy_coeff = np.linalg.solve(SPATIAL_MODEL.mass_sqrt,
                                         hierarchy.predict(future_times))
        hierarchy_future = SPATIAL_MODEL.evaluate(hierarchy_coeff, SPATIAL_MODEL.positions)
        evaluation[name] = {"frequencies": frequencies.tolist(),
            "future_harmonic_sensor_rmse": float(np.sqrt(np.mean((harmonic_future-future_clean)**2))),
            "future_hierarchy_sensor_rmse": float(np.sqrt(np.mean((hierarchy_future-future_clean)**2))),
            "common_harmonic_amplitude_order": selected["order"],
            "hierarchy_selected_k": [component.selected_k for component in hierarchy.components]}

    # A single clean-data comparison isolates misspecification from noise.
    clean_raw = fit_joint_frequencies(clean[:, :n_train], starting, half_width_cycles=.35)
    clean_nuisance = fit_nuisance_harmonics(clean[:, :n_train], starting, selected["order"])
    # Begin physical-truth scoring only after every optimization is finished.
    true_frequencies = np.asarray(generated["metadata"]["alpha"])
    for row in evaluation.values():
        error = np.angle(np.exp(1j*(np.asarray(row["frequencies"])-true_frequencies)))
        row["frequency_error_truth_only"] = error.tolist()
        row["weak_phase_drift_at_future_start_truth_only_rad"] = float(error[1]*(n_train+n_validation))
        row["weak_phase_drift_over_training_truth_only_rad"] = float(error[1]*n_train)
    payload = {"case": CASE, "source": str(SOURCE),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": saved["seed"], "settings": settings,
        "noise_sigma_truth_only": sigma,
        "saved_training_starts": starting.tolist(),
        "truth_supplied_to_fitting": False, "future_used_for_fitting_or_order_selection": False,
        "nuisance_harmonic_order_grid": list(ORDER_GRID), "order_selection": "training BIC",
        "selected_harmonic_order": selected["order"],
        "nuisance_order_fits": [summary(model) for model in fits],
        "raw_joint": summary(raw_joint), "strategies": evaluation,
        "train_score_decomposition": score_decomposition(train, selected),
        "clean_data_misspecification_control": {
            "raw_two_line_frequencies": clean_raw["frequencies"].tolist(),
            "nuisance_harmonic_frequencies": clean_nuisance["frequencies"].tolist(),
            "raw_two_line_errors_truth_only": (clean_raw["frequencies"]-true_frequencies).tolist(),
            "nuisance_harmonic_errors_truth_only": (clean_nuisance["frequencies"]-true_frequencies).tolist(),
            "harmonic_order_selected_on_noisy_training": selected["order"]},
        "true_frequencies_scoring_only": true_frequencies.tolist(),
        "limits": ["one fixture and noise realization", "training BIC is finite-model selection",
                   "nuisance model need not represent all unknown infinite families or mixed modes",
                   "no infinite-tail frequency-bias bound or calibrated post-selection confidence"]}
    OUT.mkdir(exist_ok=True)
    DESTINATION.write_text(json.dumps(payload, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"selected_order": selected["order"], "strategies": evaluation,
                     "clean_control": payload["clean_data_misspecification_control"],
                     "score": {key: value for key, value in payload["train_score_decomposition"].items()
                               if key != "by_omitted_harmonic"}}, indent=2))


if __name__ == "__main__":
    main()
