"""Train-selected harmonic nuisance fits for a small acquired peak list.

Only multiples of acquired frequencies are proposed; no physical labels,
latent rank, or true frequencies are used. The dictionary is finite and omits
general mixed characters. Reported Fisher covariance describes local iid
noise sensitivity conditional on the selected dictionary, not tail bias or
post-selection coverage.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import lstsq
from scipy.optimize import minimize
from scipy.signal import find_peaks


HARMONIC_ORDERS = (1, 2, 4, 8, 16, 24)


def _gap(first: float, second: float) -> float:
    return float(abs(np.angle(np.exp(1j*(first-second)))))


def _line_dictionary(initial: np.ndarray, order: int,
                     tolerance: float) -> tuple[np.ndarray, list[dict]]:
    """Retain acquired lines; merge nuisance duplicates in the local chart."""
    p = len(initial)
    rows = [row.copy() for row in np.eye(p, dtype=int)]
    frequencies = list(initial)
    omitted = []
    for harmonic in range(2, order+1):
        for coordinate in range(p):
            frequency = harmonic*initial[coordinate]
            edge_gap = min(_gap(frequency, 0.), _gap(frequency, np.pi))
            if edge_gap <= tolerance:
                omitted.append({"coordinate": coordinate, "harmonic": harmonic,
                                "reason": "near constant or Nyquist line",
                                "gap": edge_gap})
                continue
            distances = [min(_gap(frequency, old), _gap(frequency, -old))
                         for old in frequencies]
            nearest = int(np.argmin(distances))
            if distances[nearest] <= tolerance:
                omitted.append({"coordinate": coordinate, "harmonic": harmonic,
                                "reason": "near existing real sinusoid line",
                                "retained_label": rows[nearest].tolist(),
                                "gap": distances[nearest]})
                continue
            row = np.zeros(p, dtype=int)
            row[coordinate] = harmonic
            rows.append(row)
            frequencies.append(frequency)
    return np.asarray(rows, dtype=int), omitted


def _design(times: np.ndarray, frequencies: np.ndarray,
            labels: np.ndarray) -> np.ndarray:
    line_frequencies = np.einsum("ij,j->i", labels, frequencies, optimize=False)
    angles = times[:, None]*line_frequencies[None, :]
    return np.column_stack((np.ones(len(times)), np.cos(angles), np.sin(angles)))


def _fit_dictionary(observed: np.ndarray, initial: np.ndarray,
                    labels: np.ndarray, half_width_cycles: float,
                    noise_sigma: float | None) -> dict:
    q, n = observed.shape
    p, count = len(initial), len(labels)
    times = np.arange(n, dtype=float)-(n-1)/2
    target = observed.T

    def profile(frequencies):
        design = _design(times, frequencies, labels)
        coefficients, _, rank, _ = lstsq(design, target, lapack_driver="gelsy")
        if rank != design.shape[1]:
            raise np.linalg.LinAlgError("harmonic nuisance design is rank deficient")
        residual = target-np.einsum("nj,ji->ni", design, coefficients, optimize=False)
        line_frequencies = np.einsum("ij,j->i", labels, frequencies, optimize=False)
        angles = times[:, None]*line_frequencies[None, :]
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
    bounds = [(max(-width, (1e-8-value)*n),
               min(width, (np.pi-1e-8-value)*n)) for value in initial]
    result = minimize(objective, np.zeros(p), jac=True, method="L-BFGS-B",
                      bounds=bounds,
                      options={"ftol": 1e-13, "gtol": 1e-10, "maxiter": 200})
    frequencies = initial+result.x/n
    design, coefficients, residual, derivatives = profile(frequencies)
    flattened = derivatives.reshape(n, p*q)
    projection = np.einsum("nj,ji->ni", design,
        lstsq(design, flattened, lapack_driver="gelsy")[0], optimize=False)
    efficient = (flattened-projection).reshape(n, p, q)
    information = np.einsum("nji,nki->jk", efficient, efficient, optimize=False)
    information = (information+information.T)/2
    condition = float(np.linalg.cond(information))
    if not np.isfinite(condition) or condition > 1e14:
        raise np.linalg.LinAlgError("acquired frequencies have ill-conditioned local information")
    rss = float(np.sum(residual**2))
    parameters = q*(2*count+1)+p
    dof = q*n-parameters
    if dof <= 0:
        raise ValueError("not enough observations for nuisance dictionary")
    residual_variance = rss/dof
    local_variance = residual_variance if noise_sigma is None else noise_sigma**2
    covariance = local_variance*np.linalg.inv(information)
    covariance = (covariance+covariance.T)/2
    if not np.isfinite(covariance).all() or np.min(np.linalg.eigvalsh(covariance)) <= 0:
        raise np.linalg.LinAlgError("local nuisance-fit covariance is not positive definite")
    score = np.einsum("ni,nji->j", residual, derivatives, optimize=False)
    local_step = np.linalg.solve(information, score)
    distance = float(local_step@np.linalg.solve(covariance, local_step))
    return {"frequencies": frequencies, "covariance": covariance,
            "frequency_standard_errors": np.sqrt(np.diag(covariance)),
            "noise_variance_estimate": residual_variance,
            "local_covariance_noise_variance": local_variance,
            "rss": rss, "degrees_of_freedom": dof,
            "parameters": parameters, "bic": float(q*n*np.log(rss/(q*n))+parameters*np.log(q*n)),
            "information_condition": condition,
            "optimizer_success": bool(result.success), "optimizer_message": str(result.message),
            "stationary_score_distance_squared": distance,
            "stationary_in_frequency_uncertainty": distance < 1e-4,
            "iterations": int(result.nit), "function_evaluations": int(result.nfev),
            "coefficients": coefficients, "residual": residual.T,
            "times_center": (n-1)/2, "line_labels": labels,
            "covariance_status": "local_iid_conditional_on_selected_harmonic_dictionary_only"}


def _residual_peaks(fitted: dict, noise_sigma: float | None) -> list[dict]:
    residual = fitted["residual"]
    q, n = residual.shape
    window = np.hanning(n)
    transformed = np.fft.rfft((residual-residual.mean(axis=1, keepdims=True))*window,
                              n=4*n, axis=1)
    power = np.sum(abs(transformed)**2, axis=0)
    frequencies = np.fft.rfftfreq(4*n)*2*np.pi
    fitted_lines = np.einsum("ij,j->i", fitted["line_labels"],
                             fitted["frequencies"], optimize=False)
    indices = find_peaks(power)[0]
    eligible = [int(index) for index in indices
                if min(min(_gap(frequencies[index], old), _gap(frequencies[index], -old))
                       for old in fitted_lines) > 2*np.pi/n]
    strongest = sorted(eligible, key=lambda index: power[index], reverse=True)[:5]
    variance = fitted["local_covariance_noise_variance"]
    reference = q*variance*float(np.sum(window**2))
    return [{"frequency": float(frequencies[index]), "hann_power": float(power[index]),
             "power_over_iid_noise_expectation": float(power[index]/reference)}
            for index in strongest]


def fit_joint_frequencies_with_harmonic_nuisance(
    observed: np.ndarray, starting: np.ndarray, half_width_cycles: float = .35,
    noise_sigma: float | None = None, harmonic_orders=HARMONIC_ORDERS,
    collision_cycles: float = .25,
) -> dict:
    """Choose a tied-harmonic nuisance dictionary by training BIC.

    All acquired frequencies remain independent nonlinear parameters; no
    integer lattice relation is imposed. A global harmonic order keeps this
    small prototype's search linear in the number of declared orders. Real
    nuisance lines close to an existing line are merged in the initial local
    chart; the omissions and final design condition are exposed in the audit.
    `noise_sigma`, when supplied, must describe raw iid sensor noise, preferably
    estimated independently of this temporal fit (e.g. spatial orthogonal data).
    """
    y = np.asarray(observed, dtype=float)
    initial = np.asarray(starting, dtype=float).reshape(-1)
    orders = tuple(sorted(set(int(order) for order in harmonic_orders)))
    if (y.ndim != 2 or not np.isfinite(y).all() or not 1 <= len(initial) <= 6
            or not np.isfinite(initial).all() or np.any((initial <= 0) | (initial >= np.pi))
            or not orders or orders[0] < 1 or half_width_cycles <= 0 or collision_cycles < 0
            or (noise_sigma is not None and (not np.isfinite(noise_sigma) or noise_sigma <= 0))):
        raise ValueError("finite raw data, 1..6 interior peaks, positive budgets/noise required")
    q, n = y.shape
    if n <= 2*len(initial)+1:
        raise ValueError("too few samples for acquired peaks")
    tolerance = collision_cycles*2*np.pi/n
    fits, grid = [], []
    for order in orders:
        labels, omissions = _line_dictionary(initial, order, tolerance)
        try:
            fit = _fit_dictionary(y, initial, labels, half_width_cycles, noise_sigma)
            fit["harmonic_order"] = order
            fit["merged_or_omitted_harmonic_lines"] = omissions
            fits.append(fit)
            grid.append({key: fit[key] for key in ("harmonic_order", "bic", "rss", "parameters",
                "optimizer_success", "stationary_in_frequency_uncertainty", "information_condition")}
                | {"line_count": len(labels), "frequencies": fit["frequencies"].tolist(),
                   "merged_or_omitted_count": len(omissions)})
        except (np.linalg.LinAlgError, ValueError, FloatingPointError) as error:
            grid.append({"harmonic_order": order, "failure": f"{type(error).__name__}: {error}"})
    if not fits:
        raise np.linalg.LinAlgError("all harmonic nuisance models failed")
    fitted = min(fits, key=lambda fit: fit["bic"])
    selected_index = next(index for index, fit in enumerate(fits) if fit is fitted)
    neighbors = [fits[index] for index in (selected_index-1, selected_index+1)
                 if 0 <= index < len(fits)]
    drift = [{"harmonic_order": neighbor["harmonic_order"],
              "frequency_shift_from_selected": (neighbor["frequencies"]-fitted["frequencies"]).tolist(),
              "bic_minus_selected": neighbor["bic"]-fitted["bic"]} for neighbor in neighbors]
    fitted["nuisance_audit"] = {
        "dictionary": "acquired lines plus their tied individual harmonics",
        "order_selection": "training BIC; one common order across acquired lines",
        "order_grid": grid, "selected_harmonic_order": fitted["harmonic_order"],
        "selected_at_maximum_order": fitted["harmonic_order"] == max(orders),
        "harmonic_collision_tolerance": tolerance,
        "omissions": fitted["merged_or_omitted_harmonic_lines"],
        "amplitude_design_condition": float(np.linalg.cond(_design(
            np.arange(n)-fitted["times_center"], fitted["frequencies"], fitted["line_labels"]))),
        "adjacent_order_frequency_sensitivity": drift,
        "residual_variance_over_supplied_noise_variance": (
            fitted["noise_variance_estimate"]/noise_sigma**2 if noise_sigma is not None else None),
        "local_covariance_noise_scale": "supplied raw iid sigma" if noise_sigma is not None else "temporal residual variance",
        "remaining_residual_spectral_peaks": _residual_peaks(fitted, noise_sigma),
        "lattice_relations_imposed_on_acquired_frequencies": False,
        "physical_truth_supplied": False,
        "mixed_characters_explicitly_in_dictionary": False,
        "infinite_tail_bias_bounded": False,
        "post_selection_covariance_calibrated": False,
        "limits": ["individual harmonics omit general mixed characters",
                   "finite order sensitivity and residual spectrum are diagnostics, not tail bounds",
                   "near-collision merging may omit a separate unresolved nuisance mode",
                   "local covariance does not include omitted-tail bias or model selection"]}
    return fitted
