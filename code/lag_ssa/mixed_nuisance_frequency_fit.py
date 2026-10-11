"""Sparse train-detected pairwise nuisance rows for acquired frequencies.

This is a finite dictionary sensitivity experiment. It uses no physical mode
labels or known latent rank. General mixed Fourier tails and model-selection
uncertainty remain outside its local iid covariance.
"""

from __future__ import annotations

from itertools import combinations, product

import numpy as np
from scipy.signal import find_peaks

from nuisance_frequency_fit import (
    HARMONIC_ORDERS, _design, _fit_dictionary, _gap, _line_dictionary,
    _residual_peaks, fit_joint_frequencies_with_harmonic_nuisance,
)


def _real_frequency(label: np.ndarray, frequencies: np.ndarray) -> float:
    return float(abs(np.angle(np.exp(1j*float(label@frequencies)))))


def _pairwise_candidates(fitted: dict, max_coefficient: int,
                         max_rows: int, matching_tolerance: float,
                         collision_tolerance: float) -> tuple[list[np.ndarray], dict]:
    frequencies = fitted["frequencies"]
    p = len(frequencies)
    residual = fitted["residual"]
    n = residual.shape[1]
    window = np.hanning(n)
    spectrum = np.fft.rfft((residual-residual.mean(axis=1, keepdims=True))*window,
                           n=4*n, axis=1)
    power = np.sum(abs(spectrum)**2, axis=0)
    fft_frequencies = np.fft.rfftfreq(4*n)*2*np.pi
    peak_indices = find_peaks(power)[0]
    existing = [_real_frequency(label, frequencies) for label in fitted["line_labels"]]
    proposals, skipped, examined = [], [], 0
    for first, second in combinations(range(p), 2):
        # Real sinusoids identify a row and its negative. Canonicalize its
        # first nonzero coefficient to be positive.
        for a, b in product(range(1, max_coefficient+1),
                            tuple(range(-max_coefficient, 0))+tuple(range(1, max_coefficient+1))):
            examined += 1
            label = np.zeros(p, dtype=int)
            label[first], label[second] = a, b
            frequency = _real_frequency(label, frequencies)
            if min(frequency, np.pi-frequency) <= collision_tolerance:
                skipped.append({"label": label.tolist(), "reason": "near constant or Nyquist"})
                continue
            if min(abs(frequency-old) for old in existing) <= collision_tolerance:
                skipped.append({"label": label.tolist(), "reason": "near existing real line"})
                continue
            nearby = [int(index) for index in peak_indices
                      if abs(fft_frequencies[index]-frequency) <= matching_tolerance]
            if not nearby:
                continue
            index = max(nearby, key=lambda candidate: power[candidate])
            proposals.append({"label": label, "frequency": frequency,
                              "matched_peak": float(fft_frequencies[index]),
                              "power": float(power[index]),
                              "matching_gap": abs(float(fft_frequencies[index])-frequency)})
    proposals.sort(key=lambda row: (-row["power"], row["matching_gap"],
                                    int(np.sum(abs(row["label"])))))
    selected = []
    selected_frequencies = []
    for proposal in (proposals if max_rows else []):
        if selected_frequencies and min(abs(proposal["frequency"]-old)
                                        for old in selected_frequencies) <= collision_tolerance:
            skipped.append({"label": proposal["label"].tolist(),
                            "reason": "near earlier proposed mixed real line"})
            continue
        selected.append(proposal)
        selected_frequencies.append(proposal["frequency"])
        if len(selected) == max_rows:
            break
    return [proposal["label"] for proposal in selected], {
        "bounded_pairwise_rows_examined": examined,
        "rows_matching_training_residual_peaks": len(proposals),
        "retained_proposals": [{key: (value.tolist() if isinstance(value, np.ndarray) else value)
                                 for key, value in proposal.items()} for proposal in selected],
        "skipped_near_collisions": skipped,
        "max_coefficient": max_coefficient, "proposal_cap": max_rows,
        "matching_tolerance": matching_tolerance,
        "collision_tolerance": collision_tolerance,
        "training_spectrum_is_reused_for_model_selection": True}


def _combine_rows(harmonic_rows: np.ndarray, mixed_rows: list[np.ndarray],
                  frequencies: np.ndarray, tolerance: float) -> tuple[np.ndarray, list[dict]]:
    rows = [row.copy() for row in harmonic_rows]
    line_frequencies = [_real_frequency(row, frequencies) for row in rows]
    omissions = []
    for row in mixed_rows:
        frequency = _real_frequency(row, frequencies)
        nearest = int(np.argmin([abs(frequency-old) for old in line_frequencies]))
        if abs(frequency-line_frequencies[nearest]) <= tolerance:
            omissions.append({"label": row.tolist(), "retained_label": rows[nearest].tolist(),
                              "gap": abs(frequency-line_frequencies[nearest])})
            continue
        rows.append(row.copy())
        line_frequencies.append(frequency)
    return np.asarray(rows, dtype=int), omissions


def _grid_row(fitted: dict, mixed_count: int) -> dict:
    return {"mixed_batch_size": mixed_count, "line_count": len(fitted["line_labels"]),
            "bic": fitted["bic"], "rss": fitted["rss"],
            "parameters": fitted["parameters"], "frequencies": fitted["frequencies"].tolist(),
            "optimizer_success": fitted["optimizer_success"],
            "stationary_in_frequency_uncertainty": fitted["stationary_in_frequency_uncertainty"]}


def fit_joint_frequencies_with_sparse_mixed_nuisance(
    observed: np.ndarray, starting: np.ndarray, half_width_cycles: float = .35,
    noise_sigma: float | None = None, harmonic_orders=HARMONIC_ORDERS,
    max_mixed_coefficient: int = 4, max_mixed_rows: int = 16,
    mixed_batches=(0, 1, 2, 4, 8, 16), collision_cycles: float = .25,
    matching_cycles: float = .5,
) -> dict:
    """Augment the harmonic fit with bounded rows supported by train peaks.

    Acquired frequencies remain independent parameters. Mixed frequencies
    are tied integer combinations, so the helper's profiled full covariance
    includes their derivative contributions. No relation among acquired
    frequencies is imposed. An optional second harmonic-order selection uses
    the selected mixed rows, because previously missing mixed energy can hide
    the benefit of further individual harmonics.
    """
    if not 1 <= max_mixed_coefficient <= 4 or not 0 <= max_mixed_rows <= 16:
        raise ValueError("requires coefficients <=4 and at most16 mixed nuisance rows")
    if matching_cycles <= 0 or collision_cycles < 0:
        raise ValueError("positive matching and nonnegative collision budgets required")
    y = np.asarray(observed, dtype=float)
    harmonic = fit_joint_frequencies_with_harmonic_nuisance(y, starting,
        half_width_cycles=half_width_cycles, noise_sigma=noise_sigma,
        harmonic_orders=harmonic_orders, collision_cycles=collision_cycles)
    n = y.shape[1]
    tolerance = collision_cycles*2*np.pi/n
    candidates, proposal_audit = _pairwise_candidates(harmonic, max_mixed_coefficient,
        max_mixed_rows, matching_cycles*2*np.pi/n, tolerance)
    batches = sorted(set([0, len(candidates)]+[min(int(batch), len(candidates))
        for batch in mixed_batches if 0 <= int(batch) <= max_mixed_rows]))
    fits = [(0, harmonic)]
    grid = [_grid_row(harmonic, 0)]
    for batch in batches:
        if batch == 0:
            continue
        labels, omissions = _combine_rows(harmonic["line_labels"], candidates[:batch],
                                          harmonic["frequencies"], tolerance)
        try:
            fitted = _fit_dictionary(y, harmonic["frequencies"], labels,
                                     half_width_cycles, noise_sigma)
            fitted["harmonic_order"] = harmonic["harmonic_order"]
            fitted["mixed_collision_omissions"] = omissions
            fits.append((batch, fitted))
            grid.append(_grid_row(fitted, batch))
        except (np.linalg.LinAlgError, ValueError, FloatingPointError) as error:
            grid.append({"mixed_batch_size": batch, "failure": f"{type(error).__name__}: {error}"})
    selected_batch, fitted = min(fits, key=lambda item: item[1]["bic"])
    selected_mixed = candidates[:selected_batch]
    reselection = []
    reselection_fits = [fitted]
    if selected_batch:
        for order in sorted(set(int(order) for order in harmonic_orders)):
            harmonic_rows, harmonic_omissions = _line_dictionary(
                fitted["frequencies"], order, tolerance)
            labels, mixed_omissions = _combine_rows(harmonic_rows, selected_mixed,
                                                   fitted["frequencies"], tolerance)
            try:
                candidate = _fit_dictionary(y, fitted["frequencies"], labels,
                                            half_width_cycles, noise_sigma)
                candidate["harmonic_order"] = order
                candidate["harmonic_collision_omissions"] = harmonic_omissions
                candidate["mixed_collision_omissions"] = mixed_omissions
                reselection_fits.append(candidate)
                reselection.append(_grid_row(candidate, selected_batch)
                                   | {"harmonic_order": order})
            except (np.linalg.LinAlgError, ValueError, FloatingPointError) as error:
                reselection.append({"harmonic_order": order,
                                    "failure": f"{type(error).__name__}: {error}"})
        fitted = min(reselection_fits, key=lambda item: item["bic"])
    actual_mixed = [row for row in fitted["line_labels"] if np.count_nonzero(row) == 2]
    fitted["nuisance_audit"] = {
        "dictionary": "individual acquired harmonics plus sparse train-detected pairwise rows",
        "initial_harmonic_audit": harmonic["nuisance_audit"],
        "proposal_audit": proposal_audit, "mixed_batch_bic_grid": grid,
        "selected_mixed_batch_size": selected_batch,
        "actual_selected_mixed_rows": [row.tolist() for row in actual_mixed],
        "harmonic_reselection_bic_grid": reselection,
        "selected_harmonic_order": fitted["harmonic_order"],
        "amplitude_design_condition": float(np.linalg.cond(_design(
            np.arange(n)-fitted["times_center"], fitted["frequencies"], fitted["line_labels"]))),
        "residual_variance_over_supplied_noise_variance": (
            fitted["noise_variance_estimate"]/noise_sigma**2 if noise_sigma is not None else None),
        "remaining_residual_spectral_peaks": _residual_peaks(fitted, noise_sigma),
        "lattice_relations_imposed_on_acquired_frequencies": False,
        "physical_truth_supplied": False,
        "infinite_tail_bias_bounded": False,
        "post_selection_covariance_calibrated": False,
        "limits": ["bounded pairwise rows omit higher order and higher-dimensional mixed modes",
                   "16-row cap does not certify a negligible infinite mixed tail",
                   "covariance is local iid sensitivity conditional on the selected finite dictionary",
                   "residual spectrum and order sensitivity are diagnostics, not bias bounds"]}
    return fitted
