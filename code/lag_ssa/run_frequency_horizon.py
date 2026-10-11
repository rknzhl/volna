"""Finite-sinusoid horizon study for a near integer frequency relation.

All inference uses the noisy field. Physical character labels and the known
rotation vector are used only in post-inference diagnostics. This is a
frequency/lattice identifiability control, not an SSA method.
"""

from __future__ import annotations

from project_paths import RESULTS_DIR

from collections import Counter
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import chi2

from frequency_fit import fit_joint_frequencies
from phase_lattice import infer_character_basis_global
from run_character_ssa import (
    ALPHA, AREA, CASES, MAX_ORDER, MAX_RANK, Q, SENSORS,
    inferred_index, positive_frequency, refine_peak, spatial_profiles,
    true_index,
)


OUT = RESULTS_DIR
DESTINATION = OUT / "frequency_horizon.json"
FIGURE = OUT / "frequency_horizon.png"
CASE_NAMES = (
    "r2_2_3_each_no_fundamentals",
    "r2_primitive_no_coordinate_fundamentals",
)
HORIZONS = (1600, 3200, 6400, 12800)
NOISE_LEVELS = (.2, .4)
SEEDS = tuple(range(53000, 53004))
CONFIDENCE_LEVEL = .99
MAX_VISIBLE = 8
PAD_FACTOR = 8
SEPARATION_CYCLES = 6
NOISE_FLOOR_MULTIPLIER = 25.0
NEAR_RELATION = np.array([18, 5], dtype=int)


def generate_field(case, n: int) -> np.ndarray:
    """Use the shared finite-field profiles, with the requested local horizon."""
    phases = np.array([.23, 1.08])[:, None] + ALPHA[:, None] * np.arange(n)[None, :]
    profiles = spatial_profiles(len(case.modes))
    field = np.zeros((Q, n), dtype=float)
    for j, (mode, amplitude) in enumerate(zip(case.modes, case.amplitudes)):
        angle = np.einsum("i,in->n", np.asarray(mode), phases, optimize=False) + .21 * j
        field += amplitude * profiles[:, j, None] * np.cos(angle)[None, :]
    return field


def screen_peaks(observed: np.ndarray) -> dict:
    """The original Hann screen with all sample-size terms recomputed."""
    n = observed.shape[1]
    centered = observed - observed.mean(axis=1, keepdims=True)
    nfft = 1 << int(np.ceil(np.log2(PAD_FACTOR * n)))
    transformed = np.fft.rfft(centered * np.hanning(n), n=nfft, axis=1)
    power = np.sum(AREA[:, None] * abs(transformed)**2, axis=0)
    frequencies = 2 * np.pi * np.arange(len(power)) / nfft
    background = power[(frequencies > .03) & (frequencies < np.pi - .03)]
    floor = float(np.median(background))
    separation = max(2, int(SEPARATION_CYCLES * nfft / n))
    peaks, _ = find_peaks(power, distance=separation)
    peaks = peaks[(peaks > nfft / n) & (power[peaks] >= NOISE_FLOOR_MULTIPLIER * floor)]
    peaks = peaks[np.argsort(power[peaks])[::-1]][:MAX_VISIBLE]
    refined = np.asarray([refine_peak(observed, float(2 * np.pi * peak / nfft))
                          for peak in peaks], dtype=float)
    return {"frequencies": refined.tolist(), "powers": power[peaks].tolist(),
            "noise_floor": floor, "visible_count": len(peaks)}


def match_truth_labels(peaks: np.ndarray, case, n: int) -> dict:
    """Post-inference label matching, with one Rayleigh bin as the audit gate."""
    candidates = [positive_frequency(np.asarray(mode), ALPHA)
                  for mode in case.modes]
    resolution = 2 * np.pi / n
    labels = []
    for peak in peaks:
        options = [(frequency, label) for frequency, label in candidates
                   if abs(frequency - peak) <= resolution]
        if len(options) != 1:
            return {"matched": False, "reason": f"peak {peak} has {len(options)} truth matches",
                    "rayleigh_width": resolution}
        labels.append(options[0][1].tolist())
    seen = {tuple(label) for label in labels}
    required = {tuple(label.tolist()) for _, label in candidates}
    return {"matched": True, "labels": labels, "rayleigh_width": resolution,
            "full_base_detected": seen == required,
            "missing_base_labels": [list(label) for label in sorted(required - seen)]}


def relation_diagnostic(frequencies: np.ndarray, covariance: np.ndarray,
                        labels: np.ndarray, radius: float) -> dict:
    """Oracle evaluation in physical alpha coordinates; never used in inference."""
    true_frequencies = np.array([positive_frequency(label, ALPHA)[0]
                                 for label in labels])
    error = np.angle(np.exp(1j * (frequencies - true_frequencies)))
    precision_labels = np.linalg.solve(covariance, labels.astype(float))
    alpha_covariance = np.linalg.inv(labels.T @ precision_labels)
    alpha_correction = alpha_covariance @ labels.T @ np.linalg.solve(covariance, error)
    alpha_estimate = ALPHA + alpha_correction
    gap = float(NEAR_RELATION @ ALPHA - 2 * np.pi)
    estimated_gap = float(NEAR_RELATION @ alpha_estimate - 2 * np.pi)
    gap_se = float(np.sqrt(NEAR_RELATION @ alpha_covariance @ NEAR_RELATION))
    true_mahalanobis_squared = float(error @ np.linalg.solve(covariance, error))
    return {"physical_relation": NEAR_RELATION.tolist(), "true_gap_rad_per_sample": gap,
            "estimated_gap_rad_per_sample": estimated_gap,
            "gap_standard_error": gap_se,
            "gap_99pct_ellipsoid_halfwidth": radius * gap_se,
            "true_gap_to_halfwidth": abs(gap) / (radius * gap_se),
            "true_frequency_mahalanobis_squared": true_mahalanobis_squared,
            "true_frequency_in_ellipsoid": true_mahalanobis_squared <= radius**2,
            "oracle_labels_used_only_for_evaluation": True}


def evaluate(case, n: int, noise: float, seed: int) -> dict:
    clean = generate_field(case, n)
    scale = float(np.sqrt(np.mean(np.sum(AREA[:, None] * clean**2, axis=0))))
    observed = clean + noise * scale * np.random.default_rng(seed).standard_normal(clean.shape)
    row = {"case": case.name, "n": n, "noise": noise, "seed": seed,
           "true_rank": 2, "true_index": true_index(case.modes)}
    try:
        row["screen"] = screen_peaks(observed)
        starting = np.asarray(row["screen"]["frequencies"], dtype=float)
        if len(starting) == 0:
            row["status"] = "no_visible_peak"
            return row
        fit = fit_joint_frequencies(observed, starting)
        radius = float(np.sqrt(chi2.ppf(CONFIDENCE_LEVEL, len(starting))))
        frequencies = np.asarray(fit["frequencies"])
        covariance = np.asarray(fit["covariance"])
        row["joint_fit"] = {
            "frequencies": frequencies.tolist(), "covariance": covariance.tolist(),
            "max_frequency_standard_error": float(np.max(fit["frequency_standard_errors"])),
            "optimizer_success": fit["optimizer_success"],
            "stationary_in_frequency_uncertainty": fit["stationary_in_frequency_uncertainty"],
            "stationary_score_distance_squared": fit["stationary_score_distance_squared"],
            "information_condition": fit["information_condition"],
            "noise_variance_inflation": fit["noise_variance_estimate"] / (noise * scale)**2,
            "confidence_radius": radius,
        }
        if not fit["stationary_in_frequency_uncertainty"]:
            row["status"] = "frequency_fit_not_stationary"
            return row
        lattice = infer_character_basis_global(
            frequencies, tolerance=1e-12, max_order=MAX_ORDER, max_rank=MAX_RANK,
            frequency_covariance=covariance, confidence_radius=radius,
        )
        row["lattice"] = lattice
        audit = match_truth_labels(frequencies, case, n)
        row["truth_label_audit"] = audit
        if not audit["matched"]:
            row["status"] = "truth_match_unavailable"
            return row
        labels = np.asarray(audit["labels"], dtype=int)
        row["near_relation_audit"] = relation_diagnostic(
            frequencies, covariance, labels, radius)
        row["module_complete"] = False
        if lattice.get("rank") is not None:
            character_matrix = np.asarray(lattice["products"], dtype=int) @ labels
            row["character_matrix_in_physical_coordinates"] = character_matrix.tolist()
            row["inferred_index"] = inferred_index(character_matrix)
            row["module_complete"] = bool(
                lattice["rank"] == row["true_rank"]
                and row["inferred_index"] == row["true_index"])
        row["status"] = "evaluated"
    except Exception as exc:
        row["status"] = "exception"
        row["exception"] = f"{type(exc).__name__}: {exc}"
    return row


def summarize(rows: list[dict]) -> list[dict]:
    groups = []
    for case_name in CASE_NAMES:
        for noise in NOISE_LEVELS:
            for n in HORIZONS:
                group = [row for row in rows if row["case"] == case_name
                         and row["noise"] == noise and row["n"] == n]
                audited = [row for row in group if "near_relation_audit" in row]
                groups.append({
                    "case": case_name, "noise": noise, "n": n, "trials": len(group),
                    "module_complete": sum(row.get("module_complete", False) for row in group),
                    "wrong_accepted": sum(row.get("lattice", {}).get("rank") is not None
                                          and not row.get("module_complete", False)
                                          for row in group),
                    "rejected": sum(row.get("lattice", {}).get("rank") is None
                                    for row in group),
                    "candidate_rank_counts": dict(Counter(
                        str(row.get("lattice", {}).get("rank")) for row in group)),
                    "status_counts": dict(Counter(row["status"] for row in group)),
                    "all_base_modes_detected": sum(
                        row.get("truth_label_audit", {}).get("full_base_detected", False)
                        for row in group),
                    "median_max_frequency_se": float(np.median([
                        row["joint_fit"]["max_frequency_standard_error"]
                        for row in group if "joint_fit" in row])) if group else None,
                    "median_true_gap_to_99pct_halfwidth": float(np.median([
                        row["near_relation_audit"]["true_gap_to_halfwidth"]
                        for row in audited])) if audited else None,
                    "true_frequency_in_ellipsoid": sum(
                        row["near_relation_audit"]["true_frequency_in_ellipsoid"]
                        for row in audited),
                })
    return groups


def plot_summary(summary: list[dict]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex="col")
    for column, case_name in enumerate(CASE_NAMES):
        ax_correct, ax_gap = axes[:, column]
        for noise in NOISE_LEVELS:
            group = [row for row in summary if row["case"] == case_name
                     and row["noise"] == noise]
            n = np.array([row["n"] for row in group])
            correct = np.array([row["module_complete"] / row["trials"] for row in group])
            gap_ratio = np.array([row["median_true_gap_to_99pct_halfwidth"] for row in group])
            ax_correct.plot(n, correct, marker="o", label=f"noise {noise:g}")
            ax_gap.plot(n, gap_ratio, marker="o", label=f"noise {noise:g}")
        ax_correct.set_title(case_name.replace("r2_", ""))
        ax_correct.set_ylim(-.05, 1.05)
        ax_correct.set_ylabel("correct observable module / 4")
        ax_correct.grid(alpha=.3)
        ax_gap.axhline(1.0, color="black", linewidth=1, linestyle="--",
                       label="gap = 99% halfwidth")
        ax_gap.set_ylabel("|18α₁+5α₂−2π| / 99% halfwidth")
        ax_gap.set_xlabel("number of time samples N")
        ax_gap.set_xscale("log", base=2)
        ax_gap.set_xticks(HORIZONS, labels=[str(n) for n in HORIZONS])
        ax_gap.grid(alpha=.3)
        ax_correct.legend(fontsize=8)
        ax_gap.legend(fontsize=8)
    fig.suptitle("Frequency-lattice horizon control (finite sinusoids, same screen)")
    fig.tight_layout()
    fig.savefig(FIGURE, dpi=170)
    plt.close(fig)


def main() -> None:
    cases = [next(case for case in CASES if case.name == name) for name in CASE_NAMES]
    rows = []
    for case in cases:
        for noise in NOISE_LEVELS:
            for n in HORIZONS:
                for seed in SEEDS:
                    row = evaluate(case, n, noise, seed)
                    rows.append(row)
                    print(case.name, noise, n, seed, row["status"],
                          row.get("lattice", {}).get("rank"),
                          row.get("module_complete"), flush=True)
                DESTINATION.write_text(json.dumps({"complete": False, "rows": rows,
                    "summary": summarize(rows)}, indent=2) + "\n")
    summary = summarize(rows)
    payload = {
        "description": "Near-resonance frequency/lattice horizon control, not a new SSA method",
        "complete": True, "rows_expected": 64, "rows": rows, "summary": summary,
        "cases": CASE_NAMES, "n_values": HORIZONS, "noise_levels": NOISE_LEVELS,
        "seeds": SEEDS, "sensor_count": Q, "sensors": SENSORS.tolist(),
        "confidence_level": CONFIDENCE_LEVEL, "max_relation_order": MAX_ORDER,
        "screen": {"pad_factor": PAD_FACTOR,
                   "separation_cycles": SEPARATION_CYCLES,
                   "max_visible": MAX_VISIBLE,
                   "noise_floor_multiplier": NOISE_FLOOR_MULTIPLIER},
        "physical_near_relation_audit_only": NEAR_RELATION.tolist(),
        "physical_gap_rad_per_sample": float(NEAR_RELATION @ ALPHA - 2 * np.pi),
        "peak_selection_and_fit_on_same_record": True,
        "covariance_status": "local asymptotic under correct finite iid model; post-selection not covered",
    }
    DESTINATION.write_text(json.dumps(payload, indent=2) + "\n")
    plot_summary(summary)
    print(DESTINATION)
    print(FIGURE)


if __name__ == "__main__":
    main()
