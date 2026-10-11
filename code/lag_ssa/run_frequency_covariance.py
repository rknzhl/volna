"""Audit frequency uncertainty and lattice inference on saved peak lists.

Peaks were selected on the same raw record in the earlier experiment.
Consequently these are diagnostic asymptotic ellipsoids, not validated
post-selection confidence sets. Noisy rank remains conditional on a gap.
"""

from __future__ import annotations

from project_paths import RESULTS_DIR

import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.stats import chi2

from frequency_fit import fit_joint_frequencies
from phase_lattice import infer_character_basis_global
from run_character_ssa import (
    ALPHA, AREA, CASES, MAX_ORDER, MAX_RANK, clean_field, inferred_index,
    positive_frequency, true_index,
)


OUT = RESULTS_DIR
CONFIDENCE_LEVEL = .99


def summarize(rows):
    answer = []
    for case in CASES:
        group = [r for r in rows if r["case"] == case.name]
        if not group:
            continue
        fitted = [r for r in group if "joint_fit" in r]
        answer.append({"case": case.name, "trials": len(group),
                       "finite_model": not case.infinite,
                       "old_module_complete": sum(r["old_module_complete"] for r in group),
                       "new_module_complete": sum(r.get("new_module_complete", False) for r in group),
                       "new_inference_rejected": sum(r.get("new_lattice", {}).get("rank") is None for r in group),
                       "local_optimizer_success": sum(r["joint_fit"]["optimizer_success"] for r in fitted),
                       "true_frequencies_in_ellipsoid": sum(r["joint_fit"]["true_in_ellipsoid"] for r in fitted),
                       "median_frequency_error": float(np.median([
                           r["joint_fit"]["max_frequency_error"] for r in fitted])) if fitted else None,
                       "median_max_standardized_error": float(np.median([
                           r["joint_fit"]["max_standardized_error"] for r in fitted])) if fitted else None})
    return answer


def evaluate(saved):
    case = next(c for c in CASES if c.name == saved["case"])
    row = {"case": case.name, "noise": saved["noise"], "seed": saved["seed"],
           "old_module_complete": bool(saved.get("observable_module_complete", False)),
           "infinite_signal": case.infinite}
    clean, _ = clean_field(case, ALPHA)
    scale = np.sqrt(np.mean(np.sum(AREA[:, None]*clean**2, axis=0)))
    observed = clean+saved["noise"]*scale*np.random.default_rng(saved["seed"]).standard_normal(clean.shape)
    starting = np.asarray(saved["screen"]["refined"])
    truth_labels = saved["truth_label_audit"]["labels"]
    if any(label is None for label in truth_labels):
        row["status"] = "truth_match_unavailable"
        return row
    labels = np.asarray(truth_labels, dtype=int)
    true_frequencies = np.asarray([positive_frequency(label, ALPHA[:case.dimension])[0]
                                   for label in labels])
    start_time = time.monotonic()
    try:
        fit = fit_joint_frequencies(observed, starting)
        p = len(starting)
        radius = float(np.sqrt(chi2.ppf(CONFIDENCE_LEVEL, p)))
        frequency_error = np.angle(np.exp(1j*(fit["frequencies"]-true_frequencies)))
        mahalanobis = float(frequency_error @ np.linalg.solve(fit["covariance"], frequency_error))
        row["joint_fit"] = {
            name: fit[name] for name in ("optimizer_success", "optimizer_message",
                                        "iterations", "function_evaluations",
                                        "rss", "noise_variance_estimate",
                                        "stationary_score_distance_squared",
                                        "stationary_in_frequency_uncertainty",
                                        "information_condition", "covariance_status")}
        row["joint_fit"].update({
            "frequencies": fit["frequencies"].tolist(),
            "covariance": fit["covariance"].tolist(),
            "standard_errors": fit["frequency_standard_errors"].tolist(),
            "true_frequency_error": frequency_error.tolist(),
            "max_frequency_error": float(np.max(abs(frequency_error))),
            "max_standardized_error": float(np.max(abs(frequency_error)/fit["frequency_standard_errors"])),
            "true_frequency_mahalanobis_squared": mahalanobis,
            "radius_squared": radius**2,
            "true_in_ellipsoid": bool(mahalanobis <= radius**2),
            "actual_noise_variance": float((saved["noise"]*scale)**2),
            "noise_variance_inflation": float(fit["noise_variance_estimate"]/(saved["noise"]*scale)**2),
        })
        if not fit["stationary_in_frequency_uncertainty"]:
            row["status"] = "frequency_fit_not_stationary"
            row["new_module_complete"] = False
            row["elapsed_seconds"] = time.monotonic()-start_time
            return row
        lattice = infer_character_basis_global(
            fit["frequencies"], tolerance=1e-12, max_order=MAX_ORDER,
            max_rank=MAX_RANK, frequency_covariance=fit["covariance"],
            confidence_radius=radius)
        row["new_lattice"] = lattice
        row["new_module_complete"] = False
        if lattice.get("rank") is not None:
            character_matrix = np.asarray(lattice["products"], dtype=int) @ labels
            row["new_character_matrix"] = character_matrix.tolist()
            row["new_module_index"] = inferred_index(character_matrix)
            row["new_module_complete"] = bool(
                lattice["rank"] == case.dimension
                and row["new_module_index"] == true_index(case.modes))
        row["status"] = "evaluated"
    except Exception as exc:
        row["status"] = "exception"
        row["exception"] = f"{type(exc).__name__}: {exc}"
    row["elapsed_seconds"] = time.monotonic()-start_time
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--finite-only", action="store_true")
    args = parser.parse_args()
    source = json.loads((OUT/"character_ssa.json").read_text())
    selected = source["rows"]
    if args.pilot:
        selected = [row for row in selected if row["noise"] == .08 and row["seed"] in (40000, 40001)]
    if args.finite_only:
        finite_names = {case.name for case in CASES if not case.infinite}
        selected = [row for row in selected if row["case"] in finite_names]
    name = "frequency_covariance"+("_pilot" if args.pilot else "")+("_finite" if args.finite_only else "")+".json"
    destination = OUT/name
    rows = []
    metadata = {"source": "character_ssa.json saved selected peak lists",
                "confidence_level": CONFIDENCE_LEVEL, "same_record_peak_selection": True,
                "covariance_is_asymptotic_diagnostic": True,
                "infinite_tail_bias_not_bounded": True,
                "gap_and_visible_generator_completeness_not_certified": True,
                "rows_expected": len(selected), "max_relation_order": MAX_ORDER,
                "local_joint_frequency_fit": True}
    for saved in selected:
        row = evaluate(saved)
        rows.append(row)
        print(row["case"], row["noise"], row["seed"], row["status"],
              "complete", row.get("new_module_complete"),
              "rank", row.get("new_lattice", {}).get("rank"),
              "ellipsoid", row.get("joint_fit", {}).get("true_in_ellipsoid"),
              "seconds", round(row["elapsed_seconds"], 2), flush=True)
        destination.write_text(json.dumps({**metadata, "complete": False,
                                          "rows": rows, "summary": summarize(rows)}, indent=2)+"\n")
    payload = {**metadata, "complete": len(rows) == len(selected),
               "rows": rows, "summary": summarize(rows)}
    destination.write_text(json.dumps(payload, indent=2)+"\n")
    print(json.dumps(payload["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
