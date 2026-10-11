"""Exploratory blind SSA character front-end assay on one spatial field.

Truth enters only after the extractor returns. The first character is tested,
not a complete torus or unknown-r recovery. Acceptance gates are uncalibrated.
"""

from __future__ import annotations

from project_paths import RESULTS_DIR

from collections import Counter
import json
from pathlib import Path

import numpy as np

from method import circular_mae
from run_conditional_ssa import AREA, CASES, Q, component_fields
from ssa_character import extract_ssa_character


OUT = RESULTS_DIR
DESTINATION = OUT / "ssa_character_pilot.json"
N = 3200
WINDOWS = (64, 128)
NOISE_LEVELS = (.03, .08, .20)
SEEDS = (54000, 54001)
CASE_NAMES = (
    "b_poisson70_r2",
    "c_poisson88_r2",
    "d_poisson70_mixed_r2",
    "f_poisson70_2theta_cos3_cover",
    "pure_gaussian_noise",
)


def make_observation(case_name: str, noise: float, seed: int) -> tuple[np.ndarray, dict]:
    generator = np.random.default_rng(seed)
    if case_name == "pure_gaussian_noise":
        return noise * generator.standard_normal((Q, N)), {
            "has_character": False, "applied_noise_sigma": noise}
    case = next(item for item in CASES if item.name == case_name)
    strong, weak, meta = component_fields(case, np.arange(N), normalize_strong=True)
    clean = strong + weak
    scale = float(np.sqrt(np.mean(np.sum(AREA[:, None] * clean**2, axis=0))))
    noise_sigma = noise*scale
    observed = clean + noise_sigma*generator.standard_normal(clean.shape)
    true_phase = ((2 if meta["strong_lattice_index"] == 2 else 1)
                  * meta["theta1"])
    return observed, {"has_character": True,
                      "true_first_frequency": float(meta["first_family_frequency"]),
                      "true_first_phase": true_phase,
                      "strong_lattice_index": meta["strong_lattice_index"],
                      "applied_noise_sigma": noise_sigma}


def evaluate(case_name: str, noise: float, seed: int, window: int) -> dict:
    observed, truth = make_observation(case_name, noise, seed)
    # No frequency, rank, phase, true noise scale, or Fourier peak is supplied.
    result = extract_ssa_character(observed, AREA, window, noise_sigma=None)
    row = {"case": case_name, "noise": noise, "seed": seed, "window": window,
           "status": result["status"], "reason": result.get("reason"),
           "metrics": result.get("metrics", {}), "gates": result.get("gates", {}),
           "candidate_frequency": result.get("candidate_frequency"),
           "has_character_truth": truth["has_character"],
           "applied_noise_sigma_truth_only": truth["applied_noise_sigma"]}
    if result.get("phase") is not None:
        row["phase_coordinate_length"] = len(result["phase"])
        row["phase_first_8"] = np.asarray(result["phase"][:8]).tolist()
        row["normalized_character_first_8"] = [
            [float(value.real), float(value.imag)]
            for value in result["normalized_character"][:8]]
    if truth["has_character"]:
        target = truth["true_first_frequency"]
        row["true_first_frequency_score_only"] = target
        row["strong_lattice_index_score_only"] = truth["strong_lattice_index"]
        if row["candidate_frequency"] is not None:
            row["candidate_frequency_error"] = float(abs(np.angle(np.exp(
                1j*(row["candidate_frequency"]-target)))))
            phase = np.asarray(result["phase"])
            row["phase_mae_rad"] = circular_mae(
                phase[None, :], truth["true_first_phase"][:len(phase)][None, :])
    return row


def summarize(rows: list[dict]) -> list[dict]:
    groups = []
    for case_name in CASE_NAMES:
        for noise in NOISE_LEVELS:
            for window in WINDOWS:
                group = [row for row in rows if row["case"] == case_name
                         and row["noise"] == noise and row["window"] == window]
                errors = [row["candidate_frequency_error"] for row in group
                          if row["status"] == "resolved"
                          and "candidate_frequency_error" in row]
                phases = [row["phase_mae_rad"] for row in group
                          if row["status"] == "resolved" and "phase_mae_rad" in row]
                groups.append({
                    "case": case_name, "noise": noise, "window": window,
                    "trials": len(group),
                    "resolved": sum(row["status"] == "resolved" for row in group),
                    "status_counts": dict(Counter(row["status"] for row in group)),
                    "median_resolved_frequency_error": float(np.median(errors)) if errors else None,
                    "median_resolved_phase_mae_rad": float(np.median(phases)) if phases else None,
                    "median_pair_to_next_gap": float(np.median([
                        row["metrics"]["pair_to_next_gap"] for row in group
                        if "pair_to_next_gap" in row["metrics"]])),
                })
    return groups


def main() -> None:
    rows = []
    for case_name in CASE_NAMES:
        for noise in NOISE_LEVELS:
            for seed in SEEDS:
                for window in WINDOWS:
                    try:
                        row = evaluate(case_name, noise, seed, window)
                    except Exception as exc:
                        row = {"case": case_name, "noise": noise, "seed": seed,
                               "window": window, "status": "exception",
                               "exception": f"{type(exc).__name__}: {exc}"}
                    rows.append(row)
                    print(case_name, noise, seed, window, row["status"],
                          row.get("candidate_frequency"), flush=True)
                DESTINATION.write_text(json.dumps({"complete": False,
                    "rows": rows}, indent=2) + "\n")
    payload = {
        "description": "blind SSA top-pair first-character pilot; not unknown-r torus recovery",
        "complete": True, "rows_expected": 60, "rows": rows,
        "summary": summarize(rows), "case_names": CASE_NAMES,
        "noise_levels": NOISE_LEVELS, "seeds": SEEDS, "windows": WINDOWS,
        "samples": N, "sensor_count": Q,
        "extractor_receives": ["observed_field", "spatial_weights", "window"],
        "fft_used_in_extractor": False, "true_frequency_used_in_extractor": False,
        "known_rank_used_in_extractor": False,
        "purity_gate_calibrated": False,
        "oracle_truth_only_for_scoring": True,
        "noise_control": "pure iid Gaussian field with the same sensor geometry",
    }
    DESTINATION.write_text(json.dumps(payload, indent=2) + "\n")
    print(DESTINATION)


if __name__ == "__main__":
    main()
