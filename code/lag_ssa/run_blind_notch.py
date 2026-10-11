"""Held-out frequency-blind, known-r=2 notch ablation.

The spectral front end is fixed from earlier pilot work. New noise seeds are
used here. This is a ProjSSA baseline, not a claim of novelty.
"""

from __future__ import annotations

from project_paths import RESULTS_DIR

import json
from pathlib import Path

import numpy as np

from method import LagDesign, dmd_phase_coordinates, estimate_peaks, trajectory_matrix
from run_controlled import AREA, PROFILES, case_data, field, phase_error_permutation
from run_notch_ssa import notch
from run_spectral_baseline import refine_frequency


OUT = RESULTS_DIR
HORIZON = 32
RANK = 4
NOISE = [.08, .20, .40]
SEEDS = list(range(10000, 10020))
RELATIVE_PEAK_CUTOFF = .0005


def select_harmonic_family(observed: np.ndarray) -> dict:
    """Detect a visible half-frequency of the strongest spectral peak."""
    peaks, strengths = estimate_peaks(observed, AREA, count=16,
                                      suppress_harmonics=False)
    strong = float(peaks[0])
    resolution = 2 * np.pi / observed.shape[1]
    half_idx = int(np.argmin(abs(peaks - strong / 2)))
    ratio = float(strengths[half_idx] / strengths[0])
    if abs(peaks[half_idx] - strong/2) > 2*resolution or ratio < RELATIVE_PEAK_CUTOFF:
        return {"accepted": False, "reason": "no resolved half-frequency above cutoff",
                "strongest_peak": strong, "nearest_half": float(peaks[half_idx]),
                "half_power_ratio": ratio}
    separated = next((int(i) for i in range(1, len(peaks))
                      if i != half_idx and abs(peaks[i]-strong) > 2*resolution
                      and abs(peaks[i]-peaks[half_idx]) > 2*resolution), None)
    if separated is None:
        return {"accepted": False, "reason": "no second frequency-separated peak",
                "strongest_peak": strong, "nearest_half": float(peaks[half_idx]),
                "half_power_ratio": ratio}
    nuisance = refine_frequency(observed, strong)
    targets = np.array([refine_frequency(observed, float(peaks[half_idx])),
                        refine_frequency(observed, float(peaks[separated]))])
    return {"accepted": True, "nuisance": nuisance, "targets": targets.tolist(),
            "half_power_ratio": ratio, "coarse_peaks": peaks.tolist(),
            "separated_peak_index": separated}


def subtract_sinusoid(observed: np.ndarray, angular_frequency: float) -> np.ndarray:
    times = np.arange(observed.shape[1])
    basis = np.stack([np.cos(angular_frequency*times),
                      np.sin(angular_frequency*times)], axis=1)
    gram = np.einsum("nj,nk->jk", basis, basis, optimize=False)
    cross = np.einsum("nj,in->ji", basis, observed, optimize=False)
    coefficients = np.linalg.solve(gram, cross)
    fitted = np.einsum("nj,ji->in", basis, coefficients, optimize=False)
    return observed - fitted


def run_one(observed: np.ndarray, truth: np.ndarray, noise: float, seed: int) -> dict:
    selection = select_harmonic_family(observed)
    base = {"noise": noise, "seed": seed, "selection": selection,
            "computed": selection["accepted"]}
    if not selection["accepted"]:
        return base
    targets = np.asarray(selection["targets"])
    nuisance = float(selection["nuisance"])
    design = LagDesign(np.arange(HORIZON), np.ones(HORIZON)/HORIZON,
                       float("nan"), HORIZON)
    spatial = trajectory_matrix(observed, AREA, design)
    classical = trajectory_matrix(observed, np.ones(len(AREA))/len(AREA), design)
    projected = notch(spatial, HORIZON, len(AREA), nuisance)
    residual = trajectory_matrix(subtract_sinusoid(observed, nuisance), AREA, design)
    k = spatial.shape[1]
    errors = {"direct_spectrum": phase_error_permutation(
        targets[:, None] * np.arange(k)[None, :], truth[:, :k])}
    for label, h in [("spatial_mssa", spatial), ("classical_mssa", classical),
                     ("lag_projection", projected), ("time_regression", residual)]:
        estimated = dmd_phase_coordinates(h, targets, rank=RANK)["phases"]
        errors[label] = phase_error_permutation(estimated, truth[:, :k])
    base["phase_mae_rad"] = errors
    return base


def main() -> None:
    truth, _, nonlinear, weak = case_data("weak_primitive")
    clean = field(truth, nonlinear, weak)
    scale = np.sqrt(np.mean(np.sum(AREA[:, None] * clean**2, axis=0)))
    rows = []
    for noise in NOISE:
        for seed in SEEDS:
            observed = clean + noise * scale * np.random.default_rng(seed).standard_normal(clean.shape)
            rows.append(run_one(observed, truth, noise, seed))
        print("weak primitive", noise, "complete", flush=True)
    strict_cover = field(truth, nonlinear=0.0, weak=0.0)
    strict_scale = np.sqrt(np.mean(np.sum(AREA[:, None] * strict_cover**2, axis=0)))
    negative = []
    one_phase = PROFILES[0][:, None] * (
        np.cos(2*truth[0]) + .035*np.cos(truth[0]) + .2*np.cos(3*truth[0]))[None, :]
    one_phase_scale = np.sqrt(np.mean(np.sum(AREA[:, None] * one_phase**2, axis=0)))
    one_phase_negative = []
    for noise in NOISE:
        for seed in SEEDS:
            cover_observed = strict_cover + noise*strict_scale*np.random.default_rng(seed).standard_normal(strict_cover.shape)
            one_observed = one_phase + noise*one_phase_scale*np.random.default_rng(seed).standard_normal(one_phase.shape)
            negative.append({"noise": noise, "seed": seed,
                             "selection": select_harmonic_family(cover_observed)})
            one_phase_negative.append({"noise": noise, "seed": seed,
                                       "selection": select_harmonic_family(one_observed)})
    methods = ["direct_spectrum", "spatial_mssa", "classical_mssa",
               "lag_projection", "time_regression"]
    summary = []
    for noise in NOISE:
        chosen = [r for r in rows if r["noise"] == noise]
        valid = [r for r in chosen if r["computed"]]
        summary.append({
            "noise": noise, "accepted": len(valid), "trials": len(chosen),
            "median_mae_rad": {m: float(np.median([r["phase_mae_rad"][m] for r in valid]))
                               if valid else None for m in methods},
            "p90_mae_rad": {m: float(np.quantile([r["phase_mae_rad"][m] for r in valid], .9))
                            if valid else None for m in methods},
            "failure_rate_over_0_3rad_including_rejection": {
                m: float(sum((not r["computed"]) or r["phase_mae_rad"][m] > .3
                             for r in chosen)/len(chosen)) for m in methods},
        })
    payload = {"case": "weak_primitive", "known_r": 2,
               "front_end": "strongest peak, significant resolved half-frequency, next frequency-separated peak",
               "relative_peak_cutoff": RELATIVE_PEAK_CUTOFF,
               "rank": RANK, "horizon": HORIZON, "seeds": SEEDS,
               "rows": rows, "summary": summary,
               "strict_double_cover_negative": negative,
               "one_phase_three_harmonics_negative": one_phase_negative,
               "strict_cover_gate_acceptance_rate": {
                   str(noise): float(np.mean([x["selection"]["accepted"] for x in negative
                                              if x["noise"] == noise])) for noise in NOISE},
               "one_phase_gate_acceptance_rate": {
                   str(noise): float(np.mean([x["selection"]["accepted"]
                                              for x in one_phase_negative if x["noise"] == noise]))
                   for noise in NOISE}}
    (OUT / "blind_notch.json").write_text(json.dumps(payload, indent=2) + "\n")
    for row in summary:
        print(row, flush=True)
    print("strict double-cover gate acceptance", payload["strict_cover_gate_acceptance_rate"])
    print("one-phase three-harmonic gate acceptance", payload["one_phase_gate_acceptance_rate"])


if __name__ == "__main__":
    main()
