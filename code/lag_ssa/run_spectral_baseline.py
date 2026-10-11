"""Direct spectral phase baseline for the rigid torus-flow model.

This is a diagnostic with known phase dimension r=2 and a low-frequency
selection rule.  It does not solve general lattice identification.
"""

from __future__ import annotations

from project_paths import RESULTS_DIR

import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

from method import design_positive_lags, estimate_peaks, infer_frequency_lattice
from run_controlled import AREA, N, case_data, field, phase_error_permutation


OUT = RESULTS_DIR
OUT.mkdir(exist_ok=True)
CASES = ["two_modes", "infinite_weak", "infinite_strong",
         "weak_primitive", "aliased_sampling"]
NOISE = [0.03, 0.08, 0.10, 0.20, 0.40]


def refine_frequency(signal: np.ndarray, peak: float) -> float:
    """Maximize spatially aggregated Hann spectral power near one peak."""
    sample_count = signal.shape[1]
    times = np.arange(sample_count)
    weighted = (signal - signal.mean(axis=1, keepdims=True)) * np.hanning(sample_count)

    def negative_power(angular: float) -> float:
        wave = np.exp(-1j * angular * times)
        coefficient = np.einsum("in,n->i", weighted, wave, optimize=False)
        return -float(np.sum(AREA * abs(coefficient)**2))

    width = 2 * np.pi / sample_count
    result = minimize_scalar(negative_power, bounds=(peak-width, peak+width),
                             method="bounded", options={"xatol": 1e-11})
    if not result.success:
        raise RuntimeError(result.message)
    return float(result.x)


def evaluate(case: str, seed: int, noise: float) -> dict:
    truth, frequency, nonlinear, weak = case_data(case)
    clean = field(truth, nonlinear, weak)
    scale = np.sqrt(np.mean(np.sum(AREA[:, None] * clean**2, axis=0)))
    observed = clean + noise * scale * np.random.default_rng(1000 + seed).standard_normal(clean.shape)
    peaks, strengths = estimate_peaks(observed, AREA, count=8, suppress_harmonics=False)
    visible = peaks[strengths / strengths[0] > .0005]
    relation = infer_frequency_lattice(visible, tolerance=2*np.pi/N,
                                       max_rank=3, max_order=6)
    if len(visible) < 2:
        return {"case": case, "seed": seed, "noise": noise, "computed": False,
                "reason": "fewer than two visible peaks"}
    selected = np.sort(visible)[:2]
    refined = np.array([refine_frequency(observed, x) for x in selected])
    estimated_phase = refined[:, None] * np.arange(N)[None, :]
    # Match the columns scored in the same-horizon MSSA noise sweep.  The
    # synthetic truth is used only to define this evaluation subset.
    reference_design = design_positive_lags(np.r_[frequency, -frequency], max_lag=256)
    scored_samples = N - int(max(reference_design.lags))
    phase_mae = phase_error_permutation(estimated_phase[:, :scored_samples],
                                        truth[:, :scored_samples])
    return {
        "case": case, "seed": seed, "noise": noise, "computed": True,
        "noise_relative_rms": noise,
        "known_dimension": 2,
        "visible_peaks": visible.tolist(),
        "conditional_lattice_rank": relation["rank"],
        "selected_coarse_frequencies": selected.tolist(),
        "refined_frequencies": refined.tolist(),
        "true_frequencies_for_evaluation_only": frequency.tolist(),
        "scored_samples": scored_samples,
        "phase_mae_rad": phase_mae,
        "phase_failure_over_0_3rad": bool(phase_mae > .3),
    }


def main() -> None:
    rows = [evaluate(case, seed, noise) for case in CASES
            for noise in NOISE for seed in range(6)]
    summary = []
    for case in CASES:
        for noise in NOISE:
            selected = [r for r in rows if r["case"] == case and r["noise"] == noise]
            errors = [r["phase_mae_rad"] for r in selected if r["computed"]]
            summary.append({"case": case, "noise": noise,
                            "computed_runs": len(errors), "trials": len(selected),
                            "phase_mae_median": float(np.median(errors)) if errors else None,
                            "phase_mae_max": float(np.max(errors)) if errors else None,
                            "phase_failure_rate_over_0_3rad": float(
                                sum((not r["computed"])
                                    or r.get("phase_failure_over_0_3rad", False)
                                    for r in selected) / len(selected))})
            print(summary[-1], flush=True)
    payload = {"method": "direct Hann periodogram with local frequency refinement",
               "selection": "two smallest peaks above 0.0005 of maximum power; r=2 supplied",
               "seed_ids": list(range(6)), "noise_levels": NOISE,
               "rows": rows, "summary": summary}
    (OUT / "spectral_baseline.json").write_text(json.dumps(payload, indent=2) + "\n")

if __name__ == "__main__":
    main()
