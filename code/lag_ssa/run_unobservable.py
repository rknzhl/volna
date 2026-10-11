"""Negative control: the first torus phase is exactly unobservable."""

from __future__ import annotations

from project_paths import RESULTS_DIR

import json
from pathlib import Path

import numpy as np

from method import LagDesign, design_positive_lags, dmd_phase_coordinates, trajectory_matrix
from run_controlled import AREA, case_data, draw_case, field, phase_error_permutation


OUT = RESULTS_DIR
OUT.mkdir(exist_ok=True)


def main() -> None:
    truth, frequency, _, _ = case_data("weak_primitive")
    # No primitive (1,0) character and no mixed terms.  The observation is
    # invariant under theta_1 -> theta_1 + pi at every sensor and every lag.
    clean = field(truth, nonlinear=0.0, weak=0.0)
    shifted = field(truth + np.array([[np.pi], [0.0]]), nonlinear=0.0, weak=0.0)
    exact_ambiguity = float(np.max(abs(clean - shifted)))
    scale = np.sqrt(np.mean(np.sum(AREA[:, None] * clean**2, axis=0)))
    observed = clean + .03 * scale * np.random.default_rng(20261007).standard_normal(clean.shape)
    positive = design_positive_lags(np.r_[frequency, -frequency], max_lag=256)
    budget = len(positive.lags)
    horizon = int(max(positive.lags)) + 1
    variants = [
        ("positive_oracle", positive, AREA),
        ("mssa_equal_budget",
         LagDesign(np.arange(budget), np.ones(budget)/budget, float("nan"), budget),
         np.ones(len(AREA))/len(AREA)),
        ("mssa_equal_horizon",
         LagDesign(np.arange(horizon), np.ones(horizon)/horizon, float("nan"), horizon),
         np.ones(len(AREA))/len(AREA)),
    ]
    rows, embeddings, recovered = [], [], []
    for label, design, spatial in variants:
        h = trajectory_matrix(observed, spatial, design)
        dmd = dmd_phase_coordinates(h, frequency, rank=8)
        estimated_frequency = np.angle(dmd["eigenvalues"])
        row = {
            "design": label, "lag_count": len(design.lags), "horizon": int(max(design.lags))+1,
            "phase_mae_rad": phase_error_permutation(dmd["phases"], truth[:, :h.shape[1]]),
            "target_frequencies": frequency.tolist(),
            "selected_frequencies": estimated_frequency.tolist(),
            "first_frequency_mismatch": float(abs(estimated_frequency[0] - frequency[0])),
            "radius_cv": (np.std(dmd["radii"], axis=1)
                          / np.mean(dmd["radii"], axis=1)).tolist(),
        }
        rows.append(row); embeddings.append(h); recovered.append(dmd["phases"])
    draw_case("unobservable_double_cover", truth, rows, embeddings, recovered)
    payload = {"exact_phase_shift_ambiguity_max": exact_ambiguity,
               "noise_relative_rms": .03, "seed": 20261007, "results": rows,
               "interpretation": "The chart can represent the observable quotient torus, but cannot recover the original first phase."}
    (OUT / "unobservable_metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    for row in rows:
        print(row["design"], "MAE", round(row["phase_mae_rad"], 3),
              "frequency mismatch", round(row["first_frequency_mismatch"], 3),
              flush=True)


if __name__ == "__main__":
    main()
