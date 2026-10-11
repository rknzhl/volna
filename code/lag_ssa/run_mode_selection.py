"""Test whether canceling the strong overtone rescues a weak primitive phase."""

from __future__ import annotations

from project_paths import RESULTS_DIR

import json
from pathlib import Path

import numpy as np

from method import LagDesign, design_positive_lags, dmd_phase_coordinates, trajectory_matrix
from run_controlled import AREA, case_data, field, phase_error_permutation


OUT = RESULTS_DIR
OUT.mkdir(exist_ok=True)
MODE_SETS = [
    [(1, 0), (2, 0), (0, 1)],
    [(1, 0), (2, 0), (0, 1), (1, 1)],
    [(1, 0), (2, 0), (0, 1), (1, 1), (2, 2)],
]


def main() -> None:
    truth, frequency, nonlinear, weak = case_data("weak_primitive")
    clean = field(truth, nonlinear, weak)
    scale = np.sqrt(np.mean(np.sum(AREA[:, None] * clean**2, axis=0)))
    observed = clean + .08 * scale * np.random.default_rng(20261007).standard_normal(clean.shape)
    rows = []
    for modes in MODE_SETS:
        target = np.array([a * frequency[0] + b * frequency[1] for a, b in modes])
        positive = design_positive_lags(np.r_[target, -target], max_lag=256)
        budget = len(positive.lags)
        horizon = int(max(positive.lags)) + 1
        variants = [
            ("positive_sparse", positive, AREA),
            ("mssa_equal_budget",
             LagDesign(np.arange(budget), np.ones(budget)/budget, float("nan"), budget),
             np.ones(len(AREA))/len(AREA)),
            ("mssa_equal_horizon",
             LagDesign(np.arange(horizon), np.ones(horizon)/horizon, float("nan"), horizon),
             np.ones(len(AREA))/len(AREA)),
        ]
        for label, design, spatial in variants:
            h = trajectory_matrix(observed, spatial, design)
            for rank in (8, 10, 16):
                result = dmd_phase_coordinates(h, frequency, rank=rank)
                rows.append({
                    "selected_modes": modes, "design": label,
                    "lag_count": len(design.lags), "horizon": int(max(design.lags))+1,
                    "rank": rank,
                    "phase_mae_rad": phase_error_permutation(
                        result["phases"], truth[:, :h.shape[1]]),
                })
        print(f"modes={modes}, budget={budget}, horizon={horizon}", flush=True)
    payload = {"case": "weak_primitive", "noise_relative_rms": .08,
               "known_frequency_targets": frequency.tolist(),
               "seed": 20261007, "rows": rows}
    (OUT / "mode_selection.json").write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
