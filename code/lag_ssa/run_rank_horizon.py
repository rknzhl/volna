"""Rank/window sensitivity at the weak-primitive counterexample.

All methods receive the same known target frequencies.  The purpose is to
separate exact lag moment balancing from the effects of window length and SVD
truncation, not to validate a blind estimator.
"""

from __future__ import annotations

from project_paths import RESULTS_DIR

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from method import LagDesign, design_min_variance_lags, design_positive_lags, dmd_phase_coordinates, trajectory_matrix
from run_controlled import AREA, case_data, field, phase_error_permutation


OUT = RESULTS_DIR
OUT.mkdir(exist_ok=True)


def main() -> None:
    truth, frequency, nonlinear, weak = case_data("weak_primitive")
    clean = field(truth, nonlinear, weak)
    scale = np.sqrt(np.mean(np.sum(AREA[:, None] * clean**2, axis=0)))
    positive = design_positive_lags(np.r_[frequency, -frequency], max_lag=256)
    horizons = sorted(set([int(max(positive.lags)) + 1, 32, 64, 128]))
    rows = []
    for noise in (0.0, 0.08):
        rng = np.random.default_rng(20261007)
        observed = clean + noise * scale * rng.standard_normal(clean.shape)
        for horizon in horizons:
            uniform = LagDesign(np.arange(horizon), np.ones(horizon) / horizon,
                                float("nan"), horizon)
            dense = design_min_variance_lags(np.r_[frequency, -frequency], horizon)
            variants = [
                ("positive_dense", dense, AREA),
                ("classical_mssa", uniform, np.ones(len(AREA)) / len(AREA)),
            ]
            if horizon == horizons[0]:
                variants.insert(0, ("positive_sparse", positive, AREA))
            for name, design, spatial in variants:
                trajectory = trajectory_matrix(observed, spatial, design)
                for rank in (8, 10, 16):
                    result = dmd_phase_coordinates(trajectory, frequency, rank=rank)
                    rows.append({
                        "noise": noise, "horizon": horizon, "rank": rank,
                        "design": name, "lag_count": len(design.lags),
                        "phase_mae_rad": phase_error_permutation(
                            result["phases"], truth[:, :trajectory.shape[1]]),
                    })
            print(f"noise={noise:g}, horizon={horizon} complete", flush=True)
    payload = {"case": "weak_primitive", "known_frequency_targets": frequency.tolist(),
               "seed": 20261007, "rows": rows}
    (OUT / "rank_horizon.json").write_text(json.dumps(payload, indent=2) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    colors = {"positive_dense": "#0072B2", "classical_mssa": "#D55E00",
              "positive_sparse": "#009E73"}
    for ax, noise in zip(axes, (0.0, 0.08)):
        for rank, style in ((8, "-"), (10, "--"), (16, ":")):
            for design in ("positive_dense", "classical_mssa"):
                selected = [r for r in rows if r["noise"] == noise and r["rank"] == rank
                            and r["design"] == design]
                ax.plot([r["horizon"] for r in selected],
                        [r["phase_mae_rad"] for r in selected],
                        linestyle=style, marker="o", color=colors[design],
                        label=f"{design}, rank {rank}")
        sparse = [r for r in rows if r["noise"] == noise and r["design"] == "positive_sparse"]
        ax.scatter([r["horizon"] for r in sparse], [r["phase_mae_rad"] for r in sparse],
                   marker="x", s=65, color=colors["positive_sparse"], label="positive_sparse")
        ax.set_title(f"noise RMS / signal RMS = {noise:g}")
        ax.set_xlabel("number of consecutive lags / horizon")
        ax.grid(alpha=.2)
    axes[0].set_ylabel("phase MAE (rad)")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="lower center", frameon=False)
    fig.tight_layout(rect=(0,.16,1,1))
    fig.savefig(OUT / "rank_horizon.png", dpi=160)
    plt.close(fig)
    print(OUT / "rank_horizon.json")


if __name__ == "__main__":
    main()
