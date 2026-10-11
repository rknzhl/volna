"""Oracle delay-axis notch test against the strong overtone.

The filter is a genuine linear transformation of a full SSA trajectory matrix:
it removes the lag-axis cosine/sine subspace at twice the first frequency.
The nuisance frequency is supplied from the synthetic truth.
"""

from __future__ import annotations

from project_paths import RESULTS_DIR

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from method import LagDesign, dmd_phase_coordinates, trajectory_matrix
from run_controlled import AREA, case_data, field, phase_error_permutation


OUT = RESULTS_DIR
OUT.mkdir(exist_ok=True)
NOISE = [0.0, .08, .10, .20]
HORIZONS = [18, 32, 64]
RANKS = [4, 8]


def notch(trajectory: np.ndarray, horizon: int, sensors: int,
          angular_frequency: float) -> np.ndarray:
    delays = np.arange(horizon)
    nuisance = np.stack([np.cos(angular_frequency * delays),
                         np.sin(angular_frequency * delays)], axis=1)
    basis, _ = np.linalg.qr(nuisance)
    projector = np.eye(horizon) - basis @ basis.T
    blocks = trajectory.reshape(horizon, sensors, -1)
    return np.einsum("ab,bik->aik", projector, blocks, optimize=False).reshape(trajectory.shape)


def main() -> None:
    truth, frequency, nonlinear, weak = case_data("weak_primitive")
    clean = field(truth, nonlinear, weak)
    scale = np.sqrt(np.mean(np.sum(AREA[:, None] * clean**2, axis=0)))
    rows = []
    for noise in NOISE:
        observed = clean + noise * scale * np.random.default_rng(20261007).standard_normal(clean.shape)
        for horizon in HORIZONS:
            design = LagDesign(np.arange(horizon), np.ones(horizon)/horizon,
                               float("nan"), horizon)
            spatial = trajectory_matrix(observed, AREA, design)
            classical = trajectory_matrix(observed, np.ones(len(AREA))/len(AREA), design)
            filtered = notch(spatial, horizon, len(AREA), 2 * frequency[0])
            for label, h in [("spatial_uniform", spatial),
                             ("classical_mssa", classical),
                             ("overtone_notch", filtered)]:
                for rank in RANKS:
                    dmd = dmd_phase_coordinates(h, frequency, rank=rank)
                    rows.append({
                        "noise": noise, "horizon": horizon, "rank": rank,
                        "design": label,
                        "phase_mae_rad": phase_error_permutation(
                            dmd["phases"], truth[:, :h.shape[1]]),
                    })
            print(f"noise={noise:g}, horizon={horizon} complete", flush=True)
    payload = {"case": "weak_primitive", "oracle_nuisance_frequency": float(2*frequency[0]),
               "oracle_phase_targets": frequency.tolist(), "seed": 20261007,
               "rows": rows}
    (OUT / "notch_ssa.json").write_text(json.dumps(payload, indent=2) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    colors = {"spatial_uniform": "#0072B2", "classical_mssa": "#D55E00",
              "overtone_notch": "#009E73"}
    for ax, rank in zip(axes, RANKS):
        for label in colors:
            selected = [r for r in rows if r["noise"] == .08
                        and r["rank"] == rank and r["design"] == label]
            ax.plot([r["horizon"] for r in selected],
                    [r["phase_mae_rad"] for r in selected],
                    marker="o", color=colors[label], label=label)
        ax.set_title(f"rank {rank}, noise 8%")
        ax.set_xlabel("lag horizon")
        ax.grid(alpha=.2)
    axes[0].set_ylabel("phase MAE (rad)")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "notch_ssa.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
