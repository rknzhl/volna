"""Compare torus phase charts for the data-trained secant pilot (seed 0)."""

from __future__ import annotations

from project_paths import RESULTS_DIR

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from method import dmd_phase_coordinates, torus_xyz, trajectory_matrix
from run_controlled import AREA, case_data, field, phase_error_permutation
from run_secant_data_driven import J, L, NOISE, TRAIN_LENGTH
from run_secant_phase import lag_design


OUT = RESULTS_DIR


def main() -> None:
    trained = json.loads((OUT / "secant_data_driven.json").read_text())
    fig = plt.figure(figsize=(16, 8.2))
    headings = ["data secant SSA, 9 lags", "MSSA, 9 lags",
                "MSSA, 64 lags", "direct train frequency"]
    grid_u, grid_v = np.meshgrid(np.linspace(0, 2*np.pi, 40),
                                 np.linspace(0, 2*np.pi, 22))
    surface = torus_xyz(np.stack([grid_u, grid_v]))
    for row_id, case in enumerate(["infinite_weak", "weak_primitive"]):
        record = next(r for r in trained["rows"] if r["case"] == case and r["seed"] == 0)
        phases, _, nonlinear, weak = case_data(case)
        truth = phases[:, TRAIN_LENGTH:]
        clean = field(truth, nonlinear, weak)
        scale = np.sqrt(np.mean(np.sum(AREA[:, None]*clean*clean, axis=0)))
        observed = clean + NOISE*scale*np.random.default_rng(8100).standard_normal(clean.shape)
        targets = np.asarray(record["fit"]["frequencies"])
        k = observed.shape[1]-L+1
        variants = [
            lag_design(record["selected_lags"], record["selected_weights"]),
            lag_design(list(range(J)), [1/J]*J),
            lag_design(list(range(L)), [1/L]*L),
        ]
        estimates = [dmd_phase_coordinates(
            trajectory_matrix(observed, AREA, variant)[:, :k], targets, rank=8)["phases"]
            for variant in variants]
        estimates.append(targets[:, None]*np.arange(k)[None, :])
        for col, estimate in enumerate(estimates):
            ax = fig.add_subplot(2, 4, row_id*4+col+1, projection="3d")
            ax.plot_surface(surface[..., 0], surface[..., 1], surface[..., 2],
                            color="#d6d6d6", alpha=.1, linewidth=0, shade=False)
            xyz = torus_xyz(estimate[:, ::2])
            ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2],
                       c=np.mod(truth[0, :k:2], 2*np.pi), cmap="twilight",
                       s=3, alpha=.8, vmin=0, vmax=2*np.pi)
            error = phase_error_permutation(estimate, truth[:, :k])
            ax.set_title(f"{headings[col]}\nMAE {error:.3f} rad", fontsize=10)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
            ax.view_init(elev=25, azim=35)
        fig.text(.015, .74 if row_id == 0 else .27, case.replace("_", " "),
                 rotation=90, fontsize=11, va="center")
    fig.suptitle("Canonical torus chart from estimated phases; color = true first phase\n"
                 "The donut shape is imposed by the chart and does not prove recovery.")
    fig.tight_layout(rect=(.02, 0, 1, .92))
    fig.savefig(OUT / "secant_phase_tori.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
