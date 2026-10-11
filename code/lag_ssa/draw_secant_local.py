"""Phase charts after local nonlinear inversion of the learned lag map."""

from __future__ import annotations

from project_paths import RESULTS_DIR

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from method import torus_xyz
from run_controlled import AREA, case_data, field, phase_error_permutation
from run_secant_data_driven import J, L, NOISE, TRAIN_LENGTH, fitted_model
from run_secant_local_decoder import decode
from run_secant_phase import lag_design


OUT = RESULTS_DIR


def main() -> None:
    trained = json.loads((OUT / "secant_data_driven.json").read_text())
    axis = np.linspace(0, 2*np.pi, 64, endpoint=False)
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    grid = np.stack([xx.ravel(), yy.ravel()])
    uu, vv = np.meshgrid(np.linspace(0, 2*np.pi, 40),
                         np.linspace(0, 2*np.pi, 22))
    surface = torus_xyz(np.stack([uu, vv]))
    fig = plt.figure(figsize=(16, 8.2))
    names = ["learned secant, 9 lags", "consecutive, 9 lags",
             "full window, 64 lags", "training frequencies"]
    for row_index, case in enumerate(["infinite_weak", "weak_primitive"]):
        record = next(r for r in trained["rows"] if r["case"] == case and r["seed"] == 0)
        phases, _, nonlinear, weak = case_data(case)
        clean_train = field(phases[:, :TRAIN_LENGTH], nonlinear, weak)
        scale = np.sqrt(np.mean(np.sum(AREA[:, None]*clean_train*clean_train, axis=0)))
        train = clean_train + NOISE*scale*np.random.default_rng(6100).standard_normal(clean_train.shape)
        frequency, coefficients, _ = fitted_model(train)
        truth = phases[:, TRAIN_LENGTH:]
        clean_test = field(truth, nonlinear, weak)
        scale = np.sqrt(np.mean(np.sum(AREA[:, None]*clean_test*clean_test, axis=0)))
        test = clean_test + NOISE*scale*np.random.default_rng(8100).standard_normal(clean_test.shape)
        k = 1200-L+1
        designs = [lag_design(record["selected_lags"], record["selected_weights"]),
                   lag_design(list(range(J)), [1/J]*J),
                   lag_design(list(range(L)), [1/L]*L)]
        estimates = [decode(test, grid, frequency, coefficients, design, k)[1]
                     for design in designs]
        estimates.append(frequency[:, None]*np.arange(k)[None, :])
        for column, estimate in enumerate(estimates):
            ax = fig.add_subplot(2, 4, row_index*4+column+1, projection="3d")
            ax.plot_surface(surface[..., 0], surface[..., 1], surface[..., 2],
                            color="#d6d6d6", alpha=.1, linewidth=0, shade=False)
            xyz = torus_xyz(estimate[:, ::2])
            ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2],
                       c=np.mod(truth[0, :k:2], 2*np.pi), cmap="twilight",
                       s=3, alpha=.8, vmin=0, vmax=2*np.pi)
            error = phase_error_permutation(estimate, truth[:, :k])
            ax.set_title(f"{names[column]}\nMAE {error:.3f} rad", fontsize=10)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
            ax.view_init(elev=25, azim=35)
        fig.text(.015, .74 if row_index == 0 else .27, case.replace("_", " "),
                 rotation=90, fontsize=11, va="center")
    fig.suptitle("Learned-map inversion; color = true first phase\n"
                 "The canonical donut shape is imposed by the chart; phase error tests recovery.")
    fig.tight_layout(rect=(.02, 0, 1, .92))
    fig.savefig(OUT / "secant_local_tori.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
