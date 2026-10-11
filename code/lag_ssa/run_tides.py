"""Two-factor projection of the real Delaware River tide records.

The measured tide contains six catalogued constituents, so the M2/K1 torus
below is a factor-model diagnostic, not a claim that the full field has r=2.
"""

from __future__ import annotations

from project_paths import RESULTS_DIR

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from method import (
    LagDesign, circular_mae, design_positive_lags, dmd_phase_coordinates,
    estimate_peaks, moment_matrix, torus_xyz, trajectory_matrix,
    voronoi_weights_1d,
)


ROOT = Path(__file__).resolve().parents[2]
OUT = RESULTS_DIR
OUT.mkdir(exist_ok=True)


def field_nrmse(phases: np.ndarray, measurements: np.ndarray,
                spatial_weights: np.ndarray) -> float:
    # A two-phase, second-order Fourier decoder is fit only on the first 70%.
    terms = [np.ones(phases.shape[1])]
    for a in range(-2, 3):
        for b in range(-2, 3):
            if (a, b) <= (0, 0):
                continue
            angle = a * phases[0] + b * phases[1]
            terms.extend([np.cos(angle), np.sin(angle)])
    x = np.stack(terms, axis=1)
    split = int(.7 * len(x))
    coefficient = np.linalg.lstsq(x[:split], measurements[:, :split].T, rcond=1e-8)[0]
    prediction = np.einsum("ij,jk->ik", x[split:], coefficient, optimize=False).T
    observed = measurements[:, split:]
    centered = observed - measurements[:, :split].mean(axis=1, keepdims=True)
    numerator = np.sum(spatial_weights[:, None] * (prediction - observed)**2)
    denominator = np.sum(spatial_weights[:, None] * centered**2)
    return float(np.sqrt(numerator / denominator))


def plot(result: list[dict], coordinates: list[np.ndarray],
         reference: np.ndarray) -> None:
    fig = plt.figure(figsize=(15, 5))
    for j, (row, phase) in enumerate(zip(result, coordinates)):
        ax = fig.add_subplot(1, 3, j + 1, projection="3d")
        xyz = torus_xyz(phase[:, ::8])
        ax.scatter(*xyz.T, c=np.mod(reference[1, :phase.shape[1]:8], 2*np.pi),
                   cmap="twilight", s=2, alpha=.65)
        ax.set_title(f"{row['design']}\nmean phase MAE {row['phase_mae_rad']:.2f} rad")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.view_init(25, 35)
    fig.suptitle("Delaware tide: M2/K1 phase chart. Other constituents remain in the data.")
    fig.tight_layout()
    fig.savefig(OUT / "tides_m2_k1_factor_torus.png", dpi=150)
    plt.close(fig)


def main() -> None:
    data = np.load(ROOT / "data/real/tides.npz")
    use = 7200  # 30 days at six-minute spacing; fixed before comparisons
    y = data["eta"][:, :use].astype(float)
    t = data["t"][:use]
    distance = data["distance_km"]
    area = voronoi_weights_1d(distance, (float(distance.min()), float(distance.max())))
    step_hours = float(t[1] - t[0])
    names = data["constituents"].tolist()
    selected_names = ["M2", "K1"]
    indices = [names.index(name) for name in selected_names]
    angular = data["harcon"][0, indices, 2] * np.pi / 180 * step_hours
    detected, strengths = estimate_peaks(y, area, count=6)
    positive = design_positive_lags(np.r_[angular, -angular], max_lag=1000)
    budget = LagDesign(np.arange(len(positive.lags)),
                       np.ones(len(positive.lags))/len(positive.lags),
                       float("nan"), len(positive.lags))
    horizon = LagDesign(np.arange(max(positive.lags)+1),
                        np.ones(max(positive.lags)+1)/(max(positive.lags)+1),
                        float("nan"), max(positive.lags)+1)
    variants = [("positive_oracle", positive, area),
                ("mssa_equal_budget", budget, np.ones(len(area))/len(area)),
                ("mssa_equal_horizon", horizon, np.ones(len(area))/len(area))]
    results, coordinates = [], []
    for label, design, spatial in variants:
        h = trajectory_matrix(y, spatial, design)
        dmd = dmd_phase_coordinates(h, angular, rank=16)
        phase = dmd["phases"]
        reference = angular[:, None] * np.arange(h.shape[1])[None, :]
        mat = moment_matrix(np.r_[angular, -angular], design.lags)
        rhs = np.r_[1.0, np.zeros(mat.shape[0] - 1)]
        row = {
            "design": label, "lag_count": len(design.lags),
            "max_lag": int(max(design.lags)),
            "moment_residual": float(np.max(abs(np.einsum("ij,j->i", mat, design.weights, optimize=False) - rhs))),
            "phase_mae_rad": circular_mae(phase, reference),
            "dmd_frequency_rad_per_sample": np.angle(dmd["eigenvalues"]).tolist(),
            "heldout_field_nrmse": field_nrmse(phase, y[:, :h.shape[1]], area),
        }
        results.append(row); coordinates.append(phase)
        print(label, row, flush=True)
    plot(results, coordinates, angular[:, None] * np.arange(use)[None, :])
    payload = {
        "data": "Delaware River tide station records; six harmonic constituents in metadata",
        "record_samples": use, "sample_step_hours": step_hours,
        "selected_factor_constituents": selected_names,
        "known_angular_frequencies": angular.tolist(),
        "preliminary_peaks": detected.tolist(),
        "preliminary_peak_power": strengths.tolist(),
        "positive_lags": positive.lags.tolist(),
        "positive_weights": positive.weights.tolist(),
        "results": results,
        "interpretation": "two-factor approximation, not a full torus-identification test",
    }
    (OUT / "tides_metrics.json").write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
