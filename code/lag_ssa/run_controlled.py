"""Reproducible controlled tests of finite positive-lag SSA.

Run from the repository root with ``.venv/bin/python code/lag_ssa/run_controlled.py``.
All phase errors use known synthetic truth; that truth is not passed to DMD.
The oracle lag design does receive true frequencies and is labeled accordingly.
"""

from __future__ import annotations

from project_paths import RESULTS_DIR

import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from method import (
    LagDesign, circular_mae, design_min_variance_lags, design_positive_lags,
    dmd_phase_coordinates,
    estimate_peaks, moment_matrix, torus_xyz, trajectory_matrix,
    voronoi_weights_1d,
)


OUT = RESULTS_DIR
OUT.mkdir(exist_ok=True)
SENSORS = np.array([0.05, 0.14, 0.23, 0.34, 0.47, 0.62, 0.76, 0.90])
AREA = voronoi_weights_1d(SENSORS, (0.0, 1.0))
PROFILES = (
    1.0 + 0.4 * np.cos(2 * np.pi * SENSORS),
    0.9 + 0.3 * np.sin(2 * np.pi * SENSORS),
    0.4 + 0.2 * SENSORS,
)
N = 2400
FREQUENCY = 2 * np.pi * np.array([0.061 * np.sqrt(2), 0.073 * np.sqrt(3)])
ORIGIN = np.array([0.23, 1.08])


def poisson(angle: np.ndarray, rho: float) -> np.ndarray:
    """Poisson kernel: 1 + 2 sum_{k>=1} rho^k cos(k angle)."""
    return (1 - rho * rho) / (1 - 2 * rho * np.cos(angle) + rho * rho)


def field(phases: np.ndarray, nonlinear: float, weak: float | None = None) -> np.ndarray:
    """One sensor-sampled field, with an infinite mixed-harmonic series."""
    a, b, c = PROFILES
    theta, phi = phases
    if weak is None:
        first = np.cos(theta)
    else:
        first = np.cos(2 * theta) + weak * np.cos(theta)
    return (a[:, None] * first[None, :]
            + b[:, None] * np.cos(phi)[None, :]
            + nonlinear * c[:, None] * (poisson(theta + phi, 0.6) - 1)[None, :])


def case_data(name: str) -> tuple[np.ndarray, np.ndarray, float, float | None]:
    n = np.arange(N)
    frequency = FREQUENCY.copy()
    nonlinear = {"two_modes": 0.0, "infinite_weak": 0.05,
                 "infinite_strong": 0.20, "weak_primitive": 0.05,
                 "aliased_sampling": 0.05}[name]
    weak = 0.035 if name == "weak_primitive" else None
    if name == "aliased_sampling":
        frequency[0] = 2 * np.pi * 0.06
    phases = ORIGIN[:, None] + frequency[:, None] * n[None, :]
    return phases, frequency, nonlinear, weak


def fit_field_fourier(phases: np.ndarray, signals: np.ndarray, degree: int = 3) -> float:
    """Held-out field NRMSE from the recovered 2D phases; no true phase input."""
    basis = [np.ones(phases.shape[1])]
    for m1 in range(-degree, degree + 1):
        for m2 in range(-degree, degree + 1):
            if (m1, m2) <= (0, 0):
                continue
            angle = m1 * phases[0] + m2 * phases[1]
            basis.extend([np.cos(angle), np.sin(angle)])
    x = np.stack(basis, axis=1)
    split = int(0.7 * x.shape[0])
    coefficients = np.linalg.lstsq(x[:split], signals[:, :split].T, rcond=1e-8)[0]
    prediction = np.einsum("ij,jk->ik", x[split:], coefficients, optimize=False).T
    actual = signals[:, split:]
    numerator = np.sum(AREA[:, None] * (actual - prediction) ** 2)
    denominator = np.sum(AREA[:, None] * (actual - signals[:, :split].mean(axis=1, keepdims=True)) ** 2)
    return float(np.sqrt(numerator / denominator))


def phase_error_permutation(estimated: np.ndarray, truth: np.ndarray) -> float:
    """Allow either order of the two latent coordinates, but not their harmonics."""
    return min(circular_mae(estimated, truth), circular_mae(estimated, truth[::-1]))


def geometry_ratios(
    nonlinear: float, weak: float | None,
    design: LagDesign, frequency: np.ndarray, spatial_weights: np.ndarray,
    theta: np.ndarray, phi: np.ndarray,
) -> dict:
    """Empirical global distance ratios on off-trajectory phase pairs."""
    samples = theta.shape[1]
    q = len(SENSORS)
    first = np.empty((q * len(design.lags), samples))
    second = np.empty_like(first)
    for j, (lag, weight) in enumerate(zip(design.lags, design.weights)):
        shift = frequency[:, None] * lag
        scale = np.sqrt(weight * spatial_weights)[:, None]
        first[j*q:(j+1)*q] = scale * field(theta + shift, nonlinear, weak)
        second[j*q:(j+1)*q] = scale * field(phi + shift, nonlinear, weak)
    phase_delta = np.angle(np.exp(1j * (theta - phi)))
    distance = np.linalg.norm(phase_delta, axis=0)
    ratios = np.linalg.norm(first - second, axis=0) / distance
    return {"min": float(np.min(ratios)), "p05": float(np.quantile(ratios, .05)),
            "median": float(np.median(ratios)), "p95": float(np.quantile(ratios, .95)),
            "distortion_p95_over_p05": float(np.quantile(ratios, .95) / np.quantile(ratios, .05))}


def theoretical_tail(nonlinear: float, weak: float | None) -> dict:
    """Analytic sufficient lower bound for the selected primitive modes."""
    a, b, c = PROFILES
    norm = lambda v: float(np.sqrt(np.sum(AREA * v * v)))
    first_amplitude = weak if weak is not None else 1.0
    b_lower = np.sqrt(2) / np.pi * min(first_amplitude * norm(a), norm(b))
    rho = .6
    tail = 2 * np.sqrt(2) * nonlinear * norm(c) * rho / (1-rho)**2
    if weak is not None:
        tail += 2 * norm(a)  # the excluded ±(2,0) pair
    return {"b_lower": float(b_lower), "tail_upper": float(tail),
            "certified": bool(tail < b_lower)}


def draw_case(
    name: str, phases: np.ndarray, trial_results: list[dict],
    embeddings: list[np.ndarray], recovered: list[np.ndarray],
) -> None:
    fig = plt.figure(figsize=(15, 8))
    for column, (row, h, phase_est) in enumerate(zip(trial_results, embeddings, recovered)):
        ax = fig.add_subplot(2, 3, column + 1, projection="3d")
        stride = 2 if h.shape[1] > 1000 else 1
        x = h[:, ::stride].T.copy()
        x -= x.mean(axis=0)
        _, _, vh = np.linalg.svd(x, full_matrices=False)
        xyz = np.einsum("ij,jk->ik", x, vh[:3].T, optimize=False)
        ax.scatter(*xyz.T, c=np.mod(phases[0, :xyz.shape[0]*stride:stride][:xyz.shape[0]],2*np.pi),
                   cmap="twilight", s=2, alpha=.65)
        ax.set_title(f"{row['design']}: PCA of SSA")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax = fig.add_subplot(2, 3, column + 4, projection="3d")
        u = np.linspace(0, 2*np.pi, 36)
        v = np.linspace(0, 2*np.pi, 18)
        uu, vv = np.meshgrid(u, v)
        surface = torus_xyz(np.stack([uu, vv]))
        ax.plot_surface(surface[...,0], surface[...,1], surface[...,2],
                        color="#c8c8c8", alpha=.08, linewidth=0, shade=False)
        xyz = torus_xyz(phase_est[:, ::stride])
        ax.scatter(*xyz.T, c=np.mod(phases[0, :phase_est.shape[1]:stride],2*np.pi),
                   cmap="twilight", s=2, alpha=.65)
        ax.set_title(f"phase chart; MAE={row['phase_mae_rad']:.3f} rad")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.view_init(elev=25, azim=35)
    fig.suptitle(f"{name}: upper row is raw SSA; lower row uses recovered phases\n"
                 "Color is the known first phase; chart shape alone is not evidence of recovery.")
    fig.tight_layout()
    fig.savefig(OUT / f"{name}_torus.png", dpi=150)
    plt.close(fig)


def main() -> None:
    rng = np.random.default_rng(20261007)
    names = ["two_modes", "infinite_weak", "infinite_strong",
             "weak_primitive", "aliased_sampling"]
    all_results = {"seed": 20261007, "sample_count": N,
                   "sensor_positions": SENSORS.tolist(),
                   "spatial_weights": AREA.tolist(), "cases": []}
    for name in names:
        phases, frequency, nonlinear, weak = case_data(name)
        clean = field(phases, nonlinear, weak)
        scale = np.sqrt(np.mean(np.sum(AREA[:, None] * clean**2, axis=0)))
        noise_level = 0.03 if name != "weak_primitive" else 0.08
        observed = clean + noise_level * scale * rng.standard_normal(clean.shape)
        detected, power = estimate_peaks(observed, AREA)
        selected = np.r_[frequency, -frequency]
        design = design_positive_lags(selected, max_lag=256)
        uniform_support = LagDesign(design.lags, np.ones(len(design.lags))/len(design.lags),
                                    float('nan'), design.candidate_count)
        consecutive = LagDesign(np.arange(len(design.lags)),
                                np.ones(len(design.lags))/len(design.lags),
                                float('nan'), len(design.lags))
        full_horizon = LagDesign(np.arange(max(design.lags) + 1),
                                 np.ones(max(design.lags) + 1)/(max(design.lags) + 1),
                                 float('nan'), max(design.lags) + 1)
        blind_design = design_positive_lags(np.r_[detected, -detected], max_lag=256)
        blind_budget = LagDesign(np.arange(len(blind_design.lags)),
                                 np.ones(len(blind_design.lags))/len(blind_design.lags),
                                 float('nan'), len(blind_design.lags))
        blind_horizon_count = int(max(blind_design.lags)) + 1
        blind_horizon = LagDesign(np.arange(blind_horizon_count),
                                  np.ones(blind_horizon_count)/blind_horizon_count,
                                  float('nan'), blind_horizon_count)
        dense_design = design_min_variance_lags(selected, max(design.lags)+1)
        phase_pairs = (rng.uniform(-np.pi, np.pi, (2, 2000)),
                       rng.uniform(-np.pi, np.pi, (2, 2000)))
        variants = [
            ("positive_oracle", design, frequency, AREA),
            ("same_lags_uniform", uniform_support, frequency, AREA),
            ("spatial_uniform_budget", consecutive, frequency, AREA),
            ("spatial_uniform_horizon", full_horizon, frequency, AREA),
            ("positive_blind", blind_design, detected, AREA),
            ("positive_min_variance_horizon", dense_design, frequency, AREA),
            ("mssa_equal_budget", consecutive, frequency, np.ones(len(AREA))/len(AREA)),
            ("mssa_equal_horizon", full_horizon, frequency, np.ones(len(AREA))/len(AREA)),
            ("mssa_blind_equal_budget", blind_budget, detected, np.ones(len(AREA))/len(AREA)),
            ("mssa_blind_equal_horizon", blind_horizon, detected, np.ones(len(AREA))/len(AREA)),
        ]
        rows, embeddings, recovered = [], [], []
        for label, variant, phase_targets, spatial_weights in variants:
            h = trajectory_matrix(observed, spatial_weights, variant)
            dmd = dmd_phase_coordinates(h, phase_targets, rank=8)
            k = h.shape[1]
            phase_est = dmd["phases"]
            spectral_matrix = moment_matrix(selected, variant.lags)
            rhs = np.r_[1.0, np.zeros(spectral_matrix.shape[0]-1)]
            moment_error = float(np.max(abs(spectral_matrix @ variant.weights-rhs)))
            target_matrix = moment_matrix(np.r_[phase_targets, -phase_targets], variant.lags)
            target_rhs = np.r_[1.0, np.zeros(target_matrix.shape[0]-1)]
            row = {
                "design": label, "lag_count": len(variant.lags),
                "max_lag": int(max(variant.lags)), "moment_residual": moment_error,
                "target_moment_residual": float(np.max(abs(target_matrix @ variant.weights-target_rhs))),
                "frequency_targets": phase_targets.tolist(),
                "spatial_weighting": "voronoi" if np.array_equal(spatial_weights, AREA) else "equal",
                "phase_mae_rad": phase_error_permutation(phase_est, phases[:, :k]),
                "dmd_frequencies": np.angle(dmd["eigenvalues"]).tolist(),
                "radius_cv": (np.std(dmd["radii"],axis=1)
                              / np.mean(dmd["radii"],axis=1)).tolist(),
                "field_test_nrmse": fit_field_fourier(phase_est, observed[:, :k]),
                "geometry": geometry_ratios(nonlinear, weak, variant,
                                             frequency, spatial_weights, *phase_pairs),
            }
            rows.append(row); embeddings.append(h); recovered.append(phase_est)
        shown = [0, 6, 7]
        draw_case(name, phases, [rows[i] for i in shown],
                  [embeddings[i] for i in shown], [recovered[i] for i in shown])
        dense_shown = [0, 5, 7]
        draw_case(name + "_dense", phases, [rows[i] for i in dense_shown],
                  [embeddings[i] for i in dense_shown], [recovered[i] for i in dense_shown])
        all_results["cases"].append({
            "name": name, "frequency_rad_per_sample": frequency.tolist(),
            "sampled_orbit_dense_on_full_torus": name != "aliased_sampling",
            "preliminary_peaks": detected.tolist(),
            "preliminary_peak_power": power.tolist(),
            "noise_relative_rms": noise_level,
            "theory": theoretical_tail(nonlinear, weak),
            "positive_lags": design.lags.tolist(),
            "positive_weights": design.weights.tolist(),
            "variants": rows,
        })
        print(name, {row["design"]: round(row["phase_mae_rad"], 3) for row in rows},
              flush=True)
    (OUT / "controlled_metrics.json").write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2) + "\n")
    print(OUT / "controlled_metrics.json")


if __name__ == "__main__":
    main()
