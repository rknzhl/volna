"""Evaluate positive-lag SSA against classical MSSA on simulated wave fields.

The source frequencies are known for these simulations. Detuned frequencies
make the sampled orbit dense in the ideal infinite-record model; the original
6 Hz / 100 Hz setup does not. Source-noise is a negative model control.
"""

from __future__ import annotations

from project_paths import RESULTS_DIR

import json
import sys
from pathlib import Path

import numpy as np

from method import (
    LagDesign, circular_mae, design_positive_lags, dmd_phase_coordinates,
    estimate_peaks, moment_matrix, trajectory_matrix, voronoi_weights_grid,
)
from run_controlled import draw_case
from run_tides import field_nrmse


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/waves/synthetic"))
import scenarios  # noqa: E402; this module imports its local wave.py

OUT = RESULTS_DIR
OUT.mkdir(exist_ok=True)
ORIGINAL = tuple(scenarios.NU)
DETUNED = (6 + np.sqrt(2)/10, 6*np.sqrt(2) + np.sqrt(3)/10)
CASES = [
    ("wave_1d_original_alias", "1D, бегущая волна", False, 8.0),
    ("wave_1d_detuned", "1D, бегущая волна", True, 8.0),
    ("wave_1d_standing", "1D, стоячая волна", True, 8.0),
    ("wave_2d_lens", "2D, линза", True, 6.0),
    ("wave_3d_uniform", "3D, однородная", True, 4.0),
    ("wave_1d_source_noise", "1D, шум источника", True, 8.0),
]


def evaluate_one(slug: str, name: str, detuned: bool, duration: float) -> dict:
    scenarios.NU = DETUNED if detuned else ORIGINAL
    result = scenarios.simulate(scenarios.SCENARIOS[name], seed=0,
                                snap_times=(), t_end=scenarios.START+duration)
    t, y, sensors = result["t"], result["Y"], result["sensors"]
    area = voronoi_weights_grid(sensors, resolution={1: 100, 2: 50, 3: 25}[sensors.shape[1]])
    step = float(t[1] - t[0])
    frequency = 2 * np.pi * np.asarray(scenarios.NU) * step
    phases = 2 * np.pi * np.asarray(scenarios.NU)[:, None] * t[None, :]
    preliminary, power = estimate_peaks(y, area)
    design = design_positive_lags(np.r_[frequency, -frequency], max_lag=256)
    budget = LagDesign(np.arange(len(design.lags)),
                       np.ones(len(design.lags))/len(design.lags),
                       float("nan"), len(design.lags))
    horizon = LagDesign(np.arange(max(design.lags)+1),
                        np.ones(max(design.lags)+1)/(max(design.lags)+1),
                        float("nan"), max(design.lags)+1)
    equal_spatial = np.ones(len(area))/len(area)
    variants = [("positive_oracle", design, area),
                ("mssa_equal_budget", budget, equal_spatial),
                ("mssa_equal_horizon", horizon, equal_spatial)]
    rows, embedded, recovered = [], [], []
    for label, lag, spatial in variants:
        h = trajectory_matrix(y, spatial, lag)
        dmd = dmd_phase_coordinates(h, frequency, rank=16)
        phase_est = dmd["phases"]
        mat = moment_matrix(np.r_[frequency,-frequency], lag.lags)
        rhs = np.r_[1.0, np.zeros(mat.shape[0]-1)]
        row = {
            "design": label, "lag_count": len(lag.lags),
            "max_lag": int(max(lag.lags)),
            "moment_residual": float(np.max(abs(np.einsum("ij,j->i", mat, lag.weights)-rhs))),
            "phase_mae_rad": circular_mae(phase_est, phases[:, :h.shape[1]]),
            "dmd_frequency_rad_per_sample": np.angle(dmd["eigenvalues"]).tolist(),
            "heldout_field_nrmse": field_nrmse(phase_est, y[:, :h.shape[1]], area),
        }
        rows.append(row); embedded.append(h); recovered.append(phase_est)
    draw_case(slug, phases, rows, embedded, recovered)
    payload = {
        "slug": slug, "scenario": name, "spatial_dimension": sensors.shape[1],
        "samples": len(t), "sensors": len(sensors), "sample_step": step,
        "source_frequencies_hz": list(scenarios.NU),
        "sampled_orbit_dense_on_full_torus": bool(detuned),
        "source_noise": bool(scenarios.SCENARIOS[name].get("noise", 0)),
        "preliminary_peaks": preliminary.tolist(),
        "preliminary_peak_power": power.tolist(),
        "positive_lags": design.lags.tolist(),
        "positive_weights": design.weights.tolist(),
        "results": rows,
    }
    print(slug, {row["design"]: round(row["phase_mae_rad"],3) for row in rows},
          flush=True)
    return payload


def main() -> None:
    results = [evaluate_one(*case) for case in CASES]
    scenarios.NU = ORIGINAL
    (OUT / "wave_metrics.json").write_text(json.dumps(results, ensure_ascii=False, indent=2)+"\n")
    print(OUT / "wave_metrics.json")


if __name__ == "__main__":
    main()
