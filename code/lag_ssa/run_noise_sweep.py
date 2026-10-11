"""Paired noise sweep: positive-lag SSA versus classical MSSA baselines."""

from __future__ import annotations

from project_paths import RESULTS_DIR

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from method import LagDesign, design_positive_lags, dmd_phase_coordinates, estimate_peaks, trajectory_matrix
from run_controlled import (
    AREA, N, case_data, circular_mae, field, fit_field_fourier,
    phase_error_permutation,
)


OUT = RESULTS_DIR
OUT.mkdir(exist_ok=True)
CASES = ["two_modes", "infinite_weak", "infinite_strong", "weak_primitive"]
NOISE = [0.0, 0.03, 0.10, 0.20, 0.40]
SEEDS = list(range(6))
DESIGNS = ["positive_oracle", "positive_blind", "mssa_equal_budget",
           "mssa_equal_horizon", "mssa_blind_equal_horizon"]


def summarize(rows: list[dict]) -> list[dict]:
    groups = {}
    for row in rows:
        key = (row["case"], row["noise"], row["design"])
        groups.setdefault(key, []).append(row)
    result = []
    for (case, noise, design), values in groups.items():
        phase = np.array([v["phase_mae_rad"] for v in values])
        field_error = np.array([v["field_nrmse"] for v in values])
        result.append({
            "case": case, "noise": noise, "design": design,
            "phase_median": float(np.median(phase)),
            "phase_p10": float(np.quantile(phase, .1)),
            "phase_p90": float(np.quantile(phase, .9)),
            "phase_failure_rate_over_0_3rad": float(np.mean(phase > .3)),
            "field_nrmse_median": float(np.median(field_error)),
        })
    return result


def plot(summary: list[dict]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    colors = {"positive_oracle": "#0072B2", "positive_blind": "#56B4E9",
              "mssa_equal_budget": "#D55E00", "mssa_equal_horizon": "#009E73",
              "mssa_blind_equal_horizon": "#CC79A7"}
    for ax, case in zip(axes.flat, CASES):
        for design in DESIGNS:
            rows = sorted((r for r in summary if r["case"] == case and r["design"] == design),
                          key=lambda r:r["noise"])
            x = [r["noise"] for r in rows]
            y = [r["phase_median"] for r in rows]
            lo = [r["phase_p10"] for r in rows]
            hi = [r["phase_p90"] for r in rows]
            ax.plot(x, y, marker="o", label=design, color=colors[design])
            ax.fill_between(x, lo, hi, color=colors[design], alpha=.12)
        ax.set_title(case)
        ax.grid(alpha=.2)
        ax.set_ylabel("phase MAE (rad)")
        ax.set_xlabel("noise RMS / signal RMS")
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("Paired noise sweep, 6 seeds per point; shaded: 10–90%")
    fig.tight_layout(rect=(0,.08,1,.95))
    fig.savefig(OUT / "noise_phase_mae.png", dpi=160)
    plt.close(fig)


def main() -> None:
    rows = []
    for case in CASES:
        truth, frequency, nonlinear, weak = case_data(case)
        clean = field(truth, nonlinear, weak)
        scale = np.sqrt(np.mean(np.sum(AREA[:, None] * clean**2, axis=0)))
        positive = design_positive_lags(np.r_[frequency, -frequency], max_lag=256)
        budget = LagDesign(np.arange(len(positive.lags)),
                           np.ones(len(positive.lags))/len(positive.lags),
                           float("nan"), len(positive.lags))
        horizon = LagDesign(np.arange(max(positive.lags)+1),
                            np.ones(max(positive.lags)+1)/(max(positive.lags)+1),
                            float("nan"), max(positive.lags)+1)
        equal_area = np.ones(len(AREA))/len(AREA)
        for noise in NOISE:
            for seed in SEEDS:
                rng = np.random.default_rng(1000 + seed)
                observed = clean + noise * scale * rng.standard_normal(clean.shape)
                detected, _ = estimate_peaks(observed, AREA)
                blind = design_positive_lags(np.r_[detected, -detected], max_lag=256)
                blind_horizon_count = int(max(blind.lags)) + 1
                blind_horizon = LagDesign(np.arange(blind_horizon_count),
                                          np.ones(blind_horizon_count)/blind_horizon_count,
                                          float("nan"), blind_horizon_count)
                variants = [
                    ("positive_oracle", positive, AREA, frequency),
                    ("positive_blind", blind, AREA, detected),
                    ("mssa_equal_budget", budget, equal_area, frequency),
                    ("mssa_equal_horizon", horizon, equal_area, frequency),
                    ("mssa_blind_equal_horizon", blind_horizon, equal_area, detected),
                ]
                for label, design, area, target in variants:
                    h = trajectory_matrix(observed, area, design)
                    dmd = dmd_phase_coordinates(h, target, rank=8)
                    phase = dmd["phases"]
                    k = h.shape[1]
                    rows.append({
                        "case": case, "noise": noise, "seed": seed, "design": label,
                        "phase_mae_rad": phase_error_permutation(phase, truth[:, :k]),
                        "field_nrmse": fit_field_fourier(phase, observed[:, :k]),
                    })
        print(case, "complete", flush=True)
    summary = summarize(rows)
    payload = {"noise_levels": NOISE, "seeds": SEEDS, "rows": rows, "summary": summary}
    (OUT / "noise_sweep.json").write_text(json.dumps(payload, indent=2) + "\n")
    plot(summary)
    print(OUT / "noise_sweep.json")


if __name__ == "__main__":
    main()
