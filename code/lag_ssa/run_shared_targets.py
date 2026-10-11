"""Paired data-driven phase readout using identical frequency targets.

The direct spectrum, positive-lag SSA, and equal-horizon MSSA see the same
noisy field.  The phase dimension and low-peak selection rule remain supplied
model choices, so this is a controlled comparison, not a general estimator.
"""

from __future__ import annotations

from project_paths import RESULTS_DIR

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from method import LagDesign, design_positive_lags, dmd_phase_coordinates, trajectory_matrix
from run_controlled import AREA, case_data, field, phase_error_permutation


OUT = RESULTS_DIR
NOISE = [.03, .10, .20, .40]
CASES = ["two_modes", "infinite_weak", "infinite_strong", "weak_primitive"]
SEEDS = list(range(6))


def main() -> None:
    spectral = json.loads((OUT / "spectral_baseline.json").read_text())
    by_key = {(r["case"], r["noise"], r["seed"]): r for r in spectral["rows"]}
    rows = []
    for case in CASES:
        truth, _, nonlinear, weak = case_data(case)
        clean = field(truth, nonlinear, weak)
        scale = np.sqrt(np.mean(np.sum(AREA[:, None] * clean**2, axis=0)))
        for noise in NOISE:
            for seed in SEEDS:
                direct_row = by_key[(case, noise, seed)]
                if not direct_row["computed"]:
                    rows.append({"case": case, "noise": noise, "seed": seed,
                                 "computed": False, "reason": "spectral target absent"})
                    continue
                targets = np.asarray(direct_row["refined_frequencies"])
                observed = clean + noise * scale * np.random.default_rng(1000 + seed).standard_normal(clean.shape)
                try:
                    positive = design_positive_lags(np.r_[targets, -targets], max_lag=256)
                except RuntimeError as exc:
                    rows.append({"case": case, "noise": noise, "seed": seed,
                                 "computed": False, "reason": str(exc),
                                 "targets": targets.tolist()})
                    continue
                horizon = int(max(positive.lags)) + 1
                k = observed.shape[1] - horizon + 1
                classical_design = LagDesign(np.arange(horizon),
                                             np.ones(horizon)/horizon,
                                             float("nan"), horizon)
                direct_phase = targets[:, None] * np.arange(k)[None, :]
                errors = {"direct_spectrum": phase_error_permutation(direct_phase, truth[:, :k])}
                for label, design, area in [
                    ("positive_ssa", positive, AREA),
                    ("classical_mssa", classical_design, np.ones(len(AREA))/len(AREA)),
                ]:
                    h = trajectory_matrix(observed, area, design)
                    phase = dmd_phase_coordinates(h, targets, rank=8)["phases"]
                    errors[label] = phase_error_permutation(phase, truth[:, :k])
                rows.append({"case": case, "noise": noise, "seed": seed,
                             "computed": True, "targets": targets.tolist(),
                             "horizon": horizon, "scored_samples": k,
                             "phase_mae_rad": errors})
        print(case, "complete", flush=True)
    summary = []
    for case in CASES:
        for noise in NOISE:
            selected = [r for r in rows if r["case"] == case and r["noise"] == noise]
            valid = [r for r in selected if r["computed"]]
            summary.append({
                "case": case, "noise": noise, "computed_runs": len(valid),
                "trials": len(selected),
                "median_mae_rad": {method: float(np.median([r["phase_mae_rad"][method]
                                                          for r in valid])) if valid else None
                                   for method in ["direct_spectrum", "positive_ssa", "classical_mssa"]},
                "failure_rate_over_0_3rad": {method: float(sum(
                    (not r["computed"]) or r["phase_mae_rad"][method] > .3
                    for r in selected) / len(selected))
                    for method in ["direct_spectrum", "positive_ssa", "classical_mssa"]},
            })
    payload = {"description": "identical data-driven spectral targets, observations, and scored time indices",
               "known_dimension": 2, "rank": 8, "seeds": SEEDS, "rows": rows, "summary": summary}
    (OUT / "shared_targets.json").write_text(json.dumps(payload, indent=2) + "\n")
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    styles = [("direct_spectrum", "#0072B2", "direct spectrum"),
              ("positive_ssa", "#009E73", "positive SSA"),
              ("classical_mssa", "#D55E00", "MSSA, same horizon")]
    for ax, case in zip(axes.flat, CASES):
        selected = sorted((r for r in summary if r["case"] == case),
                          key=lambda r: r["noise"])
        for method, color, label in styles:
            ax.plot([r["noise"] for r in selected],
                    [r["median_mae_rad"][method] for r in selected],
                    marker="o", color=color, label=label)
        ax.set_title(case.replace("_", " "))
        ax.set_yscale("log")
        ax.set_ylim(1e-4, 2)
        ax.grid(alpha=.2)
        ax.set_xlabel("noise RMS / signal RMS")
        ax.set_ylabel("median phase MAE (rad)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Known r=2; identical data-driven targets and time indices")
    fig.tight_layout(rect=(0, .07, 1, .95))
    fig.savefig(OUT / "shared_targets.png", dpi=160)
    plt.close(fig)
    for r in summary:
        print(r, flush=True)


if __name__ == "__main__":
    main()
