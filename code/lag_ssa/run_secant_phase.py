"""Held-out phase readout for data-trained secant lag designs.

Designs and frequency targets come from the independent first-half record.
All readouts see the same second-half observations, target frequencies,
time indices, and DMD rank. Direct frequency propagation is a baseline.
"""

from __future__ import annotations

from project_paths import RESULTS_DIR

import json
from pathlib import Path

import numpy as np

from method import LagDesign, dmd_phase_coordinates, trajectory_matrix
from run_controlled import AREA, case_data, field, phase_error_permutation
from run_secant_data_driven import J, L, NOISE, SEEDS, TRAIN_LENGTH


OUT = RESULTS_DIR
RANK = 8


def lag_design(lags: list[int], weights: list[float]) -> LagDesign:
    x = np.asarray(lags, dtype=int)
    w = np.asarray(weights, dtype=float)
    keep = w > 1e-8
    return LagDesign(x[keep], w[keep]/w[keep].sum(), float("nan"), L)


def main() -> None:
    trained = json.loads((OUT / "secant_data_driven.json").read_text())
    rows = []
    for row in trained["rows"]:
        case, seed = row["case"], row["seed"]
        if not row["computed"]:
            rows.append({"case": case, "seed": seed, "computed": False,
                         "reason": "training design failed"})
            continue
        phases, _, nonlinear, weak = case_data(case)
        true_phase = phases[:, TRAIN_LENGTH:]
        clean = field(true_phase, nonlinear, weak)
        scale = np.sqrt(np.mean(np.sum(AREA[:, None]*clean*clean, axis=0)))
        observed = clean + NOISE*scale*np.random.default_rng(8100+seed).standard_normal(clean.shape)
        targets = np.asarray(row["fit"]["frequencies"])
        k = observed.shape[1]-L+1
        truth = true_phase[:, :k]
        selected = [(lag, weight) for lag, weight in
                    zip(row["selected_lags"], row["selected_weights"])
                    if weight > 1e-8]
        support = [lag for lag, _ in selected]
        equal_sensors = np.ones(len(AREA))/len(AREA)
        designs = {
            "data_secant_9": (lag_design(row["selected_lags"], row["selected_weights"]), AREA),
            "same_support_uniform": (lag_design(support, [1/len(support)]*len(support)), AREA),
            "oracle_secant_9": (lag_design(row["oracle_lags_for_evaluation_only"],
                                            row["oracle_weights_for_evaluation_only"]), AREA),
            "consecutive_9": (lag_design(list(range(J)), [1/J]*J), AREA),
            "spread_9": (lag_design(np.linspace(0, L-1, J, dtype=int).tolist(), [1/J]*J), AREA),
            "uniform_full_64": (lag_design(list(range(L)), [1/L]*L), AREA),
            "classical_mssa_9": (lag_design(list(range(J)), [1/J]*J), equal_sensors),
            "classical_mssa_64": (lag_design(list(range(L)), [1/L]*L), equal_sensors),
        }
        errors = {"direct_frequency_train": phase_error_permutation(
            targets[:, None]*np.arange(k)[None, :], truth)}
        failures = {}
        for label, (design, spatial_weights) in designs.items():
            try:
                h = trajectory_matrix(observed, spatial_weights, design)[:, :k]
                recovered = dmd_phase_coordinates(h, targets, rank=RANK)["phases"]
                errors[label] = phase_error_permutation(recovered, truth)
            except (FloatingPointError, np.linalg.LinAlgError, StopIteration, ValueError) as exc:
                errors[label] = None
                failures[label] = str(exc)
        rows.append({"case": case, "seed": seed, "computed": True,
                     "phase_mae_rad": errors, "failures": failures,
                     "same_test_indices": k, "rank": RANK})
    summary = []
    labels = ["direct_frequency_train", "data_secant_9", "same_support_uniform",
              "oracle_secant_9", "consecutive_9", "spread_9", "uniform_full_64",
              "classical_mssa_9", "classical_mssa_64"]
    for case in ["two_modes", "infinite_weak", "infinite_strong", "weak_primitive"]:
        selected = [r for r in rows if r["case"] == case]
        median = {label: float(np.median([r["phase_mae_rad"][label]
                 for r in selected if r["computed"] and r["phase_mae_rad"].get(label) is not None]))
                 for label in labels}
        failures = {label: sum((not r["computed"]) or r["phase_mae_rad"].get(label) is None
                               or r["phase_mae_rad"][label] > .3 for r in selected)
                    for label in labels}
        summary.append({"case": case, "trials": len(selected),
                        "median_phase_mae_rad": median,
                        "failures_over_0_3rad": failures})
        print(case, {label: round(value, 4) for label, value in median.items()}, flush=True)
    payload = {"description": "held-out noisy test record; common train frequencies, test samples, and DMD rank",
               "training_samples": TRAIN_LENGTH, "test_samples_scored": 1200-L+1,
               "noise_rms_fraction": NOISE, "rank": RANK, "lag_budget": J,
               "horizon": L, "seeds": SEEDS, "rows": rows, "summary": summary}
    (OUT / "secant_phase.json").write_text(json.dumps(payload, indent=2)+"\n")


if __name__ == "__main__":
    main()
