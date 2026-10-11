"""Oracle pilot for finite-budget global-secant delay design.

This diagnoses geometry only. The optimization sees the true field and
frequencies; the independent validation pairs are never used for design.
It neither estimates phases from a noisy record nor certifies the true
infimum over the continuous torus.
"""

from __future__ import annotations

from project_paths import RESULTS_DIR

import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

from run_controlled import AREA, case_data, field


OUT = RESULTS_DIR
L = 64
J = 9


def phase_pairs(rng: np.random.Generator, count: int) -> tuple[np.ndarray, np.ndarray]:
    """Mix global, axial/antipodal, near-diagonal, and tangent secants."""
    n_global = count // 4
    n_axes = count // 4
    n_near = count // 4
    n_tangent = count - n_global - n_axes - n_near
    first = rng.uniform(-np.pi, np.pi, (2, count))
    second = first.copy()
    second[:, :n_global] = rng.uniform(-np.pi, np.pi, (2, n_global))
    axes = np.array([[np.pi, 0.0], [0.0, np.pi], [np.pi, np.pi],
                     [np.pi/2, 0.0], [0.0, np.pi/2]], dtype=float).T
    indices = rng.integers(0, axes.shape[1], n_axes)
    second[:, n_global:n_global+n_axes] += axes[:, indices]
    directions = rng.normal(size=(2, n_near + n_tangent))
    directions /= np.linalg.norm(directions, axis=0)
    radii = np.r_[rng.uniform(.01, .25, n_near),
                  np.full(n_tangent, 1e-4)]
    start = n_global + n_axes
    second[:, start:] = first[:, start:] + directions * radii
    return first, second


def secant_matrix(
    first: np.ndarray, second: np.ndarray,
    frequency: np.ndarray, nonlinear: float, weak: float | None,
    horizon: int = L,
) -> np.ndarray:
    delta = np.angle(np.exp(1j * (first - second)))
    denominator = np.sum(delta * delta, axis=0)
    out = np.empty((first.shape[1], horizon))
    for lag in range(horizon):
        shift = frequency[:, None] * lag
        difference = field(first + shift, nonlinear, weak) - field(second + shift, nonlinear, weak)
        out[:, lag] = np.sum(AREA[:, None] * difference * difference, axis=0) / denominator
    return out


def max_min_weights(secants: np.ndarray, allowed: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """Maximize the minimum sampled squared stretch by a linear program."""
    a = secants[:, allowed]
    objective = np.r_[np.zeros(len(allowed)), -1.0]
    inequalities = np.column_stack((-a, np.ones(a.shape[0])))
    result = linprog(objective, A_ub=inequalities, b_ub=np.zeros(a.shape[0]),
                     A_eq=np.r_[np.ones(len(allowed)), 0.0][None, :],
                     b_eq=[1.0], bounds=[(0.0, None)] * len(allowed) + [(0.0, None)],
                     method="highs")
    if not result.success:
        raise RuntimeError(result.message)
    weights = np.zeros(secants.shape[1])
    weights[allowed] = result.x[:-1]
    return weights, float(result.x[-1]), -result.ineqlin.marginals


def greedy_support(secants: np.ndarray, budget: int) -> tuple[np.ndarray, float, list[int]]:
    """Column generation, stopped after `budget` distinct lag columns."""
    allowed = [int(np.argmax(np.min(secants, axis=0)))]
    for _ in range(budget - 1):
        weights, objective, dual = max_min_weights(secants, np.asarray(allowed))
        scores = np.einsum("i,ij->j", dual, secants, optimize=False)
        scores[allowed] = -np.inf
        candidate = int(np.argmax(scores))
        if scores[candidate] <= objective + 1e-10:
            break
        allowed.append(candidate)
    weights, objective, _ = max_min_weights(secants, np.asarray(allowed))
    return weights, objective, allowed


def score(secants: np.ndarray, weights: np.ndarray) -> dict:
    stretches = np.sqrt(np.einsum("ij,j->i", secants, weights, optimize=False))
    return {"sample_min": float(stretches.min()),
            "p01": float(np.quantile(stretches, .01)),
            "p05": float(np.quantile(stretches, .05)),
            "median": float(np.median(stretches)),
            "support": int(np.sum(weights > 1e-8)),
            "max_weight": float(weights.max())}


def main() -> None:
    training = phase_pairs(np.random.default_rng(7201), 3500)
    validation = phase_pairs(np.random.default_rng(9201), 20000)
    rows = []
    for case in ["two_modes", "infinite_weak", "infinite_strong", "weak_primitive"]:
        _, frequency, nonlinear, weak = case_data(case)
        train = secant_matrix(*training, frequency, nonlinear, weak)
        test = secant_matrix(*validation, frequency, nonlinear, weak)
        full, objective, _ = max_min_weights(train, np.arange(L))
        top = np.sort(np.argsort(full)[-J:])
        sparse, sparse_objective, _ = max_min_weights(train, top)
        greedy, greedy_objective, greedy_lags = greedy_support(train, J)
        consecutive = np.zeros(L); consecutive[:J] = 1 / J
        spread = np.zeros(L); spread[np.linspace(0, L-1, J, dtype=int)] = 1 / J
        uniform_full = np.ones(L) / L
        variants = {"secant_lp_top9_oracle": sparse,
                    "secant_greedy_9_oracle": greedy,
                    "consecutive_9": consecutive,
                    "spread_9": spread,
                    "uniform_full_64": uniform_full,
                    "unconstrained_lp_64_oracle": full}
        random_rng = np.random.default_rng(3001)
        random_uniform_mins = []
        random_reweighted_mins = []
        for _ in range(40):
            support = np.sort(random_rng.choice(L, J, replace=False))
            random_weights = np.zeros(L); random_weights[support] = 1/J
            random_uniform_mins.append(score(test, random_weights)["sample_min"])
            optimized_weights, _, _ = max_min_weights(train, support)
            random_reweighted_mins.append(score(test, optimized_weights)["sample_min"])
        row = {"case": case,
               "lp_train_lower_on_sample": objective,
               "sparse_lp_train_lower_on_sample": sparse_objective,
               "greedy_train_lower_on_sample": greedy_objective,
               "greedy_lags": greedy_lags,
               "selected_lags": top.tolist(),
               "random_40_median_min": float(np.median(random_uniform_mins)),
               "random_40_reweighted_median_min": float(np.median(random_reweighted_mins)),
               "random_40_reweighted_best_min": float(np.max(random_reweighted_mins)),
               "scores_independent_pairs": {name: score(test, w) for name, w in variants.items()}}
        rows.append(row)
        print(case, {name: round(value["sample_min"], 4)
                     for name, value in row["scores_independent_pairs"].items()}, flush=True)
    payload = {"description": "oracle sampled-secant geometry pilot; validation minima are upper bounds on true gamma",
               "lag_budget": J, "horizon": L, "training_pairs": training[0].shape[1],
               "independent_validation_pairs": validation[0].shape[1],
               "train_seed": 7201, "validation_seed": 9201, "rows": rows}
    (OUT / "secant_pilot.json").write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
