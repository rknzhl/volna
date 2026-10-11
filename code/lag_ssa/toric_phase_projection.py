"""Weighted projection of observed characters onto an inferred toric image.

The integer label matrix A and its left inverse B are learned elsewhere.
Training clocks determine constant origins and phase-dispersion weights only.
Every returned phase is fitted independently to the observed characters; the
projection method has no time or clock argument.  Multistart is finite and
does not certify global optimality or statistical injectivity under noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product

import numpy as np


def wrap(phase):
    return np.angle(np.exp(1j * np.asarray(phase)))


def _unit_characters(values, dimension):
    z = np.asarray(values, dtype=complex)
    if (z.ndim != 2 or z.shape[0] != dimension or not z.shape[1]
            or not np.isfinite(z).all() or np.any(abs(z) < 1e-12)):
        raise ValueError("matching finite nonzero observed character rows required")
    return z / abs(z)


def _cover_offsets(labels: np.ndarray, max_lifts: int = 128):
    rank = labels.shape[1]
    subsets = []
    for active in combinations(range(len(labels)), rank):
        square = labels[list(active)]
        determinant = int(round(np.linalg.det(square)))
        if determinant:
            subsets.append((abs(determinant), float(np.linalg.cond(square)), active, square))
    if not subsets:
        raise ValueError("learned character labels have deficient column rank")
    determinant, _, active, square = min(subsets, key=lambda row: row[:3])
    inverse = np.linalg.inv(square)
    offsets, keys = [], set()
    exhausted = False
    for winding in product(range(determinant), repeat=rank):
        offset = wrap(2*np.pi*np.einsum("ij,j->i", inverse, winding, optimize=False))
        canonical = np.mod(offset, 2*np.pi)
        canonical[np.isclose(canonical, 2*np.pi, atol=1e-9, rtol=0)] = 0.
        key = tuple(np.round(canonical, 10))
        if key in keys:
            continue
        keys.add(key)
        offsets.append(offset)
        if len(offsets) == determinant:
            break
        if len(offsets) >= max_lifts:
            exhausted = True
            break
    return active, inverse, np.asarray(offsets), {
        "square_character_rows": list(active), "absolute_minor_determinant": determinant,
        "distinct_cover_lifts_searched": len(offsets),
        "cover_lift_budget_exhausted": exhausted,
    }


def _objective(theta: np.ndarray, observed_phase: np.ndarray,
               labels: np.ndarray, weights: np.ndarray):
    residual = np.einsum("pr,rn->pn", labels, theta, optimize=False)-observed_phase
    return 2*np.einsum("p,pn->n", weights, 1-np.cos(residual), optimize=False)


def _refine_starts(starts: np.ndarray, observed_phase: np.ndarray,
                   labels: np.ndarray, weights: np.ndarray, max_iterations=100):
    """Batched damped Gauss--Newton with individual decreasing steps."""
    count, rank, columns = starts.shape
    theta = np.transpose(starts, (1, 0, 2)).reshape(rank, count*columns).copy()
    phase = np.tile(observed_phase, (1, count))
    gram = np.einsum("pi,p,pj->ij", labels, weights, labels, optimize=False)
    inverse = np.linalg.inv(gram)
    cost = _objective(theta, phase, labels, weights)
    iterations = 0
    for iteration in range(max_iterations):
        residual = np.einsum("pr,rn->pn", labels, theta, optimize=False)-phase
        gradient = np.einsum("pr,p,pn->rn", labels, weights, np.sin(residual), optimize=False)
        delta = -np.einsum("ij,jn->in", inverse, gradient, optimize=False)
        norm = np.linalg.norm(delta, axis=0)
        delta *= np.minimum(1., np.pi/np.maximum(norm, 1e-30))[None, :]
        active = norm > 1e-10
        iterations = iteration+1
        if not np.any(active):
            break
        scale = np.ones(theta.shape[1])
        proposed = theta+delta
        proposed_cost = _objective(proposed, phase, labels, weights)
        for _ in range(14):
            reject = proposed_cost > cost+1e-14
            if not np.any(reject):
                break
            scale[reject] *= .5
            proposed[:, reject] = theta[:, reject]+delta[:, reject]*scale[None, reject]
            proposed_cost[reject] = _objective(proposed[:, reject], phase[:, reject], labels, weights)
        improve = proposed_cost <= cost+1e-14
        theta[:, improve] = proposed[:, improve]
        cost[improve] = proposed_cost[improve]
    return theta.reshape(rank, count, columns).transpose(1, 0, 2), cost.reshape(count, columns), iterations


@dataclass(frozen=True)
class ToricPhaseProjector:
    labels: np.ndarray
    products: np.ndarray
    basis_frequencies: np.ndarray
    character_origins: np.ndarray
    weights: np.ndarray
    training_phase_variances: np.ndarray
    training_samples: int
    weight_ratio_cap: float = 100.

    def aligned_phases(self, characters):
        z = _unit_characters(characters, len(self.labels))
        return wrap(np.angle(z)-self.character_origins[:, None])

    def product_phases(self, characters):
        return wrap(np.einsum("rp,pn->rn", self.products, self.aligned_phases(characters), optimize=False))

    def project(self, characters, max_iterations=100):
        phase = self.aligned_phases(characters)
        initial = wrap(np.einsum("rp,pn->rn", self.products, phase, optimize=False))
        active, inverse, offsets, lift_audit = _cover_offsets(self.labels)
        square_initial = np.einsum("rp,pn->rn", inverse, phase[list(active)], optimize=False)
        starts = [initial]
        starts.extend(square_initial+offset[:, None] for offset in offsets)
        # Fixed offsets supplement algebraic cover lifts without clocks.
        for offset in product((0., 2*np.pi/3, -2*np.pi/3), repeat=self.labels.shape[1]):
            if any(offset):
                starts.append(initial+np.asarray(offset)[:, None])
        starts = np.asarray(starts)
        solved, costs, iterations = _refine_starts(starts, phase, self.labels, self.weights, max_iterations)
        best = np.argmin(costs, axis=0)
        columns = np.arange(phase.shape[1])
        projected = wrap(solved[best, :, columns].T)
        product_cost = _objective(initial, phase, self.labels, self.weights)
        projection_cost = costs[best, columns]
        branch_distance = np.sqrt(np.sum(abs(np.exp(1j*solved)-np.exp(1j*projected[None, :, :]))**2, axis=1))
        distinct = branch_distance > .25
        alternatives = np.where(distinct, costs, np.inf)
        second = np.min(alternatives, axis=0)
        finite = np.isfinite(second)
        gaps = second[finite]-projection_cost[finite]
        selected_residual = np.einsum("pr,rn->pn", self.labels, projected, optimize=False)-phase
        selected_gradient = np.einsum("pr,p,pn->rn", self.labels, self.weights,
                                      np.sin(selected_residual), optimize=False)
        selected_gradient_norm = np.linalg.norm(selected_gradient, axis=0)
        return {"phase": projected, "product_phase": initial,
                "product_weighted_character_cost": product_cost,
                "projection_weighted_character_cost": projection_cost,
                "diagnostics": {
                    "starts_per_point": len(starts), "optimizer_iterations": iterations,
                    "projection_cost_exceeds_product_cost_count": int(np.sum(projection_cost > product_cost+1e-10)),
                    "best_start_is_product_start_count": int(np.sum(best == 0)),
                    "refined_product_start_worse_than_best_count": int(np.sum(costs[0] > projection_cost+1e-8)),
                    "points_with_distinct_alternative_multistart_solution": int(np.sum(finite)),
                    "distinct_branch_cost_gap_min": float(gaps.min()) if len(gaps) else None,
                    "distinct_branch_cost_gap_median": float(np.median(gaps)) if len(gaps) else None,
                    "near_tied_distinct_branch_count": int(np.sum(gaps < 1e-6)),
                    "selected_solution_gradient_norm_max": float(selected_gradient_norm.max()),
                    "selected_solution_gradient_norm_median": float(np.median(selected_gradient_norm)),
                    "finite_multistart_global_optimum_certified": False,
                    "future_clock_or_time_used": False, **lift_audit}}

    def metadata(self):
        return {"labels": self.labels.tolist(), "products": self.products.tolist(),
                "integer_left_inverse_BA_is_identity": bool(np.array_equal(self.products @ self.labels,
                    np.eye(self.labels.shape[1], dtype=int))),
                "basis_frequencies_training_estimated_only": self.basis_frequencies.tolist(),
                "character_origins_training_only": self.character_origins.tolist(),
                "weights_training_only": self.weights.tolist(),
                "phase_variances_training_only": self.training_phase_variances.tolist(),
                "weight_ratio_cap": self.weight_ratio_cap, "training_samples": self.training_samples,
                "output_phase_source": "per-point observed-character toric projection; no output clocks",
                "global_closest_point_or_noise_injectivity_certified": False,
                "novelty_claimed": False}


def fit_toric_phase_projector(train_characters, labels, products,
                             basis_frequencies, weight_ratio_cap=100.):
    a, b = np.asarray(labels, dtype=int), np.asarray(products, dtype=int)
    frequencies = np.asarray(basis_frequencies, dtype=float)
    if (a.ndim != 2 or b.shape != (a.shape[1], a.shape[0])
            or frequencies.shape != (a.shape[1],) or not np.isfinite(frequencies).all()
            or not np.array_equal(b @ a, np.eye(a.shape[1], dtype=int))
            or weight_ratio_cap < 1):
        raise ValueError("finite inferred basis frequencies and integer BA=I required")
    z = _unit_characters(train_characters, len(a))
    times = np.arange(z.shape[1])
    learned_peak_frequency = np.einsum("pr,r->p", a, frequencies, optimize=False)
    residual = z*np.exp(-1j*learned_peak_frequency[:, None]*times[None, :])
    origins = np.angle(np.mean(residual, axis=1))
    error = wrap(np.angle(residual)-origins[:, None])
    variances = np.mean(error**2, axis=1)
    inverse = 1/np.maximum(variances, 1e-8)
    inverse = np.minimum(inverse, weight_ratio_cap*inverse.min())
    weights = inverse/inverse.sum()
    return ToricPhaseProjector(a.copy(), b.copy(), frequencies.copy(), origins, weights,
                               variances, z.shape[1], float(weight_ratio_cap))
