"""One-relation compact-group Hankel projection, with masked data-only seeds.

Relation rows remain unsaturated. For a=q*w with primitive integer w, the
zero-gauge group has q components w.theta=2*pi*j/q. Each component is searched
with the original finite GN budget in null(a), using only weighted free-span
coefficients for initialization. No absolute-time or raw full-window seed is
accepted. Finite noisy support cannot certify an exact sampled orbit group.
"""

from collections import deque
from math import gcd

import numpy as np
from scipy.linalg import null_space, solve, svd
from scipy.linalg.blas import dgemm
from sympy import Matrix
from sympy.matrices.normalforms import hermite_normal_form

from hankel_torus_projection import HankelTorusProjector, ProjectionSettings


def _relation(value, rank):
    original = np.asarray(value)
    if original.ndim == 2:
        if original.shape[0] != 1:
            raise ValueError("unsupported_multiple_relations: use the continuous comparator")
        original = original[0]
    row = np.asarray(original, dtype=int)
    if row.shape != (rank,) or not np.array_equal(original, row) or not np.any(row):
        raise ValueError("one nonzero integer relation row per ambient phase dimension required")
    return row


def snap_rotation(alpha, relation):
    alpha = np.asarray(alpha, dtype=float)
    row = _relation(relation, len(alpha))
    winding = int(np.rint(np.dot(row, alpha)/(2*np.pi)))
    residue = float(np.dot(row, alpha)-2*np.pi*winding)
    snapped = alpha-row*residue/float(np.dot(row, row))
    order = 0
    for value in row:
        order = gcd(order, abs(int(value)))
    return {"frequencies": snapped, "integer_winding": winding,
        "relation": row.copy(), "primitive_direction": row//order, "branch_order": order,
        "rotation_change_l2": float(np.linalg.norm(snapped-alpha)),
        "relation_residue_before_snap": residue,
        "relation_residue_after_snap": float(np.dot(row, snapped)-2*np.pi*winding),
        "exact_group_identification_certified": False}


def _row_multiple(value, row):
    """Membership in Z*row, without dividing the row gcd."""
    index = int(np.flatnonzero(row)[0])
    if int(value[index]) % int(row[index]):
        return None
    multiplier = int(value[index])//int(row[index])
    return multiplier if np.array_equal(value, multiplier*row) else None


def _representative(value, row):
    nearest = int(np.rint(np.dot(value, row)/np.dot(row, row)))
    choices = []
    for multiplier in (nearest-1, nearest, nearest+1):
        candidate = value-multiplier*row
        if np.any(candidate) and next(int(x) for x in candidate if x) < 0:
            candidate = -candidate
        choices.append(candidate)
    return min(choices, key=lambda item: (int(np.sum(item*item)), tuple(int(x) for x in item)))


def refit_snapped_template(raw_train, times, alpha, labels, temporal_ridge, relation):
    """TRAIN-only real regression, merging exact quotient aliases of Z*a.

Duplicate-column multiplicities preserve the original ridge penalty on their
sum. Order-two characters retain their cosine; their sine is identically zero
on the group. The clock-refit gauge is group origin zero.
"""
    values = np.asarray(raw_train, dtype=float)
    times = np.asarray(times, dtype=float)
    labels = np.asarray(labels, dtype=int)
    snapped = snap_rotation(alpha, relation)
    row = snapped["relation"]
    if values.ndim != 2 or values.shape[1] != len(times) or not np.isfinite(values).all():
        raise ValueError("finite matching TRAIN sensor observations and sample indices required")
    if len(times) > 1 and not np.allclose(np.diff(times), 1., atol=1e-12, rtol=0):
        raise ValueError("use unit-spaced sample indices with radians-per-sample rotations")
    if temporal_ridge < 0:
        raise ValueError("nonnegative frozen temporal ridge required")
    representatives = []
    original_mapping = []
    constant_count = 0
    for label in labels:
        if _row_multiple(label, row) is not None:
            original_mapping.append({"original_label": label.tolist(), "quotient": "constant"})
            constant_count += 1
            continue
        candidate = _representative(label, row)
        key = tuple(int(x) for x in candidate)
        if key not in representatives:
            representatives.append(key)
        sign = 1 if _row_multiple(label-candidate, row) is not None else -1
        if _row_multiple(label-sign*candidate, row) is None:
            raise ArithmeticError("exact quotient alias mapping failed")
        original_mapping.append({"original_label": label.tolist(), "quotient_label": list(key), "sine_sign": sign})
    representatives.sort(key=lambda item: (sum(abs(x) for x in item), item))
    quotient = np.asarray(representatives, dtype=int).reshape(-1, len(alpha))
    count = len(quotient)
    multiplicity = np.asarray([sum(tuple(item.get("quotient_label", ())) == key for item in original_mapping)
                               for key in representatives], dtype=int)
    nyquist = np.asarray([_row_multiple(2*label, row) is not None for label in quotient], dtype=bool)
    frequency = np.einsum("ij,j->i", quotient, snapped["frequencies"], optimize=False)
    angles = times[:, None]*frequency[None, :]
    cosines, sines = np.cos(angles), np.sin(angles)
    for index in np.flatnonzero(nyquist):
        multiple = _row_multiple(2*quotient[index], row)
        parity = multiple*snapped["integer_winding"]
        cosines[:, index] = np.cos(np.pi*parity*times)
        sines[:, index] = 0.
    full_design = np.column_stack((np.ones(len(times)), cosines, sines))
    active = np.r_[0, np.arange(1, 1+count), 1+count+np.flatnonzero(~nyquist)]
    design = full_design[:, active]
    penalty = np.r_[0., 1/np.maximum(multiplicity, 1), 1/np.maximum(multiplicity[~nyquist], 1)]
    gram = np.einsum("tn,tm->nm", design, design, optimize=False)/len(times)
    right = np.einsum("tn,qt->nq", design, values, optimize=False)/len(times)
    fitted = solve(gram+temporal_ridge*np.diag(penalty), right, assume_a="pos")
    coefficients = np.zeros((1+2*count, values.shape[0]))
    coefficients[active] = fitted
    return {"frequencies": snapped["frequencies"], "labels": quotient,
        "coefficients_sensor": coefficients,
        "metadata": {"relation": row.tolist(), "primitive_direction": snapped["primitive_direction"].tolist(),
            "branch_order": snapped["branch_order"], "integer_winding": snapped["integer_winding"],
            "rotation_change_l2": snapped["rotation_change_l2"], "group_phase_origin": 0.,
            "group_origin_absorbed_by_train_clock_regression": True,
            "original_dictionary_lines": len(labels), "quotient_dictionary_lines": count,
            "constant_alias_lines_absorbed_into_unpenalized_mean": constant_count,
            "nyquist_cosine_only_labels": quotient[nyquist].tolist(),
            "quotient_multiplicities": multiplicity.tolist(), "original_to_quotient": original_mapping,
            "relation_lattice_saturated": False, "alias_tests": "k±j belongs to Z*a; exact integer row membership",
            "ridge_scale_frozen": float(temporal_ridge),
            "duplicate_ridge_penalty_preserved_by_inverse_multiplicity": True,
            "full_nonorthogonal_temporal_gram_preserved": True,
            "true_frequency_or_group_information_supplied": False}}


def _integer_decks(square, cap):
    matrix = Matrix(np.asarray(square, dtype=int).tolist())
    inverse = matrix.inv()
    rank = matrix.rows
    order = abs(int(matrix.det()))
    generators = [tuple(inverse[i, j] % 1 for i in range(rank)) for j in range(rank)]
    origin = tuple([0]*rank);queue = deque([origin]);seen = {origin};representatives = []
    while queue and len(representatives) < cap:
        current = queue.popleft();representatives.append(current)
        for generator in generators:
            following = tuple((x+y) % 1 for x, y in zip(current, generator))
            if following not in seen:
                seen.add(following);queue.append(following)
    offsets = 2*np.pi*np.asarray([[float(x) for x in row] for row in representatives])
    return offsets, {"seed_integer_matrix": np.asarray(square, dtype=int).tolist(),
        "seed_deck_order": order, "seed_deck_starts": len(offsets), "seed_deck_enumeration_complete": len(offsets) == order}


def _character_index(labels, relation):
    matrix = np.vstack((labels, np.asarray(relation)[None, :]))
    basis = hermite_normal_form(Matrix(matrix.T.tolist()))
    if basis.cols != matrix.shape[1]:
        return None
    return abs(int(basis.det()))


class WeightedHankelProjector(HankelTorusProjector):
    def __init__(self, labels, coefficients_mass, frequencies, weights=None, relation=None,
                 settings=ProjectionSettings(), max_branch_order=100):
        super().__init__(labels, coefficients_mass, frequencies, settings)
        weight = np.ones(settings.window) if weights is None else np.asarray(weights, dtype=float)
        if weight.shape != (settings.window,) or not np.isfinite(weight).all() or np.any(weight < 0) or weight.sum() <= 0:
            raise ValueError("matching nonnegative lag weights with nonempty observed support required")
        self.weights = weight/weight.sum()
        self.weighted_bank = np.asarray((self.bank*np.sqrt(self.weights)[None, None, :]).reshape(2*self.count, -1), order="F")
        self.gram = dgemm(1., self.weighted_bank, self.weighted_bank, trans_b=True) if self.count else np.empty((0, 0))
        self.relation = None if relation is None else _relation(relation, self.rank)
        self.primitive_direction = None
        self.branch_order = 1
        self.tangent = np.eye(self.rank)
        energy = np.sum(self.coefficients[1:1+self.count]**2+self.coefficients[1+self.count:]**2, axis=1)
        selected, indices = [], []
        if self.relation is not None:
            if abs(np.angle(np.exp(1j*np.dot(self.relation, self.frequencies)))) > 1e-8:
                raise ValueError("rotation must be snapped before imposing its compact-group relation")
            order = 0
            for value in self.relation:
                order = gcd(order, abs(int(value)))
            if order > max_branch_order:
                raise ValueError("unsupported_branch_order_above_declared_cap: use continuous comparator")
            self.branch_order = order
            self.primitive_direction = self.relation//order
            self.tangent = null_space(self.relation.astype(float)[None, :])
            selected = [self.primitive_direction.tolist()]
            target = self.rank-1
            for index in np.argsort(-energy, kind="stable"):
                trial = selected+[self.labels[index].tolist()]
                if Matrix(trial).rank() > len(selected):
                    selected = trial;indices.append(int(index))
                if len(indices) == target:
                    break
            if len(indices) != target:
                raise ValueError("unsupported_unobserved_group_tangent: use continuous comparator")
            square = np.asarray([self.labels[index] for index in indices]+[self.primitive_direction], dtype=int)
            artificial = 0
        else:
            for index in np.argsort(-energy, kind="stable"):
                trial = selected+[self.labels[index].tolist()]
                if Matrix(trial).rank() > len(selected):
                    selected = trial;indices.append(int(index))
                if len(indices) == self.rank:
                    break
            artificial = 0
            for row in np.eye(self.rank, dtype=int):
                if len(selected) == self.rank:
                    break
                trial = selected+[row.tolist()]
                if Matrix(trial).rank() > len(selected):
                    selected = trial;artificial += 1
            square = np.asarray(selected, dtype=int)
        self.seed_indices = np.asarray(indices, dtype=int)
        self.seed_matrix = square
        self.offsets, seed_information = _integer_decks(square, settings.max_multistarts)
        character_index = None if self.relation is None else _character_index(self.labels, self.relation)
        self.multistart_metadata = {**seed_information,
            "seed_source": "free coefficients from the weighted observed lag support only",
            "raw_hann_or_absolute_time_seed_used": False,
            "artificial_flat_direction_seed_rows": artificial,
            "branch_order": self.branch_order, "all_compact_components_enumerated": True,
            "maximum_component_order": max_branch_order,
            "total_attempts_per_window": self.branch_order*len(self.offsets),
            "group_origin": 0., "group_character_lattice_index": character_index,
            "dictionary_characters_separate_presented_group": character_index == 1 if self.relation is not None else None,
            "physical_group_identified": False, "global_optimum_certified": False}

    def _statistics(self, mass_values, starts):
        if self.count:
            return self.sufficient_statistics(mass_values, starts)
        values = np.asarray(mass_values)-self.coefficients[0, :, None]
        windows = np.lib.stride_tricks.sliding_window_view(values, self.settings.window, axis=1)[:, :starts, :]
        return np.empty((0, starts)), np.einsum("qnl,l,qnl->n", windows, self.weights, windows, optimize=False)

    def _cost(self, phase, right, constant):
        if not self.count:
            return float(constant), np.zeros(self.rank), np.empty((0, self.rank))
        return self.cost_gradient(phase, right, constant)

    def _free_coefficients(self, mass_values, starts):
        if not self.count:
            return np.empty((0, starts))
        return self.free_linear_projection(mass_values, starts)["free_coefficients"]

    def _optimize_tangent(self, initial, right, constant):
        phase = np.asarray(initial, dtype=float).copy()
        basis = self.tangent
        damping = self.settings.initial_damping
        initial_cost = self._cost(phase, right, constant)[0]
        if not basis.shape[1]:
            return phase, {"initial_cost": initial_cost, "final_cost": initial_cost,
                "iterations": 0, "accepted_steps": 0, "converged_gradient": True, "final_gradient_max": 0.}
        accepted = 0;converged = False
        for iteration in range(self.settings.max_iterations):
            cost, full_gradient, full_jacobian = self._cost(phase, right, constant)
            gradient = np.einsum("ji,j->i", basis, full_gradient, optimize=False)
            jacobian = np.einsum("di,ij->dj", full_jacobian, basis, optimize=False)
            if np.max(abs(gradient)) <= self.settings.gradient_tolerance:
                converged = True;break
            gram_jacobian = dgemm(1., self.gram, np.asarray(jacobian, order="F"))
            hessian = 2*np.einsum("di,dj->ij", jacobian, gram_jacobian, optimize=False)
            regularizer = np.maximum(np.diag(hessian), 1e-10)
            try:
                step = np.linalg.solve(hessian+damping*np.diag(regularizer), -gradient)
            except np.linalg.LinAlgError:
                damping *= 10;continue
            full_step = np.einsum("ij,j->i", basis, step, optimize=False)
            full_step *= min(1., self.settings.max_phase_step/max(np.max(abs(full_step)), 1e-15))
            changed = False
            for index in range(self.settings.max_line_search):
                trial = phase+2.**(-index)*full_step
                if self._cost(trial, right, constant)[0] < cost:
                    phase = trial;accepted += 1;damping = max(1e-10, damping*.5);changed = True;break
            if not changed:
                damping *= 10
                if damping > 1e10:
                    break
        cost, full_gradient, _ = self._cost(phase, right, constant)
        gradient = np.einsum("ji,j->i", basis, full_gradient, optimize=False)
        return phase, {"initial_cost": initial_cost, "final_cost": cost,
            "iterations": iteration+1, "accepted_steps": accepted, "converged_gradient": converged,
            "final_gradient_max": float(np.max(abs(gradient)))}

    def project(self, mass_values):
        values = np.asarray(mass_values, dtype=float)
        if values.ndim != 2 or not np.isfinite(values).all() or values.shape[1] < self.settings.window:
            raise ValueError("finite matching observed field with at least one full weighted window required")
        starts = values.shape[1]-self.settings.window+1
        right, constant = self._statistics(values, starts)
        free = self._free_coefficients(values, starts)
        angles = np.arctan2(free[self.count+self.seed_indices], free[self.seed_indices])
        phases, diagnostics = [], []
        for column in range(starts):
            attempts = []
            for branch in range(self.branch_order):
                rhs = (np.r_[angles[:, column], 2*np.pi*branch/self.branch_order]
                       if self.relation is not None else
                       np.r_[angles[:, column], np.zeros(self.rank-len(self.seed_indices))])
                seed = np.linalg.solve(self.seed_matrix.astype(float), rhs)
                for deck, offset in enumerate(self.offsets):
                    phase, information = self._optimize_tangent(seed+offset, right[:, column], constant[column])
                    attempts.append((phase, information, branch, deck))
            selected = min(range(len(attempts)), key=lambda index: attempts[index][1]["final_cost"])
            phase, information, branch, deck = attempts[selected]
            wrapped = np.angle(np.exp(1j*phase));phases.append(wrapped)
            residual = None if self.relation is None else float(abs(np.angle(np.exp(1j*np.dot(self.relation, wrapped)))))
            diagnostics.append({**information, "selected_start": selected, "selected_branch": branch,
                "selected_seed_deck": deck, "group_constraint_residual_rad": residual})
        phase = np.asarray(phases).T
        field = self.predict_windows(phase, np.asarray([0]))[:, :, 0]
        return {"phase": phase, "mass_field": field, "diagnostics": diagnostics,
            "multistart": self.multistart_metadata}

    def predict_windows(self, phase, lags=None):
        phase = np.asarray(phase, dtype=float)
        lags = np.arange(self.settings.window) if lags is None else np.asarray(lags, dtype=int)
        angles = np.einsum("ij,jt->it", self.labels, phase, optimize=False)
        features = np.r_[np.cos(angles), np.sin(angles)]
        return self.coefficients[0, :, None, None]+np.einsum("dql,dn->qnl", self.bank[:, :, lags], features, optimize=False)

    def compression_check(self, mass_values, phase):
        right, constant = self._statistics(mass_values, 1)
        cost, gradient, _ = self._cost(phase, right[:, 0], constant[0])
        prediction = self.predict_windows(np.asarray(phase)[:, None])[:, 0, :]
        difference = np.asarray(mass_values)[:, :self.settings.window]-prediction
        explicit = float(np.einsum("ql,l,ql->", difference, self.weights, difference, optimize=False))
        numeric = [];epsilon = 1e-6
        for index in range(self.rank):
            delta = np.zeros(self.rank);delta[index] = epsilon
            numeric.append((self._cost(phase+delta, right[:, 0], constant[0])[0]-
                            self._cost(phase-delta, right[:, 0], constant[0])[0])/(2*epsilon))
        return {"compressed_cost": cost, "explicit_window_cost": explicit,
            "cost_absolute_difference": abs(cost-explicit),
            "analytic_gradient_max_difference_from_finite_difference": float(np.max(abs(gradient-numeric)))}
