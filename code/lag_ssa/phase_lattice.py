"""Conditional character-lattice inference from resolved temporal frequencies.

Relations are searched within a fixed integer budget. A small residual is
not a certificate of an exact relation without a separation assumption.
The primitive basis need not coincide with any observed Fourier peak.
"""

from __future__ import annotations

from fractions import Fraction
from itertools import product
from math import gcd, lcm

import numpy as np


def extended_gcd(a: int, b: int) -> tuple[int, int, int]:
    old_r, r = a, b
    old_s, s = 1, 0
    old_t, t = 0, 1
    while r:
        q = old_r // r
        old_r, r = r, old_r-q*r
        old_s, s = s, old_s-q*s
        old_t, t = t, old_t-q*t
    if old_r < 0:
        return -old_r, -old_s, -old_t
    return old_r, old_s, old_t


def exact_integer_inverse(matrix: np.ndarray) -> np.ndarray:
    n = matrix.shape[0]
    augmented = [[Fraction(int(matrix[i, j])) for j in range(n)]
                 + [Fraction(int(i == j)) for j in range(n)] for i in range(n)]
    for col in range(n):
        pivot = next(i for i in range(col, n) if augmented[i][col])
        augmented[col], augmented[pivot] = augmented[pivot], augmented[col]
        scale = augmented[col][col]
        augmented[col] = [value/scale for value in augmented[col]]
        for row in range(n):
            if row != col:
                scale = augmented[row][col]
                augmented[row] = [a-scale*b for a, b in zip(augmented[row], augmented[col])]
    values = [row[n:] for row in augmented]
    if any(value.denominator != 1 for row in values for value in row):
        raise ValueError("matrix is not unimodular")
    inverse = np.asarray([[int(value) for value in row] for row in values], dtype=np.int64)
    if not np.array_equal(matrix @ inverse, np.eye(n, dtype=np.int64)):
        raise ArithmeticError("integer inverse check failed")
    return inverse


def relation_coordinates(relation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return U,U^-1 with first row(U)=primitive relation."""
    v = np.asarray(relation, dtype=np.int64).copy()
    reduction = np.eye(len(v), dtype=np.int64)
    for i in range(1, len(v)):
        a, b = int(v[0]), int(v[i])
        if b == 0:
            continue
        g, s, t = extended_gcd(a, b)
        first = reduction[0].copy()
        other = reduction[i].copy()
        reduction[0] = s*first+t*other
        reduction[i] = (-b//g)*first+(a//g)*other
        v[0], v[i] = g, 0
    if abs(int(v[0])) != 1:
        raise ValueError("relation must be primitive")
    if v[0] == -1:
        reduction[0] *= -1
    inverse = reduction.T
    transform = exact_integer_inverse(reduction).T
    if not np.array_equal(transform[0], relation):
        raise ArithmeticError("unimodular relation completion failed")
    return transform, inverse


def short_relation(frequencies: np.ndarray, order: int, tolerance: float) -> dict | None:
    """Find a primitive bounded relation involving the newest generator."""
    n = len(frequencies)
    candidates = []
    for first in product(range(-order, order+1), repeat=n-1):
        for last in range(1, order+1):
            relation = np.asarray((*first, last), dtype=np.int64)
            divisor = 0
            for value in relation:
                divisor = gcd(divisor, abs(int(value)))
            if divisor != 1:
                continue
            raw = float(np.dot(relation, frequencies))
            residual = abs(float(np.angle(np.exp(1j*raw))))
            if residual <= tolerance:
                score = (int(np.sum(abs(relation))), int(np.max(abs(relation))), residual)
                candidates.append((score, relation, residual, int(np.rint(raw/(2*np.pi)))))
    if not candidates:
        return None
    candidates.sort(key=lambda row: row[0])
    _, relation, residual, winding = candidates[0]
    return {"relation": relation, "residual": residual,
            "winding": winding, "candidate_count": len(candidates)}


def infer_character_basis(
    peaks: np.ndarray, tolerance: float, max_order: int = 6, max_rank: int = 3,
) -> dict:
    """Infer an abstract primitive basis of the supplied visible module.

    Incremental unimodular changes refine the lattice when a new peak is a
    rational combination of previous ones (e.g. 2w and 3w). Completeness and
    exact rank remain conditional on the order budget and relation gap.
    """
    frequencies = np.asarray(peaks, dtype=float).reshape(-1)
    if not len(frequencies):
        raise ValueError("at least one resolved peak is required")
    basis = frequencies[:1].copy()
    labels = np.ones((1, 1), dtype=np.int64)
    products_matrix = np.ones((1, 1), dtype=np.int64)
    relations = np.zeros((0, 1), dtype=np.int64)
    history = []
    for index in range(1, len(frequencies)):
        augmented_basis = np.r_[basis, frequencies[index]]
        r = len(basis)
        augmented_labels = np.zeros((index+1, r+1), dtype=np.int64)
        augmented_labels[:-1, :-1] = labels
        augmented_labels[-1, -1] = 1
        augmented_products = np.zeros((r+1, index+1), dtype=np.int64)
        augmented_products[:-1, :-1] = products_matrix
        augmented_products[-1, -1] = 1
        relations = np.column_stack((relations, np.zeros(len(relations), dtype=np.int64)))
        found = short_relation(augmented_basis, max_order, tolerance)
        if found is None:
            if r >= max_rank:
                return {"rank": None, "reason": "rank exceeds declared search budget",
                        "history": history, "max_rank": max_rank}
            basis = augmented_basis
            labels = augmented_labels
            products_matrix = augmented_products
            history.append({"peak_index": index, "action": "new_generator"})
        else:
            transform, inverse = relation_coordinates(found["relation"])
            basis = np.angle(np.exp(1j*(transform[1:] @ augmented_basis)))
            labels = augmented_labels @ inverse[:, 1:]
            products_matrix = transform[1:] @ augmented_products
            relations = np.vstack((relations, found["relation"] @ augmented_products))
            history.append({"peak_index": index, "action": "lattice_refinement",
                            "relation": found["relation"].tolist(),
                            "residual": found["residual"], "winding": found["winding"],
                            "candidate_count": found["candidate_count"]})
    # A primitive generator has many integer representations differing by
    # known relations. Use a short one to limit phase-error amplification.
    for row in range(len(products_matrix)):
        for _ in range(8):
            current = products_matrix[row].copy()
            best = current
            best_score = (int(np.sum(abs(best))), int(np.max(abs(best))))
            for relation in relations:
                for multiple in range(-max_order, max_order+1):
                    candidate = current+multiple*relation
                    score = (int(np.sum(abs(candidate))), int(np.max(abs(candidate))))
                    if score < best_score:
                        best, best_score = candidate, score
            if np.array_equal(best, current):
                break
            products_matrix[row] = best
    # Products are the actual returned coordinate map; re-express frequency
    # from that map so phase propagation and phase products have one gauge.
    product_frequency = np.angle(np.exp(1j*(products_matrix @ frequencies)))
    for coordinate in range(len(product_frequency)):
        if product_frequency[coordinate] < 0:
            products_matrix[coordinate] *= -1
            labels[:, coordinate] *= -1
            product_frequency[coordinate] *= -1
    if not np.array_equal(products_matrix @ labels, np.eye(len(product_frequency), dtype=np.int64)):
        raise ArithmeticError("primitive-coordinate identity failed")
    reconstruction = np.angle(np.exp(1j*(labels @ product_frequency-frequencies)))
    return {"rank": len(basis), "basis": product_frequency.tolist(),
            "peak_labels": labels.tolist(), "products": products_matrix.tolist(),
            "relations": relations.tolist(),
            "max_peak_residual": float(np.max(abs(reconstruction))),
            "max_product_l1": int(np.max(np.sum(abs(products_matrix), axis=1))),
            "max_order": max_order, "tolerance": tolerance,
            "history": history, "status": "conditional_on_visible_peaks_and_relation_gap"}


def phase_products(phases: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    return np.angle(np.exp(1j*np.einsum("ij,jk->ik", coefficients, phases, optimize=False)))


def short_product_coordinates(labels: np.ndarray, products_matrix: np.ndarray,
                              coefficient_budget: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Short integer left inverses, followed by a unimodular basis reduction."""
    from scipy.optimize import Bounds, LinearConstraint, milp
    from sympy import Matrix

    labels = labels.copy()
    products_matrix = products_matrix.copy()
    p, r = labels.shape
    for pass_number in range(2):
        for coordinate in range(r):
            equality = np.column_stack((labels.T, np.zeros((r, p))))
            inequalities = np.vstack((np.column_stack((np.eye(p), -np.eye(p))),
                                      np.column_stack((-np.eye(p), -np.eye(p)))))
            target = np.eye(r)[coordinate]
            constraints = [LinearConstraint(equality, target, target),
                           LinearConstraint(inequalities, -np.inf, 0.0)]
            result = milp(np.r_[np.zeros(p), np.ones(p)],
                          integrality=np.r_[np.ones(p), np.zeros(p)],
                          bounds=Bounds(np.r_[-np.full(p, coefficient_budget), np.zeros(p)],
                                        np.full(2*p, coefficient_budget)),
                          constraints=constraints,
                          options={"time_limit": 1.0, "mip_rel_gap": 0.0})
            if result.x is not None:
                candidate = np.rint(result.x[:p]).astype(np.int64)
                if (np.array_equal(candidate @ labels, target.astype(np.int64))
                        and np.sum(abs(candidate)) <= np.sum(abs(products_matrix[coordinate]))):
                    products_matrix[coordinate] = candidate
        if pass_number == 0 and r > 1:
            _, transform = Matrix(products_matrix.tolist()).lll_transform()
            inverse = transform.inv()
            if any(not value.is_Integer for value in inverse):
                raise ArithmeticError("LLL basis transform is not unimodular")
            transform_integer = np.asarray(transform.tolist(), dtype=np.int64)
            inverse_integer = np.asarray(inverse.tolist(), dtype=np.int64)
            products_matrix = transform_integer @ products_matrix
            labels = labels @ inverse_integer
    return labels, products_matrix


def infer_character_basis_global(
    peaks: np.ndarray, tolerance: float, max_order: int = 6, max_rank: int = 3,
    *, frequency_covariance: np.ndarray | None = None,
    confidence_radius: float = 1.0,
) -> dict:
    """Use all bounded relations in the original peak coordinates.

    Meet-in-the-middle enumeration avoids an order-dependent bound in a
    changing basis. Smith decomposition constructs the quotient generators.
    Correctness still requires a relation gap and a complete visible module.
    Near rational sampling resonances are reported, not interpreted as tori.
    Optional covariance uses a simultaneous frequency ellipsoid, not a
    Rayleigh-bin tolerance. Compatibility with this ellipsoid is not proof
    of an exact integer relation; finite-model bias and gaps remain external
    assumptions. Without covariance the previous scalar rule is retained.
    """
    from sympy import Matrix, ZZ
    from sympy.polys.matrices import DomainMatrix
    from sympy.polys.matrices.normalforms import smith_normal_decomp

    frequencies = np.asarray(peaks, dtype=float).reshape(-1)
    p = len(frequencies)
    if not p:
        raise ValueError("at least one resolved peak is required")
    covariance = None
    search_tolerance = tolerance
    if frequency_covariance is not None:
        covariance = np.asarray(frequency_covariance, dtype=float)
        if (covariance.shape != (p, p) or confidence_radius <= 0
                or not np.isfinite(covariance).all()
                or not np.allclose(covariance, covariance.T)
                or np.min(np.linalg.eigvalsh(covariance)) <= 0):
            raise ValueError("positive definite frequency covariance required")
        search_tolerance += confidence_radius*max_order*np.sqrt(np.sum(abs(covariance)))
        if search_tolerance >= np.pi:
            return {"rank": None, "reason": "frequency uncertainty exceeds local circular search budget"}

    def relation_tolerance(vector):
        if covariance is None:
            return tolerance
        variance = float(np.einsum("i,ij,j->", vector, covariance, vector, optimize=False))
        return tolerance+confidence_radius*np.sqrt(max(0.0, variance))

    split = p//2
    left = np.asarray(list(product(range(-max_order, max_order+1), repeat=split)), dtype=np.int64)
    right = np.asarray(list(product(range(-max_order, max_order+1), repeat=p-split)), dtype=np.int64)
    if split == 0:
        left = np.empty((1, 0), dtype=np.int64)
    left_angles = np.mod(np.einsum("ij,j->i", left, frequencies[:split], optimize=False), 2*np.pi)
    right_angles = np.mod(np.einsum("ij,j->i", right, frequencies[split:], optimize=False), 2*np.pi)
    permutation = np.argsort(right_angles)
    sorted_angles = right_angles[permutation]
    rows = []
    null_columns = np.eye(p, dtype=np.int64)
    examined = 0
    for left_coefficients, left_angle in zip(left, left_angles):
        target = float((-left_angle) % (2*np.pi))
        intervals = [(target-search_tolerance, target+search_tolerance)]
        if target < search_tolerance:
            intervals.append((target-search_tolerance+2*np.pi, 2*np.pi))
        if target+search_tolerance > 2*np.pi:
            intervals.append((0.0, target+search_tolerance-2*np.pi))
        for lower, upper in intervals:
            begin = np.searchsorted(sorted_angles, max(0.0, lower), side="left")
            end = np.searchsorted(sorted_angles, min(2*np.pi, upper), side="right")
            if begin == end:
                continue
            matching = right[permutation[begin:end]]
            block = np.column_stack((np.broadcast_to(left_coefficients, (len(matching), split)), matching))
            block = block[np.any(block != 0, axis=1)]
            examined += len(block)
            if covariance is not None and len(block):
                residuals = abs(np.angle(np.exp(1j*np.einsum(
                    "ij,j->i", block, frequencies, optimize=False))))
                variances = np.einsum("bi,ij,bj->b", block, covariance, block,
                                      optimize=False)
                limits = tolerance+confidence_radius*np.sqrt(np.maximum(0.0, variances))
                block = block[residuals <= limits]
            # Once the kernel span is known, millions of redundant harmonic
            # relations can be rejected together, before integer arithmetic.
            while len(block):
                membership = block @ null_columns
                independent = np.flatnonzero(np.any(membership != 0, axis=1))
                if not len(independent):
                    break
                selected = int(independent[0])
                vector = block[selected].copy()
                block = np.delete(block, selected, axis=0)
                divisor = 0
                for value in vector:
                    divisor = gcd(divisor, abs(int(value)))
                primitive = vector//divisor
                primitive_residual = abs(float(np.angle(np.exp(1j*np.dot(primitive, frequencies)))))
                # A nonprimitive relation can indicate torsion (e.g. 2*pi=0).
                if primitive_residual <= relation_tolerance(primitive):
                    vector = primitive
                first_nonzero = next(int(value) for value in vector if value)
                if first_nonzero < 0:
                    vector = -vector
                candidate_rows = rows+[vector.tolist()]
                if Matrix(candidate_rows).rank() <= len(rows):
                    continue
                rows = candidate_rows
                if len(rows) == p:
                    return {"rank": None, "reason": "bounded relations imply no nonresonant torus; tolerance or sampling resonance",
                            "relations": rows, "examined_matches": examined,
                            "max_order": max_order, "tolerance": tolerance}
                columns = []
                for vector in Matrix(rows).nullspace():
                    denominator = 1
                    for value in vector:
                        denominator = lcm(denominator, int(value.q))
                    integer_vector = [int(value*denominator) for value in vector]
                    divisor = 0
                    for value in integer_vector:
                        divisor = gcd(divisor, abs(value))
                    columns.append([value//divisor for value in integer_vector])
                largest = max(abs(value) for column in columns for value in column)
                dtype = np.int64 if largest*max_order*p < np.iinfo(np.int64).max else object
                null_columns = np.asarray(columns, dtype=dtype).T
    relation_rank = len(rows)
    rank = p-relation_rank
    if rank > max_rank:
        return {"rank": None, "reason": "rank exceeds declared search budget",
                "relation_rank": relation_rank, "examined_matches": examined,
                "max_order": max_order, "tolerance": tolerance}
    if not rows:
        labels = np.eye(p, dtype=np.int64)
        products_matrix = np.eye(p, dtype=np.int64)
        kernel = np.zeros((0, p), dtype=np.int64)
        invariants = []
    else:
        original = Matrix(rows)
        reduced = original.lll()
        diagonal, _, right_transform = smith_normal_decomp(DomainMatrix.from_Matrix(reduced).convert_to(ZZ))
        right_matrix = right_transform.to_Matrix()
        inverse_matrix = right_matrix.inv()
        values = list(right_matrix)+list(inverse_matrix)
        if any(not value.is_Integer or abs(int(value)) > 10**12 for value in values):
            return {"rank": None, "reason": "integer coordinate transform exceeds numerical budget"}
        right_integer = np.asarray(right_matrix.tolist(), dtype=np.int64)
        inverse_integer = np.asarray(inverse_matrix.tolist(), dtype=np.int64)
        labels = right_integer[:, relation_rank:]
        products_matrix = inverse_integer[relation_rank:]
        kernel = inverse_integer[:relation_rank]
        invariants = [abs(int(diagonal.to_Matrix()[i, i])) for i in range(relation_rank)]
        constraint_phase = np.angle(np.exp(1j*(kernel @ frequencies)))
        # Saturation is valid only in the assumed torsion-free frequency group.
        for value, invariant, relation in zip(constraint_phase, invariants, kernel):
            limit = (relation_tolerance(relation) if covariance is not None
                     else tolerance*max(1, int(np.sum(abs(relation)))))
            if invariant > 1 and abs(value) > limit:
                return {"rank": None, "reason": "possible rational sampling resonance or unsaturated noisy relation",
                        "smith_invariants": invariants, "constraint_residuals": constraint_phase.tolist()}
    for row in range(rank):
        for _ in range(12):
            current = products_matrix[row].copy()
            best = current
            best_score = (int(np.sum(abs(current))), int(np.max(abs(current))))
            for relation in kernel:
                for multiple in range(-max_order, max_order+1):
                    candidate = current+multiple*relation
                    score = (int(np.sum(abs(candidate))), int(np.max(abs(candidate))))
                    if score < best_score:
                        best, best_score = candidate, score
            if np.array_equal(best, current):
                break
            products_matrix[row] = best
    labels, products_matrix = short_product_coordinates(labels, products_matrix)
    basis = np.angle(np.exp(1j*(products_matrix @ frequencies)))
    for coordinate in range(rank):
        if basis[coordinate] < 0:
            basis[coordinate] *= -1
            products_matrix[coordinate] *= -1
            labels[:, coordinate] *= -1
    if not np.array_equal(products_matrix @ labels, np.eye(rank, dtype=np.int64)):
        raise ArithmeticError("Smith coordinate identity failed")
    ellipsoid_fit = None
    if covariance is not None:
        # One common fit must satisfy all accepted relations, including their
        # correlations. Individual near-relations can be jointly inconsistent.
        factor = np.linalg.cholesky(covariance)
        winding = np.rint((np.einsum("ij,j->i", labels, basis, optimize=False)
                           -frequencies)/(2*np.pi)).astype(np.int64)
        lifted = frequencies+2*np.pi*winding
        whitened_labels = np.linalg.solve(factor, labels.astype(float))
        whitened_lifted = np.linalg.solve(factor, lifted)
        fitted_basis = np.linalg.lstsq(whitened_labels, whitened_lifted, rcond=None)[0]
        fitted_residual = (np.einsum("ij,j->i", whitened_labels, fitted_basis,
                                     optimize=False)-whitened_lifted)
        distance_squared = float(np.sum(fitted_residual**2))
        ellipsoid_fit = {"distance_squared": distance_squared,
                         "radius_squared": confidence_radius**2,
                         "local_winding": winding.tolist(),
                         "compatibility_only": True}
        if distance_squared > confidence_radius**2*(1+1e-7):
            return {"rank": None, "reason": "accepted relations jointly inconsistent with frequency ellipsoid",
                    "candidate_rank": rank, "ellipsoid_fit": ellipsoid_fit,
                    "peak_labels": labels.tolist(), "products": products_matrix.tolist(),
                    "relations": kernel.tolist()}
        basis = np.angle(np.exp(1j*fitted_basis))
    reconstruction = np.angle(np.exp(1j*(labels @ basis-frequencies)))
    answer = {"rank": rank, "basis": basis.tolist(), "products": products_matrix.tolist(),
            "peak_labels": labels.tolist(), "relations": kernel.tolist(),
            "smith_invariants": invariants, "relation_rank": relation_rank,
            "max_peak_residual": float(np.max(abs(reconstruction))),
            "max_product_l1": int(np.max(np.sum(abs(products_matrix), axis=1))),
            "examined_matches": examined, "max_order": max_order, "tolerance": tolerance,
            "status": "conditional_on_relation_gap_complete_kernel_and_visible_generators"}
    if ellipsoid_fit is not None:
        answer["ellipsoid_fit"] = ellipsoid_fit
        answer["frequency_threshold_rule"] = "tolerance + radius*sqrt(v.T covariance v)"
        answer["confidence_radius"] = confidence_radius
    return answer
