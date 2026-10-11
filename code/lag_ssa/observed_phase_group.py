"""A bounded observed-character diagnostic for a sampled orbit group.

For exact characters z_j(n), constant products z(n)^a define the integer
relation lattice K.  The character group is Z^p/K.  Smith invariants retain
finite cyclic factors: the row (50, 0) must never be replaced by (1, 0).

Here K is only a candidate learned from finite noisy phase products.  Search
coverage, exact relations, and density in the candidate group are unresolved.
No source frequencies, phase rank, generator law or true phases are inputs.
"""

from __future__ import annotations

from itertools import combinations, product
from math import gcd

import numpy as np
from sympy import Matrix, ZZ
from sympy.matrices.normalforms import hermite_normal_form, smith_normal_form


def _characters(values: np.ndarray, name: str) -> np.ndarray:
    z = np.asarray(values, dtype=complex)
    if (z.ndim != 2 or not z.shape[0] or z.shape[1] < 3
            or not np.isfinite(z).all() or np.any(abs(z) < 1e-12)):
        raise ValueError(name + " must contain finite nonzero character rows and at least three columns")
    return z / abs(z)


def _positive_compositions(total: int, size: int, bound: int):
    if size == 1:
        if 1 <= total <= bound:
            yield (total,)
        return
    for first in range(1, min(bound, total-size+1) + 1):
        for rest in _positive_compositions(total-first, size-1, bound):
            yield (first,) + rest


def _mixed_directions(dimension: int, coefficient_bound: int,
                      support_bound: int, budget: int):
    """Sparse primitive directions in increasing l1 order; a finite budget."""
    count = 0
    for norm in range(2, coefficient_bound * min(support_bound, dimension) + 1):
        for support in range(2, min(support_bound, dimension, norm) + 1):
            for active in combinations(range(dimension), support):
                for magnitudes in _positive_compositions(norm, support, coefficient_bound):
                    divisor = 0
                    for value in magnitudes:
                        divisor = gcd(divisor, value)
                    if divisor != 1:
                        continue
                    for signs in product((-1, 1), repeat=support-1):
                        if count >= budget:
                            return
                        row = np.zeros(dimension, dtype=int)
                        for index, magnitude, sign in zip(active, magnitudes, (1,) + signs):
                            row[index] = sign * magnitude
                        count += 1
                        yield row


def _score_angles(train_angles: np.ndarray, heldout_angles: np.ndarray,
                  row: np.ndarray, threshold: float) -> dict:
    train = np.exp(1j * np.einsum("i,it->t", row, train_angles, optimize=False))
    heldout = np.exp(1j * np.einsum("i,it->t", row, heldout_angles, optimize=False))
    train_mean, heldout_mean = np.mean(train), np.mean(heldout)
    origin = float(np.angle(train_mean))
    aligned = heldout * np.exp(-1j * origin)
    train_coherence = float(abs(train_mean))
    heldout_coherence = float(abs(heldout_mean))
    agreement = float(np.real(np.mean(aligned)))
    return {
        "relation": np.asarray(row, dtype=int).tolist(),
        "train_constant_origin_rad": origin,
        "train_coherence": train_coherence,
        "heldout_coherence": heldout_coherence,
        "heldout_frozen_origin_agreement": agreement,
        "heldout_frozen_origin_mae_rad": float(np.mean(abs(np.angle(aligned)))),
        "supported": bool(min(train_coherence, heldout_coherence, agreement) >= threshold),
    }


def _presentation(rows: list, dimension: int, train_angles: np.ndarray,
                  heldout_angles: np.ndarray, threshold: float) -> dict:
    """Present the unsaturated row lattice, then recheck its HNF basis."""
    if rows:
        relation_matrix = Matrix(rows)
        basis = hermite_normal_form(relation_matrix.T).T
        smith = smith_normal_form(basis, domain=ZZ)
        invariant_factors = [abs(int(smith[index, index]))
                             for index in range(min(smith.shape)) if smith[index, index]]
        basis_rows = [[int(value) for value in row] for row in basis.tolist()]
    else:
        basis_rows, invariant_factors = [], []
    rank = len(invariant_factors)
    checks = [_score_angles(train_angles, heldout_angles, np.asarray(row, dtype=int), threshold)
              for row in basis_rows]
    compatible = all(check["supported"] for check in checks)
    finite_factors = [value for value in invariant_factors if value > 1]
    connected_dimension = dimension-rank
    terms = (["T^" + str(connected_dimension)] if connected_dimension else [])
    terms.extend("Z/" + str(value) + "Z" for value in finite_factors)
    symbol = " x ".join(terms) if terms else "trivial group"
    status = ("unresolved_incompatible_approximate_constraints" if not compatible else
              "supported_bounded_candidate" if rows else "no_supported_relations_within_search_budget")
    return {
        "ambient_character_count": dimension,
        "integer_relation_row_basis_HNF": basis_rows,
        "smith_nonzero_invariant_factors": invariant_factors,
        "relation_rank": rank,
        "candidate_connected_component_dimension": connected_dimension,
        "candidate_finite_cyclic_factors": finite_factors,
        "candidate_component_count": int(np.prod(np.asarray(finite_factors, dtype=object))) if finite_factors else 1,
        "candidate_character_group": "Z^p / K_hat",
        "candidate_compact_group_symbol": symbol,
        "basis_relations_rechecked_on_observations": checks,
        "candidate_basis_empirically_compatible": compatible,
        "candidate_presentation_has_supported_constraints": bool(rows),
        "relation_lattice_saturated_or_gcd_divided": False,
        "interpretation_status": status,
        "actual_observable_group_identified": False,
        "density_in_candidate_group_certified": False,
        "connected_dimension_claim": "dimension of the group presented by the candidate relations; only if those relations are exact is the true closure a subgroup; finite noisy support can also accept a false near relation",
    }


def estimate_observed_group(acquired_train: np.ndarray,
                            acquired_validation: np.ndarray,
                            max_integer_bound: int = 100,
                            mixed_coefficient_bound: int = 6,
                            max_mixed_support: int | None = None,
                            max_primitive_directions: int = 4096,
                            power_proposals_per_mixed_direction: int = 3,
                            coherence_threshold: float = .95) -> dict:
    """Estimate a candidate sampled-orbit group from observed phase products.

    Rows correspond to independently obtained, aligned, unit-circle readouts.
    Training proposes all single-coordinate powers through max_integer_bound
    and short primitive mixed directions.  A few possible finite orders of
    mixed directions are proposed from observed phase increments.  Acceptance
    always uses actual products on training and independently frozen heldout
    readouts, with the constant origin fitted only on training.

    Relations include products whose accumulated frequency is nonzero modulo
    2*pi before sampling.  Their integer rows remain unsaturated.  A finite
    search cannot determine completeness or certify an exact resonance.
    """
    train = _characters(acquired_train, "acquired_train")
    heldout = _characters(acquired_validation, "acquired_validation")
    if train.shape[0] != heldout.shape[0]:
        raise ValueError("training and heldout must contain the same acquired characters")
    if (max_integer_bound < 1 or mixed_coefficient_bound < 1
            or (max_mixed_support is not None and max_mixed_support < 2)
            or max_primitive_directions < 1 or power_proposals_per_mixed_direction < 1
            or not 0 < coherence_threshold <= 1):
        raise ValueError("positive finite search bounds and a coherence threshold in (0,1] required")
    dimension = train.shape[0]
    support_bound = dimension if max_mixed_support is None else min(max_mixed_support, dimension)
    train_angles, heldout_angles = np.angle(train), np.angle(heldout)
    accepted, train_only_candidates, tested = [], [], set()
    tested_count, mixed_direction_count = 0, 0

    def test(row: np.ndarray, proposal: str, base=None, power=None):
        nonlocal tested_count
        key = tuple(int(value) for value in row)
        if key in tested:
            return
        tested.add(key)
        tested_count += 1
        angle = np.einsum("i,it->t", row, train_angles, optimize=False)
        coherence = float(abs(np.mean(np.exp(1j * angle))))
        if coherence < coherence_threshold:
            return
        check = _score_angles(train_angles, heldout_angles, row, coherence_threshold)
        check.update(proposal=proposal, primitive_direction=None if base is None else base.tolist(),
                     proposed_power=power, origin_fitted_on_training_only=True,
                     original_frequency_or_clock_used=False)
        train_only_candidates.append(check)
        if check["supported"]:
            accepted.append(check)

    # Keep nonprimitive powers: q e_j may produce a finite cyclic factor.
    for coordinate in range(dimension):
        base = np.zeros(dimension, dtype=int)
        base[coordinate] = 1
        for power in range(1, max_integer_bound+1):
            test(power * base, "exhaustive bounded single-coordinate power", base, power)

    # Budgeted sparse directions avoid an exponential [-100,100]^p cube.
    for base in _mixed_directions(dimension, min(mixed_coefficient_bound, max_integer_bound),
                                  support_bound, max_primitive_directions):
        mixed_direction_count += 1
        test(base, "direct bounded primitive mixed product", base, 1)
        order_bound = max_integer_bound // int(np.max(abs(base)))
        if order_bound < 2:
            continue
        product_character = np.exp(1j * np.einsum("i,it->t", base, train_angles, optimize=False))
        observed_step = float(np.angle(np.mean(product_character[1:] * np.conj(product_character[:-1]))))
        powers = np.arange(2, order_bound+1)
        residue = abs(np.angle(np.exp(1j * powers * observed_step)))
        proposals = sorted(range(len(powers)), key=lambda index: (residue[index], powers[index]))
        for index in proposals[:power_proposals_per_mixed_direction]:
            power = int(powers[index])
            test(power * base, "finite-order proposal from observed product increments, checked on actual products", base, power)

    accepted.sort(key=lambda row: (sum(abs(value) for value in row["relation"]), row["relation"]))
    presentation = _presentation([row["relation"] for row in accepted], dimension,
                                 train_angles, heldout_angles, coherence_threshold)
    return {
        "status": presentation["interpretation_status"],
        "input_scope": "observed aligned acquired train/heldout unit-circle phase arrays only",
        "train_columns": train.shape[1], "heldout_columns": heldout.shape[1],
        "coherence_threshold": coherence_threshold,
        "accepted_relations": accepted,
        "train_candidate_relations_rejected_on_heldout": [row for row in train_only_candidates if not row["supported"]],
        "presentation": presentation,
        "search": {
            "maximum_absolute_coefficient": max_integer_bound,
            "short_mixed_coefficient_bound": min(mixed_coefficient_bound, max_integer_bound),
            "maximum_mixed_support": support_bound,
            "mixed_support_parameter": max_mixed_support,
            "all_acquired_characters_allowed_in_one_relation": support_bound == dimension,
            "mixed_direction_budget": max_primitive_directions,
            "mixed_directions_examined": mixed_direction_count,
            "mixed_direction_budget_reached": mixed_direction_count >= max_primitive_directions,
            "mixed_power_proposals_per_direction": power_proposals_per_mixed_direction,
            "integer_product_candidates_tested_on_train": tested_count,
            "mixed_power_search_exhaustive": False,
            "relation_search_complete": False,
        },
        "exact_relation_or_torsion_certified": False,
        "continuous_time_phase_rank_identified": False,
        "finite_noise_and_near_resonance_resolved": False,
        "novelty_established": False,
        "interpretation": "candidate sampled orbit closure, distinct from the unsampled continuous-time phase torus",
    }
