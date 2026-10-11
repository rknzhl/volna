"""Bounded skip/include alternatives for the observed integer relation lattice.

This is an exact-algebra search probe, not a fitted group selector or rank
certificate. No field fits, evaluator, physical label or new VAL score enters.
Complexity is measured in one fixed LLL-reduced free-character gauge; it is
not an optimization over all GL(Z) bases or a necessary identifiability bound.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isqrt
import hashlib

from sympy import Matrix, Rational

from ledger_group_selection import row_hnf, smith_presentation


@dataclass(frozen=True)
class AlternativeSettings:
    beam_width: int = 64
    maximum_proposals: int = 4096
    maximum_observed_lines: int = 48
    lll_delta_numerator: int = 3
    lll_delta_denominator: int = 4
    whole_representation_budget_rule: str = "floor(sqrt(N_train))"
    finite_component_budget_rule: str = "N_train"
    expand_code_ineligible_intermediates: bool = True
    allocation: str = "one state per reachable relation rank per round, ascending rank; no cross-rank score"


SETTINGS = AlternativeSettings()


def _key(relation, winding):
    return tuple(tuple(int(v) for v in row) for row in relation.tolist()), tuple(int(v) for v in winding)


def _identifier(key):
    return hashlib.sha256(repr(key).encode()).hexdigest()[:24]


def _exact_append(relation, winding, row, new_winding, dimension):
    """Lift the source winding homomorphism to the new UNSATURATED HNF.

    Original rows have unique integer coordinates in the HNF basis. Solve
    those coordinate equations exactly for basis windings. No large HNF row
    is multiplied by noisy float frequencies and independently re-rounded.
    """
    source_rows = [[int(v) for v in r] for r in relation.tolist()]+[list(map(int, row))]
    source_windings = list(map(int, winding))+[int(new_winding)]
    basis = row_hnf(source_rows, dimension)
    coordinates = []
    for source in source_rows:
        coordinate, parameters = basis.T.gauss_jordan_solve(Matrix(source))
        if parameters.rows or any(not v.is_Integer for v in coordinate):
            raise ArithmeticError("source row lacks exact integer HNF coordinates")
        coordinates.append([int(v) for v in coordinate])
    try:
        lifted, parameters = Matrix(coordinates).gauss_jordan_solve(Matrix(source_windings))
    except ValueError:
        return None, None, "inconsistent_dependent_source_windings"
    if parameters.rows or any(not v.is_Integer for v in lifted):
        return None, None, "source_windings_do_not_define_integer_homomorphism"
    if Matrix(coordinates)*lifted != Matrix(source_windings):
        raise ArithmeticError("exact lifted winding equations failed")
    return basis, tuple(int(v) for v in lifted), "exact_winding_homomorphism_lifted"


def representation(relation, winding, dimension, n_train, settings=SETTINGS):
    presentation = smith_presentation(relation.tolist(), dimension, list(winding))
    rank = relation.rows
    v = Matrix(presentation["V"])
    inverse = Matrix(presentation["V_inverse"])
    u = Matrix(rank, rank, [int(v) for row in presentation["U"] for v in row])
    free = v[:, rank:]
    free_dimension = dimension-rank
    if free_dimension:
        reduced_rows, row_transform = free.T.lll_transform(
            delta=Rational(settings.lll_delta_numerator, settings.lll_delta_denominator))
        column_transform = row_transform.T
        if row_transform*free.T != reduced_rows or abs(int(column_transform.det())) != 1:
            raise ArithmeticError("free-character LLL transformation is not exact unimodular")
        transform = Matrix.eye(dimension)
        transform[rank:, rank:] = column_transform
        transform_inverse = Matrix.eye(dimension)
        transform_inverse[rank:, rank:] = column_transform.inv()
    else:
        column_transform = Matrix.eye(0)
        transform = transform_inverse = Matrix.eye(dimension)
    v_reduced, inverse_reduced = v*transform, transform_inverse*inverse
    diagonal = Matrix.zeros(rank, dimension)
    for i,value in enumerate(presentation["smith_diagonal"]):diagonal[i, i] = value
    if (u*relation*v_reduced != diagonal or v_reduced*inverse_reduced != Matrix.eye(dimension)
            or v_reduced[:, :rank] != v[:, :rank]
            or any(not value.is_Integer for value in inverse_reduced)):
        raise ArithmeticError("reduced gauge changed lattice, inverse or finite character block")
    free_reduced = v_reduced[:, rank:]
    finite_indices = presentation["finite_smith_indices"]
    orders = presentation["smith_diagonal"]
    finite_centered = Matrix(dimension, len(finite_indices),
        [(int(v[j,i]) % orders[i] if 2*(int(v[j,i]) % orders[i]) <= orders[i]
          else (int(v[j,i]) % orders[i])-orders[i])
         for j in range(dimension) for i in finite_indices])
    row_costs = [sum(abs(int(value)) for value in free_reduced.row(j))
                 +sum(abs(int(value)) for value in finite_centered.row(j)) for j in range(dimension)]
    complexity = max(row_costs, default=0)
    budget = isqrt(int(n_train))
    components = presentation["component_count"]
    reasons = []
    if complexity > budget:reasons.append("whole_character_weight_budget_exceeded")
    if components > n_train:reasons.append("finite_component_budget_exceeded")
    if not presentation["finite_rotation_visits_all_presented_components"]:
        reasons.append("incomplete_self_domain_finite_rotation")
    convert = lambda matrix:[[int(value) for value in row] for row in matrix.tolist()]
    return dict(relation_rows_HNF=convert(relation), integer_windings=list(winding),
        relation_rank=rank, free_dimension=free_dimension,
        smith_diagonal=presentation["smith_diagonal"], finite_orders=presentation["finite_orders"],
        finite_component_count=components, finite_rotation_orbit_order=presentation["finite_rotation_orbit_order"],
        finite_rotation_visits_all_components=presentation["finite_rotation_visits_all_presented_components"],
        U=convert(u), V_reduced=convert(v_reduced), V_inverse_reduced=convert(inverse_reduced),
        free_column_transform_unimodular=convert(column_transform),
        free_character_weights_reduced=convert(free_reduced),
        finite_character_weights_centered=convert(finite_centered),
        whole_character_row_costs=row_costs, whole_representation_complexity=complexity,
        whole_representation_budget=budget, finite_component_budget=int(n_train),
        representation_sieve_passed=not reasons, representation_sieve_rejection_reasons=reasons,
        finite_character_block_unchanged=True, exact_reduced_transforms_verified=True,
        lattice_saturated=False, minimum_over_GL_bases_computed=False,
        actual_rotation_not_refitted_or_snapped=True, actual_group_or_rank_certified=False)


def _within_rank_order(state):
    return (state["whole_representation_complexity"], state["finite_component_count"],
            tuple(tuple(row) for row in state["relation_rows_HNF"]), tuple(state["integer_windings"]))


def _rank_balanced_beam(states, width):
    buckets = {}
    for state in states:
        buckets.setdefault(state["relation_rank"], []).append(state)
    for values in buckets.values():values.sort(key=_within_rank_order)
    selected, round_index = [], 0
    while len(selected) < width:
        added = False
        for rank in sorted(buckets):
            if round_index < len(buckets[rank]) and len(selected) < width:
                selected.append(buckets[rank][round_index])
                added = True
        if not added:break
        round_index += 1
    return selected


def search_alternatives(eligible_proposals, dimension, n_train, settings=SETTINGS, progress=None):
    if (not 1 <= dimension <= settings.maximum_observed_lines
            or settings.beam_width < dimension+1 or n_train <= 1):
        raise ValueError("finite ledger dimension and beam preserving every reachable rank required")
    proposals = eligible_proposals[:settings.maximum_proposals]
    states, cache, transitions, stages = [], {}, [], []

    def register(relation, winding, parent=None, proposal_index=None):
        key = _key(relation, winding)
        if key in cache:return cache[key]
        state = representation(relation, winding, dimension, n_train, settings)
        state.update(state_id=_identifier(key), first_parent_id=parent,
            first_included_proposal_index=proposal_index,
            first_generated_stage=(proposal_index+1 if proposal_index is not None else 0))
        cache[key] = state
        states.append(state)
        return state

    initial = register(row_hnf([], dimension), ())
    beam = [initial]
    for index, proposal in enumerate(proposals):
        expanded = {state["state_id"]:state for state in beam}
        for state in beam:
            relation = Matrix(len(state["relation_rows_HNF"]), dimension,
                              [v for row in state["relation_rows_HNF"] for v in row])
            try:
                child_relation, child_winding, status = _exact_append(relation, state["integer_windings"],
                    proposal["row"], proposal["initial_integer_winding"], dimension)
                if child_relation is None:
                    transitions.append(dict(stage=index+1, parent_id=state["state_id"], action="include",
                                            status=status))
                    continue
                child = register(child_relation, child_winding, state["state_id"], index)
                expanded[child["state_id"]] = child
                transitions.append(dict(stage=index+1, parent_id=state["state_id"], action="include",
                    status=status, child_id=child["state_id"], exact_lattice_changed=child["state_id"] != state["state_id"]))
            except (ArithmeticError, ValueError, AssertionError) as error:
                transitions.append(dict(stage=index+1, parent_id=state["state_id"], action="include",
                    status="unresolved_exact_algebra_or_LLL_failure", exception=f"{type(error).__name__}: {error}"))
        beam = _rank_balanced_beam(list(expanded.values()), settings.beam_width)
        stages.append(dict(stage=index+1, proposal_index=index, row=proposal["row"],
            expanded_unique_state_count=len(expanded), retained_state_count=len(beam),
            retained_relation_ranks=sorted({state["relation_rank"] for state in beam}),
            retained_state_ids=[state["state_id"] for state in beam],
            retained_sieve_passing_count=sum(state["representation_sieve_passed"] for state in beam),
            generated_unique_state_count=len(states)))
        if progress is not None and ((index+1)%10 == 0 or index+1 == len(proposals)):
            progress(dict(stage=index+1, proposals=len(proposals), unique_states=len(states), beam=len(beam)))
    final_ids = {state["state_id"] for state in beam}
    for state in states:state["retained_in_final_beam"] = state["state_id"] in final_ids
    return dict(settings=asdict(settings), observed_line_count=dimension, n_train=n_train,
        eligible_proposal_count=len(eligible_proposals), processed_proposal_count=len(proposals),
        proposal_budget_exhausted=len(eligible_proposals) > len(proposals),
        source_proposal_order_preserved=True, source_conditional_covariance_screen_not_recalibrated=True,
        all_registered_states=states, transitions=transitions, stages=stages,
        final_beam_state_ids=[state["state_id"] for state in beam],
        registered_sieve_passing_state_ids=[state["state_id"] for state in states if state["representation_sieve_passed"]],
        same_rank_sort="whole LLL-gauge character complexity, finite count, HNF lexicographic",
        different_rank_allocation=settings.allocation,
        code_ineligible_intermediates_expanded=True,
        no_group_rank_selected=True, no_new_field_fit=True, no_new_validation_selection=True,
        new_rotation_equality_fit_not_performed=True, exact_source_winding_homomorphisms_preserved=True,
        incomplete_bounded_search=True,
        complexity_is_computational_eligibility_rule_not_identifiability_certificate=True,
        correctness_or_new_SSA_claimed=False)
