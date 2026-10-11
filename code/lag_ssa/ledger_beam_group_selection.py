"""RAW validation of all code-eligible states in a frozen bounded final beam.

No proposal discovery, representation budget, fit settings or equality backend
is changed here. Candidate windings and reduced Smith gauges are frozen exact
integers. This posthoc computational repair is not a physical-rank certificate.
"""

from dataclasses import asdict
from math import gcd, isqrt, lcm

import numpy as np
from sympy import Matrix

from ledger_group_selection import (SETTINGS, fit_tied_lines, _prediction,
                                    row_hnf, smith_presentation)


def _matrix(rows, height, width):
    if len(rows) != height or any(len(row) != width for row in rows):
        raise ValueError("exact matrix shape differs from frozen beam state")
    return Matrix(height, width, [int(value) for row in rows for value in row])


def presentation_from_state(state, dimension, n_train, settings=SETTINGS):
    """Validate and preserve the final state's UNSATURATED reduced Smith gauge."""
    if not state["retained_in_final_beam"] or not state["representation_sieve_passed"]:
        raise ValueError("only final-beam representation-sieve states are eligible")
    rank = int(state["relation_rank"])
    relation = _matrix(state["relation_rows_HNF"], rank, dimension)
    u = _matrix(state["U"], rank, rank)
    v = _matrix(state["V_reduced"], dimension, dimension)
    inverse = _matrix(state["V_inverse_reduced"], dimension, dimension)
    diagonal_values = [int(value) for value in state["smith_diagonal"]]
    if (len(diagonal_values) != rank or any(value <= 0 for value in diagonal_values)
            or any(diagonal_values[i+1] % diagonal_values[i] for i in range(rank-1))):
        raise ArithmeticError("invalid frozen positive Smith invariant factors")
    diagonal = Matrix.zeros(rank, dimension)
    for i,value in enumerate(diagonal_values):diagonal[i, i] = value
    if (row_hnf(relation.tolist(), dimension) != relation or relation.rank() != rank
            or u*relation*v != diagonal or v*inverse != Matrix.eye(dimension)
            or abs(int(u.det())) != 1 or abs(int(v.det())) != 1):
        raise ArithmeticError("exact HNF/Smith/unimodular inverse checks failed")
    free = v[:, rank:]
    finite_indices = [i for i,value in enumerate(diagonal_values) if value > 1]
    finite_orders = [diagonal_values[i] for i in finite_indices]
    finite = [[int(v[j,i]) % diagonal_values[i] for i in finite_indices] for j in range(dimension)]
    centered = [[value if 2*value <= order else value-order
                 for value,order in zip(row, finite_orders)] for row in finite]
    convert = lambda matrix:[[int(value) for value in row] for row in matrix.tolist()]
    row_costs = [sum(abs(int(value)) for value in free.row(j))+sum(abs(value) for value in centered[j])
                 for j in range(dimension)]
    complexity, components = max(row_costs, default=0), 1
    for order in finite_orders:components *= order
    winding = [int(value) for value in state["integer_windings"]]
    if len(winding) != rank:raise ArithmeticError("frozen equality windings shape mismatch")
    advance = u*Matrix(winding) if rank else Matrix.zeros(0, 1)
    orbit_order = 1
    for i in finite_indices:
        orbit_order = lcm(orbit_order, diagonal_values[i]//gcd(diagonal_values[i], int(advance[i])))
    if (convert(free) != state["free_character_weights_reduced"]
            or centered != state["finite_character_weights_centered"]
            or complexity != state["whole_representation_complexity"]
            or row_costs != state["whole_character_row_costs"]
            or int(state["whole_representation_budget"]) != isqrt(n_train)
            or int(state["finite_component_budget"]) != n_train
            or complexity > isqrt(n_train) or components > n_train
            or components != int(state["finite_component_count"])
            or orbit_order != components or orbit_order != int(state["finite_rotation_orbit_order"])):
        raise ArithmeticError("frozen representation or finite-orbit eligibility failed")
    return dict(relation_rows_HNF=convert(relation), U=convert(u), V=convert(v), V_inverse=convert(inverse),
        smith_diagonal=diagonal_values, relation_rank=rank, continuous_dimension=dimension-rank,
        finite_smith_indices=finite_indices, finite_orders=finite_orders, component_count=components,
        all_free_character_weights=convert(free), all_finite_character_weights=finite,
        fixed_smith_rotation_steps=[int(advance[i]) % diagonal_values[i] for i in range(rank)],
        finite_rotation_orbit_order=orbit_order, finite_rotation_visits_all_presented_components=True,
        finite_part_cyclic=len(finite_orders) <= 1, free_rotation_density_certified=False,
        coordinate_equations="U R V_reduced=D; ambient phi=V_reduced z, z=V_inverse_reduced phi",
        zero_origin_group_domain="D z in 2*pi*Z; finite components retained",
        every_finite_component_retained=True, lattice_saturated=False,
        observer_component_budget_available=components <= settings.maximum_observer_component_count,
        observer_status="eligible_for_later_integration" if components <= settings.maximum_observer_component_count
                        else "unresolved_component_count_exceeds_observer_budget",
        exact_algebra_verified=True, original_coordinate_characters_faithful_on_presented_group=True,
        frozen_free_LLL_gauge_used=True, beam_state_id=state["state_id"],
        whole_representation_complexity=complexity, representation_budget=isqrt(n_train),
        representation_budget_is_computational_not_physical_identifiability=True,
        actual_observable_group_certified=False)


def select_beam_group(raw_train, raw_validation, t_train, t_validation, parameters,
                      noise_sigma, final_states, settings=SETTINGS, progress=None):
    """Apply the unchanged old RAW one-bin backend to EVERY frozen eligible state."""
    nu = np.asarray(parameters["frequencies"], dtype=float)
    baseline = {key:np.asarray(parameters[key]).copy() for key in
                ("frequencies", "coefficients_sensor_center", "time_center",
                 "complex_profiles_sensor_origin", "temporal_gram")}
    baseline["presentation"] = smith_presentation([], len(nu), [], settings)
    baseline["model_kind"] = "frozen_free_ledger_baseline"
    baseline["validation_raw_sensor_mse"] = float(np.mean((raw_validation-_prediction(baseline, t_validation))**2))
    models, best = [baseline], 0
    attempts = [dict(status="frozen_full_free_baseline", stored_model_index=0,
                     validation_raw_sensor_mse=baseline["validation_raw_sensor_mse"])]
    free_attempt = dict(candidate_id="matched_unconstrained_refit", relation_rows_HNF=[])
    try:
        free = fit_tied_lines(raw_train, t_train, nu, [], [], noise_sigma, settings)
        free_attempt["fit_audit"] = free["audit"]
        if not free["audit"]["optimizer_success"] or not free["audit"]["equality_feasible"]:
            free_attempt["status"] = "unresolved_or_unconverged_free_refit"
        else:
            free["presentation"], free["model_kind"] = baseline["presentation"], "matched_unconstrained_refit"
            free["validation_raw_sensor_mse"] = float(np.mean((raw_validation-_prediction(free, t_validation))**2))
            models.append(free)
            free_attempt.update(status="admissible_free_refit", stored_model_index=len(models)-1,
                                validation_raw_sensor_mse=free["validation_raw_sensor_mse"])
            if free["validation_raw_sensor_mse"] < models[best]["validation_raw_sensor_mse"]:best = len(models)-1
    except (ValueError, np.linalg.LinAlgError, ArithmeticError, RuntimeError) as error:
        free_attempt.update(status="unresolved_free_refit_exception", exception=f"{type(error).__name__}: {error}")
    attempts.append(free_attempt)
    state_ids = [state["state_id"] for state in final_states]
    if len(set(state_ids)) != len(state_ids):raise ValueError("duplicate frozen final state")
    for index,state in enumerate(final_states):
        attempt = dict(candidate_id=state["state_id"], relation_rows_HNF=state["relation_rows_HNF"],
                       integer_windings=state["integer_windings"], frozen_final_state=state)
        try:
            presentation = presentation_from_state(state, len(nu), raw_train.shape[1], settings)
            attempt["presentation"] = presentation
            fitted = fit_tied_lines(raw_train, t_train, nu, state["relation_rows_HNF"],
                                    state["integer_windings"], noise_sigma, settings)
            attempt["fit_audit"] = fitted["audit"]
            if not fitted["audit"]["optimizer_success"] or not fitted["audit"]["equality_feasible"]:
                attempt["status"] = "unresolved_or_unconverged_equality_fit"
            else:
                fitted["presentation"], fitted["model_kind"] = presentation, "frozen_beam_relation_constrained_ledger"
                fitted["beam_state_id"] = state["state_id"]
                fitted["validation_raw_sensor_mse"] = float(np.mean((raw_validation-_prediction(fitted, t_validation))**2))
                models.append(fitted)
                attempt.update(status="admissible_tied_model", stored_model_index=len(models)-1,
                               validation_raw_sensor_mse=fitted["validation_raw_sensor_mse"])
                if fitted["validation_raw_sensor_mse"] < models[best]["validation_raw_sensor_mse"]:best = len(models)-1
        except (ValueError, np.linalg.LinAlgError, ArithmeticError, RuntimeError) as error:
            attempt.update(status="unresolved_model_exception", exception=f"{type(error).__name__}: {error}")
        attempts.append(attempt)
        if progress is not None:
            progress(dict(candidate=index+1, total=len(final_states), state_id=state["state_id"],
                          status=attempt["status"], selected_model_index=best))
    return dict(models=models, selected_model_index=best,
        audit=dict(settings=asdict(settings), observed_line_count=len(nu),
            candidate_state_ids=state_ids, all_model_attempts=attempts,
            all_frozen_final_sieve_states_attempted=True,
            selected_model_index=best, selected_presentation=models[best]["presentation"],
            selected_model_kind=models[best]["model_kind"],
            selected_validation_raw_sensor_mse=models[best]["validation_raw_sensor_mse"],
            selection_rule="strict minimum RAW sensor VAL MSE; exact ties retain earlier/free model",
            backend="unchanged ledger_group_selection.fit_tied_lines/_prediction/GroupSettings",
            final_state_windings_used_without_rerounding=True, reduced_Smith_gauge_preserved=True,
            no_proposal_research_or_truthmatched_subset=True, all_observed_lines_retained=True,
            no_new_frequency_or_primitive_divisor_discovery=True,
            clock_validation_only=True, observed_window_phase_recovery_scored=False,
            unknown_true_relation_lattice_or_phase_rank_supplied=False,
            exact_group_or_relation_completeness_certified=False,
            finite_near_relations_can_pass_and_win_validation=True,
            computational_code_budget_not_identifiability_certificate=True,
            posthoc_repair_after_prior_case_evaluation=True, novelty_claimed=False))
