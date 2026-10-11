"""Bounded whole-ledger relation hypotheses and constrained raw-field refits.

No physical rank, character labels, generator law or evaluator is an input.
LLL rows are proposals only. HNF preserves the exact UNSATURATED row lattice;
primitive divisors are independently screened alternatives, never replacements.
Smith transformations expose every free and finite coordinate of each model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import gcd, lcm

import numpy as np
from scipy.linalg import lstsq, qr, solve_triangular
from scipy.optimize import LinearConstraint, minimize
from sympy import Matrix, Rational, ZZ
from sympy.matrices.normalforms import hermite_normal_form
from sympy.polys.matrices import DomainMatrix
from sympy.polys.matrices.normalforms import smith_normal_decomp

from observed_phase_group import _score_angles
from raw_spectral_ledger import _design


@dataclass(frozen=True)
class GroupSettings:
    lll_scales: tuple[int, ...] = (10**5, 10**6, 10**7, 10**8, 10**9)
    mixed_coefficient_bound: int = 12
    mixed_l1_bound: int = 48
    coordinate_power_bound: int = 100
    conditional_z_bound: float = 3.
    observed_product_threshold: float = .95
    observed_products_are_acceptance_gate: bool = False
    maximum_nested_models: int = 64
    frequency_half_width_bins: float = 1.
    maximum_frequency_iterations: int = 100
    optimizer_ftol: float = 1e-11
    design_rcond: float = 1e-12
    constraint_rank_rcond: float = 1e-12
    constraint_absolute_tolerance: float = 1e-8
    minimum_interior_frequency: float = 1e-8
    maximum_observer_component_count: int = 100


SETTINGS = GroupSettings()


def row_hnf(rows, dimension):
    matrix = Matrix(len(rows), dimension, [int(value) for row in rows for value in row])
    return hermite_normal_form(matrix.T).T


def canonical_sign(row):
    values = tuple(int(value) for value in row)
    first = next((value for value in values if value), 0)
    return tuple(-value for value in values) if first < 0 else values


def propose_relations(frequencies, covariance, train_characters, validation_characters, settings=SETTINGS):
    nu = np.asarray(frequencies, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    train, validation = np.asarray(train_characters), np.asarray(validation_characters)
    count = len(nu)
    if (not count or not np.isfinite(nu).all() or covariance.shape != (count, count)
            or not np.isfinite(covariance).all() or train.ndim != 2 or validation.ndim != 2
            or train.shape[0] != count or validation.shape[0] != count
            or not np.isfinite(train).all() or not np.isfinite(validation).all()
            or np.any(abs(train) < 1e-12) or np.any(abs(validation) < 1e-12)):
        raise ValueError("full conditional ledger covariance and observed phase rows required")
    train_angles, validation_angles = np.angle(train), np.angle(validation)
    proposed = {}
    lll_audit = []

    def register(values, provenance):
        key = canonical_sign(values)
        if not any(key):
            return False
        support = sum(value != 0 for value in key)
        maximum, norm = max(abs(value) for value in key), sum(abs(value) for value in key)
        if ((support == 1 and maximum > settings.coordinate_power_bound)
                or (support > 1 and (maximum > settings.mixed_coefficient_bound
                                     or norm > settings.mixed_l1_bound))):
            return False
        proposed.setdefault(key, []).append(provenance)
        return True

    for scale in settings.lll_scales:
        lattice = Matrix.eye(count+1)
        for j, frequency in enumerate(nu):
            lattice[j, count] = int(round(scale*float(frequency)/(2*np.pi)))
        lattice[count, count] = int(scale)
        try:
            reduced = lattice.lll(delta=Rational(3, 4))
            lll_audit.append(dict(scale=int(scale), status="reduced", row_count=reduced.rows,
                                 reduced_integer_rows=[[int(v) for v in row] for row in reduced.tolist()]))
            for index, row in enumerate(reduced.tolist()):
                relation = tuple(int(value) for value in row[:count])
                source = dict(kind="LLL reduced row", scale=int(scale), row_index=index,
                              reduced_integer_last_entry=int(row[-1]))
                original_registered = register(relation, source)
                divisor = 0
                for value in relation:
                    divisor = gcd(divisor, abs(value))
                if divisor > 1:
                    register(tuple(value//divisor for value in relation),
                        dict(kind="independent primitive-divisor proposal", parent_row=list(relation),
                             divisor=divisor, scale=int(scale),
                             original_row_retained_as_bounded_proposal=original_registered,
                             original_row_recorded_in_LLL_audit=True))
        except (ValueError, AssertionError, ArithmeticError) as error:
            lll_audit.append(dict(scale=int(scale), status="failed", exception=f"{type(error).__name__}: {error}"))
    for j in range(count):
        for power in range(1, settings.coordinate_power_bound+1):
            row = [0]*count
            row[j] = power
            register(row, dict(kind="independent coordinate-power proposal", coordinate=j, power=power))
    rows = []
    for key, provenance in proposed.items():
        a = np.asarray(key, dtype=float)
        value = float(a@nu)
        residue = float(np.angle(np.exp(1j*value)))
        variance = float(a@covariance@a)
        deviation = float(np.sqrt(variance)) if variance > 0 else None
        z = abs(residue)/deviation if deviation is not None else None
        frequency_supported = bool(z is not None and np.isfinite(z) and z <= settings.conditional_z_bound)
        check = _score_angles(train_angles, validation_angles, np.asarray(key, dtype=int),
                              settings.observed_product_threshold) if frequency_supported else None
        eligible = frequency_supported and (not settings.observed_products_are_acceptance_gate
                                            or check["supported"])
        rows.append(dict(row=list(key), provenance=provenance, wrapped_frequency_residue=residue,
            initial_integer_winding=int(round(value/(2*np.pi))), conditional_standard_deviation=deviation,
            conditional_z=z if z is not None and np.isfinite(z) else None,
            conditional_frequency_screen_passed=frequency_supported, observed_product_check=check,
            observed_product_support_required=settings.observed_products_are_acceptance_gate,
            eligible_for_nested_refit=eligible, exact_relation_certified=False,
            gcd_divided_for_normalization=False))
    eligible = sorted([row for row in rows if row["eligible_for_nested_refit"]],
        key=lambda row:(row["conditional_z"], sum(abs(v) for v in row["row"]), tuple(row["row"])))
    return dict(all_proposals=rows, eligible=eligible, lll_audit=lll_audit,
        candidate_order="increasing conditional z, then l1, then lexicographic",
        covariance_screen_calibrated=False, phase_product_check_is_diagnostic=not settings.observed_products_are_acceptance_gate,
        bounded_search_complete=False)


def _initial_windings_consistent(source_rows, basis, frequencies):
    winding = [int(round(float(sum(int(a)*float(nu) for a,nu in zip(row, frequencies)))/(2*np.pi)))
               for row in basis.tolist()]
    checks = []
    for source in source_rows:
        coordinates, parameters = basis.T.gauss_jordan_solve(Matrix(source))
        if parameters.rows or any(not value.is_Integer for value in coordinates):
            raise ArithmeticError("source row not in the exact HNF integer row module")
        expected = int(round(float(sum(int(a)*float(nu) for a,nu in zip(source, frequencies)))/(2*np.pi)))
        represented = sum(int(a)*b for a,b in zip(coordinates, winding))
        checks.append(dict(source_row=list(source), source_winding=expected, represented_winding=represented,
                           consistent=expected == represented))
    return winding, all(check["consistent"] for check in checks), checks


def nested_lattice_plans(proposals, frequencies, settings=SETTINGS):
    count = len(frequencies)
    accepted_rows, plans, decisions = [], [], []
    previous = row_hnf([], count)
    for index, proposal in enumerate(proposals["eligible"]):
        candidate_rows = accepted_rows+[proposal["row"]]
        basis = row_hnf(candidate_rows, count)
        decision = dict(proposal_index=index, row=proposal["row"])
        if basis == previous:
            decision["status"] = "already_in_exact_integer_row_lattice"
            decisions.append(decision)
            continue
        winding, consistent, checks = _initial_windings_consistent(candidate_rows, basis, frequencies)
        if not consistent:
            decision.update(status="unresolved_inconsistent_initial_windings", winding_checks=checks)
            decisions.append(decision)
            continue
        if len(plans) >= settings.maximum_nested_models:
            decision["status"] = "nested_model_budget_exhausted"
            decisions.append(decision)
            break
        accepted_rows, previous = candidate_rows, basis
        plan = dict(model_index=len(plans)+1, relation_rows_HNF=[[int(v) for v in row] for row in basis.tolist()],
            source_relation_rows=[list(row) for row in accepted_rows], integer_windings=winding,
            winding_checks=checks, relation_rank=basis.rows,
            rank_increased=basis.rows > (plans[-1]["relation_rank"] if plans else 0),
            exact_row_lattice_changed=True, row_lattice_saturated=False,
            added_proposal=proposal)
        plans.append(plan)
        decision.update(status="new_unsaturated_lattice_plan", model_index=plan["model_index"])
        decisions.append(decision)
    return dict(plans=plans, decisions=decisions, every_eligible_relation_used=False,
        nested_model_budget_exhausted=any(d["status"] == "nested_model_budget_exhausted" for d in decisions))


def smith_presentation(relation_rows, dimension, windings, settings=SETTINGS):
    relation = row_hnf(relation_rows, dimension)
    rank = relation.rows
    if rank:
        domain = DomainMatrix.from_Matrix(relation).convert_to(ZZ)
        d_domain, u_domain, v_domain = smith_normal_decomp(domain)
        diagonal, u, v = d_domain.to_Matrix(), u_domain.to_Matrix(), v_domain.to_Matrix()
        for index in range(rank):
            if diagonal[index, index] < 0:
                diagonal.row_op(index, lambda value,_: -value)
                u.row_op(index, lambda value,_: -value)
    else:
        diagonal, u, v = Matrix.zeros(0, dimension), Matrix.eye(0), Matrix.eye(dimension)
    inverse = v.inv()
    if (u*relation*v != diagonal or abs(int(u.det())) != 1 or abs(int(v.det())) != 1
            or any(not value.is_Integer for value in inverse)
            or v*inverse != Matrix.eye(dimension)):
        raise ArithmeticError("Smith exact transforms or unimodularity failed")
    d = [int(diagonal[i, i]) for i in range(rank)]
    if any(value <= 0 for value in d) or any(d[i+1] % d[i] for i in range(rank-1)):
        raise ArithmeticError("invalid positive Smith invariant factors")
    finite_indices = [i for i, value in enumerate(d) if value > 1]
    finite_orders = [d[i] for i in finite_indices]
    component_count = 1
    for order in finite_orders:
        component_count *= order
    advance = u*Matrix(windings) if rank else Matrix.zeros(0, 1)
    finite_rotation_order = 1
    for index in finite_indices:
        finite_rotation_order = lcm(finite_rotation_order, d[index]//gcd(d[index], int(advance[index])))
    finite_self_domain_complete = finite_rotation_order == component_count
    convert = lambda matrix:[[int(value) for value in row] for row in matrix.tolist()]
    return dict(relation_rows_HNF=convert(relation), U=convert(u), V=convert(v), V_inverse=convert(inverse),
        smith_diagonal=d, relation_rank=rank, continuous_dimension=dimension-rank,
        finite_smith_indices=finite_indices, finite_orders=finite_orders, component_count=component_count,
        all_free_character_weights=convert(v[:, rank:]),
        all_finite_character_weights=[[int(v[j, i]) % d[i] for i in finite_indices] for j in range(dimension)],
        fixed_smith_rotation_steps=[int(advance[i]) % d[i] for i in range(rank)],
        finite_rotation_orbit_order=finite_rotation_order,
        finite_rotation_visits_all_presented_components=finite_self_domain_complete,
        finite_part_cyclic=len(finite_orders) <= 1,
        free_rotation_density_certified=False,
        coordinate_equations="U R V=D; ambient phi=V z, z=V_inverse phi",
        zero_origin_group_domain="D z in 2*pi*Z; z_i=2*pi*component_i/d_i for i<rank",
        every_finite_component_retained=True, lattice_saturated=False,
        observer_component_budget_available=component_count <= settings.maximum_observer_component_count,
        observer_status="eligible_for_later_integration" if component_count <= settings.maximum_observer_component_count
                        else "unresolved_component_count_exceeds_observer_budget",
        exact_algebra_verified=True, actual_observable_group_certified=False)


def _checked_integer_float(values, shape, name):
    flat = [int(value) for row in values for value in row] if len(shape) == 2 else [int(value) for value in values]
    floating = np.asarray(flat, dtype=float)
    if not np.isfinite(floating).all() or any(int(value) != exact for value,exact in zip(floating, flat)):
        raise ArithmeticError(name+" is not exactly representable in numeric equality solver")
    return floating.reshape(shape)


def fit_tied_lines(raw_train, times, initial_frequencies, relation_rows, windings, noise_sigma, settings=SETTINGS):
    y, times, initial = np.asarray(raw_train, dtype=float), np.asarray(times, dtype=float), np.asarray(initial_frequencies)
    q, n = y.shape
    count = len(initial)
    relation = _checked_integer_float(relation_rows, (len(relation_rows), count), "HNF coefficients")
    winding = _checked_integer_float(windings, (len(windings),), "integer windings")
    center, target = float(times.mean()), y.T
    tau = times-center

    def profile(frequencies):
        design = _design(tau, frequencies)
        coefficient, _, rank, _ = lstsq(design, target, cond=settings.design_rcond, lapack_driver="gelsy")
        if rank != design.shape[1]:
            raise np.linalg.LinAlgError("tied ledger full amplitude design rank deficient")
        residual = target-np.einsum("nj,jq->nq", design, coefficient, optimize=False)
        angles = tau[:, None]*frequencies[None, :]
        derivative = tau[:, None, None]*(
            -np.sin(angles)[:, :, None]*coefficient[1:1+count][None, :, :]
            +np.cos(angles)[:, :, None]*coefficient[1+count:][None, :, :])
        return design, coefficient, residual, derivative

    def objective(displacement):
        _, _, residual, derivative = profile(initial+displacement/n)
        gradient = -2*np.einsum("nq,njq->j", residual, derivative, optimize=False)/n
        return float(np.sum(residual**2))/(q*n), gradient/(q*n)

    width = 2*np.pi*settings.frequency_half_width_bins
    lower = np.maximum(-width, (settings.minimum_interior_frequency-initial)*n)
    upper = np.minimum(width, (np.pi-settings.minimum_interior_frequency-initial)*n)
    # Orthonormal row constraints improve SLSQP conditioning without changing
    # the exact integer lattice used for the group's presentation.
    if len(relation):
        basis, triangular = qr(relation.T, mode="economic")
        singular = np.linalg.svd(triangular, compute_uv=False)
        if singular[-1] <= settings.constraint_rank_rcond*singular[0]:
            raise np.linalg.LinAlgError("numerically unresolved exact relation equality basis")
        rhs = n*(2*np.pi*winding-relation@initial)
        transformed_rhs = solve_triangular(triangular.T, rhs, lower=True)
        constraints = LinearConstraint(basis.T, transformed_rhs, transformed_rhs)
        feasible_start = basis@transformed_rhs
    else:
        constraints, feasible_start = (), np.zeros(count)
    result = minimize(objective, np.clip(feasible_start, lower, upper), jac=True, method="SLSQP",
        bounds=list(zip(lower, upper)), constraints=constraints,
        options=dict(ftol=settings.optimizer_ftol, maxiter=settings.maximum_frequency_iterations))
    frequency = initial+result.x/n
    design, coefficient, residual, _ = profile(frequency)
    equality_residual = relation@frequency-2*np.pi*winding
    equality_max = float(np.max(abs(equality_residual))) if len(equality_residual) else 0.
    feasible = bool(equality_max <= settings.constraint_absolute_tolerance
                    and np.all(result.x >= lower-1e-8) and np.all(result.x <= upper+1e-8))
    profiles_center = .5*(coefficient[1:1+count]-1j*coefficient[1+count:])
    profiles_origin = profiles_center*np.exp(-1j*frequency*center)[:, None]
    audit = dict(optimizer_success=bool(result.success), optimizer_message=str(result.message),
        iterations=int(result.nit), function_evaluations=int(result.nfev), equality_feasible=feasible,
        equality_max_absolute_residual=equality_max,
        frequency_search_half_width=width/n, raw_train_sensor_mse=float(np.mean(residual**2)),
        source_noise_sigma_raw=float(noise_sigma), fit_objective="unpenalized RAW sensor RSS/(q*N)",
        all_observed_line_coefficients_retained=True, full_temporal_gram=True,
        information_or_postselection_confidence_calibrated=False,
        exact_integer_equalities_float_coefficients_checked=True,
        numeric_equalities_only_machine_precision_not_exact_real_arithmetic=True,
        phase_domain_gauge="origin phi_n=nu*n; coefficients_origin=c_center*exp(-i*nu*time_center)",
        centered_phase_used_as_zero_group_domain=False)
    return dict(frequencies=frequency, coefficients_sensor_center=coefficient, time_center=np.asarray(center),
        complex_profiles_sensor_origin=profiles_origin, temporal_gram=design.T@design, audit=audit)


def _prediction(model, times):
    design = _design(np.asarray(times)-float(model["time_center"]), model["frequencies"])
    return np.einsum("nj,jq->qn", design, model["coefficients_sensor_center"], optimize=False)


def select_ledger_group(raw_train, raw_validation, t_train, t_validation, parameters, noise_sigma, settings=SETTINGS):
    nu = np.asarray(parameters["frequencies"], dtype=float)
    proposals = propose_relations(nu, parameters["conditional_frequency_covariance"],
        parameters["observed_train_characters"], parameters["observed_validation_characters"], settings)
    nested = nested_lattice_plans(proposals, nu, settings)
    baseline = {key:np.asarray(parameters[key]).copy() for key in
                ("frequencies", "coefficients_sensor_center", "time_center", "complex_profiles_sensor_origin", "temporal_gram")}
    baseline["presentation"] = smith_presentation([], len(nu), [], settings)
    baseline["model_kind"] = "frozen_free_ledger_baseline"
    baseline["validation_raw_sensor_mse"] = float(np.mean((raw_validation-_prediction(baseline, t_validation))**2))
    models = [baseline]
    attempts = [dict(model_index=0, status="frozen_full_free_baseline", relation_rows_HNF=[],
        validation_raw_sensor_mse=baseline["validation_raw_sensor_mse"], presentation=baseline["presentation"])]
    best = 0
    # A matched unconstrained refinement uses the same final-ledger initializers,
    # one-bin cells, SLSQP iterations and objective as all relation-constrained
    # candidates. Keep the untouched baseline as an additional comparator.
    free_attempt = dict(candidate_id="matched_unconstrained_refit", relation_rows_HNF=[])
    try:
        free = fit_tied_lines(raw_train, t_train, nu, [], [], noise_sigma, settings)
        free_attempt["fit_audit"] = free["audit"]
        if not free["audit"]["optimizer_success"] or not free["audit"]["equality_feasible"]:
            free_attempt["status"] = "unresolved_or_unconverged_free_refit"
        else:
            free["presentation"] = baseline["presentation"]
            free["model_kind"] = "matched_unconstrained_refit"
            free["validation_raw_sensor_mse"] = float(np.mean((raw_validation-_prediction(free, t_validation))**2))
            models.append(free)
            free_attempt.update(status="admissible_free_refit", stored_model_index=len(models)-1,
                                validation_raw_sensor_mse=free["validation_raw_sensor_mse"])
            if free["validation_raw_sensor_mse"] < models[best]["validation_raw_sensor_mse"]:
                best = len(models)-1
    except (ValueError, np.linalg.LinAlgError, ArithmeticError, RuntimeError) as error:
        free_attempt.update(status="unresolved_free_refit_exception", exception=f"{type(error).__name__}: {error}")
    attempts.append(free_attempt)
    for plan in nested["plans"]:
        attempt = dict(plan)
        try:
            presentation = smith_presentation(plan["relation_rows_HNF"], len(nu), plan["integer_windings"], settings)
            attempt["presentation"] = presentation
            if not presentation["finite_rotation_visits_all_presented_components"]:
                attempt["status"] = "rejected_incomplete_self_domain_finite_orbit"
                attempts.append(attempt)
                continue
            fitted = fit_tied_lines(raw_train, t_train, nu, plan["relation_rows_HNF"],
                                    plan["integer_windings"], noise_sigma, settings)
            attempt["fit_audit"] = fitted["audit"]
            if not fitted["audit"]["optimizer_success"] or not fitted["audit"]["equality_feasible"]:
                attempt["status"] = "unresolved_or_unconverged_equality_fit"
            else:
                fitted["presentation"] = presentation
                fitted["model_kind"] = "relation_constrained_ledger"
                fitted["validation_raw_sensor_mse"] = float(np.mean((raw_validation-_prediction(fitted, t_validation))**2))
                fitted["plan_index"] = plan["model_index"]
                models.append(fitted)
                attempt.update(status="admissible_tied_model", stored_model_index=len(models)-1,
                               validation_raw_sensor_mse=fitted["validation_raw_sensor_mse"])
                if fitted["validation_raw_sensor_mse"] < models[best]["validation_raw_sensor_mse"]:
                    best = len(models)-1
        except (ValueError, np.linalg.LinAlgError, ArithmeticError, RuntimeError) as error:
            attempt.update(status="unresolved_model_exception", exception=f"{type(error).__name__}: {error}")
        attempts.append(attempt)
    return dict(models=models, selected_model_index=best,
        audit=dict(settings=asdict(settings), observed_line_count=len(nu), proposals=proposals,
            lattice_plans=nested, all_model_attempts=attempts, selected_model_index=best,
            selected_validation_raw_sensor_mse=models[best]["validation_raw_sensor_mse"],
            selected_presentation=models[best]["presentation"],
            selected_model_kind=models[best]["model_kind"],
            same_search_cell_objective_optimizer_free_comparator_included=True,
            selection_rule="strict minimum RAW sensor VAL MSE; exact ties retain earlier/free model",
            observed_product_checks_are_diagnostic=not settings.observed_products_are_acceptance_gate,
            unknown_true_relation_lattice_or_phase_rank_supplied=False,
            previous_harmonic_labels_reused=False, conditional_z_is_not_confidence_certificate=True,
            exact_group_or_relation_completeness_certified=False,
            finite_near_relations_can_pass_and_win_validation=True,
            unvisited_finite_components_candidates_rejected=True,
            exact_free_angle_density_not_certified=True,
            nearrelation_uncertainty="all candidate residue/z/product diagnostics retained; finite data do not prove exact rationality",
            no_frequency_split_or_new_line_discovery=True,
            finite_group_model_does_not_certify_missing_undetected_phases=True,
            raw_profiles_origin_gauge_preserves_all_finite_components=True,
            novelty_claimed=False))


def parameter_arrays(model):
    arrays = {key:np.asarray(model[key]) for key in ("frequencies", "coefficients_sensor_center", "time_center",
                                                   "complex_profiles_sensor_origin", "temporal_gram")}
    presentation = model["presentation"]
    for key in ("relation_rows_HNF", "U", "V", "V_inverse", "all_free_character_weights", "all_finite_character_weights"):
        values = presentation[key]
        # Unicode decimal strings preserve arbitrary-size integers without pickle.
        shape = (len(values), len(values[0]) if values else (len(model["frequencies"]) if key == "relation_rows_HNF" else 0))
        arrays[key+"_integer_text"] = np.asarray([[str(v) for v in row] for row in values], dtype=str).reshape(shape)
    arrays["smith_diagonal_integer_text"] = np.asarray([str(v) for v in presentation["smith_diagonal"]], dtype=str)
    arrays["finite_smith_indices"] = np.asarray(presentation["finite_smith_indices"], dtype=np.int64)
    arrays["finite_orders_integer_text"] = np.asarray([str(v) for v in presentation["finite_orders"]], dtype=str)
    arrays["fixed_smith_rotation_steps_integer_text"] = np.asarray([str(v) for v in presentation["fixed_smith_rotation_steps"]], dtype=str)
    arrays["continuous_dimension"] = np.asarray(presentation["continuous_dimension"], dtype=np.int64)
    for key, value in arrays.items():
        if value.dtype.kind not in "US" and not np.isfinite(value).all():
            raise FloatingPointError("nonfinite group parameter: "+key)
    return arrays
