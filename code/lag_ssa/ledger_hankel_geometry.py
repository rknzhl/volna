"""Bounded, model-only reach upper diagnostics for frozen ledger embeddings.

Federer (1959), Curvature Measures, Theorem 4.18, p.449:
reach(M) <= ||y-x||^2/(2 dist(y-x,T_x M)) for any distinct pair.
The formula is evaluated in the FULL normalized mass-window bank, including
all finite-window cross terms. Sampling supplies upper candidates only.
No true phase, evaluator, catalogue, fit or model selection is an input.
"""

from dataclasses import asdict, dataclass
from math import lcm
import heapq
import random

import numpy as np
from scipy.spatial import cKDTree
from sympy import Matrix, Rational

from torus_lattice_reach_witness import projected_lattice_chords


MATH_REFERENCE = dict(author="Herbert Federer", title="Curvature Measures", year=1959,
    theorem="4.18", page=449,
    url="https://users.math.msu.edu/users/iwenmark/Teaching/MTH995/Papers/MMod_Federer.pdf",
    formula="reach(M)=inf_{x!=y} ||y-x||^2/(2||P_Nx(y-x)||)",
    verification="primary article theorem retrieved by web search before source freeze",
    application="compact C2 embedded group image; numeric tangents are conditional, not interval certified")


@dataclass(frozen=True)
class GeometrySettings:
    window: int = 128
    bank_rank_rcond: float = 1e-10
    tangent_rank_rcond: float = 1e-10
    relation_rank_rcond: float = 1e-12
    one_window_alpha: float = .01
    model_tail_bound: float = 0.
    random_seed: int = 870211
    generic_states: int = 4096
    finite_states: int = 65536
    continuous_binary_denominator: int = 2**53
    circle_full_enumeration_cap: int = 2000000
    circle_sample_count: int = 65536
    circle_excluded_wrapped_delta: float = .1
    circle_retained_deltas: int = 32
    circle_base_points: int = 16
    circle_ranking_chunk: int = 16384
    curvature_base_points: int = 16
    curvature_random_directions: int = 16
    lattice_base_points: int = 16
    pair_batch: int = 256
    normal_relative_floor: float = 1e-12


SETTINGS = GeometrySettings()


def _integer_matrix(values, rows, columns):
    return Matrix(rows, columns, [int(x) for row in values for x in row])


def _exact_presentation(presentation, count):
    rank = presentation["relation_rank"]
    relation = _integer_matrix(presentation["relation_rows_HNF"], rank, count)
    u = _integer_matrix(presentation["U"], rank, rank)
    v = _integer_matrix(presentation["V"], count, count)
    inverse = _integer_matrix(presentation["V_inverse"], count, count)
    diagonal = Matrix.zeros(rank, count)
    for i, value in enumerate(presentation["smith_diagonal"]):
        diagonal[i, i] = int(value)
    if (u*relation*v != diagonal or v*inverse != Matrix.eye(count)
            or abs(int(v.det())) != 1 or abs(int(u.det())) != 1):
        raise ValueError("frozen exact Smith transforms are inconsistent")
    return relation, v


def _fractions_to_phase(numerators, denominator):
    flat = [float(int(value)/denominator) for value in np.asarray(numerators, dtype=object).flat]
    return 2*np.pi*np.asarray(flat).reshape(np.shape(numerators))


def _distinct_codes(order, maximum, seed):
    """Distinct finite states, arbitrary-size Python integer order."""
    size = min(int(order), maximum)
    if order <= maximum:
        return list(range(int(order))), True
    rng = random.Random(seed)
    result = set()
    while len(result) < size:
        result.add(rng.randrange(int(order)))
    return sorted(result), False


def rational_group_states(presentation, size, seed, binary_denominator):
    """Uniform finite coordinates and standard 53-bit Haar approximation.

    Exact integer modular products avoid multiplying huge SNF weights by
    floating angles. Only the final bounded fraction is converted to double.
    """
    v = np.asarray(presentation["V"], dtype=object)
    count = len(v)
    rank, free = presentation["relation_rank"], presentation["continuous_dimension"]
    rng = np.random.default_rng(seed)
    finite_rng = random.Random(seed+1)
    orders = [int(presentation["smith_diagonal"][i]) for i in range(rank)]
    denominator = lcm(binary_denominator, *orders) if free else lcm(1, *orders)
    numerator = np.zeros((size, count), dtype=object)
    free_codes = rng.integers(0, binary_denominator, (size, free), dtype=np.int64)
    if free:
        reduced = np.asarray([[int(x) % binary_denominator for x in row]
                              for row in v[:, rank:].tolist()], dtype=object)
        numerator += (free_codes.astype(object)@reduced.T % binary_denominator)*(denominator//binary_denominator)
    finite_codes = []
    for i, order in enumerate(orders):
        codes = [finite_rng.randrange(order) if order > 1 else 0 for _ in range(size)]
        finite_codes.append(codes)
        if order > 1:
            weights = np.asarray([int(x) % order for x in v[:, i]], dtype=object)
            numerator += (np.asarray(codes, dtype=object)[:, None]*weights[None, :] % order)*(denominator//order)
    numerator %= denominator
    return _fractions_to_phase(numerator, denominator), dict(
        state_sampling="uniform finite factors, fixed-seed binary-rational Haar approximation",
        exact_modular_integer_products=True, continuous_denominator=binary_denominator,
        common_denominator=str(denominator), sampled_state_count=size,
        finite_coordinate_order_product=int(presentation["component_count"]),
        no_claim_sampled_net_complete=True)


def finite_group_states(presentation, maximum, seed):
    indices, orders = presentation["finite_smith_indices"], presentation["finite_orders"]
    if len(indices) > 1:
        raise ValueError("finite-only sampler requires the presented cyclic finite group")
    order = int(orders[0]) if orders else 1
    codes, complete = _distinct_codes(order, maximum, seed)
    count = len(presentation["V"])
    if not indices:
        phase = np.zeros((1, count))
    else:
        weights = np.asarray([int(row[indices[0]]) % order for row in presentation["V"]], dtype=object)
        residues = np.asarray(codes, dtype=object)[:, None]*weights[None, :] % order
        phase = _fractions_to_phase(residues, order)
    return phase, dict(sampled_state_count=len(codes), finite_group_order=str(order),
        distinct_finite_states=True, all_finite_states_enumerated=complete,
        exact_modular_integer_products=True, fixed_seed=seed)


def lattice_pair_proposals(presentation, settings=SETTINGS):
    """Exact identity-component proposals; algebra failures remain unresolved."""
    count = len(presentation["V"])
    try:
        audit = projected_lattice_chords(presentation["relation_rows_HNF"], count)
        points = audit["points"]
        if not points:
            return [], audit
        rank, free = presentation["relation_rank"], presentation["continuous_dimension"]
        denominator = settings.continuous_binary_denominator
        rng = np.random.default_rng(settings.random_seed+37)
        codes = rng.integers(0, denominator, (settings.lattice_base_points, free), dtype=np.int64)
        codes[0] = 0
        v = presentation["V"]
        pairs = []
        for code in codes:
            base = [Rational(sum(int(v[j][rank+i])*int(code[i]) for i in range(free)) % denominator,
                             denominator) for j in range(count)]
            for point in points:
                turns = [Rational(value) for value in point["wrapped_phase_turns"]]
                endpoint = [(first+second) % 1 for first,second in zip(base, turns)]
                pairs.append(dict(phase_first=[2*np.pi*float(value) for value in base],
                                  phase_second=[2*np.pi*float(value) for value in endpoint]))
        audit.update(base_points=settings.lattice_base_points, origin_included=True,
            pair_count=len(pairs), phase_addition_and_modulo_exact_before_float=True,
            base_sampling="fixed-seed binary-rational identity-component coordinates",
            continuous_denominator=denominator)
        return pairs, audit
    except Exception as error:
        return [], dict(status="exact_lattice_proposals_unavailable",
                        exception_type=type(error).__name__, exception=str(error),
                        algebra_failure_does_not_reject_or_accept_model=True)


def harmonic_bank(parameters, presentation, sigma, settings=SETTINGS):
    nu = np.asarray(parameters["frequencies"], dtype=float)
    profile = np.asarray(parameters["complex_profiles_sensor_origin"], dtype=complex)
    mass, root = np.asarray(parameters["physical_mass"]), np.asarray(parameters["spatial_mass_sqrt"])
    count, sensors = profile.shape
    if (nu.shape != (count,) or root.shape != (sensors, sensors) or mass.shape != root.shape
            or not np.isfinite(sigma) or sigma <= 0):
        raise ValueError("matching frozen profiles/frequencies/mass and positive estimated raw sigma required")
    relation, _ = _exact_presentation(presentation, count)
    mass_profile = profile@root.T
    complex_bank = (2/np.sqrt(settings.window))*np.exp(1j*np.arange(settings.window)[None, :, None]*nu[:, None, None])*mass_profile[:, None, :]
    positive = complex_bank.transpose(1, 2, 0).reshape(settings.window*sensors, count)
    bank = np.column_stack((positive.real, -positive.imag))
    q, singular, vt = np.linalg.svd(bank, full_matrices=False)
    thin = singular[:, None]*vt
    bank_rank = int(np.sum(singular > settings.bank_rank_rcond*singular[0])) if len(singular) else 0
    q_blocks = q.reshape(settings.window, sensors, len(singular))
    covariance = (sigma**2/settings.window)*np.einsum("lqi,qk,lkj->ij", q_blocks, mass, q_blocks, optimize=False)
    covariance = (covariance+covariance.T)/2
    eigenvalues = np.linalg.eigvalsh(covariance)
    radius = np.sqrt(max(float(np.trace(covariance)), 0.))+np.sqrt(
        2*max(float(eigenvalues[-1]), 0.)*np.log(1/settings.one_window_alpha))
    if relation.rows:
        values = np.asarray(relation.tolist(), dtype=float)
        if not np.isfinite(values).all() or any(int(x) != int(y) for x,y in zip(values.flat, relation)):
            raise ValueError("relation coefficients not exactly representable in tangent solver")
        _, relation_singular, relation_vt = np.linalg.svd(values, full_matrices=True)
        resolved_rank = int(np.sum(relation_singular > settings.relation_rank_rcond*relation_singular[0]))
        tangent = relation_vt[relation.rows:].T
        relation_residual = float(np.linalg.norm(values@tangent))
    else:
        resolved_rank, relation_residual = 0, 0.
        tangent = np.eye(count)
    profile_nonzero = bool(np.all(np.linalg.norm(mass_profile, axis=1) > 0))
    signed = np.r_[nu, -nu]
    gaps = abs(np.angle(np.exp(1j*(signed[:, None]-signed[None, :]))))
    np.fill_diagonal(gaps, np.inf)
    distinct = bool(np.min(gaps) > 1e-12 and np.all(abs(np.angle(np.exp(1j*signed))) > 1e-12))
    theoretical = bool(settings.window >= 2*count and profile_nonzero and distinct)
    resolved = bool(bank_rank == 2*count and resolved_rank == relation.rows)
    info = dict(real_bank_columns=2*count, bank_ambient_rows=bank.shape[0], uniform_lags=settings.window,
        bank_row_order="lag-major/sensor-minor", bank_column_order="cos(allJ),sin(allJ)",
        bank_singular_values=singular.tolist(), bank_numerical_rank=bank_rank,
        all_columns_kept_in_thin_bank=True, no_svd_truncation_used_for_distances=True,
        svd_reconstruction_relative_defect=float(np.linalg.norm(bank-q@thin)/max(np.linalg.norm(bank), 1e-30)),
        full_gram_thin_relative_defect=float(np.linalg.norm(bank.T@bank-thin.T@thin)/max(np.linalg.norm(bank.T@bank), 1e-30)),
        mass_root_relative_defect=float(np.linalg.norm(root@root.T-mass)/max(np.linalg.norm(mass), 1e-30)),
        original_coordinate_characters_generate_quotient_exact=True, exact_Smith_transform_checked=True,
        theoretical_vandermonde_embedding_conditions=theoretical,
        vandermonde_scope="distinct nonzero signed temporal frequencies, nonzero profiles, L>=2J",
        numerical_embedding_witness_available=resolved,
        relation_numerical_rank=resolved_rank, relation_exact_rank=relation.rows,
        tangent_dimension=tangent.shape[1], orthonormal_phase_tangent_used=True,
        giant_Smith_weights_not_used_for_floating_tangent= True, tangent_relation_residual=relation_residual,
        covariance_trace=float(np.trace(covariance)), covariance_maximum_eigenvalue=float(eigenvalues[-1]),
        one_untouched_window_alpha=settings.one_window_alpha, projected_noise_radius=float(radius),
        estimated_raw_sigma=float(sigma), estimated_sigma_calibrated=False,
        noise_covariance="sigma_train^2/L Q.T (I_L tensor M_s) Q",
        model_tail_bound=settings.model_tail_bound, model_tail_bound_is_optimistic_assumption=True,
        true_domain_or_true_field_bound_available=False)
    return thin, tangent, 2*np.sum(abs(mass_profile)**2, axis=1), info


def rho(phase):
    return np.column_stack((np.cos(phase), np.sin(phase)))


def _tangent_bases(thin, tangent, phase, settings):
    count, dimension = phase.shape[1], tangent.shape[1]
    if not dimension:
        return np.empty((len(phase), len(thin), 0)), np.ones(len(phase), dtype=bool)
    derivative = np.concatenate((-np.sin(phase)[:, :, None]*tangent[None, :, :],
                                  np.cos(phase)[:, :, None]*tangent[None, :, :]), axis=1)
    jacobian = np.einsum("dk,nkr->ndr", thin, derivative, optimize=False)
    u, singular, _ = np.linalg.svd(jacobian, full_matrices=False)
    valid = singular[:, -1] > settings.tangent_rank_rcond*singular[:, 0]
    return u, valid


def pair_bounds(thin, tangent, first, second, settings=SETTINGS):
    """Any valid pair is an upper candidate; no double-normal condition needed."""
    best = None
    for start in range(0, len(first), settings.pair_batch):
        x, y = first[start:start+settings.pair_batch], second[start:start+settings.pair_batch]
        chord = (rho(y)-rho(x))@thin.T
        tangent_bases, valid = _tangent_bases(thin, tangent, x, settings)
        projection = np.einsum("ndr,nd->nr", tangent_bases, chord, optimize=False)
        normal = chord-np.einsum("ndr,nr->nd", tangent_bases, projection, optimize=False)
        square = np.sum(chord**2, axis=1)
        normal_norm = np.linalg.norm(normal, axis=1)
        usable = valid & (square > 0) & (normal_norm > settings.normal_relative_floor*np.sqrt(square))
        for index in np.flatnonzero(usable):
            upper = float(square[index]/(2*normal_norm[index]))
            if best is not None and upper >= best["upper"]:
                continue
            best = dict(upper=upper,
                chord_distance=float(np.sqrt(square[index])), normal_distance=float(normal_norm[index]),
                phase_first=x[index].tolist(), phase_second=y[index].tolist(),
                source="Federer chord/tangent pair", global_minimum_certified=False)
    return best


def circle_witness(thin, tangent, energy, presentation, settings=SETTINGS):
    weights = [int(row[0]) for row in presentation["all_free_character_weights"]]
    active = [i for i,k in enumerate(weights) if k]
    strongest = max(active, key=lambda i:energy[i])
    order = abs(weights[strongest])
    if order < 2:
        return None, dict(status="strongest_character_has_no_nontrivial_deck_deltas", strongest_weight=order)
    codes, complete = _distinct_codes(order, settings.circle_full_enumeration_cap if order <= settings.circle_full_enumeration_cap
        else settings.circle_sample_count, settings.random_seed+11)
    heap = []
    considered = 0
    for start in range(0, len(codes), settings.circle_ranking_chunk):
        h = [value for value in codes[start:start+settings.circle_ranking_chunk]
             if 2*np.pi*float(min(value, order-value)/order) >= settings.circle_excluded_wrapped_delta]
        if not h:
            continue
        residues = np.asarray(h, dtype=object)[:, None]*np.asarray(weights, dtype=object)[None, :] % order
        angle = _fractions_to_phase(residues, order)
        score = (2-2*np.cos(angle))@energy
        considered += len(h)
        take = np.argsort(score, kind="stable")[:settings.circle_retained_deltas]
        for index in take:
            item = (-float(score[index]), -int(h[index]))
            if len(heap) < settings.circle_retained_deltas:
                heapq.heappush(heap, item)
            elif item > heap[0]:
                heapq.heapreplace(heap, item)
    chosen = sorted([(-score, -h) for score,h in heap])
    first, second = [], []
    bases = settings.circle_base_points
    denominator = bases*order
    for _, h in chosen:
        for base in range(bases):
            first.append([2*np.pi*float(((k*base*order) % denominator)/denominator) for k in weights])
            second.append([2*np.pi*float(((k*(base*order+bases*h)) % denominator)/denominator) for k in weights])
    witness = pair_bounds(thin, tangent, np.asarray(first), np.asarray(second), settings) if first else None
    return witness, dict(status="bounded_deck_delta_search", strongest_line_index=strongest,
        strongest_integer_weight=str(weights[strongest]), deck_denominator=str(order),
        all_h_enumerated=complete, nonlocal_deltas_ranked=considered, retained_deltas=len(chosen),
        base_points=bases, exact_modular_phase_arithmetic=True,
        ranking="Haar average sum_j 2||c_j||_Ms^2 |exp(i k_j delta)-1|^2; heuristic ordering only",
        finite_window_cross_terms_in_final_formula=True)


def curvature_witness(thin, tangent, phase, settings=SETTINGS):
    dimension = tangent.shape[1]
    if not dimension:
        return None
    bases = phase[:settings.curvature_base_points]
    tangent_bases, valid = _tangent_bases(thin, tangent, bases, settings)
    rng = np.random.default_rng(settings.random_seed+23)
    directions = np.vstack((np.eye(dimension), rng.standard_normal((settings.curvature_random_directions, dimension))))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    answers = []
    for index, x in enumerate(bases):
        if not valid[index]:
            continue
        velocity = directions@tangent.T
        first = np.column_stack((-np.sin(x)[None, :]*velocity, np.cos(x)[None, :]*velocity))@thin.T
        second = np.column_stack((-np.cos(x)[None, :]*velocity**2, -np.sin(x)[None, :]*velocity**2))@thin.T
        normal = second-(second@tangent_bases[index])@tangent_bases[index].T
        numerator, denominator = np.sum(first**2, axis=1), np.linalg.norm(normal, axis=1)
        for j in np.flatnonzero((numerator > 0) & (denominator > settings.normal_relative_floor*numerator)):
            answers.append(dict(upper=float(numerator[j]/denominator[j]), source="analytic directional normal-curvature limit",
                phase_first=x.tolist(), phase_direction=velocity[j].tolist(),
                global_maximum_curvature_certified=False))
    return min(answers, key=lambda item:item["upper"]) if answers else None


def diagnose_model(parameters, presentation, sigma, settings=SETTINGS, extra_pairs=None):
    thin, tangent, energy, bank = harmonic_bank(parameters, presentation, sigma, settings)
    dimension = presentation["continuous_dimension"]
    witnesses, methods = [], {}
    if dimension == 0:
        phase, methods["sampling"] = finite_group_states(presentation, settings.finite_states, settings.random_seed)
    else:
        phase, methods["sampling"] = rational_group_states(presentation, settings.generic_states,
            settings.random_seed, settings.continuous_binary_denominator)
    if len(phase) >= 2:
        points = rho(phase)@thin.T
        _, neighbour = cKDTree(points).query(points, k=2, workers=1)
        pair = pair_bounds(thin, tangent, phase, phase[neighbour[:, 1]], settings)
        if pair:
            witnesses.append(pair)
        methods["kdtree"] = dict(sampled_points=len(phase), directed_nearest_pairs=len(phase),
            zero_dimensional_tangent=dimension == 0, arbitrary_half_chord_used_only_for_0D=True,
            sampled_pairs_cannot_establish_reach_lower_bound=True)
    if dimension:
        curvature = curvature_witness(thin, tangent, phase, settings)
        if curvature:
            witnesses.append(curvature)
    if dimension == 1:
        circle, methods["circle_decks"] = circle_witness(thin, tangent, energy, presentation, settings)
        if circle:
            witnesses.append(circle)
    if extra_pairs:
        first, second = [], []
        for pair in extra_pairs:
            first.append(pair["phase_first"]);second.append(pair["phase_second"])
        extra = pair_bounds(thin, tangent, np.asarray(first), np.asarray(second), settings)
        if extra:
            extra["source"] = "Federer pair from exact projected-integer-lattice proposal"
            witnesses.append(extra)
        methods["extra_exact_lattice_pairs"] = dict(pair_count=len(extra_pairs), proposals_do_not_fit_or_select_model=True)
    best = min(witnesses, key=lambda item:item["upper"]) if witnesses else None
    upper = best["upper"] if best else None
    radius = bank["projected_noise_radius"]+settings.model_tail_bound
    witness_resolved = bank["numerical_embedding_witness_available"] and bank["theoretical_vandermonde_embedding_conditions"]
    rejection = bool(witness_resolved and upper is not None and upper <= radius)
    return dict(settings=asdict(settings), bank=bank, methods=methods, best_witness=best,
        reach_upper_bound_numerical=upper, reach_upper_to_projected_noise_ratio=upper/radius if upper is not None and radius else None,
        model_only_noise_scale_rejection_numerical=rejection,
        geometry_status="conditional_model_only_rejection_witness" if rejection else
            "bounded_search_no_rejection_found" if witness_resolved else "unresolved_numerical_embedding_witness",
        rejection_interpretation="cannot satisfy declared reach>noise sufficient condition, conditional on embedded C2 template and accurate floating arithmetic",
        reach_lower_bound_or_global_minimum_established=False, no_positive_acceptance_certificate=True,
        rejection_is_not_general_necessity_for_phase_recovery=True,
        nonsmooth_or_zero_reach_maps_may_still_have_stable_inverse=True,
        true_group_or_true_r_inferred=False, true_field_uniform_error_certified=False,
        no_evaluation_or_catalogue_used=True, no_model_refit_or_selection_retuning=True,
        validated_floating_point_certificate=False, math_reference=MATH_REFERENCE)
