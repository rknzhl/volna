"""Fields-first finite-complex periods. No initial circle coordinates are computed.

Public arrays use states by ambient features, edges by coframe components.
prepare_period_complex extracts integer COCHAINS ONLY after the caller's W fit.
fit_periods projects caller-supplied beta integrals onto closed cochains and
fits the period matrix and integrated real potentials in a single linear LS.
Certificates concern the supplied finite complex, never the true manifold.
"""
from dataclasses import dataclass, replace
import numpy as np
from scipy import linalg, sparse
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse.linalg import lsmr
from scipy.spatial.distance import cdist
from dreimac import ToroidalCoords
from dreimac.utils import CohomologyUtils
from sklearn.neighbors import NearestNeighbors

# Local copies of the four metric/finite-complex utilities keep this module
# independent of the angle-producing geometric frontend. Geometry can select
# the dimension; the common standard range and exact triangle checks remain.
from dreimac.combinatorial import combinatorial_number_system_d1_forward

if __package__:
    from .cohomology_primitivity import certify_primitivity
else:
    from cohomology_primitivity import certify_primitivity


@dataclass(frozen=True)
class _PeriodSettings:
    n_landmarks: int = 192
    prime: int = 47
    cover_percent: float = 0.5
    lifetime_gap: float = 2.0
    rank_selection: str = "largest lifetime gap among common-standard-range prefixes"


SETTINGS = _PeriodSettings()


def _distances(left, right=None):
    """Columns are states; the full Euclidean Hankel metric is retained."""
    left = np.asarray(left, dtype=float)
    right = left if right is None else np.asarray(right, dtype=float)
    if (left.ndim != 2 or right.ndim != 2 or left.shape[0] != right.shape[0]
            or not np.isfinite(left).all() or not np.isfinite(right).all()):
        raise ValueError("finite feature-by-state arrays of matching dimension required")
    ll = np.einsum("ij,ij->j", left, left)
    rr = np.einsum("ij,ij->j", right, right)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        product = left.T @ right
    squared = ll[:, None] + rr[None, :] - 2 * product
    if not np.isfinite(squared).all():
        raise FloatingPointError("nonfinite full-Hankel distances")
    tolerance = 128 * np.finfo(float).eps * max(1.0, float(ll.max()), float(rr.max()))
    if squared.min() < -tolerance:
        raise FloatingPointError("invalid Gram distance beyond roundoff")
    result = np.sqrt(np.maximum(squared, 0.0))
    if right is left:
        np.fill_diagonal(result, 0.0)
        result = (result + result.T) / 2
    return result

def _select_prefix(diagram, settings, preferred_count=None):
    """Select a common standard range, with no topological rank promise.

    An independently estimated geometry/frame dimension fixes the prefix size;
    its lifetime gap is reported but need not exceed the legacy gap threshold.
    Without that estimate, retain the largest eligible lifetime-gap rule.
    """
    dgm = np.asarray(diagram, dtype=float)
    if not len(dgm) or not np.isfinite(dgm).all():
        return None, dict(reason="no_finite_H1_diagram")
    if preferred_count is not None:
        if isinstance(preferred_count, bool) or int(preferred_count) != preferred_count or preferred_count < 1:
            raise ValueError('positive preferred dimension required')
        preferred_count = int(preferred_count)
        if preferred_count > len(dgm):
            return None, dict(reason='fewer_persistent_classes_than_preferred_dimension',
                              preferred_count=preferred_count)
    lives = dgm[:, 1] - dgm[:, 0]
    candidates = []
    counts = (preferred_count,) if preferred_count is not None else range(1, len(dgm) + 1)
    for count in counts:
        birth = float(np.max(dgm[:count, 0]))
        death = float(np.min(dgm[:count, 1]))
        if not 2 * birth < death:
            continue
        # Include all discarded bars when measuring the gap, not just eligible ones.
        denominator = float(lives[count]) if count < len(lives) else 0.0
        ratio = float(lives[count - 1] / denominator) if denominator > 0 else float("inf")
        if preferred_count is not None or ratio >= settings.lifetime_gap:
            candidates.append((ratio, -count, count, birth, death))
    if not candidates:
        if preferred_count is not None:
            return None, dict(reason='preferred_dimension_has_no_common_standard_range',
                              preferred_count=preferred_count)
        return None, dict(reason="no_declared_persistence_gap_with_common_standard_range")
    ratio, _, count, birth, death = max(candidates)
    return count, dict(selected_count=count, lifetime_gap=None if np.isinf(ratio) else ratio,
                       common_birth=birth, common_death=death,
                       rank_certified_for_true_manifold=False,
                       lifetime_gap_required=preferred_count is None,
                       criterion=settings.rank_selection if preferred_count is None else
                           'independent dimension with common standard range')


def _select_scale(diagram, count, selection, landmark_to_data, metric):
    """Select a standard scale with exactly count classes and a TRAIN buffer.

    If the TRAIN fill distance is eps and the selected radius is R, every
    point within R-eps of TRAIN is covered. Future states need not lie in
    that neighbourhood; this is a finite coverage bound, not a global claim.
    """
    eps=float(np.max(np.min(landmark_to_data,axis=0)))
    neighbour=np.min(metric,axis=0,where=metric>0,initial=np.inf)
    neighbour=neighbour[np.isfinite(neighbour)]
    report=dict(train_fill_distance=eps,scale_selection=
        'midpoint of widest standard interval with exactly r active classes and TRAIN buffer')
    if not len(neighbour):
        return None,report|dict(reason='no_positive_training_sampling_distance')
    delta=.5*float(np.median(neighbour))
    birth,death=selection['common_birth'],selection['common_death']
    lower=max(2*birth,2*(eps+delta))
    report.update(training_sampling_buffer=delta,standard_scale_interval=[2*birth,death],
        allowed_scale_interval=[lower,death])
    endpoints=np.unique(np.r_[lower,death,np.asarray(diagram).ravel()])
    endpoints=endpoints[(endpoints>=lower)&(endpoints<=death)]
    candidates=[]
    for left,right in zip(endpoints[:-1],endpoints[1:]):
        middle=(left+right)/2
        active=int(np.sum((diagram[:,0]<middle)&(diagram[:,1]>middle)))
        if right>left and active==count:
            candidates.append((right-left,middle,float(left),float(right)))
    if not candidates:
        return None,report|dict(reason='no_standard_scale_with_exact_rank_and_training_cover_buffer')
    _,threshold,left,right=max(candidates)
    report.update(selected_scale_interval=[left,right],rips_threshold=float(threshold),
        cover_radius=float(threshold/2),active_b1_mod_prime=int(count),
        train_cover_margin=float(threshold/2-eps),
        coverage_guarantee='covers the TRAIN neighbourhood of radius R minus TRAIN fill distance; no promise for all future states')
    return float(threshold),report


def _local_dimension_hint(X, neighbors=24, anchors=64, gap_threshold=2.):
    """First median local singular gap, estimated from coordinates only."""
    x = np.asarray(X, dtype=float)
    if x.ndim != 2 or len(x) < 4 or x.shape[1] < 1 or not np.isfinite(x).all():
        raise ValueError('finite X[states,features] with at least four states required')
    k = min(int(neighbors), len(x)-1)
    maximum = min(x.shape[1]-1, k-2)
    ids = np.unique(np.linspace(0, len(x)-1, min(int(anchors), len(x)), dtype=int))
    diagnostic = dict(rule='first median local singular gap at least 2',
        neighbors=k, anchors=len(ids), gap_threshold=float(gap_threshold),
        maximum_dimension=max(maximum, 0), automatic_dimension_is_heuristic=True)
    if maximum < 1:
        return None, diagnostic | dict(selected_dimension=None, median_singular_gaps=[])
    nearest = NearestNeighbors(n_neighbors=k+1).fit(x).kneighbors(x[ids], return_distance=False)
    values = []
    for anchor, candidates in zip(ids, nearest):
        js = candidates[candidates != anchor][:k]
        patch = x[js] - x[js].mean(axis=0)
        values.append(linalg.svdvals(patch)[:maximum+1])
    values = np.asarray(values)
    gaps = np.median(values[:, :-1] / np.maximum(values[:, 1:], 1e-14), axis=0)
    eligible = np.flatnonzero(gaps >= gap_threshold)
    dimension = int(eligible[0]+1) if len(eligible) else None
    return dimension, diagnostic | dict(selected_dimension=dimension, median_singular_gaps=gaps)

def _edge_representation(distances, threshold, lookup):
    edges = np.argwhere(np.triu(distances < threshold, 1)).astype(np.int64)
    indices = np.array([combinatorial_number_system_d1_forward(int(i), int(j), lookup)
                        for i, j in edges], dtype=np.int64)
    return edges, indices

def _closed_integer_cocycles(cocycles, edges, n_vertices):
    """Verify every triangle of exactly this strict-threshold Rips complex."""
    values = np.asarray(cocycles)
    if not np.issubdtype(values.dtype, np.integer):
        raise ValueError("closed-cocycle check requires integer arrays")
    largest_entry = max((abs(int(value)) for value in values.flat), default=0)
    if largest_entry > np.iinfo(np.int64).max // 3:
        raise OverflowError("integer closure arithmetic exceeds the declared exact int64 range")
    matrices = np.zeros((values.shape[0], n_vertices, n_vertices), dtype=np.int64)
    adjacency = np.zeros((n_vertices, n_vertices), dtype=bool)
    for index, (i, j) in enumerate(edges):
        matrices[:, i, j] = values[:, index]
        matrices[:, j, i] = -values[:, index]
        adjacency[i, j] = adjacency[j, i] = True
    triangles, largest = 0, 0
    for i, j in edges:
        third = np.flatnonzero(adjacency[i] & adjacency[j] & (np.arange(n_vertices) > j))
        triangles += len(third)
        if len(third):
            defect = matrices[:, i, j, None] + matrices[:, j, third] - matrices[:, i, third]
            largest = max(largest, int(np.max(np.abs(defect))))
    return largest == 0, dict(triangle_count=int(triangles), maximum_integer_closure_defect=largest,
                              all_triangles_streamed=True)

def _fail(reason, diagnostics=None):
    return dict(status='unresolved_' + reason, diagnostics=diagnostics or {},
                no_initial_circle_coordinates_computed=True)


def _triangle_incidence(edges, vertices):
    """All oriented flag triangles, columns in the supplied i<j edge order."""
    lookup = np.full((vertices, vertices), -1, dtype=np.int64)
    for index, (u, v) in enumerate(edges):
        lookup[u, v] = lookup[v, u] = index
    row, col, val = [], [], []
    faces = 0
    for index, (u, v) in enumerate(edges):
        third = np.flatnonzero((lookup[u] >= 0) & (lookup[v] >= 0)
                               & (np.arange(vertices) > v))
        for w in third:
            row.extend((faces, faces, faces))
            col.extend((index, int(lookup[v, w]), int(lookup[u, w])))
            val.extend((1, 1, -1))
            faces += 1
    return sparse.csr_matrix((np.asarray(val, dtype=np.int64), (row, col)),
                             shape=(faces, len(edges)))


def _infer_torus_dimension_once(X, nland=192, *, prime=47, lifetime_gap=2.0,
                              preferred_count=None, filtration_cap=None):
    """Return (r_or_None, diagnostics) from persistence and TRAIN coverage.

    This is a finite noisy-data dimension HEURISTIC conditional on the model
    M being a torus: dim(T^r)=b1(T^r).  It is not valid as an intrinsic-dimension
    formula for arbitrary manifolds and does not certify torus topology.
    A supplied geometric dimension fixes the prefix; otherwise use the largest
    eligible lifetime gap. Both require the common standard range, exactly that
    many active classes, and a buffered training cover at the selected scale.
    No integer lifts, cochain basis, harmonic smoothing, angles, frequencies,
    or vector-field fit are computed.  The persistence constructor internally
    produces mod-p representatives, but none are read or used by this probe.
    Actual integer period-basis preparation remains AFTER the caller's W fit.
    """
    x=np.asarray(X,dtype=float)
    if x.ndim!=2 or len(x)<4 or x.shape[1]<1 or not np.isfinite(x).all():
        raise ValueError('finite X[states,features] with at least four states required')
    if int(nland)!=nland or int(nland)<4:
        raise ValueError('at least four requested landmarks required')
    count_landmarks=min(int(nland),len(x))
    settings=replace(SETTINGS,n_landmarks=count_landmarks,prime=int(prime),
                     lifetime_gap=float(lifetime_gap))
    metric=_distances(x.T)
    control=np.unique(np.linspace(0,len(x)-1,9).astype(int))
    direct=cdist(x[control],x[control])
    metric_error=float(np.max(abs(direct-metric[np.ix_(control,control)])))
    if metric_error>2e-7*max(1.,float(direct.max())):
        raise FloatingPointError('dimension-probe metric disagrees with direct distances')
    if filtration_cap is None:
        persistence=ToroidalCoords(metric,n_landmarks=count_landmarks,
            distance_matrix=True,prime=settings.prime,maxdim=1,verbose=False)
    else:
        from fields_first_bounded_persistence import bounded_persistence
        persistence=bounded_persistence(metric,count_landmarks,settings.prime,filtration_cap)
    diagram=np.asarray(persistence._dgms[1],dtype=float)
    censored=np.asarray(getattr(persistence,'_h1_right_censored',np.zeros(len(diagram),dtype=bool)))
    selected,selection=_select_prefix(diagram,settings,preferred_count=preferred_count)
    if np.any(censored):
        selection['truncated_lifetime_gap_score']=selection.pop('lifetime_gap',None)
        selection['full_lifetime_gap_available']=False
        if preferred_count is None:
            selected=None
            selection['reason']='censored_lifetime_gap_requires_local_dimension_hint'
    selection['common_death_lower_bound']=selection.get('common_death')
    selection['selected_classes_include_censored_deaths']=bool(np.any(censored[:selected])) if selected is not None else False
    admissible=selected is not None and selected<x.shape[1]
    dimension_bound_satisfied = admissible
    scale_diagnostic = {}
    failure = None
    if selected is not None and not admissible:
        failure = 'selected_b1_not_below_ambient_dimension_for_compact_embedded_torus'
    if admissible:
        threshold,scale_diagnostic=_select_scale(diagram,selected,selection,
            np.asarray(persistence._dist_land_data),metric)
        if threshold is None:failure=scale_diagnostic['reason']
        else:
            radius=threshold/2
            covered=np.min(np.asarray(persistence._dist_land_data),axis=0)<radius
            scale_diagnostic['train_uncovered_count']=int(np.sum(~covered))
        admissible = failure is None
    diagnostic=dict(status='conditional_torus_dimension_heuristic' if admissible
        else 'unresolved_torus_dimension_probe',selected_dimension=selected if admissible else None,
        candidate_dimension=selected, preferred_count=preferred_count,
        selection=selection,states=len(x),ambient_dimension=x.shape[1],
        landmarks=count_landmarks,prime=settings.prime,h1_diagram=diagram,
        filtration_cap=filtration_cap,h1_right_censored=censored,
        direct_metric_control_max_error=metric_error,
        model_assumption='underlying smooth compact state manifold is T^r, so dimension equals b1',
        result_is_dimension_hypothesis_not_topology_or_noise_certificate=True,
        compact_torus_embedding_dimension_bound_satisfied=bool(dimension_bound_satisfied),
        diagram_only_used=False,integer_cocycle_lifts_computed=False,
        integer_period_basis_computed=False,cocycle_representatives_used=False,
        initial_phase_coordinates_computed=False,harmonic_smoothing_computed=False,
        fields_or_frequencies_computed=False,
        mod_prime_representatives_internally_produced_by_persistence_constructor=True,
        period_basis_must_still_be_prepared_after_W=True, **scale_diagnostic)
    if failure is not None:
        diagnostic['failure_reason']=failure
    return (int(selected) if admissible else None),diagnostic


def infer_torus_dimension(X, nland=192, *, prime=47, lifetime_gap=2.0):
    """Cross-check local geometry and persistence; refine only unresolved cases.

    The default landmark schedule is 192, 384, 768, 1024, 1536, 2048. No generator dimension,
    phase coordinates or frequencies enter this bounded finite-data heuristic.
    """
    if isinstance(nland, bool) or int(nland) != nland or nland < 4:
        raise ValueError('at least four requested landmarks required')
    hint, geometry = _local_dimension_hint(X)
    schedule=(192,384,768,1024,1536,2048)
    count = min(int(nland), schedule[-1], len(X)); attempts = [];filtration_cap=None
    while True:
        r, diagnostic = _infer_torus_dimension_once(X, count,
            prime=prime, lifetime_gap=lifetime_gap, preferred_count=hint,
            **({} if filtration_cap is None else dict(filtration_cap=filtration_cap)))
        attempt_keys = ('landmarks', 'status', 'selected_dimension', 'candidate_dimension',
            'selection', 'failure_reason', 'active_b1_mod_prime',
            'cover_radius', 'rips_threshold', 'train_uncovered_count',
            'train_fill_distance','training_sampling_buffer','train_cover_margin',
            'selected_scale_interval','allowed_scale_interval','filtration_cap')
        attempts.append({key: diagnostic[key] for key in attempt_keys if key in diagnostic})
        refined=min(next((value for value in schedule if value>count),count),len(X))
        if r is not None or refined <= count:
            break
        if refined>1536:
            death=diagnostic.get('selection',{}).get('common_death')
            filtration_cap=float(1.1*death) if death is not None and np.isfinite(death) and death>0 else max(float(np.sqrt(np.mean(np.sum(np.asarray(X)**2,axis=1)))),np.finfo(float).eps)
        count = refined
    diagnostic.update(geometry_dimension_hint=hint, geometry_dimension_diagnostic=geometry,
        diagram_only_used=False, dimension_inputs='local singular gaps and persistence diagram',
        maximum_topology_landmarks=schedule[-1])
    if len(attempts) > 1:
        diagnostic['resolution_attempts'] = attempts
    return r, diagnostic


def prepare_period_complex(X, r, nland=192, *, prime=47,
                           cover_percent=0.5, lifetime_gap=2.0,
                           repair_integer_lifts=True, milp_seconds=30.0,
                           filtration_cap=None):
    """Return connected finite K and a certified full primitive integer H1 dual.

    X[N,p] must use exactly the same processed coordinates/metric as W.
    r is supplied by the already fitted frame, never inferred from truth.
    ToroidalCoords is used ONLY for its persistence constructor/cocycle API.
    No call to get_coordinates, smoothing, or circle integration is made.
    cover_percent is retained for API compatibility; scale selection uses the
    same TRAIN coverage/rank rule as the dimension probe.
    """
    x = np.asarray(X, dtype=float)
    if x.ndim != 2 or len(x) < 4 or not np.isfinite(x).all():
        raise ValueError('finite X[states,features] with at least four states required')
    if isinstance(r, bool) or int(r) != r or r < 1:
        raise ValueError('positive supplied frame dimension required')
    r = int(r)
    nland = min(int(nland), len(x))
    settings = replace(SETTINGS, n_landmarks=nland, prime=int(prime),
                       cover_percent=float(cover_percent),
                       lifetime_gap=float(lifetime_gap))
    # Same verified full Euclidean metric as the existing geometric frontend.
    # Precomputed distances avoid repeated sklearn BLAS warning flags in
    # greedy landmark selection; direct coordinate-distance controls remain.
    metric = _distances(x.T)
    control = np.unique(np.linspace(0,len(x)-1,9).astype(int))
    direct = cdist(x[control],x[control])
    metric_error = float(np.max(abs(direct-metric[np.ix_(control,control)])))
    if metric_error > 2e-7*max(1.,float(direct.max())):
        raise FloatingPointError('full metric disagrees with direct coordinate distances')
    if filtration_cap is None:
        tc = ToroidalCoords(metric, n_landmarks=nland, distance_matrix=True,
                           prime=settings.prime, maxdim=1, verbose=False)
    else:
        from fields_first_bounded_persistence import bounded_persistence
        tc=bounded_persistence(metric,nland,settings.prime,filtration_cap)
    diagram = np.asarray(tc._dgms[1], dtype=float)
    censored=np.asarray(getattr(tc,'_h1_right_censored',np.zeros(len(diagram),dtype=bool)))
    count, selection = _select_prefix(diagram, settings, preferred_count=r)
    if np.any(censored):
        selection['truncated_lifetime_gap_score']=selection.pop('lifetime_gap',None)
        selection['full_lifetime_gap_available']=False
    selection['common_death_lower_bound']=selection.get('common_death')
    selection['selected_classes_include_censored_deaths']=bool(np.any(censored[:r]))
    diagnostics = dict(selected_topological_count=count,
        supplied_frame_dimension=r, selection=selection,
        filtration_cap=filtration_cap,h1_right_censored=censored,
        direct_metric_control_max_error=metric_error,
        no_initial_circle_coordinates_computed=True,
        no_harmonic_coordinate_smoothing_computed=True,
        topology_used_only_for_integer_period_basis=True,
        true_manifold_topology_certified=False)
    if count != r:
        return _fail('topological_count_does_not_match_frame', diagnostics)
    threshold,scale_diagnostic=_select_scale(diagram,r,selection,
        np.asarray(tc._dist_land_data),metric)
    diagnostics.update(scale_diagnostic)
    if threshold is None:return _fail(scale_diagnostic['reason'],diagnostics)
    radius=threshold/2
    landmark_indices = np.asarray(tc._idx_land, dtype=np.int64)
    distances = np.asarray(tc._dist_land_land, dtype=float)
    n = len(landmark_indices)
    edges, indices = _edge_representation(distances, threshold, tc._cns_lookup_table)
    delta1 = _triangle_incidence(edges, n)
    repair_reports, cocycles = [], []
    for j in range(r):
        _, _, representative = tc.get_representative_cocycle(j, 1)
        lifted = CohomologyUtils.lift_to_integer_cocycle(
            representative, prime=settings.prime)
        full = CohomologyUtils.sparse_cocycle_to_vector(
            lifted, tc._cns_lookup_table, n, int)
        c = np.asarray(full[indices], dtype=np.int64)
        defect = np.asarray(delta1 @ c, dtype=np.int64)
        before = int(np.max(abs(defect))) if len(defect) else 0
        report = dict(class_index=j, defect_before=before, repair_used=False)
        if before:
            if np.any(defect % settings.prime):
                return _fail('representative_not_closed_mod_prime', diagnostics | report)
            if not repair_integer_lifts:
                return _fail('integer_lift_requires_repair', diagnostics | report)
            # Corrections are SIGNED integers; imposing nonnegative bounds is
            # not justified by the integer-lift problem.
            target = defect // settings.prime
            result = milp(np.zeros(len(edges)), integrality=np.ones(len(edges)),
                bounds=Bounds(-np.inf, np.inf),
                constraints=LinearConstraint(delta1.astype(float), target, target),
                options=dict(time_limit=float(milp_seconds)))
            report.update(repair_used=True, milp_status=int(result.status),
                          milp_success=bool(result.success))
            if not result.success or result.x is None:
                return _fail('integer_lift_repair_failed', diagnostics | report)
            correction = np.rint(result.x)
            if not np.isfinite(correction).all() or np.max(abs(result.x-correction)) > 1e-5:
                return _fail('integer_lift_repair_noninteger', diagnostics | report)
            if np.max(abs(correction)) > np.iinfo(np.int64).max // (3*settings.prime):
                return _fail('integer_lift_repair_overflow', diagnostics | report)
            c = c - settings.prime * correction.astype(np.int64)
        cocycles.append(c)
        repair_reports.append(report)
    cocycles = np.asarray(cocycles, dtype=np.int64)
    closed, closure = _closed_integer_cocycles(cocycles, edges, n)
    active = int(np.sum((diagram[:,0] < threshold) & (diagram[:,1] > threshold)))
    adjacency64 = distances < threshold
    adjacency32 = distances.astype(np.float32).astype(float) < threshold
    changed = int(np.sum(np.triu(adjacency64 != adjacency32, 1)))
    endpoint = bool(np.any(diagram == threshold))
    same = changed == 0 and not endpoint
    certificate = certify_primitivity(edges, cocycles, active, vertex_count=n,
        integer_closure_verified=closed, same_complex_verified=same,
        prime=settings.prime, return_cycle_matrix=True)
    cycle_periods = np.asarray(certificate.pop('graph_cycle_period_matrix_integer',
                              np.empty((r,0),dtype=np.int64)),
                              dtype=np.int64)
    nearest = np.min(np.asarray(tc._dist_land_data), axis=0)
    covered = nearest < radius
    diagnostics.update(integer_repairs=repair_reports, integer_closure=closure,
        active_b1_mod_prime=active, cover_radius=float(radius),
        rips_threshold=float(threshold), train_uncovered_count=int(np.sum(~covered)),
        float32_float64_filtration_disagreements=changed,
        coordinate_scale_equals_bar_endpoint=endpoint)
    if not certificate['full_primitive_free_cohomology_basis_certified']:
        return _fail('full_primitive_finite_basis_not_certified', diagnostics | {'certificate':certificate})
    if not covered.all():
        return _fail('landmark_cover_incomplete_on_training_data', diagnostics | {'certificate':certificate})
    return dict(status='prepared_full_primitive_finite_period_complex',
        vertex_count=n, r=r, edges=edges, edge_indices=indices,
        integer_cocycles=cocycles, delta1=delta1,
        landmark_indices=landmark_indices, landmarks=x[landmark_indices].copy(),
        landmark_distances=distances,
        cover_radius=float(radius), rips_threshold=float(threshold),
        h1_diagram=diagram, graph_cycle_period_matrix_integer=cycle_periods,
        certificate=certificate, diagnostics=diagnostics,
        no_initial_circle_coordinates_computed=True)


def fit_periods(raw_edge_integrals, period_complex, *, edge_weights=None,
                tolerance=1e-11, maxiter=20000, maximum_period_condition=1e8):
    """Fit b_e = (f_head-f_tail) + Pi C_e by sparse weighted linear LS.

    raw_edge_integrals[E,r] are caller-computed beta integrals along the
    oriented landmark edges, potentially via refined fine-graph paths.
    For a connected complex with full H1 basis C, [Inc,C.T] spans ALL closed
    real cochains. No independent topological angles are used as targets.
    """
    cx = period_complex
    if cx.get('status') != 'prepared_full_primitive_finite_period_complex':
        return _fail('period_complex_unresolved', {'input_status':cx.get('status')})
    edges, c = np.asarray(cx['edges']), np.asarray(cx['integer_cocycles'])
    n, r, e = int(cx['vertex_count']), int(cx['r']), len(edges)
    raw = np.asarray(raw_edge_integrals, dtype=float)
    if raw.shape != (e,r) or not np.isfinite(raw).all():
        raise ValueError('raw_edge_integrals[E,r] must be finite and match complex edge order')
    weights = np.ones(e) if edge_weights is None else np.asarray(edge_weights, dtype=float)
    if weights.shape != (e,) or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError('positive finite edge weights required')
    inc = sparse.csr_matrix((np.r_[-np.ones(e),np.ones(e)],
        (np.r_[np.arange(e),np.arange(e)],np.r_[edges[:,0],edges[:,1]])),shape=(e,n))
    design = sparse.hstack((inc[:,1:], sparse.csr_matrix(c.T)), format='csr')
    weighted = design.multiply(np.sqrt(weights)[:,None]).tocsr()
    coeff, reports = [], []
    for axis in range(r):
        out = lsmr(weighted, raw[:,axis]*np.sqrt(weights),
                   atol=tolerance, btol=tolerance, maxiter=int(maxiter))
        coeff.append(out[0])
        reports.append(dict(coordinate=axis, stop=int(out[1]), iterations=int(out[2]),
            residual_norm=float(out[3]), normal_residual_norm=float(out[4]),
            estimated_design_condition=float(out[6])))
    coefficient = np.column_stack(coeff)
    potential = np.vstack((np.zeros((1,r)), coefficient[:n-1]))
    periods = coefficient[n-1:].T
    fitted = np.asarray(inc @ potential) + c.T @ periods.T
    defect = fitted-raw
    raw_curl = np.asarray(cx['delta1'] @ raw)
    fit_curl = np.asarray(cx['delta1'] @ fitted)
    singular = np.linalg.svd(periods, compute_uv=False)
    condition = float(singular[0]/singular[-1]) if singular[-1] > 0 else float('inf')
    diagnostics = dict(solver=reports, gauge='real integration potential at landmark 0 is zero',
        projection_relative=float(np.linalg.norm(defect)/max(np.linalg.norm(raw),1e-30)),
        weighted_projection_relative=float(np.linalg.norm(defect*np.sqrt(weights)[:,None])
            /max(np.linalg.norm(raw*np.sqrt(weights)[:,None]),1e-30)),
        raw_triangle_curl_rms=float(np.sqrt(np.mean(raw_curl**2))) if raw_curl.size else 0.,
        fitted_triangle_curl_max=float(np.max(abs(fit_curl))) if fit_curl.size else 0.,
        period_singular_values=singular.tolist(), period_condition=condition,
        cochain_basis_is_full_primitive_finite_basis=True,
        true_manifold_period_lattice_certified=False,
        quadrature_bias_not_detectable_from_closure_alone=True,
        independent_circle_coordinates_used=False,
        node_potentials_are_not_automatically_equal_to_smooth_partition_phase_values=True)
    if any(report['stop'] not in (0,1,2,4,5) for report in reports):
        return _fail('period_least_squares_not_converged',diagnostics)
    if not np.isfinite(condition) or condition > maximum_period_condition:
        return _fail('singular_or_ill_conditioned_period_matrix',diagnostics)
    normalized = np.linalg.solve(periods,fitted.T)
    normalized_potential = np.linalg.solve(periods,potential.T)
    algebra_error = normalized-c-np.asarray(inc @ normalized_potential.T).T
    diagnostics['normalized_integer_period_decomposition_max']=float(np.max(abs(algebra_error)))
    return dict(status='fitted_fields_first_finite_periods',periods=periods,
        Pi=periods,potential=potential,normalized_potential=normalized_potential,
        closed_edge_integrals=fitted,normalized_edge_integrals=normalized,
        node_phases=np.mod(2*np.pi*normalized_potential.T,2*np.pi),
        diagnostics=diagnostics,no_initial_circle_coordinates_computed=True)


def reduce_period_basis(period_model, period_complex, frame_gram):
    """Return dictionary copies with a shorter integral period basis.

    One exact unimodular change preserves the finite cohomology certificate.
    Raw frame-unit potentials and edge integrals are shared without mutation.
    No geometry, fields, periods, or phase targets are refitted.
    """
    from dreimac.toroidalcoords import _lll
    from sympy import Integer, Matrix

    if (period_model.get('status') != 'fitted_fields_first_finite_periods'
            or not period_complex.get('certificate',{}).get(
                'full_primitive_free_cohomology_basis_certified',False)):
        raise ValueError('fitted periods and a certified primitive basis required')
    pi=np.asarray(period_model['Pi'],dtype=float);r=len(pi)
    metric=np.asarray(frame_gram,dtype=float)
    if (r < 1 or pi.shape != (r,r) or metric.shape != (r,r)
            or not np.isfinite(pi).all() or not np.isfinite(metric).all()
            or not np.allclose(metric,metric.T,rtol=1e-10,atol=1e-12)):
        raise ValueError('finite square periods and symmetric frame Gram required')
    gram=pi.T@((metric+metric.T)/2)@pi;gram=(gram+gram.T)/2
    eigenvalues=np.linalg.eigvalsh(gram)
    if (not np.isfinite(eigenvalues).all() or eigenvalues[0] <= 0
            or eigenvalues[-1]/eigenvalues[0] > 1e12):
        raise ValueError('period Gram is not numerically positive and well conditioned')
    upper=linalg.cholesky(gram,lower=False)
    reduced,change=_lll(upper)
    if (not np.isfinite(change).all() or np.any(change != np.rint(change))
            or np.max(abs(change)) > 2**53-1):
        raise ValueError('lattice reduction did not return an exact integer transform')
    exact=Matrix([[int(value) for value in row] for row in change])
    if abs(exact.det()) != 1:
        raise ValueError('lattice reduction transform is not unimodular')
    inverse=exact.inv()
    if any(not value.is_Integer for value in inverse):
        raise ArithmeticError('unimodular inverse is not integral')

    def integer_array(value):
        source=np.asarray(value,dtype=object)
        if any(isinstance(v,(bool,np.bool_)) or not isinstance(v,(int,np.integer,Integer))
               for v in source.flat):
            raise ValueError('exact integer cohomology arrays required')
        if any(abs(int(v)) > np.iinfo(np.int64).max for v in source.flat):
            raise OverflowError('reduced integer cohomology exceeds int64 storage')
        return source.astype(np.int64)

    U=integer_array(exact.tolist());Ui=integer_array(inverse.tolist())
    if not np.allclose(upper@U,reduced,rtol=1e-10,atol=1e-10):
        raise ArithmeticError('lattice reduction orientation is inconsistent')
    pm=dict(period_model);cx=dict(period_complex);new_pi=pi@U
    pm.update(Pi=new_pi,periods=new_pi,
        normalized_potential=Ui@period_model['normalized_potential'],
        normalized_edge_integrals=Ui@period_model['normalized_edge_integrals'])
    pm['node_phases']=np.mod(2*np.pi*pm['normalized_potential'].T,2*np.pi)
    for key in ('integer_cocycles','graph_cycle_period_matrix_integer'):
        original=integer_array(period_complex[key]).astype(object)
        cx[key]=integer_array(Ui.astype(object)@original)
    after=U.T@gram@U
    report=dict(method='LLL in the transported-field Gram metric',
        U=U.tolist(),Ui=Ui.tolist(),determinant=int(exact.det()),
        exact_unimodular_change_verified=True,
        gram_condition_before=float(eigenvalues[-1]/eigenvalues[0]),
        gram_condition_after=float(np.linalg.cond(after)),
        frame_gram_condition=float(np.linalg.cond(metric)),
        gram_trace_before=float(np.trace(gram)),gram_trace_after=float(np.trace(after)),
        no_phase_targets_used=True)
    singular=np.linalg.svd(new_pi,compute_uv=False)
    pm['diagnostics']=dict(period_model.get('diagnostics',{}),
        period_basis_reduction=report,period_singular_values=singular.tolist(),
        period_condition=float(singular[0]/singular[-1]))
    edge=np.asarray(cx['edges']);potential=pm['normalized_potential']
    error=pm['normalized_edge_integrals']-cx['integer_cocycles']-(
        potential[:,edge[:,1]]-potential[:,edge[:,0]])
    pm['diagnostics']['normalized_integer_period_decomposition_max']=float(np.max(abs(error)))
    cx['diagnostics']=dict(period_complex.get('diagnostics',{}),period_basis_reduction=report)
    cx['certificate']=dict(period_complex['certificate'],
        basis_change_exact_unimodular=True,basis_change_U=U.tolist(),basis_change_Ui=Ui.tolist())
    return pm,cx


def extend_period_phases(model, period_complex, X_query, *, with_jacobian=False):
    """Smooth chart/partition integration of the FITTED beta cochain.

    Use this same map on TRAIN and held-out points. It is generally different
    from the raw node potential at a training vertex, because partition
    weights are not cardinal. No query changes topology, periods, or fit.
    Output phi[queries,r], jacobian[queries,r,ambient] when requested.
    """
    if model.get('status') != 'fitted_fields_first_finite_periods':
        return _fail('period_model_unresolved',{'input_status':model.get('status')})
    cx=period_complex
    points=np.asarray(X_query,dtype=float)
    centers=np.asarray(cx['landmarks'],dtype=float)
    if points.ndim != 2 or points.shape[1] != centers.shape[1] or not np.isfinite(points).all():
        raise ValueError('finite query states by ambient features required')
    n,r=len(centers),int(cx['r'])
    edge=np.asarray(cx['edges'])
    h=np.zeros((r,n,n))
    h[:,edge[:,0],edge[:,1]]=model['normalized_edge_integrals']
    h[:,edge[:,1],edge[:,0]]=-model['normalized_edge_integrals']
    potential=np.asarray(model['normalized_potential'])
    radius=float(cx['cover_radius']); radius2=radius**2
    phi=np.full((len(points),r),np.nan)
    jac=np.full((len(points),r,points.shape[1]),np.nan) if with_jacobian else None
    coverage=np.zeros(len(points),dtype=bool)
    for start in range(0,len(points),64):
        query=points[start:start+64]
        d2=cdist(centers,query,'sqeuclidean')
        active=d2<radius2
        covered=active.any(axis=0)
        root=np.argmax(active,axis=0)
        logweight=np.full_like(d2,-np.inf)
        logweight[active]=radius2/(d2[active]-radius2)
        scale=np.max(logweight,axis=0)
        scale[~covered]=0
        weight=np.exp(logweight-scale[None,:])
        denominator=weight.sum(axis=0)
        weight/=np.where(denominator>0,denominator,1)[None,:]
        connection=h[:,root,:]
        value=potential[:,root]+np.einsum('rqn,nq->rq',connection,weight,optimize=False)
        value[:,~covered]=np.nan
        phi[start:start+len(query)]=np.mod(2*np.pi*value.T,2*np.pi)
        coverage[start:start+len(query)]=covered
        if with_jacobian:
            difference=query[None,:,:]-centers[:,None,:]
            multiplier=np.zeros_like(d2)
            multiplier[active]=-2*radius2/(d2[active]-radius2)**2
            loggradient=multiplier[:,:,None]*difference
            mean=np.einsum('nq,nqp->qp',weight,loggradient,optimize=False)
            gradient=weight[:,:,None]*(loggradient-mean[None,:,:])
            derivative=2*np.pi*np.einsum('rqn,nqp->qrp',connection,gradient,optimize=False)
            derivative[~covered]=np.nan
            jac[start:start+len(query)]=derivative
    return dict(status='frozen_fields_first_period_extension',phi=phi,covered=coverage,
                jacobian=jac,topology_or_periods_refitted=False,
                no_initial_circle_coordinates_computed=True)
