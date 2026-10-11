"""Conditional source-graph lift and fixed-window phase averaging.

The graph must connect SOURCE states x_n, n=0,...,len(phi)-2.  It must
not be built on a possibly folded image of the circular coordinates phi.
The algorithm lifts arg(exp(i*(phi[n+1]-phi[n]))) across that graph, never
by temporal unwrap.  Its sufficient assumption is that the unknown true
real increment lift varies by strictly less than pi on every graph edge.
Finite graph consistency CANNOT certify that assumption or exclude hidden
turns between nodes.  It also cannot certify a primitive cohomology basis,
an observed torus, or absence of finite components.  All successful results
are conditional graph lifts, not topology/physical-phase certificates.

Only NumPy is required.  ``python phase_lift_average.py --self-test`` runs
analytic controls defined in this file; it reads no external data.
"""

from collections import deque
import argparse
import json
import numpy as np


TAU = 2.0 * np.pi


def wrap_angles(values):
    """Principal angles in [-pi, pi); no temporal branch selection."""
    return (np.asarray(values, dtype=float) + np.pi) % TAU - np.pi


def _angles(phi):
    phi = np.asarray(phi, dtype=float)
    if phi.ndim == 1:
        phi = phi[:, None]
    if phi.ndim != 2 or phi.shape[0] < 2 or phi.shape[1] < 1:
        raise ValueError("phi must have shape (n>=2, r>=1)")
    if not np.all(np.isfinite(phi)):
        raise ValueError("phi contains nonfinite values")
    return wrap_angles(phi)


def _source_graph(source_edges, node_count, edge_distances, max_edge_distance):
    edges = np.asarray(source_edges)
    if edges.size == 0:
        edges = np.empty((0, 2), dtype=np.int64)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("source_edges must be an undirected edge list (E,2)")
    if not np.issubdtype(edges.dtype, np.integer):
        raise ValueError("source_edges must contain integer vertex indices")
    edges = edges.astype(np.int64, copy=False)
    if np.any(edges < 0) or np.any(edges >= node_count):
        raise ValueError("source graph vertices must be in 0,...,len(phi)-2")
    original_count = len(edges)
    keep = np.ones(len(edges), dtype=bool)
    if edge_distances is not None:
        distances = np.asarray(edge_distances, dtype=float)
        if distances.shape != (len(edges),) or not np.all(np.isfinite(distances)):
            raise ValueError("edge_distances must be finite and have shape (E,)")
        if np.any(distances < 0):
            raise ValueError("edge distances cannot be negative")
        if max_edge_distance is not None:
            bound = float(max_edge_distance)
            if not np.isfinite(bound) or bound < 0:
                raise ValueError("max_edge_distance must be finite and nonnegative")
            keep &= distances <= bound
    elif max_edge_distance is not None:
        raise ValueError("max_edge_distance requires edge_distances")
    keep &= edges[:, 0] != edges[:, 1]
    edges = np.unique(np.sort(edges[keep], axis=1), axis=0)
    return edges, {"input_edge_count": original_count,
                   "used_edge_count": int(len(edges)),
                   "distance_filter_applied": max_edge_distance is not None}


def _edge_data(ratio_angles, edges, branch_margin):
    margin = float(branch_margin)
    if not np.isfinite(margin) or not 0 <= margin < np.pi:
        raise ValueError("branch_margin must lie in [0,pi)")
    raw = ratio_angles[edges[:, 1]] - ratio_angles[edges[:, 0]]
    principal = wrap_angles(raw)
    turns = np.rint((raw - principal) / TAU).astype(np.int64)
    ambiguous = np.any(np.abs(principal) >= np.pi - margin, axis=1)
    return principal, turns, ambiguous


def _base_diagnostics(graph_info, ratio_angles, principal, ambiguous):
    return dict(graph_info, source_node_count=int(len(ratio_angles)),
                coordinate_count=int(ratio_angles.shape[1]),
                ambiguous_edge_count=int(np.count_nonzero(ambiguous)),
                max_principal_edge_difference=float(np.max(np.abs(principal)))
                if principal.size else 0.0,
                hidden_continuous_branches_certified=False,
                primitive_cohomology_certified=False,
                connected_observed_torus_assumed=True,
                finite_component_extension_implemented=False,
                sufficient_edge_condition="true real lift variation < pi on every edge")


def lift_step_ratios(phi, source_edges, *, edge_distances=None,
                     max_edge_distance=None, branch_margin=1e-6,
                     numeric_tolerance=1e-8):
    """Lift one-step circle ratios over a connected SOURCE graph.

    A successful ``delta`` has shape (n-1,r), up to a common 2*pi integer
    vector.  Nonzero cycle winding, ambiguous edges, or disconnected graphs
    return status='unresolved' and delta=None.  Local edge increments are
    principal ARGUMENT DIFFERENCES OF RATIOS, not temporal phase increments.
    ``max_edge_distance`` only filters supplied distances; it is not a
    certificate of the unknown variation condition.
    """
    phi = _angles(phi)
    ratio = wrap_angles(phi[1:] - phi[:-1])
    edges, graph_info = _source_graph(source_edges, len(ratio), edge_distances,
                                      max_edge_distance)
    principal, turns, ambiguous = _edge_data(ratio, edges, branch_margin)
    diag = _base_diagnostics(graph_info, ratio, principal, ambiguous)
    adjacency = [[] for _ in range(len(ratio))]
    for i, (u, v) in enumerate(edges):
        adjacency[int(u)].append((int(v), i, 1))
        adjacency[int(v)].append((int(u), i, -1))
    branches = np.zeros_like(ratio, dtype=np.int64)
    components = np.full(len(ratio), -1, dtype=np.int64)
    count = 0
    for anchor in range(len(ratio)):
        if components[anchor] >= 0:
            continue
        components[anchor] = count
        queue = deque([anchor])
        while queue:
            u = queue.popleft()
            for v, i, orientation in adjacency[u]:
                if components[v] < 0:
                    branches[v] = branches[u] - orientation * turns[i]
                    components[v] = count
                    queue.append(v)
        count += 1
    winding = branches[edges[:, 1]] - branches[edges[:, 0]] + turns
    bad_cycles = np.any(winding != 0, axis=1)
    delta = ratio + TAU * branches
    residual = delta[edges[:, 1]] - delta[edges[:, 0]] - principal
    noninteger_residual = residual - TAU * winding
    numeric_defect = float(np.max(np.abs(noninteger_residual))) if residual.size else 0.0
    diag.update(component_count=count, cycle_inconsistent_edge_count=int(bad_cycles.sum()),
                max_abs_cycle_winding=int(np.max(np.abs(winding))) if winding.size else 0,
                max_noninteger_numeric_cycle_residual=numeric_defect,
                cycle_examples=[{"edge": edges[i].tolist(),
                                 "integer_winding": winding[i].tolist()}
                                for i in np.flatnonzero(bad_cycles)[:12]])
    failures = []
    if count != 1:
        failures.append("disconnected_source_graph")
    if np.any(ambiguous):
        failures.append("edge_near_branch_cut")
    if np.any(bad_cycles):
        failures.append("nonzero_integer_cycle_winding")
    if numeric_defect > numeric_tolerance:
        failures.append("noninteger_numeric_cycle_error")
    return {"status": "unresolved" if failures else "conditional_graph_lift",
            "failure_reasons": failures, "delta": None if failures else delta,
            "ratio_angles": ratio, "branch_integers": branches,
            "source_edges": edges, "diagnostics": diag}


def validate_step_lift(phi, source_edges, delta, *, edge_distances=None,
                       max_edge_distance=None, branch_margin=1e-6,
                       numeric_tolerance=1e-8):
    """Reject varying supplied branches under the same conditional edge rule.

    A common branch integer is allowed.  Passing is still conditional: finite
    endpoints cannot exclude hidden continuous turns between graph nodes.
    """
    result = lift_step_ratios(phi, source_edges, edge_distances=edge_distances,
                              max_edge_distance=max_edge_distance,
                              branch_margin=branch_margin,
                              numeric_tolerance=numeric_tolerance)
    proposed = np.asarray(delta, dtype=float)
    ratio = result["ratio_angles"]
    if proposed.shape != ratio.shape or not np.all(np.isfinite(proposed)):
        raise ValueError("delta must be finite and have shape (n-1,r)")
    ratio_defect = float(np.max(np.abs(wrap_angles(proposed - ratio))))
    branch = np.rint((proposed - ratio) / TAU).astype(np.int64)
    edges = result["source_edges"]
    principal, turns, _ = _edge_data(ratio, edges, branch_margin)
    winding = branch[edges[:, 1]] - branch[edges[:, 0]] + turns
    bad = np.any(winding != 0, axis=1)
    reasons = list(result["failure_reasons"])
    if ratio_defect > numeric_tolerance:
        reasons.append("supplied_lift_not_congruent_to_ratios")
    if np.any(bad):
        reasons.append("supplied_lift_has_varying_graph_branches")
    return {"status": "unresolved" if reasons else "conditional_graph_lift",
            "failure_reasons": reasons, "ratio_congruence_defect": ratio_defect,
            "inconsistent_supplied_edge_count": int(bad.sum()),
            "diagnostics": result["diagnostics"]}


def triangular_average(phi, delta, window, *, numeric_tolerance=1e-8):
    """Fixed-Q operator using an ALREADY CONSISTENT real increment lift.

    This low-level function checks ratio congruence, not graph continuity.
    Use phase_lift_average or validate_step_lift before relying on it.
    Output start n uses phase samples n,...,n+Q-1.  If each state is an
    L-sample Hankel window, the raw support is L+Q-1, not L.
    """
    phi = _angles(phi)
    if isinstance(window, (bool, np.bool_)) or int(window) != window:
        raise ValueError("window must be an integer")
    window = int(window)
    if not 1 <= window <= len(phi):
        raise ValueError("window must lie in [1,len(phi)]")
    delta = np.asarray(delta, dtype=float)
    if delta.shape != (len(phi) - 1, phi.shape[1]) or not np.all(np.isfinite(delta)):
        raise ValueError("delta must be finite and have shape (n-1,r)")
    defect = float(np.max(np.abs(wrap_angles(delta - (phi[1:] - phi[:-1])))))
    if defect > numeric_tolerance:
        raise ValueError("delta is not congruent to the observed one-step ratios")
    size = len(phi) - window + 1
    unwrapped = phi[:size].copy()
    if window > 1:
        weights = np.arange(window - 1, 0, -1, dtype=float) / window
        for j in range(phi.shape[1]):
            unwrapped[:, j] += np.correlate(delta[:, j], weights, mode="valid")
    starts = np.arange(size, dtype=np.int64)
    return {"phases": wrap_angles(unwrapped), "local_real_representatives": unwrapped,
            "source_indices": starts, "end_indices": starts + window - 1,
            "window": window, "phase_sample_support": window,
            "raw_sample_support_formula": "L+Q-1",
            "ratio_congruence_defect": defect,
            "fixed_window_global_branch_gauge_only": True}


def phase_lift_average(phi, source_edges, window, *, edge_distances=None,
                       max_edge_distance=None, branch_margin=1e-6,
                       numeric_tolerance=1e-8):
    """Graph-lift then average, failing closed on graph inconsistencies.

    This function uses every supplied graph node.  Its train/transductive
    scope is the caller's responsibility; it performs no heldout extension,
    field fit, frequency fit, topology discovery, or degree certification.
    """
    result = lift_step_ratios(phi, source_edges, edge_distances=edge_distances,
                              max_edge_distance=max_edge_distance,
                              branch_margin=branch_margin,
                              numeric_tolerance=numeric_tolerance)
    result["average"] = None
    if result["status"] != "unresolved":
        result["average"] = triangular_average(phi, result["delta"], window,
                                               numeric_tolerance=numeric_tolerance)
    return result


def analytic_controls():
    """Deterministic synthetic mathematical identities, not field benchmarks."""
    checks = []

    def check(name, condition, error=None):
        if not bool(condition):
            raise AssertionError(name)
        checks.append({"name": name, "passed": True, "error": error})

    def circle_graph(theta):
        order = np.argsort(wrap_angles(theta[:-1]))
        return np.column_stack((order, np.roll(order, -1)))

    def distance(a, b):
        return float(np.max(np.abs(wrap_angles(np.asarray(a) - np.asarray(b)))))

    n, q, alpha = 1025, 64, TAU * (np.sqrt(5.0) - 1.0) / 2.0
    theta = 0.31 + alpha * np.arange(n)
    graph = circle_graph(theta)
    fast = phase_lift_average(theta[:, None], graph, q)
    check("constant_fast_rotation_graph_lift", fast["status"] == "conditional_graph_lift")
    error = distance(np.diff(fast["average"]["phases"], axis=0), alpha)
    check("fast_rotation_step_modulo_2pi", error < 2e-11, error)

    p = 2.0 * np.sin(theta)
    nonlinear = phase_lift_average((theta + p)[:, None], graph, q)
    delta_true = alpha + np.diff(p)
    shift = float(nonlinear["delta"][0, 0] - delta_true[0])
    error = float(np.max(np.abs(nonlinear["delta"][:, 0] - delta_true - shift)))
    check("nonlinear_lift_is_one_global_branch", error < 2e-11, error)
    p_mean = np.correlate(p, np.ones(q) / q, mode="valid")
    expected = theta[:len(p_mean)] + (q - 1) * (alpha + shift) / 2.0 + p_mean
    error = distance(nonlinear["average"]["phases"][:, 0], expected)
    check("folded_degree_one_exact_triangular_identity", error < 2e-11, error)
    derivative_floor = 1.0 - 2.0 * abs(np.mean(np.exp(1j * alpha * np.arange(q))))
    check("analytic_averaged_map_unfolded_at_fixed_Q", derivative_floor > 0,
          float(derivative_floor))

    cover_a = phase_lift_average((2.0 * theta)[:, None], graph, q)
    cover_b = phase_lift_average((2.0 * (theta + np.pi))[:, None], graph, q)
    error = distance(cover_a["average"]["phases"], cover_b["average"]["phases"])
    check("nonprimitive_cover_is_not_repaired_or_certified", error < 2e-11 and
          not cover_a["diagnostics"]["primitive_cohomology_certified"], error)

    e = 0.04 * np.cos(3.0 * theta)
    perturbed = phase_lift_average((theta + p + e)[:, None], graph, q)
    b = float(perturbed["delta"][0, 0] - nonlinear["delta"][0, 0] - np.diff(e)[0])
    expected_error = np.correlate(e, np.ones(q) / q, mode="valid")
    measured_error = wrap_angles(perturbed["average"]["phases"][:, 0] -
                                 nonlinear["average"]["phases"][:, 0] - (q - 1) * b / 2)
    error = float(np.max(np.abs(measured_error - expected_error)))
    check("coherent_coordinate_errors_telescope", error < 2e-11, error)
    check("coherent_error_bound_does_not_grow_with_Q", np.max(np.abs(measured_error)) <= .04)

    wrong = nonlinear["delta"].copy()
    wrong[wrap_angles(theta[:-1]) > 0.0, 0] += TAU
    validation = validate_step_lift(theta[:, None] + p[:, None], graph, wrong)
    check("deliberate_varying_branch_error_rejected", validation["status"] == "unresolved"
          and "supplied_lift_has_varying_graph_branches" in validation["failure_reasons"])
    shifted = validate_step_lift(theta[:, None] + p[:, None], graph, nonlinear["delta"] + TAU)
    check("common_integer_branch_is_allowed", shifted["status"] == "conditional_graph_lift")

    m = 12
    prescribed_ratio = TAU * np.arange(m) / m
    phase = np.concatenate(([0.0], np.cumsum(prescribed_ratio)))[:, None]
    cycle = np.column_stack((np.arange(m), np.roll(np.arange(m), -1)))
    inconsistent = lift_step_ratios(phase, cycle)
    check("nonzero_cycle_winding_fails_closed", inconsistent["status"] == "unresolved"
          and inconsistent["diagnostics"]["max_abs_cycle_winding"] == 1)
    disconnected = lift_step_ratios(theta[:5, None], np.array([[0, 1], [2, 3]]))
    check("disconnected_source_graph_unresolved", "disconnected_source_graph" in
          disconnected["failure_reasons"])
    identity = triangular_average(theta[:, None], fast["delta"], 1)
    check("Q1_is_identity", distance(identity["phases"], theta[:, None]) < 1e-12)
    return {"status": "passed", "control_count": len(checks), "controls": checks,
            "external_data_read": False, "field_benchmark": False,
            "hidden_branch_or_primitive_class_certificate": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(analytic_controls(), indent=2))
    else:
        parser.print_help()
