"""Independent, public-only audit of the frozen geometric phase encoder.

No fitting, persistent-homology replay, truth/evaluator archives, or inference
helper imports. --prepare hashes the terminal six-case artifacts; --run checks
those hashes before reading public Y,s,t and reconstructs all encoder algebra.
Active Betti counts are checked against the FROZEN barcode, not recomputed PH.
"""

from project_paths import RESULTS_DIR

import argparse
from datetime import datetime, timezone
import hashlib
import json
from math import factorial
from pathlib import Path
import time

import numpy as np
from scipy.linalg.blas import dgemm
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist
from sympy import Integer, Matrix
from sympy.matrices.normalforms import hermite_normal_form


HERE = Path(__file__).resolve().parent
SOURCE = RESULTS_DIR / 'geometric_phase_frontend.json'
PROTOCOL = RESULTS_DIR / 'geometric_phase_frontend_audit_protocol.json'
OUTPUT = RESULTS_DIR / 'geometric_phase_frontend_audit.json'
TAU = 2 * np.pi


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, payload):
    if Path(path).exists():
        raise FileExistsError("preserve every existing audit artifact")
    Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def wrap(x):
    return (np.asarray(x) + np.pi) % TAU - np.pi


def max_error(a, b, angular=False):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or not np.array_equal(np.isfinite(a), np.isfinite(b)):
        raise AssertionError("shape or finite-value mask differs")
    if not np.any(np.isfinite(a)):
        return 0.0
    if a.dtype == bool or b.dtype == bool:
        return float(not np.array_equal(a, b))
    difference = a[np.isfinite(a)] - b[np.isfinite(b)]
    return float(np.max(np.abs(wrap(difference) if angular else difference)))


def assert_error(a, b, tolerance=2e-7, angular=False):
    error = max_error(a, b, angular)
    if error > tolerance:
        raise AssertionError("independent algebra defect " + str(error))
    return error


def terminal_source():
    source = json.loads(SOURCE.read_text())
    if (source["status"] != "finished_inference_only" or len(source["rows"]) != 6
            or not source["all_six_attempts_and_outputs_frozen_before_truth"]
            or source["evaluation_files_opened"]):
        raise ValueError("all six public-only attempts must be terminal and frozen")
    return source


def protected_paths(source):
    paths = {str(SOURCE): digest(SOURCE), source["protocol_file"]: source["protocol_sha256"]}
    for name, expected in source["source_fingerprints"].items():
        paths[str(HERE / name)] = expected
    for item in source["library_fingerprints"]["sources"].values():
        paths[item["path"]] = item["sha256"]
    for entry in source["manifest"]:
        paths[entry["input"]] = entry["input_sha256"]
    for row in source["rows"]:
        for key in ("frozen_parameters", "frozen_output"):
            if key in row:
                paths[row[key]] = row[key + "_sha256"]
    return paths


def verify_hashes(paths):
    for path, expected in paths.items():
        if digest(path) != expected:
            raise ValueError("frozen artifact changed: " + path)


def prepare():
    source = terminal_source()
    paths = protected_paths(source)
    verify_hashes(paths)
    declaration = dict(status="prepared_not_run", created_at=datetime.now(timezone.utc).isoformat(),
        audit_source=str(Path(__file__).resolve()), audit_source_sha256=digest(__file__),
        protected_fingerprints=paths, attempted_cases=6,
        expected_run_names=[entry["run"] for entry in source["manifest"]],
        public_input_keys=["Y", "s", "t"], no_true_parameters_or_evaluator_read=True,
        no_fit_or_persistent_homology_rerun=True, helper_functions_imported=False,
        independent_mass="P1 simplex integration from known sensor positions",
        independent_tree="depth-first forest, arbitrary-precision integer potentials",
        active_b1_scope="count from frozen F47 barcode only; no independent PH computation",
        tolerance=2e-7, raw_support=159, Q=32, L=128)
    write_json(PROTOCOL, declaration)
    print(json.dumps(dict(status=declaration["status"], protocol=str(PROTOCOL),
                         audit_source_sha256=declaration["audit_source_sha256"])))


def mass_matrix(positions):
    n, dimension = positions.shape
    if dimension == 1:
        order = np.argsort(positions[:, 0])
        cells = np.column_stack((order[:-1], order[1:]))
    else:
        cells = Delaunay(positions).simplices
    mass = np.zeros((n, n))
    scale = float(np.prod(np.ptp(positions, axis=0))) / factorial(dimension)
    for cell in cells:
        volume = abs(float(np.linalg.det(positions[cell[1:]] - positions[cell[0]]))) / factorial(dimension)
        if dimension > 1 and volume <= 64 * np.finfo(float).eps * scale:
            continue
        for i in cell:
            for j in cell:
                mass[i, j] += volume * (2 if i == j else 1) / ((dimension + 1) * (dimension + 2))
    return mass


def metric(left, right=None):
    same = right is None
    right = left if same else right
    norms_l = np.sum(left * left, axis=0)
    norms_r = np.sum(right * right, axis=0)
    gram = dgemm(1.0, np.asfortranarray(left), np.asfortranarray(right), trans_a=True)
    result = np.sqrt(np.maximum(norms_l[:, None] + norms_r[None, :] - 2 * gram, 0))
    if same:
        result = (result + result.T) / 2
        np.fill_diagonal(result, 0)
    return result


def reconstruct(entry, params, adapter):
    with np.load(entry["input"], allow_pickle=False) as public:
        if set(public.files) != {"Y", "s", "t"}:
            raise ValueError("only public observation keys are permitted")
        y, s, t = [public[name].copy() for name in ("Y", "s", "t")]
    n_train = adapter["n_train"]
    lag = adapter["lag_window"]
    mass = mass_matrix(s)
    root = params["mass_sqrt"]
    mean = y[:, :n_train].mean(axis=1)
    diagnostics = dict(mass_defect=assert_error(mass, params["physical_mass"], 2e-12),
        mass_root_defect=assert_error(np.einsum("ij,jk->ik", root, root, optimize=False), mass, 2e-12),
        train_mean_defect=assert_error(mean, params["mean_sensor_train"], 2e-12),
        sensors_defect=assert_error(s, params["sensors"], 0),
        public_times_defect=assert_error(t, params["public_sample_times"], 0))
    if not all(np.isfinite(value).all() for value in (y, s, t, root)):
        raise AssertionError("nonfinite public observations or frozen mass root")
    mass_y = np.einsum("ij,jt->it", root, y - mean[:, None], optimize=False)
    count = y.shape[1] - lag + 1
    H = np.vstack([mass_y[:, j:j + count] for j in range(lag)]) / np.sqrt(lag)
    n_columns = n_train - lag + 1
    endpoints = np.arange(lag - 1, len(t))
    blocks = (("train", 0, n_columns),
              ("validation", n_columns, n_columns + adapter["n_validation"]),
              ("future", n_columns + adapter["n_validation"], count))
    for name, lo, hi in blocks:
        assert_error(params[name + "_window_start_index"], endpoints[lo:hi] - lag + 1, 0)
        assert_error(params[name + "_window_end_index"], endpoints[lo:hi], 0)
        assert_error(params[name + "_time_at_window_start"], t[endpoints[lo:hi] - lag + 1], 0)
        assert_error(params[name + "_time_at_window_end"], t[endpoints[lo:hi]], 0)
    return H, n_columns, blocks, endpoints, diagnostics


def closed_triangles(edges, cocycles, vertices):
    # Checked range makes each THREE-term int64 operation mathematically exact.
    if max((abs(int(v)) for v in cocycles.flat), default=0) > np.iinfo(np.int64).max // 3:
        raise OverflowError("triangle audit requires safe exact integer range")
    connection = np.zeros((cocycles.shape[0], vertices, vertices), dtype=np.int64)
    adjacent = np.zeros((vertices, vertices), dtype=bool)
    for index, (u, v) in enumerate(edges):
        adjacent[u, v] = adjacent[v, u] = True
        connection[:, u, v] = cocycles[:, index]
        connection[:, v, u] = -cocycles[:, index]
    count, defect = 0, 0
    for u in range(vertices):
        upper = np.flatnonzero(adjacent[u] & (np.arange(vertices) > u))
        for j, v in enumerate(upper):
            third = upper[j + 1:][adjacent[v, upper[j + 1:]]]
            count += len(third)
            if len(third):
                values = connection[:, u, v, None] + connection[:, v, third] - connection[:, u, third]
                defect = max(defect, int(np.max(np.abs(values))))
    return count, defect


def integer_period_hnf(edges, cocycles, vertices):
    # Different tree from inference's scipy BFS; every arithmetic value is Python int.
    adjacency = [[] for _ in range(vertices)]
    for index, (u, v) in enumerate(edges):
        adjacency[int(u)].append((int(v), index, 1))
        adjacency[int(v)].append((int(u), index, -1))
    r = len(cocycles)
    potential = [[0] * vertices for _ in range(r)]
    seen, tree, components = set(), set(), 0
    for root in range(vertices):
        if root in seen:
            continue
        components += 1
        seen.add(root)
        stack = [root]
        while stack:
            u = stack.pop()
            for v, index, sign in reversed(adjacency[u]):
                if v not in seen:
                    seen.add(v)
                    tree.add(index)
                    for j in range(r):
                        potential[j][v] = potential[j][u] + sign * int(cocycles[j, index])
                    stack.append(v)
    chords = [index for index in range(len(edges)) if index not in tree]
    periods = Matrix(r, len(chords), [potential[j][int(edges[i, 0])] + int(cocycles[j, i])
                    - potential[j][int(edges[i, 1])] for j in range(r) for i in chords])
    hnf = hermite_normal_form(periods) if r else Matrix.zeros(0, 0)
    return components, hnf, len(chords)


def replay_phases(params, distances):
    radius = float(params["cover_radius"])
    active = distances < radius
    weights = np.zeros_like(distances)
    weights[active] = np.exp(radius ** 2 / (distances[active] ** 2 - radius ** 2))
    total = weights.sum(axis=0)
    covered = np.isfinite(total) & (total > 0)
    weights /= np.where(covered, total, 1)[None, :]
    membership = np.argmax(active, axis=0)
    r, vertices = params["integrals"].shape
    native = np.empty((r, distances.shape[1]))
    for j in range(r):
        connection = np.zeros((vertices, vertices))
        edge_values = params["harmonics"][j, params["edge_indices"]]
        u, v = params["edges"].T
        connection[u, v], connection[v, u] = edge_values, -edge_values
        native[j] = TAU * (params["integrals"][j, membership] +
            np.einsum("ij,ji->i", connection[membership], weights, optimize=False))
    phases = np.mod(np.einsum("ij,jt->it", params["change_basis"], native, optimize=False), TAU)
    phases[:, ~covered] = np.nan
    return phases, covered


def source_lift(phi, edges):
    ratios = wrap(np.diff(phi, axis=1))
    u, v = edges.T
    raw = ratios[:, v] - ratios[:, u]
    edge_delta = wrap(raw)
    turns = np.rint((raw - edge_delta) / TAU).astype(np.int64)
    branch = np.zeros(ratios.shape, dtype=np.int64)
    adjacency = [[] for _ in range(ratios.shape[1])]
    for i, (a, b) in enumerate(edges):
        adjacency[int(a)].append((int(b), i, -1))
        adjacency[int(b)].append((int(a), i, 1))
    seen, components = set(), 0
    for root in range(len(adjacency)):
        if root in seen:
            continue
        components += 1
        seen.add(root)
        stack = [root]
        while stack:
            a = stack.pop()
            for b, index, sign in reversed(adjacency[a]):
                if b not in seen:
                    branch[:, b] = branch[:, a] + sign * turns[:, index]
                    seen.add(b)
                    stack.append(b)
    winding = branch[:, v] - branch[:, u] + turns
    ambiguous = bool(np.any(np.abs(edge_delta) >= np.pi - 1e-6))
    valid = components == 1 and not np.any(winding) and not ambiguous
    return ratios, ratios + TAU * branch, valid, dict(
        components=components, nonzero_cycle_winding_detected=bool(np.any(winding)),
        near_branch_cut_edge_detected=ambiguous,
        hidden_continuous_turns_certified=False)


def causal_average(phi, delta, Q):
    output = np.full_like(phi, np.nan)
    weights = np.arange(Q - 1, 0, -1, dtype=float) / Q
    for endpoint in range(Q - 1, phi.shape[1]):
        start = endpoint - Q + 1
        if np.isfinite(phi[:, start:endpoint + 1]).all() and np.isfinite(delta[:, start:endpoint]).all():
            output[:, endpoint] = wrap(wrap(phi[:, start]) +
                np.sum(delta[:, start:endpoint] * weights[None, :], axis=1))
    return output


def audit_case(entry, row, adapter, settings):
    result = dict(run=row["run"], stored_run_status=row["run_status"], preserved_failure=True)
    if "frozen_parameters" not in row:
        if row["run_status"] != "fit_exception":
            raise AssertionError("missing model archive without a recorded fit failure")
        return dict(result, audit_status="recorded_fit_failure_no_archive", exception=row["exception"])
    with np.load(row["frozen_parameters"], allow_pickle=False) as archive:
        params = {key: archive[key].copy() for key in archive.files}
    H, n_train, blocks, endpoints, geometry = reconstruct(entry, params, adapter)
    landmark = params["landmark_indices"].astype(int)
    if len(np.unique(landmark)) != len(landmark) or np.any(landmark >= n_train):
        raise AssertionError("landmark indices are not distinct TRAIN states")
    diagram = params["h1_diagram"]
    assert_error(diagram, np.asarray(row["h1_diagram"]), 0)
    result.update(geometry=geometry, selected_rank=row["selected_coordinate_count"],
                  frozen_H1_bar_count=len(diagram), landmark_count=len(landmark))
    if "train_phases" not in params:
        return dict(result, audit_status="recorded_unresolved_topology_preserved",
                    frontend_status=row["frontend_diagnostics"]["status"])
    land_H = H[:, landmark]
    geometry["landmark_columns_defect"] = assert_error(land_H, params["landmark_columns"], 2e-12)
    land_dist = metric(land_H)
    geometry["landmark_metric_defect"] = assert_error(land_dist, params["landmark_distances"])
    control = np.unique(np.linspace(0, len(landmark) - 1, 9).astype(int))
    geometry["direct_distance_control_defect"] = assert_error(
        cdist(land_H[:, control].T, land_H[:, control].T), land_dist[np.ix_(control, control)])
    threshold = float(params["rips_threshold"])
    stored_land_dist = params["landmark_distances"]
    expected_edges = np.argwhere(np.triu(stored_land_dist < threshold, 1))
    assert_error(params["edges"], expected_edges, 0)
    float32_equal = np.array_equal(np.triu(stored_land_dist < threshold, 1),
                                  np.triu(stored_land_dist.astype(np.float32).astype(float) < threshold, 1))
    endpoint_tie = bool(np.any(diagram == threshold))
    active_b1 = int(np.sum((diagram[:, 0] < threshold) & (diagram[:, 1] > threshold)))
    triangle_count, closure = closed_triangles(params["edges"], params["integer_cocycles"], len(landmark))
    components, hnf, chords = integer_period_hnf(params["edges"], params["integer_cocycles"], len(landmark))
    stored_hnf = hermite_normal_form(Matrix(params["graph_cycle_period_matrix"].tolist()))
    if hnf != stored_hnf:
        raise AssertionError("independent DFS cycle image differs from stored BFS cycle image")
    cert = row["frontend_diagnostics"]["finite_complex_certificate"]
    finite_certificate = components == 1 and closure == 0 and float32_equal and not endpoint_tie and (
        hnf == Matrix.eye(params["train_phases"].shape[0])) and active_b1 == params["train_phases"].shape[0]
    if finite_certificate != cert["full_primitive_free_cohomology_basis_certified"]:
        raise AssertionError("finite-complex primitive certificate differs")
    if active_b1 != row["frontend_diagnostics"]["active_H1_at_coordinate_scale"]:
        raise AssertionError("active frozen-barcode count differs")
    if (triangle_count != row["frontend_diagnostics"]["integer_closure"]["triangle_count"]
            or closure != row["frontend_diagnostics"]["integer_closure"]["maximum_integer_closure_defect"]):
        raise AssertionError("exact all-triangle closure differs")
    expected_index = params["edges"][:, 0] + params["edges"][:, 1] * (params["edges"][:, 1] - 1) // 2
    assert_error(params["edge_indices"], expected_index, 0)
    change_det = int(Matrix(params["change_basis"].tolist()).det())
    if abs(change_det) != 1:
        raise AssertionError("coordinate basis action is not unimodular")
    result["finite_complex"] = dict(triangle_count=triangle_count, maximum_exact_closure_defect=closure,
        connected_components=components, independent_DFS_graph_cycle_count=chords,
        period_image_HNF=hnf.tolist(), active_b1_from_frozen_F47_barcode=active_b1,
        float32_float64_edge_masks_equal=float32_equal, no_bar_endpoint_tie=not endpoint_tie,
        full_primitive_free_basis_of_finite_complex=finite_certificate,
        true_manifold_or_phase_faithfulness_certified=False)
    with np.load(row["frozen_output"], allow_pickle=False) as archive:
        encoded = {key: archive[key].copy() for key in archive.files}
    for name, _, _ in blocks:
        for key in ("window_start_index", "window_end_index",
                    "time_at_window_start", "time_at_window_end"):
            assert_error(encoded[name + "_" + key], params[name + "_" + key], 0)
    if "window_end_index" in encoded:
        assert_error(encoded["window_end_index"], endpoints, 0)
    phi, covered = replay_phases(params, metric(land_H, H))
    replay_defects = {}
    for name, lo, hi in blocks:
        replay_defects[name] = assert_error(phi[:, lo:hi], encoded["initial_" + name + "_coordinates"], angular=True)
        if name != "train":
            assert_error(covered[lo:hi], encoded[name + "_covered"], 0)
    assert_error(phi[:, :n_train], params["train_phases"], angular=True)
    result["phase_replay_max_errors"] = replay_defects
    train_dist = metric(H[:, :n_train])[:-1, :-1]
    edges = encoded["source_edges"]
    edge_distances = train_dist[edges[:, 0], edges[:, 1]]
    result["source_edge_distance_defect"] = assert_error(edge_distances, encoded["source_edge_distances"])
    # Validate fixed-k union neighbour graph, allowing only numerical kth-distance ties.
    d = train_dist.copy()
    np.fill_diagonal(d, np.inf)
    k = min(settings["source_neighbours"], len(d) - 1)
    near = np.argpartition(d, k - 1, axis=1)[:, :k]
    expected = np.unique(np.sort(np.column_stack((np.repeat(np.arange(len(d)), k), near.ravel())), axis=1), axis=0)
    edge_set, expected_set = set(map(tuple, edges)), set(map(tuple, expected))
    differing = edge_set.symmetric_difference(expected_set)
    cut = np.partition(d, k - 1, axis=1)[:, k - 1]
    for u, v in differing:
        if min(abs(d[u, v] - cut[u]), abs(d[u, v] - cut[v])) > 2e-7:
            raise AssertionError("source neighbour graph differs beyond a numerical distance tie")
    result["source_graph_numerical_tie_edge_count"] = len(differing)
    ratios, train_delta, lift_valid, lift_diag = source_lift(phi[:, :n_train], edges)
    assert_error(ratios, encoded["ratio_angles"], angular=True)
    stored_lift_valid = "train_step_lift" in encoded
    if lift_valid != stored_lift_valid:
        raise AssertionError("independent source-graph lift status differs")
    result["source_lift"] = dict(lift_diag, usable=lift_valid)
    if not lift_valid:
        if any(name + "_coordinates" in encoded for name, _, _ in blocks):
            raise AssertionError("unresolved graph produced averaged phases")
        return dict(result, audit_status="passed_unresolved_graph_preserved",
                    encoding_status=row["encoding_diagnostics"]["status"])
    assert_error(train_delta, encoded["train_step_lift"])
    queries = H[:, n_train - 1:-1]
    distances = metric(H[:, :n_train - 1], queries)
    nearest = encoded["extension_nearest_source"].astype(int)
    chosen_distance = distances[nearest, np.arange(len(nearest))]
    assert_error(chosen_distance, encoded["extension_nearest_source_distance"])
    if np.max(chosen_distance - distances.min(axis=0)) > 2e-7:
        raise AssertionError("frozen source extension is not nearest in full Hankel metric")
    reference = train_delta[:, nearest]
    principal = wrap(np.diff(phi[:, n_train - 1:], axis=1))
    proposed = principal + TAU * np.rint((reference - principal) / TAU)
    margin = np.pi - np.abs(proposed - reference)
    support = float(np.max(encoded["source_edge_distances"]))
    valid = np.isfinite(principal).all(axis=0) & (margin.min(axis=0) > 1e-6) & (chosen_distance <= support)
    assert_error(margin, encoded["extension_branch_margin"])
    assert_error(valid, encoded["extension_valid"], 0)
    proposed[:, ~valid] = np.nan
    delta = np.concatenate((train_delta, proposed), axis=1)
    assert_error(delta, encoded["extended_step_lift"])
    averaged = causal_average(phi, delta, settings["averaging_window"])
    counts, averages = {}, {}
    for name, lo, hi in blocks:
        averages[name] = assert_error(averaged[:, lo:hi], encoded[name + "_coordinates"], angular=True)
        mask = np.isfinite(averaged[:, lo:hi]).all(axis=0)
        assert_error(mask, encoded[name + "_encoded_valid"], 0)
        raw_start = np.where(mask, endpoints[lo:hi] - adapter["lag_window"] - settings["averaging_window"] + 2, -1)
        raw_end = np.where(mask, endpoints[lo:hi], -1)
        assert_error(raw_start, encoded[name + "_encoded_raw_start_index"], 0)
        assert_error(raw_end, encoded[name + "_encoded_raw_end_index"], 0)
        if np.any(raw_start[mask] < 0) or np.any(raw_end[mask] - raw_start[mask] + 1 != 159):
            raise AssertionError("encoded support is not exactly 159 causal observations")
        counts[name] = int(mask.sum())
    if counts != row["encoded_coordinate_valid_counts"]:
        raise AssertionError("encoded coverage counts differ")
    result.update(audit_status="passed_conditional_encoder", encoding_status=row["encoding_diagnostics"]["status"],
        encoded_counts=counts, average_replay_max_errors=averages,
        invalid_extension_count=int(np.sum(~valid)), fixed_Q=32,
        encoded_raw_support=159, causal_no_later_observation_used=True)
    return result


def run():
    if OUTPUT.exists():
        raise FileExistsError("preserve completed audit")
    declaration = json.loads(PROTOCOL.read_text())
    if digest(__file__) != declaration["audit_source_sha256"]:
        raise ValueError("audit source changed after preparation")
    verify_hashes(declaration["protected_fingerprints"])
    source = terminal_source()
    started, rows = time.perf_counter(), []
    for entry, row in zip(source["manifest"], source["rows"]):
        if entry["run"] != row["run"]:
            raise ValueError("manifest/result case order differs")
        print("auditing " + row["run"], flush=True)
        rows.append(audit_case(entry, row, source["adapter_settings"], source["frontend_settings"]))
        print(json.dumps(dict(run=rows[-1]["run"], status=rows[-1]["audit_status"],
                              counts=rows[-1].get("encoded_counts"))), flush=True)
    verify_hashes(declaration["protected_fingerprints"])
    result = dict(status="finished_independent_public_only_audit", rows=rows,
        audit_source_sha256=digest(__file__), protocol_sha256=digest(PROTOCOL),
        source_result_sha256=digest(SOURCE), elapsed_seconds=time.perf_counter() - started,
        all_frozen_sources_inputs_archives_unchanged=True,
        true_parameters_or_evaluation_archives_read=False, inference_helper_calls=False,
        persistent_homology_or_field_fits_rerun=False,
        scope="numerical/algebraic replay of frozen finite-complex encoder; no true topology or recovery certificate",
        summary=dict(attempted_cases=len(rows), passed_cases=len(rows),
            usable_conditional_encoders=sum(row["audit_status"] == "passed_conditional_encoder" for row in rows),
            preserved_unresolved_cases=sum(row["audit_status"] != "passed_conditional_encoder" for row in rows)))
    # SymPy HNF entries are normalized only for serialization, never numerical arithmetic.
    def exact_json_integer(value):
        if isinstance(value, Integer):
            return int(value)
        raise TypeError("unexpected audit JSON value " + type(value).__name__)
    result = json.loads(json.dumps(result, default=exact_json_integer))
    write_json(OUTPUT, result)
    print(json.dumps(result["summary"], indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true")
    action.add_argument("--run", action="store_true")
    args = parser.parse_args()
    prepare() if args.prepare else run()
