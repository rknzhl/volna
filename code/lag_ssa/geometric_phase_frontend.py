"""Conditional toroidal coordinates of the full joint Hankel metric.

Known persistent-cohomology/DREiMac methods are used as a viability baseline.
No frequencies, phase count, harmonic labels, or field equation enter fitting.
The rank rule is a fixed persistence-gap heuristic. Integer periods certify
the finite complex only; its equivalence to the true manifold is unproved.
"""

from dataclasses import asdict, dataclass
import hashlib
import importlib
from importlib.metadata import version
from pathlib import Path
import warnings

import numpy as np
from scipy.spatial.distance import cdist
from sympy import Matrix
from dreimac import ToroidalCoords, PartUnity
from dreimac.combinatorial import combinatorial_number_system_d1_forward

from cohomology_primitivity import certify_primitivity


@dataclass(frozen=True)
class FrontendSettings:
    n_landmarks: int = 384
    prime: int = 47
    lifetime_gap: float = 2.0
    cover_percent: float = 0.5
    standard_range: bool = True
    check_integer_cocycles: bool = True
    source_neighbours: int = 12
    averaging_window: int = 32
    rank_selection: str = "largest lifetime gap among common-standard-range prefixes"
    partition: str = "exponential"


SETTINGS = FrontendSettings()


def library_fingerprints():
    modules = ["dreimac.toroidalcoords", "dreimac.emcoords", "dreimac.utils",
               "dreimac.combinatorial", "ripser.ripser"]
    sources = {}
    for name in modules:
        path = Path(importlib.import_module(name).__file__)
        sources[name] = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return dict(versions={name: version(name) for name in
                         ("dreimac", "ripser", "numpy", "scipy", "numba", "sympy")},
                sources=sources)


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


def _select_prefix(diagram, settings):
    """Heuristic fixed before fields are fitted; no topological rank promise."""
    dgm = np.asarray(diagram, dtype=float)
    if not len(dgm) or not np.isfinite(dgm).all():
        return None, dict(reason="no_finite_H1_diagram")
    lives = dgm[:, 1] - dgm[:, 0]
    candidates = []
    for count in range(1, len(dgm) + 1):
        birth = float(np.max(dgm[:count, 0]))
        death = float(np.min(dgm[:count, 1]))
        if not 2 * birth < death:
            continue
        # Include all discarded bars when measuring the gap, not just eligible ones.
        denominator = float(lives[count]) if count < len(lives) else 0.0
        ratio = float(lives[count - 1] / denominator) if denominator > 0 else float("inf")
        if ratio >= settings.lifetime_gap:
            candidates.append((ratio, -count, count, birth, death))
    if not candidates:
        return None, dict(reason="no_declared_persistence_gap_with_common_standard_range")
    ratio, _, count, birth, death = max(candidates)
    return count, dict(selected_count=count, lifetime_gap=None if np.isinf(ratio) else ratio,
                       common_birth=birth, common_death=death,
                       rank_certified_for_true_manifold=False,
                       criterion=settings.rank_selection)


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


def _partition(dist_land_data, radius):
    active = dist_land_data < radius
    raw = np.zeros_like(dist_land_data)
    raw[active] = PartUnity.exp(dist_land_data[active], radius)
    sums = np.sum(raw, axis=0)
    covered = (sums > 0) & np.isfinite(sums)
    weights = raw / np.where(covered, sums, 1.0)[None, :]
    membership = np.argmax(active, axis=0)  # first active chart, as in DREiMac
    return weights, membership, covered


def _integrate(harmonic, integral, weights, membership, edges, edge_indices):
    """Integrate the closed cochain in overlapping charts, in angle units.

    Antisymmetry permits any active root. DREiMac's first-active-root formula
    uses only upper entries and is equivalent under that root convention.
    """
    count = len(integral)
    connection = np.zeros((count, count))
    connection[edges[:, 0], edges[:, 1]] = harmonic[edge_indices]
    connection[edges[:, 1], edges[:, 0]] = -harmonic[edge_indices]
    values = integral[membership] + np.einsum("ij,ji->i", connection[membership], weights)
    return np.mod(2 * np.pi * values, 2 * np.pi)


def fit_hankel_topology(H_train, settings=SETTINGS):
    if isinstance(settings, dict):
        settings = FrontendSettings(**settings)
    H = np.asarray(H_train, dtype=float)
    distances = _distances(H)
    # Direct distance controls avoid interpreting an exact-metric representation as a truncation.
    control = np.unique(np.linspace(0, H.shape[1] - 1, 9).astype(int))
    direct = cdist(H[:, control].T, H[:, control].T)
    error = float(np.max(np.abs(direct - distances[np.ix_(control, control)])))
    if error > 2e-7 * max(1.0, float(direct.max())):
        raise FloatingPointError("full metric failed direct independent distance check")
    tc = ToroidalCoords(distances, n_landmarks=min(settings.n_landmarks, H.shape[1]),
                        distance_matrix=True, prime=settings.prime, maxdim=1, verbose=False)
    dgm = np.asarray(tc._dgms[1])
    selected, rank_diag = _select_prefix(dgm, settings)
    diag = dict(status="unresolved", settings=asdict(settings), selected_rank=selected,
                rank_selection=rank_diag, direct_distance_control_error=error,
                geometric_rank_certified=False, primitive_true_phase_basis_certified=False,
                finite_complex_equivalence_to_manifold_certified=False,
                energy_truncation_applied=False, train_state_count=H.shape[1],
                landmark_count=len(tc._idx_land), library_warnings=[])
    model = dict(diagnostics=diag, h1_diagram=dgm, train_distances=distances,
                 landmark_indices=tc._idx_land.copy(), train_phases=None)
    if selected is None:
        return model
    birth, death = rank_diag["common_birth"], rank_diag["common_death"]
    radius, threshold = tc.get_cover_radius(settings.cover_percent, birth, death,
                                            settings.standard_range)
    diag.update(cover_radius=float(radius), rips_threshold=float(threshold),
                train_coverage_radius=float(np.max(np.min(tc._dist_land_data, axis=0))))
    _, _, covered = _partition(tc._dist_land_data, radius)
    diag["train_uncovered_count"] = int(np.sum(~covered))
    if not np.all(covered):
        diag["failure_reason"] = "selected_class_cover_does_not_cover_all_train_states"
        return model
    edges, indices = _edge_representation(tc._dist_land_land, threshold, tc._cns_lookup_table)
    captured = []
    module = importlib.import_module("dreimac.toroidalcoords")
    original_smooth, original_integrate = module._integrate_harmonic_representative, module._sparse_integrate

    def capture_smooth(cocycle, boundary, sqrt_inner):
        if not np.issubdtype(np.asarray(cocycle).dtype, np.integer):
            raise ValueError("integer cocycle was not represented by exact integer entries")
        harmonic, integral = original_smooth(cocycle, boundary, sqrt_inner)
        if not np.isfinite(harmonic).all() or not np.isfinite(integral).all():
            raise FloatingPointError("nonfinite harmonic cocycle or chart integral")
        captured.append(dict(integer_cocycle=np.asarray(cocycle, dtype=np.int64).copy(),
                             harmonic=harmonic.copy(), integral=integral.copy()))
        return harmonic, integral

    def frozen_integrate(harmonic, integral, weights, membership, *_):
        return _integrate(harmonic, integral, weights, membership, edges, indices)

    try:
        module._integrate_harmonic_representative = capture_smooth
        module._sparse_integrate = frozen_integrate
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            train_phi = np.asarray(tc.get_coordinates(
                perc=settings.cover_percent, cocycle_idxs=list(range(selected)),
                partunity_fn=PartUnity.exp, standard_range=settings.standard_range,
                check_cocycle_condition=settings.check_integer_cocycles))
        diag["library_warnings"] = [str(item.message) for item in caught]
    except Exception as error:
        diag["failure_reason"] = "integer_circular_coordinate_construction_failed"
        diag["coordinate_construction_exception"] = f"{type(error).__name__}: {error}"
        return model
    finally:
        module._integrate_harmonic_representative = original_smooth
        module._sparse_integrate = original_integrate
    if len(captured) != selected or train_phi.shape != (selected, H.shape[1]):
        raise ValueError("unexpected DREiMac harmonic-coordinate dimensions")
    if not np.isfinite(train_phi).all():
        diag["failure_reason"] = "nonfinite_training_circular_coordinates"
        return model
    change = np.asarray(getattr(tc, "_change_basis", np.eye(selected)), dtype=float)
    if not np.isfinite(change).all():
        raise FloatingPointError("nonfinite lattice-coordinate change")
    if not np.allclose(change, np.rint(change), atol=1e-9, rtol=0):
        raise ValueError("lattice reduction returned noninteger coordinate change")
    change = np.rint(change).astype(np.int64)
    determinant = int(Matrix(change.tolist()).det())
    if abs(determinant) != 1:
        raise ValueError("lattice reduction returned nonunimodular coordinate change")
    integer_values = np.asarray([row["integer_cocycle"][indices] for row in captured])
    closed, closure = _closed_integer_cocycles(integer_values, edges, len(tc._idx_land))
    active = int(np.sum((dgm[:, 0] < threshold) & (dgm[:, 1] > threshold)))
    # Ripser computes with float32 edge filtration values. The integer cochains
    # are checked on DREiMac's float64 strict-threshold graph; certify equality.
    adjacency64 = tc._dist_land_land < threshold
    adjacency32 = tc._dist_land_land.astype(np.float32).astype(float) < threshold
    changed_edges = int(np.sum(np.triu(adjacency64 != adjacency32, 1)))
    at_bar_endpoint = bool(np.any(dgm == threshold))
    same_complex = changed_edges == 0 and not at_bar_endpoint
    certificate = certify_primitivity(edges, integer_values, active,
        vertex_count=len(tc._idx_land), integer_closure_verified=closed,
        same_complex_verified=same_complex, prime=settings.prime, return_cycle_matrix=True)
    cycle_periods = np.asarray(certificate.pop("graph_cycle_period_matrix_integer", np.empty((selected, 0))),
                               dtype=np.int64)
    diag.update(integer_closure=closure, finite_complex_certificate=certificate,
                active_H1_at_coordinate_scale=active, lattice_change_determinant=determinant,
                float32_float64_filtration_edge_disagreements=changed_edges,
                coordinate_scale_equals_bar_endpoint=at_bar_endpoint,
                same_finite_complex_verified=same_complex)
    if not closed:
        diag["failure_reason"] = "repaired_integer_cocycle_failed_exact_triangle_closure"
        return model
    model.update(landmark_columns=H[:, tc._idx_land].copy(),
        landmark_distances=tc._dist_land_land.copy(), cover_radius=float(radius),
        rips_threshold=float(threshold), edges=edges, edge_indices=indices,
        integer_cocycles=integer_values, harmonics=np.asarray([row["harmonic"] for row in captured]),
        integrals=np.asarray([row["integral"] for row in captured]), change_basis=change,
        graph_cycle_period_matrix=cycle_periods,
        train_phases=train_phi, certificate=certificate)
    replay = transform_hankel_topology(model, H)
    if not replay["covered"].all() or not np.isfinite(replay["phi"]).all():
        raise FloatingPointError("frozen phase extension returned nonfinite training coordinates")
    wrapped = (replay["phi"] - train_phi + np.pi) % (2 * np.pi) - np.pi
    replay_error = float(np.max(np.abs(wrapped)))
    diag["frozen_training_replay_error"] = replay_error
    if replay_error > 2e-7:
        raise FloatingPointError("frozen phase extension did not reproduce training coordinates")
    diag["status"] = "conditional_finite_complex_coordinates"
    return model


def transform_hankel_topology(model, H_block):
    if model.get("train_phases") is None:
        return dict(phi=None, covered=None, diagnostics={"status": "unresolved_model"})
    distances = _distances(model["landmark_columns"], H_block)
    weights, membership, covered = _partition(distances, model["cover_radius"])
    native = np.array([_integrate(harmonic, integral, weights, membership,
                                  model["edges"], model["edge_indices"])
                       for harmonic, integral in zip(model["harmonics"], model["integrals"])])
    phi = np.mod(np.einsum("ij,jk->ik", model["change_basis"], native, optimize=False), 2 * np.pi)
    phi[:, ~covered] = np.nan
    nearest = np.min(distances, axis=0)
    return dict(phi=phi, covered=covered, nearest_landmark_distance=nearest,
                diagnostics=dict(status="frozen_coordinate_transform",
                                 uncovered_count=int(np.sum(~covered)),
                                 maximum_nearest_landmark_distance=float(nearest.max()),
                                 fit_or_topology_recomputed=False))


def frozen_arrays(model):
    result = dict(h1_diagram=model["h1_diagram"], landmark_indices=model["landmark_indices"])
    for key in ("landmark_columns", "landmark_distances", "edges", "edge_indices", "integer_cocycles",
                "harmonics", "integrals", "change_basis", "graph_cycle_period_matrix", "train_phases"):
        if key in model and model[key] is not None:
            result[key] = model[key]
    if "cover_radius" in model:
        result["cover_radius"] = np.asarray(model["cover_radius"])
        result["rips_threshold"] = np.asarray(model["rips_threshold"])
    return result
