"""Capped landmark persistence for finite-complex period construction.

An H1 class alive at the filtration cap has an unknown death beyond that cap.
Its capped death is a lower bound, never an observed death or lifetime.
"""
import numpy as np
from ripser import ripser
from dreimac.combinatorial import combinatorial_number_system_table
from dreimac.utils import CohomologyUtils


class _BoundedPersistence:
    def get_representative_cocycle(self, cohomology_class, homological_dimension):
        diagram = self._dgms[homological_dimension]
        return (diagram[cohomology_class, 0], diagram[cohomology_class, 1],
                self._cocycles[homological_dimension][cohomology_class])


def bounded_persistence(metric, n_landmarks, prime, filtration_cap):
    """Return the geometry/cocycle subset of the ToroidalCoords interface.

    The source metric and greedy landmark convention are unchanged. H1 bars
    are sorted by their observed or conservative cap-bounded lifetime;
    representatives and censoring flags follow exactly the same permutation.
    Cocycles use landmark-local vertex indices, as in DREiMac.
    """
    distance = np.asarray(metric, dtype=float)
    if (distance.ndim != 2 or distance.shape[0] != distance.shape[1]
            or len(distance) < 4 or not np.isfinite(distance).all()
            or np.any(distance < 0)
            or not np.allclose(distance, distance.T, rtol=1e-12, atol=1e-12)
            or np.any(np.diag(distance) != 0)):
        raise ValueError('finite symmetric source metric with zero diagonal required')
    if (isinstance(n_landmarks, (bool, np.bool_)) or int(n_landmarks) != n_landmarks
            or not 4 <= n_landmarks <= len(distance)):
        raise ValueError('landmark count must lie between four and the source size')
    if (isinstance(prime, (bool, np.bool_)) or int(prime) != prime or prime < 2):
        raise ValueError('integer prime coefficient at least two required')
    if isinstance(filtration_cap, (bool, np.bool_)):
        raise ValueError('positive finite filtration cap required')
    cap = float(filtration_cap)
    if not np.isfinite(cap) or cap <= 0:
        raise ValueError('positive finite filtration cap required')
    count = int(n_landmarks)
    result = ripser(distance, distance_matrix=True, n_perm=count,
        maxdim=1, coeff=int(prime), thresh=cap, do_cocycles=True)
    model = _BoundedPersistence()
    model._prime = int(prime); model._filtration_cap = cap
    model._idx_land = np.asarray(result['idx_perm'], dtype=np.int64).copy()
    model._n_landmarks = len(model._idx_land)
    model._dist_land_data = np.asarray(result['dperm2all'], dtype=float)
    model._dist_land_land = model._dist_land_data[:, model._idx_land]
    model._coverage = float(np.max(np.min(model._dist_land_data, axis=0)))
    raw = [np.asarray(diagram, dtype=float).copy() for diagram in result['dgms']]
    cocycles = [[np.asarray(cocycle, dtype=np.int64).copy() for cocycle in dimension]
                for dimension in result['cocycles']]
    # Ripser returns source-global vertex IDs after greedy subsampling.
    # DREiMac applies this same reindexing before using combinatorial indices.
    CohomologyUtils.reindex_cocycles(cocycles, model._idx_land, len(distance))
    for dimension in cocycles:
        for cocycle in dimension:
            vertices = cocycle[:, :-1]
            if np.any(vertices < 0) or np.any(vertices >= model._n_landmarks):
                raise ArithmeticError('cocycle contains a vertex outside the selected landmarks')
    diagram = raw[1].copy(); censored = np.isposinf(diagram[:, 1])
    if not np.isfinite(diagram[:, 0]).all() or np.isneginf(diagram[:, 1]).any():
        raise FloatingPointError('invalid H1 persistence endpoints')
    diagram[censored, 1] = cap
    if not np.isfinite(diagram).all() or np.any(diagram[:, 1] < diagram[:, 0]):
        raise FloatingPointError('invalid conservative capped H1 diagram')
    order = np.argsort(diagram[:, 0]-diagram[:, 1], kind='stable')
    model._dgms = [raw[0].copy(), diagram[order]]
    model.dgms_ = model._dgms
    model._cocycles = [cocycles[0], [cocycles[1][i] for i in order]]
    model._h1_right_censored = censored[order]
    model._raw_h1_diagram = raw[1][order].copy()
    model.raw_diagrams = raw
    model._h1_sort_order = order
    model._cns_lookup_table = combinatorial_number_system_table(model._n_landmarks, 2)
    model._bounded_diagnostics = dict(filtration_cap=cap,
        right_censored_h1_count=int(np.sum(censored)),
        h1_deaths_at_cap_are_lower_bounds=True,
        sort_rule='decreasing conservative cap-bounded lifetime',
        raw_h1_diagram=model._raw_h1_diagram,
        h1_right_censored=model._h1_right_censored,
        no_phase_coordinates_computed=True)
    return model
