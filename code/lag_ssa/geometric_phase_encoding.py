"""Frozen source-graph phase lift and causal averaging of learned coordinates.

This is a conditional viability encoder, not a global topology certificate.
Its training graph is built in the original full joint-Hankel metric. New
step-ratio branches are extended from training source states; future states
never change training topology, integer classes, or graph branch assignments.
"""

import numpy as np

from geometric_phase_frontend import _distances, transform_hankel_topology
from phase_lift_average import phase_lift_average, triangular_average, wrap_angles


def source_neighbour_graph(distances, neighbours):
    d = np.asarray(distances, dtype=float).copy()
    if d.ndim != 2 or d.shape[0] != d.shape[1] or len(d) < 2:
        raise ValueError("square source-state distances required")
    count = min(int(neighbours), len(d) - 1)
    if count < 1:
        raise ValueError("at least one source neighbour required")
    np.fill_diagonal(d, np.inf)
    near = np.argpartition(d, count - 1, axis=1)[:, :count]
    edges = np.column_stack((np.repeat(np.arange(len(d)), count), near.ravel()))
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    return edges, np.asarray(distances)[edges[:, 0], edges[:, 1]]


def _average_valid_segments(phi, delta, window):
    """Same fixed window everywhere; output is indexed by its last observation."""
    phi = np.asarray(phi, dtype=float)
    delta = np.asarray(delta, dtype=float)
    result = np.full_like(phi, np.nan)
    finite_phi = np.isfinite(phi).all(axis=0)
    finite_delta = np.isfinite(delta).all(axis=0)
    start = 0
    while start < phi.shape[1]:
        if not finite_phi[start]:
            start += 1
            continue
        stop = start + 1
        while stop < phi.shape[1] and finite_phi[stop] and finite_delta[stop - 1]:
            stop += 1
        if stop - start >= window:
            averaged = triangular_average(phi[:, start:stop].T,
                                          delta[:, start:stop - 1].T, window)
            result[:, start + averaged["end_indices"]] = averaged["phases"].T
        start = stop
    return result


def encode_hankel_blocks(model, blocks, settings):
    """Fit the increment graph on TRAIN, then transform/encode fixed blocks."""
    if model.get("train_phases") is None:
        return dict(arrays={}, diagnostics=dict(status="unresolved_coordinate_frontend"))
    H_train = blocks["train"]["H"]
    phi_train = model["train_phases"]
    states = phi_train.shape[1]
    source_distances = model["train_distances"][:-1, :-1]
    edges, edge_distances = source_neighbour_graph(source_distances, settings.source_neighbours)
    fitted_lift = phase_lift_average(phi_train.T, edges, settings.averaging_window,
                                    edge_distances=edge_distances)
    transformed = {name: transform_hankel_topology(model, blocks[name]["H"])
                   for name in ("validation", "future")}
    phi = np.concatenate((phi_train, transformed["validation"]["phi"],
                          transformed["future"]["phi"]), axis=1)
    endpoints = np.concatenate([blocks[name]["window_end_index"]
                                for name in ("train", "validation", "future")])
    arrays = dict(initial_train_coordinates=phi_train,
                  initial_validation_coordinates=transformed["validation"]["phi"],
                  initial_future_coordinates=transformed["future"]["phi"],
                  validation_covered=transformed["validation"]["covered"],
                  future_covered=transformed["future"]["covered"],
                  source_edges=edges, source_edge_distances=edge_distances,
                  ratio_angles=fitted_lift["ratio_angles"].T,
                  ratio_branch_integers=fitted_lift["branch_integers"].T,
                  window_end_index=endpoints)
    diag = dict(status="unresolved_training_increment_lift", train_lift=fitted_lift["diagnostics"],
                train_lift_failure_reasons=fitted_lift["failure_reasons"],
                validation_transform=transformed["validation"]["diagnostics"],
                future_transform=transformed["future"]["diagnostics"],
                future_used_for_training_lift=False,
                hidden_increment_turns_certified_absent=False,
                true_manifold_primitive_basis_certified=False,
                phase_averaging_window=settings.averaging_window,
                output_indexing="last observed state of fixed-Q causal average",
                raw_observation_support="L+Q-1",
                independent_noise_averaging_rate_claimed=False)
    if fitted_lift["delta"] is None:
        return dict(arrays=arrays, diagnostics=diag)
    delta = np.full((phi.shape[0], phi.shape[1] - 1), np.nan)
    delta[:, :states - 1] = fitted_lift["delta"].T
    H_queries = np.concatenate([blocks[name]["H"] for name in ("train", "validation", "future")], axis=1)
    # First query is the last training state, whose outgoing ratio was not
    # used during fitting; remaining queries are held-out source states.
    query_start = states - 1
    distance = _distances(H_train[:, :-1], H_queries[:, query_start:-1])
    nearest = np.argmin(distance, axis=0)
    nearest_distance = distance[nearest, np.arange(distance.shape[1])]
    reference = fitted_lift["delta"][nearest].T
    principal = wrap_angles((phi[:, query_start + 1:] - phi[:, query_start:-1]).T).T
    proposed = principal + 2 * np.pi * np.rint((reference - principal) / (2 * np.pi))
    margin = np.pi - np.abs(proposed - reference)
    support_radius = float(np.max(edge_distances))
    valid = (np.isfinite(principal).all(axis=0)
             & (np.min(margin, axis=0) > 1e-6)
             & (nearest_distance <= support_radius))
    proposed[:, ~valid] = np.nan
    delta[:, query_start:] = proposed
    averaged = _average_valid_segments(phi, delta, settings.averaging_window)
    train_stop = states
    validation_stop = states + blocks["validation"]["H"].shape[1]
    arrays.update(train_step_lift=fitted_lift["delta"].T,
        extended_step_lift=delta, extension_nearest_source=nearest,
        extension_nearest_source_distance=nearest_distance,
        extension_branch_margin=margin, extension_valid=valid,
        train_coordinates=averaged[:, :train_stop],
        validation_coordinates=averaged[:, train_stop:validation_stop],
        future_coordinates=averaged[:, validation_stop:])
    for name in ("train", "validation", "future"):
        encoded_valid = np.isfinite(arrays[name + "_coordinates"]).all(axis=0)
        arrays[name + "_encoded_valid"] = encoded_valid
        arrays[name + "_window_start_index"] = blocks[name]["window_start_index"]
        arrays[name + "_window_end_index"] = blocks[name]["window_end_index"]
        arrays[name + "_encoded_raw_start_index"] = np.where(encoded_valid,
            blocks[name]["window_start_index"] - settings.averaging_window + 1, -1)
        arrays[name + "_encoded_raw_end_index"] = np.where(encoded_valid,
            blocks[name]["window_end_index"], -1)
    diag.update(status="conditional_source_graph_phase_encoding",
        source_neighbour_support_radius=support_radius,
        extension_invalid_ratio_count=int(np.sum(~valid)),
        maximum_extension_nearest_source_distance=float(nearest_distance.max()),
        nearest_source_extension_is_a_heuristic=True,
        train_encoded_count=int(np.sum(np.isfinite(arrays["train_coordinates"]).all(axis=0))),
        validation_encoded_count=int(np.sum(np.isfinite(arrays["validation_coordinates"]).all(axis=0))),
        future_encoded_count=int(np.sum(np.isfinite(arrays["future_coordinates"]).all(axis=0))))
    diag["window_index_keys_describe"] = "initial L-sample Hankel windows; encoded_raw keys include the Q-phase average"
    if diag["future_encoded_count"] == 0:
        diag["status"] = "unresolved_no_future_states_encoded"
    return dict(arrays=arrays, diagnostics=diag)
