"""Version 2: low-subspace angle probe from a supplied invariant tangent frame.

Single estimator correction: joint derivative consistency is measured inside
the low spectral subspace, and candidates are ordered by positive graph
Dirichlet energy. Full collocation leakage is retained as a diagnostic.
Acceptance tolerances and low-mode budget are unchanged from version 1.

No sensor coordinates, frequencies, phase labels, or persistence enter this
module.  The frame must already have been estimated by the caller.  This is
an r=1/2 numerical probe, not an estimator of the frame or its dimension.
On an exact flat circle/2-torus the shortest independent dual-lattice vectors
form a primitive basis.  Finite graph consistency does NOT certify that the
estimated metric is flat, that all low modes were found, or that the source
cloud represents the true manifold faithfully.  Failure returns no phases.
"""

import numpy as np
from scipy.linalg import eigh, qr
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path
from scipy.spatial import cKDTree


def _failure(reason, diagnostic):
    diagnostic["status"] = "unresolved_" + reason
    return {"phases": None, "characters": None, "diagnostics": diagnostic}


def extract_tangent_overlap_angles(X, V, neighbours=None, *, neighbour_count=24,
                                  mode_count=24, modulus_cv_limit=0.20,
                                  derivative_residual_limit=0.30,
                                  independent_sine_min=0.20, max_states=3000):
    """Return guarded approximate S1 coordinates for a SUPPLIED frame.

    X[N,p]: window states. V[N,p,r]: invariant tangent fields. neighbours[N,k]
    optionally supplies source-state neighbours (self entries are ignored).
    r is read from V, never inferred here. The returned angles have arbitrary
    origin and orientation. There is no physical zero or true-manifold
    certificate. Bounds are fixed numerical acceptance heuristics.
    """
    x, v = np.asarray(X, dtype=float), np.asarray(V, dtype=float)
    if x.ndim != 2 or v.ndim != 3 or v.shape[:2] != x.shape:
        raise ValueError("X[N,p] and V[N,p,r] must have matching first dimensions")
    n, p = x.shape
    r = v.shape[2]
    diagnostic = dict(status="started", states=n, ambient_dimension=p,
        supplied_frame_dimension=r, dimension_inferred_here=False,
        requested_low_mode_count=int(mode_count), no_truth_or_frequency_input=True,
        no_persistence_or_Betti_computation=True,
        angle_basis_rule="lowest positive graph Dirichlet energy, independent joint weights",
        derivative_acceptance_space="projected nonconstant low spectral subspace",
        full_collocation_leakage_used_for_acceptance=False,
        true_manifold_primitive_basis_certified=False,
        global_phase_faithfulness_certified=False,
        exact_primitive_basis_statement_conditional_on_flat_r1_or_r2=True,
        numerical_acceptance_limits=dict(modulus_cv=float(modulus_cv_limit),
            derivative_residual=float(derivative_residual_limit),
            independent_sine=float(independent_sine_min)))
    if r not in (1, 2):
        return _failure("frame_dimension_not_supported", diagnostic)
    if n < max(32, 4 * r) or n > max_states:
        return _failure("state_count_outside_probe_limit", diagnostic)
    if not np.isfinite(x).all() or not np.isfinite(v).all():
        return _failure("nonfinite_inputs", diagnostic)
    if mode_count < 2 * r + 2:
        raise ValueError("mode_count must include constants and complete phase pairs")
    singular = np.linalg.svd(v, compute_uv=False)
    relative = singular[:, -1] / np.maximum(singular[:, 0], np.finfo(float).tiny)
    diagnostic["minimum_frame_relative_singular_value"] = float(relative.min())
    if relative.min() < 1e-7 or singular[:, -1].min() <= 0:
        return _failure("degenerate_supplied_frame", diagnostic)
    beta = np.linalg.pinv(v, rcond=1e-10)  # [N,r,p], zero on normal directions.
    if neighbours is None:
        count = min(int(neighbour_count), n - 1)
        if count < max(6, 3 * r):
            raise ValueError("too few neighbours for directional fits")
        neighbours = cKDTree(x).query(x, k=count + 1)[1]
    neighbours = np.asarray(neighbours)
    if neighbours.ndim != 2 or len(neighbours) != n or not np.issubdtype(neighbours.dtype, np.integer):
        raise ValueError("neighbours must be an integer [N,k] array")
    if neighbours.size and (neighbours.min() < 0 or neighbours.max() >= n):
        raise ValueError("neighbour indices outside source cloud")
    pairs = sorted({(min(i, int(j)), max(i, int(j)))
        for i in range(n) for j in neighbours[i] if j != i})
    if not pairs:
        return _failure("empty_source_graph", diagnostic)
    edges = np.asarray(pairs, dtype=int)
    tails, heads = edges.T
    dx = x[heads] - x[tails]
    increments = np.einsum("erp,ep->er", (beta[tails] + beta[heads]) * .5, dx)
    lengths = np.linalg.norm(increments, axis=1)
    if not np.isfinite(lengths).all() or np.any(lengths <= 1e-12):
        return _failure("degenerate_frame_edge_lengths", diagnostic)
    rows = np.r_[tails, heads]
    cols = np.r_[heads, tails]
    graph = csr_matrix((np.r_[lengths, lengths], (rows, cols)), shape=(n, n))
    components = connected_components(graph, directed=False, return_labels=False)
    diagnostic.update(source_edges=len(edges), connected_components=int(components))
    if components != 1:
        return _failure("disconnected_source_graph", diagnostic)

    # Positive energy based on estimated flat distances. Squaring central
    # first-difference matrices would create checkerboard near-zero modes.
    distances = shortest_path(graph, directed=False)
    k_radius = min(max(8, int(neighbour_count)), n - 1)
    radii = np.partition(distances, k_radius, axis=1)[:, k_radius]
    bandwidth = float(np.median(radii))
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        return _failure("invalid_flat_bandwidth", diagnostic)
    kernel = np.exp(-.5 * (distances / bandwidth) ** 2)
    kernel[distances > 4 * bandwidth] = 0
    np.fill_diagonal(kernel, 0)
    density = kernel.sum(axis=1)
    if np.any(density <= 0):
        return _failure("empty_flat_kernel_rows", diagnostic)
    kernel /= density[:, None] * density[None, :]
    mass = kernel.sum(axis=1)
    transition = kernel / np.sqrt(mass[:, None] * mass[None, :])
    laplacian = np.eye(n) - transition
    if not np.isfinite(laplacian).all():
        return _failure("nonfinite_flat_laplacian", diagnostic)
    eigenvalues, eigenvectors = eigh(laplacian,
        subset_by_index=(0, min(int(mode_count), n - 1) - 1), check_finite=False)
    functions = eigenvectors / np.sqrt(mass[:, None])
    functions -= functions.mean(axis=0)
    # The centred first eigenfunction vanishes; discard it explicitly.
    column_norm = np.linalg.norm(functions, axis=0)
    usable = column_norm > 1e-7 * max(float(column_norm.max()), 1.)
    basis, upper = qr(functions[:, usable], mode="economic")
    if basis.shape[1] < 2 * r:
        return _failure("too_few_nonconstant_low_modes", diagnostic)
    if np.min(np.abs(np.diag(upper))) < 1e-10:
        return _failure("dependent_low_mode_basis", diagnostic)
    diagnostic.update(flat_bandwidth=bandwidth,
        graph_laplacian_eigenvalues=eigenvalues.tolist(),
        nonconstant_low_basis_dimension=basis.shape[1])

    # Directional derivatives: local Taylor fits in dual-frame coordinates.
    derivative = np.zeros((r, n, n), dtype=float)
    condition = []
    for i in range(n):
        jj = np.unique(neighbours[i][neighbours[i] != i])
        if len(jj) < 2 * r + 1:
            return _failure("insufficient_local_neighbours", diagnostic)
        local = np.einsum("jap,jp->ja", .5 * (beta[i][None] + beta[jj]), x[jj] - x[i])
        gram = local.T @ local
        values = np.linalg.eigvalsh(gram)
        ratio = values[0] / max(values[-1], np.finfo(float).tiny)
        condition.append(float(ratio))
        if ratio < 1e-6:
            return _failure("degenerate_directional_neighbourhood", diagnostic)
        coefficients = np.linalg.solve(gram, local.T)
        derivative[:, i, jj] = coefficients
        derivative[:, i, i] = -coefficients.sum(axis=1)
    # Haar-skew symmetrisation and exact removal of the constant function.
    for a in range(r):
        matrix = .5 * (derivative[a] - derivative[a].T)
        row_sum = matrix.sum(axis=1)
        matrix += (row_sum[None, :] - row_sum[:, None]) / n
        derivative[a] = matrix
    projected = np.asarray([basis.T @ matrix @ basis for matrix in derivative])
    if not np.isfinite(projected).all():
        return _failure("nonfinite_projected_derivatives", diagnostic)
    diagnostic["minimum_directional_neighbourhood_relative_eigenvalue"] = min(condition)
    scale = max(float(np.linalg.norm(projected)), 1e-15)
    if r == 2:
        commutator = projected[0] @ projected[1] - projected[1] @ projected[0]
        diagnostic["projected_commutator_relative_norm"] = float(np.linalg.norm(commutator) / scale ** 2)
    # Fixed generic direction resolves equal Laplacian eigenvalues. Its
    # coefficients are not temporal frequencies or fitted phase information.
    direction = np.asarray([1., np.sqrt(2.)][:r])
    hermitian = -1j * np.einsum("a,aij->ij", direction, projected)
    joint_values, joint_vectors = eigh(hermitian, check_finite=False)
    candidates = []
    reports = []
    for j in range(len(joint_values)):
        function = basis @ joint_vectors[:, j]
        norm = float(np.linalg.norm(function))
        amplitude = np.abs(function)
        cv = float(amplitude.std() / max(amplitude.mean(), 1e-15))
        coefficients = joint_vectors[:, j]
        projected_applied = np.asarray([matrix @ coefficients for matrix in projected])
        weights = np.asarray([np.vdot(coefficients, projected_applied[a]).imag
            / np.linalg.norm(coefficients) ** 2 for a in range(r)])
        residual = float(np.linalg.norm(projected_applied - 1j * weights[:, None] * coefficients)
            / max(np.linalg.norm(coefficients) * np.linalg.norm(weights), 1e-15))
        applied = np.asarray([matrix @ function for matrix in derivative])
        full_residual = float(np.linalg.norm(applied - 1j * weights[:, None] * function)
            / max(norm * np.linalg.norm(weights), 1e-15))
        lifted_projected = np.asarray([basis @ item for item in projected_applied])
        full_leakage = float(np.linalg.norm(applied - lifted_projected)
            / max(norm * np.linalg.norm(weights), 1e-15))
        mass_function = np.sqrt(mass) * function
        graph_energy = float(np.vdot(mass_function, laplacian @ mass_function).real
            / max(np.vdot(mass_function, mass_function).real, 1e-15))
        mean_weight = float(direction @ weights)
        passed = bool(mean_weight > 1e-6 and np.linalg.norm(weights) > 1e-6
            and cv <= modulus_cv_limit and residual <= derivative_residual_limit
            and amplitude.min() > 1e-8 * amplitude.mean())
        report = dict(index=j, directional_weights=weights.tolist(),
            squared_weight_norm=float(weights @ weights), modulus_cv=cv,
            graph_dirichlet_rayleigh=graph_energy,
            derivative_relative_residual=residual,
            full_collocation_derivative_relative_residual=full_residual,
            full_collocation_leakage_relative_norm=full_leakage,
            accepted=passed)
        reports.append(report)
        if passed:
            candidates.append((graph_energy, j, weights, function))
    diagnostic["joint_character_candidates"] = reports
    candidates.sort(key=lambda item: (item[0], item[1]))
    selected = []
    for candidate in candidates:
        if not selected:
            selected.append(candidate)
        elif r == 2:
            u, w = selected[0][2], candidate[2]
            sine = abs(float(np.linalg.det(np.stack((u, w))))) / (np.linalg.norm(u) * np.linalg.norm(w))
            if sine >= independent_sine_min:
                selected.append(candidate)
        if len(selected) == r:
            break
    if len(selected) != r:
        return _failure("no_complete_accepted_phase_basis", diagnostic)
    characters = np.column_stack([item[3] / np.abs(item[3]) for item in selected])
    characters /= characters[0]
    phases = np.angle(characters)
    phase_step = np.angle(characters[heads] / characters[tails])
    diagnostic.update(status="conditional_r1_r2_angles", selected_candidate_indices=[item[1] for item in selected],
        selected_directional_weights=[item[2].tolist() for item in selected],
        maximum_source_edge_phase_step=float(np.max(np.abs(phase_step))),
        origin="first supplied state", complete_low_spectrum_verified=False,
        finite_checks_are_heuristics_not_global_certificate=True)
    if np.max(np.abs(phase_step)) >= np.pi / 2:
        return _failure("phase_edges_too_large", diagnostic)
    return {"phases": phases, "characters": characters, "diagnostics": diagnostic,
        "selected_weights": np.asarray([item[2] for item in selected])}


angles_from_invariant_frame = extract_tangent_overlap_angles
