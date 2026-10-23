"""Rejected experiment: local primitives of the field's fitted closed cochain.

These charts only extend the field-derived integration potentials. They do not
estimate vector fields or use independent phase coordinates as targets.
"""
import numpy as np

from fields_first_tangent_atlas import evaluate_tangent_frame, _exponents, _poly


def fit_phase_charts(model):
    """Fit quadratic local primitives on the existing Rips neighbours.

    A chart at landmark i has value q_i there. On a neighbour j its lifted
    target is q_j + C_ij, where q and C are the fitted normalized potential
    and integer cocycle. Coordinates use the frozen tangent frame and existing
    cover radius. A rank-deficient design retains the old linear beta jet.
    """
    cx = model['complex']; centers = np.asarray(cx['landmarks'], dtype=float)
    potential = np.asarray(model['period_model']['normalized_potential'], dtype=float).T
    slopes = np.asarray(model['center_normalized_beta'], dtype=float)
    q = evaluate_tangent_frame(model['atlas'], centers)['Q']
    n, p = centers.shape; r = int(model['r'])
    if potential.shape != (n, r) or slopes.shape != (n, r, p) or q.shape != (n, p, r):
        raise ValueError('phase chart inputs have incompatible shapes')
    radius = float(cx['cover_radius'])
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError('positive finite phase cover radius required')
    powers = _exponents(r, degree=2)
    coefficients = np.zeros((n, len(powers), r))
    neighbours = [[] for _ in range(n)]
    for (i, j), label in zip(cx['edges'], np.asarray(cx['integer_cocycles']).T):
        neighbours[i].append((int(j), label))
        neighbours[j].append((int(i), -label))
    fallback = np.zeros(n, dtype=bool); ranks = np.zeros(n, dtype=int)
    residuals = np.full(n, np.nan)
    for i, adjacent in enumerate(neighbours):
        if adjacent:
            ids = np.asarray([j for j, _ in adjacent], dtype=int)
            labels = np.asarray([label for _, label in adjacent])
            u = (centers[ids] - centers[i]) @ q[i] / radius
            design = _poly(u, powers)[0]
            target = potential[ids] + labels - potential[i]
            coefficient, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
            ranks[i] = rank
            if rank == len(powers) and np.isfinite(coefficient).all():
                coefficients[i] = coefficient
                residuals[i] = np.sqrt(np.mean((design @ coefficient - target)**2))
                continue
        fallback[i] = True
        # Store the tangent representation too; evaluation below preserves the
        # original ambient linear jet exactly in every fallback chart.
        for a in range(r):
            k = np.flatnonzero(np.all(powers == np.eye(r, dtype=int)[a], axis=1))[0]
            coefficients[i, k] = radius * (slopes[i] @ q[i, :, a])
    fitted = residuals[~fallback]
    return dict(centers=centers.copy(), potential=potential.copy(), Q=q,
        coefficients=coefficients, powers=powers, radius=radius,
        linear_fallback=fallback, linear_slopes=slopes.copy(),
        diagnostics=dict(degree=2, chart_count=n, quadratic_chart_count=int(np.sum(~fallback)),
            linear_fallback_count=int(np.sum(fallback)), minimum_design_rank=int(ranks.min()),
            design_columns=len(powers),
            local_fit_rms_quantiles=np.quantile(fitted, [0, .5, .9, 1.]) if len(fitted) else [],
            targets='local lifts of the field-derived closed integration potential',
            fields_refitted=False, independent_phase_targets_used=False))


def evaluate_phase_charts(charts, query, derivative=False):
    """Return all chart values [query,landmark,r] and optional Jacobians.

    Values already include q_i. The encoder adds its usual integer chart
    transition C_root,i before applying the existing partition weights.
    Jacobians have shape [query,landmark,r,ambient].
    """
    x = np.atleast_2d(np.asarray(query, dtype=float))
    centers = charts['centers']; q = charts['Q']; radius = charts['radius']
    if x.shape[1] != centers.shape[1] or not np.isfinite(x).all():
        raise ValueError('finite phase chart query with matching ambient dimension required')
    delta = x[:, None] - centers[None]
    u = np.einsum('nlp,lpa->nla', delta, q) / radius
    monomials, first, _ = _poly(u, charts['powers'])
    local = charts['potential'][None] + np.einsum('nlt,ltr->nlr', monomials, charts['coefficients'])
    fallback = charts['linear_fallback']
    if np.any(fallback):
        local[:, fallback] = charts['potential'][None, fallback] + np.einsum(
            'lrp,nlp->nlr', charts['linear_slopes'][fallback], delta[:, fallback])
    jacobian = None
    if derivative:
        jacobian = np.einsum('nlta,ltr,lpa->nlrp', first, charts['coefficients'], q) / radius
        if np.any(fallback):
            jacobian[:, fallback] = charts['linear_slopes'][None, fallback]
    return local, jacobian
