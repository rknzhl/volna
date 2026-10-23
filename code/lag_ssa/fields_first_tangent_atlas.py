"""Frozen smooth tangent projectors fitted from observed window coordinates."""
from __future__ import annotations

import itertools
import numpy as np
from scipy import linalg
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors


def _exponents(r, degree=3):
    return np.asarray([a for total in range(1, degree + 1)
                       for a in itertools.product(range(total + 1), repeat=r)
                       if sum(a) == total], dtype=int)


def _poly(u, powers, second=False):
    """Monomials and their first (and optionally second) derivatives."""
    shape = u.shape[:-1]; r = u.shape[-1]; t = len(powers)
    value = np.ones(shape + (t,))
    first = np.zeros(shape + (t, r))
    hess = np.zeros(shape + (t, r, r)) if second else None
    degree = int(np.max(powers)) if t else 0
    coordinate_powers = []
    for a in range(r):
        table = np.empty(shape + (degree + 1,))
        table[..., 0] = 1.
        for k in range(1, degree + 1):
            table[..., k] = table[..., k - 1] * u[..., a]
        coordinate_powers.append(table)
        value *= table[..., powers[:, a]]
    for a in range(r):
        active = powers[:, a] > 0
        if np.any(active):
            term = np.ones(shape + (int(np.sum(active)),))
            for c in range(r):
                term *= coordinate_powers[c][..., powers[active, c] - int(c == a)]
            first[..., active, a] = powers[active, a] * term
        if second:
            for b in range(a, r):
                factor = powers[:, a] * (powers[:, b] - int(a == b))
                active = factor > 0
                if np.any(active):
                    term = np.ones(shape + (int(np.sum(active)),))
                    for c in range(r):
                        exponent = powers[active, c] - int(c == a) - int(c == b)
                        term *= coordinate_powers[c][..., exponent]
                    hess[..., active, a, b] = factor[active] * term
                    if b != a:
                        hess[..., active, b, a] = hess[..., active, a, b]
    return value, first, hess


def fit_atlas(X, r, k=24, kernel="compact", degree=3):
    """Fit polynomial normal graph jets at every source point.

    X has shape (n,p).  The fixed bandwidth is the median source eighth-neighbour
    distance.  Compact support is fixed at four times this bandwidth. Two
    graph-based tangent refinements precede the stored final fit.
    This is a geometric estimator, not a finite-data tangent accuracy guarantee.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or not np.isfinite(X).all():
        raise ValueError("X must be a finite state-by-coordinate matrix")
    n, p = X.shape
    if not 1 <= r < p or n < 5:
        raise ValueError("require 1 <= r < ambient dimension and at least 5 states")
    degree = int(degree)
    if degree < 1:
        raise ValueError("degree must be a positive integer")
    powers = _exponents(r, degree)
    k = min(max(int(k), len(powers) + 2, r + 2), n - 1)
    if k < len(powers):
        raise ValueError("too few states for the specified local graph degree")
    distances, indices = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X)
    distances, indices = distances[:, 1:], indices[:, 1:]
    positive = distances[:, min(7, k - 1)]
    positive = positive[positive > 0]
    if not len(positive):
        raise ValueError("source has no positive neighbour distances")
    bandwidth = float(np.median(positive))
    radii = np.maximum(distances[:, -1], bandwidth * 1e-6)
    bases = np.empty((n, p, r)); coefficients = np.empty((n, len(powers), p))
    ranks = []; residuals = []
    linear = [int(np.flatnonzero(np.all(powers == np.eye(r, dtype=int)[a], axis=1))[0])
              for a in range(r)]
    for i, js in enumerate(indices):
        delta = X[js] - X[i]
        _, _, vh = linalg.svd(delta - delta.mean(axis=0), full_matrices=False)
        q = vh[:r].T
        root_weight = np.exp(-0.5 * (distances[i] / radii[i]) ** 2)

        def graph_fit(frame):
            coord = np.einsum("kp,pa->ka", delta, frame, optimize=False) / radii[i]
            design = _poly(coord, powers)[0]
            normal = delta / radii[i] - np.einsum("ka,pa->kp", coord, frame, optimize=False)
            coef, _, rank, _ = linalg.lstsq(
                design * root_weight[:, None], normal * root_weight[:, None],
                cond=1e-10, lapack_driver="gelsy")
            normal_coord = np.einsum("tp,pa->ta", coef, frame, optimize=False)
            coef -= np.einsum("ta,pa->tp", normal_coord, frame, optimize=False)
            return coef, int(rank), design, normal

        for _ in range(2):
            coef, _, _, _ = graph_fit(q)
            new, _ = linalg.qr(q + coef[linear].T, mode="economic")
            left, _, right = linalg.svd(np.einsum("pa,pb->ab", new, q, optimize=False))
            align = np.einsum("ab,bc->ac", left, right, optimize=False)
            q = np.einsum("pa,ab->pb", new, align, optimize=False)
        coef, rank, design, normal = graph_fit(q)
        bases[i] = q; coefficients[i] = coef
        ranks.append(rank)
        residuals.append(float(linalg.norm(np.einsum("kt,tp->kp", design, coef, optimize=False) - normal)
                               / max(linalg.norm(delta / radii[i]), 1e-15)))
    if kernel not in ("compact", "gaussian"):
        raise ValueError("kernel must be 'compact' or 'gaussian'")
    return dict(X=X.copy(), r=int(r), powers=powers, Q=bases,
                coefficients=coefficients, radius=radii, bandwidth=bandwidth,
                kernel=kernel, kernel_radius=4 * bandwidth,
                batch_size=16, patch_batch_size=128,
                diagnostics=dict(neighbours=k, degree=degree, tangent_refinements=2,
                    kernel=kernel, compact_support_radius=4 * bandwidth,
                    bandwidth_eighth_neighbour=bandwidth,
                    minimum_graph_design_rank=int(min(ranks)),
                    graph_design_columns=len(powers),
                    graph_relative_residual_quantiles=np.quantile(residuals, [0, .5, .9, 1])))


def _evaluate_projector_gaussian(atlas, queryX, directions=None):
    """Evaluate one frozen projector on source, held-out, or ODE states.

    Returns P[n,p,p], Q[n,p,r], gap[n].  Optional directions[n,p] also return
    dP[n,p,p], the analytic directional derivative.  Frames Q may change sign;
    P and dP are the smooth, gauge-independent objects.  A one-dimensional query
    is treated as a batch of size one, so all return arrays keep their batch axis.
    """
    query = np.asarray(queryX, dtype=float)
    if query.ndim == 1:
        query = query[None, :]
    X = atlas["X"]; n, p = query.shape; r = atlas["r"]
    if query.ndim != 2 or p != X.shape[1] or not np.isfinite(query).all():
        raise ValueError("query must have finite rows of the source dimension")
    derivative = directions is not None
    if derivative:
        directions = np.asarray(directions, dtype=float)
        if directions.ndim == 1:
            directions = np.broadcast_to(directions, query.shape)
        if directions.shape != query.shape or not np.isfinite(directions).all():
            raise ValueError("directions must match queryX")
    out_p = np.empty((n, p, p)); out_q = np.empty((n, p, r))
    gap = np.empty(n); nearest = np.empty(n)
    out_dp = np.empty_like(out_p) if derivative else None
    h2 = atlas["bandwidth"] ** 2
    qb = int(atlas.get("batch_size", 16)); cb = int(atlas.get("patch_batch_size", 128))
    source_norm = np.sum(X * X, axis=1)
    for start in range(0, n, qb):
        stop = min(start + qb, n); qx = query[start:stop]; b = len(qx)
        d2 = np.maximum(np.sum(qx * qx, axis=1)[:, None] + source_norm[None, :]
                        - 2 * np.einsum("bp,cp->bc", qx, X, optimize=False), 0)
        nearest[start:stop] = np.sqrt(d2.min(axis=1))
        logw = -d2 / (2 * h2)
        logw -= logw.max(axis=1, keepdims=True)
        weights = np.exp(logw); weights /= weights.sum(axis=1, keepdims=True)
        accum = np.zeros((b, p, p)); daccum = np.zeros_like(accum) if derivative else None
        if derivative:
            direction = directions[start:stop]
            g = -(np.einsum("bp,bp->b", qx, direction)[:, None]
                  - np.einsum("bp,cp->bc", direction, X, optimize=False)) / h2
            dweights = weights * (g - np.sum(weights * g, axis=1, keepdims=True))
        for cs in range(0, len(X), cb):
            ce = min(cs + cb, len(X)); w = weights[:, cs:ce]
            if not np.any(w):
                continue
            frame = atlas["Q"][cs:ce]
            coef = atlas["coefficients"][cs:ce]
            delta = qx[:, None, :] - X[None, cs:ce, :]
            u = np.einsum("bcp,cpa->bca", delta, frame) / atlas["radius"][None, cs:ce, None]
            _, du, ddu = _poly(u, atlas["powers"], second=derivative)
            jet = frame[None, :, :, :] + np.einsum("bcta,ctp->bcpa", du, coef)
            gram = np.einsum("bcpa,bcpd->bcad", jet, jet)
            inverse = np.linalg.inv(gram)
            left = np.einsum("bcpa,bcad->bcpd", jet, inverse)
            # Sum low-rank factors directly; never allocate [query,patch,p,p].
            accum += np.einsum("bc,bcpa,bcqa->bpq", w, left, jet, optimize=True)
            if derivative:
                udot = np.einsum("bp,cpa->bca", direction, frame) / atlas["radius"][None, cs:ce, None]
                jdot = np.einsum("bctad,bcd,ctp->bcpa", ddu, udot, coef)
                tangent_jdot = np.einsum("bcpa,bcpd->bcad", jet, jdot)
                normal_jdot = jdot - np.einsum("bcpa,bcad->bcpd", left, tangent_jdot)
                half = np.einsum("bc,bcpa,bcqa->bpq", w, normal_jdot, left, optimize=True)
                daccum += np.einsum("bc,bcpa,bcqa->bpq", dweights[:, cs:ce], left, jet, optimize=True)
                daccum += half + half.transpose(0, 2, 1)
        accum = (accum + accum.transpose(0, 2, 1)) / 2
        eigenvalue, eigenvector = np.linalg.eigh(accum)
        top = eigenvector[:, :, -r:]
        out_q[start:stop] = top
        out_p[start:stop] = np.einsum("bpa,bqa->bpq", top, top)
        gap[start:stop] = eigenvalue[:, -r] - eigenvalue[:, -r-1]
        if derivative:
            bottom = eigenvector[:, :, :-r]
            cross = np.einsum("bpi,bpq,bqj->bij", bottom, daccum, top)
            denom = eigenvalue[:, None, -r:] - eigenvalue[:, :-r, None]
            with np.errstate(divide="ignore", invalid="ignore"):
                cross = cross / denom
            half = np.einsum("bpi,bij,bqj->bpq", bottom, cross, top)
            out_dp[start:stop] = half + half.transpose(0, 2, 1)
    result = dict(P=out_p, Q=out_q, gap=gap, nearest_distance=nearest)
    if derivative:
        result["dP"] = out_dp
    return result


def evaluate_projector_sparse(atlas, queryX, directions=None, frame_only=False,
                              graph_position=False):
    """Compact C-infinity Gaussian/bump partition; only active pairs evaluated.

    Inside R=atlas['kernel_radius'], log b = -d2/(2h2)-d2/(R2-d2).
    The kernel and every derivative vanish at its fixed support boundary.  This
    is an exact compact partition, not nearest-neighbour truncation of Gaussian
    weights. Queries outside the union of support balls raise ValueError.
    frame_only=True returns Q and gap without forming the output P or dP.
    """
    query = np.asarray(queryX, dtype=float)
    if query.ndim == 1:
        query = query[None, :]
    X = atlas["X"]
    if query.ndim != 2 or query.shape[1] != X.shape[1] or not np.isfinite(query).all():
        raise ValueError("query must have finite rows of the source dimension")
    n, p = query.shape; r = atlas["r"]
    derivative = directions is not None
    if frame_only and derivative:
        raise ValueError("frame_only does not return projector derivatives")
    if derivative:
        directions = np.asarray(directions, dtype=float)
        if directions.ndim == 1:
            directions = np.broadcast_to(directions, query.shape)
        if directions.shape != query.shape or not np.isfinite(directions).all():
            raise ValueError("directions must match queryX")
    tree = atlas.get("_compact_tree")
    if tree is None:
        tree = cKDTree(X); atlas["_compact_tree"] = tree
    support = float(atlas.get("kernel_radius", 4 * atlas["bandwidth"]))
    support2 = support ** 2; h2 = atlas["bandwidth"] ** 2
    P = None if frame_only else np.empty((n, p, p))
    Q = np.empty((n, p, r))
    gap = np.empty(n); nearest = np.empty(n); active_count = np.empty(n, dtype=int)
    dP = np.empty_like(P) if derivative else None
    graph_average = np.empty_like(query) if graph_position else None
    batch = int(atlas.get("sparse_batch_size", 32))
    for start in range(0, n, batch):
        stop = min(start + batch, n); qx = query[start:stop]; count = len(qx)
        neighbours = tree.query_ball_point(qx, support, return_sorted=True)
        lengths = np.asarray([len(js) for js in neighbours], dtype=int)
        if np.any(lengths == 0):
            bad = start + int(np.flatnonzero(lengths == 0)[0])
            raise ValueError(f"query {bad} outside compact tangent-atlas support")
        pair_q = np.repeat(np.arange(count), lengths)
        pair_c = np.concatenate(neighbours).astype(int)
        delta = qx[pair_q] - X[pair_c]
        d2 = np.einsum("mp,mp->m", delta, delta)
        valid = d2 < support2
        pair_q, pair_c, delta, d2 = pair_q[valid], pair_c[valid], delta[valid], d2[valid]
        lengths = np.bincount(pair_q, minlength=count)
        if np.any(lengths == 0):
            bad = start + int(np.flatnonzero(lengths == 0)[0])
            raise ValueError(f"query {bad} outside open compact tangent-atlas support")
        begins = np.r_[0, np.cumsum(lengths)[:-1]]
        denominator = support2 - d2
        logw = -d2 / (2 * h2) - d2 / denominator
        maximum = np.full(count, -np.inf)
        np.maximum.at(maximum, pair_q, logw)
        weight = np.exp(logw - maximum[pair_q])
        weight /= np.bincount(pair_q, weights=weight, minlength=count)[pair_q]
        nearest[start:stop] = np.sqrt(np.minimum.reduceat(d2, begins))
        active_count[start:stop] = lengths
        frames = atlas["Q"][pair_c]; coef = atlas["coefficients"][pair_c]
        u = np.einsum("mp,mpa->ma", delta, frames) / atlas["radius"][pair_c, None]
        poly, du, ddu = _poly(u, atlas["powers"], second=derivative)
        if graph_position:
            position = X[pair_c] + atlas["radius"][pair_c, None] * (
                np.einsum("mpa,ma->mp", frames, u)
                + np.einsum("mt,mtp->mp", poly, coef))
            graph_average[start:stop] = np.add.reduceat(
                weight[:, None] * position, begins, axis=0)
        jet = frames + np.einsum("mta,mtp->mpa", du, coef)
        gram = np.einsum("mpa,mpd->mad", jet, jet)
        inverse = np.linalg.inv(gram)
        left = np.einsum("mpa,mad->mpd", jet, inverse)

        # One segment per query; large ambient dimensions avoid pairwise p*p arrays.
        accum = np.empty((count, p, p))
        if p <= 16 or (p <= 64 and not frame_only):
            patch = np.einsum("mpa,mqa->mpq", left, jet)
            accum[:] = np.add.reduceat(weight[:, None, None] * patch, begins, axis=0)
        else:
            for j, lo in enumerate(begins):
                hi = lo + lengths[j]
                accum[j] = np.einsum("m,mpa,mqa->pq", weight[lo:hi], left[lo:hi], jet[lo:hi], optimize=True)
        if derivative:
            direction = directions[start:stop]
            logdot = (-1 / h2 - 2 * support2 / denominator ** 2) * np.einsum("mp,mp->m", delta, direction[pair_q])
            mean = np.bincount(pair_q, weights=weight * logdot, minlength=count)
            weight_dot = weight * (logdot - mean[pair_q])
            udot = np.einsum("mp,mpa->ma", direction[pair_q], frames) / atlas["radius"][pair_c, None]
            jdot = np.einsum("mtad,md,mtp->mpa", ddu, udot, coef)
            tangent_jdot = np.einsum("mpa,mpd->mad", jet, jdot)
            normal_jdot = jdot - np.einsum("mpa,mad->mpd", left, tangent_jdot)
            daccum = np.empty_like(accum)
            if p <= 64:
                half = np.einsum("mpa,mqa->mpq", normal_jdot, left)
                pair_dot = weight_dot[:, None, None] * patch
                pair_dot += weight[:, None, None] * (half + half.transpose(0, 2, 1))
                daccum[:] = np.add.reduceat(pair_dot, begins, axis=0)
            else:
                for j, lo in enumerate(begins):
                    hi = lo + lengths[j]
                    half = np.einsum("m,mpa,mqa->pq", weight[lo:hi], normal_jdot[lo:hi], left[lo:hi], optimize=True)
                    daccum[j] = np.einsum("m,mpa,mqa->pq", weight_dot[lo:hi], left[lo:hi], jet[lo:hi], optimize=True)
                    daccum[j] += half + half.T
        accum = (accum + accum.transpose(0, 2, 1)) / 2
        if frame_only:
            # The same symmetric matrix defines the frozen projector. Only its
            # top r eigenvectors and the adjacent eigenvalue are needed here.
            for j in range(count):
                value, vector = linalg.eigh(accum[j],
                    subset_by_index=(p-r-1, p-1), driver="evr")
                Q[start+j] = vector[:, 1:]
                gap[start+j] = value[1] - value[0]
        else:
            value, vector = np.linalg.eigh(accum)
            top = vector[:, :, -r:]
            Q[start:stop] = top
            P[start:stop] = np.einsum("bpa,bqa->bpq", top, top)
            gap[start:stop] = value[:, -r] - value[:, -r-1]
        if derivative:
            bottom = vector[:, :, :-r]
            cross = np.einsum("bpi,bpq,bqj->bij", bottom, daccum, top)
            divisor = value[:, None, -r:] - value[:, :-r, None]
            with np.errstate(divide="ignore", invalid="ignore"):
                cross /= divisor
            half = np.einsum("bpi,bij,bqj->bpq", bottom, cross, top)
            dP[start:stop] = half + half.transpose(0, 2, 1)
    result = dict(Q=Q, gap=gap, nearest_distance=nearest,
                  active_patches=active_count, covered=np.ones(n, dtype=bool))
    if not frame_only:
        result["P"] = P
    if derivative:
        result["dP"] = dP
    if graph_position:
        result["graph_average"] = graph_average
    return result


def evaluate_projector(atlas, queryX, directions=None):
    """Evaluate the frozen data-supported implicit or local tangent geometry."""
    if atlas.get('kind')=='implicit_quadratic':
        from fields_first_implicit_geometry import evaluate_implicit_geometry
        return evaluate_implicit_geometry(atlas,queryX,directions=directions)
    if atlas.get("kernel", "gaussian") == "compact":
        return evaluate_projector_sparse(atlas, queryX, directions=directions)
    return _evaluate_projector_gaussian(atlas, queryX, directions=directions)


def evaluate_tangent_frame(atlas, queryX):
    """Return the frozen tangent frame and gap without a full projector."""
    if atlas.get('kind')=='implicit_quadratic':
        from fields_first_implicit_geometry import evaluate_implicit_geometry
        return evaluate_implicit_geometry(atlas,queryX,frame_only=True)
    if atlas.get("kernel", "gaussian") == "compact":
        return evaluate_projector_sparse(atlas, queryX, frame_only=True)
    result = evaluate_projector(atlas, queryX)
    result.pop("P", None)
    return result


def normal_graph_residual(atlas, queryX):
    """Normal residual of the weighted local graph positions.

    This uses the SAME frozen chart kernel as the tangent estimator. Solving
    (I-P(x)) (graph_average(x)-x)=0 defines an approximate geometric constraint;
    a single chart average is not a projection onto that constraint.
    """
    x = np.atleast_2d(np.asarray(queryX, dtype=float))
    if atlas.get("kernel", "gaussian") != "compact":
        raise ValueError("normal_graph_constraint_requires_compact_atlas")
    geometry = evaluate_projector_sparse(atlas, x, frame_only=True, graph_position=True)
    displacement = geometry.pop("graph_average") - x
    q = geometry["Q"]
    residual = displacement - np.einsum("npa,na->np", q,
        np.einsum("npa,np->na", q, displacement))
    return residual, geometry


def retract_to_atlas(atlas, queryX, *, tolerance=1e-9, max_iterations=20):
    """Solve the normal graph constraint, without any phase/end-point target.

    Every accepted row satisfies the residual tolerance. Nonconvergence or loss
    of geometric support is a refusal, not a nearest-neighbour snap. The map
    estimates geometry from noisy windows; it is not an exact data manifold.
    """
    original = np.atleast_2d(np.asarray(queryX, dtype=float))
    if tolerance <= 0 or max_iterations < 1:
        raise ValueError("positive retraction tolerance and iteration count required")
    states = original.copy()
    scale = max(float(np.sqrt(np.mean(np.sum(atlas["X"] ** 2, axis=1)))), 1e-12)
    active = np.arange(len(states))
    final_error = np.zeros(len(states))
    for iteration in range(max_iterations + 1):
        residual, geometry = normal_graph_residual(atlas, states[active])
        if not np.isfinite(residual).all():
            raise FloatingPointError("nonfinite_normal_graph_residual")
        if np.min(geometry["gap"]) < .1:
            raise ValueError("retraction_tangent_projector_gap_lost")
        error = np.linalg.norm(residual, axis=1) / scale
        final_error[active] = error
        remaining = error > tolerance
        if not np.any(remaining):
            return states, dict(iterations=iteration,
                maximum_normal_residual_relative=float(np.max(final_error)),
                maximum_displacement_relative=float(np.max(np.linalg.norm(states-original, axis=1))/scale))
        if iteration < max_iterations:
            states[active[remaining]] += residual[remaining]
            active = active[remaining]
    raise ValueError("normal_graph_retraction_did_not_converge")
