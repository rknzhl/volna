"""Pilot of tangent transport obtained from overlapping joint Hankel windows.

Inference uses synchronized readings only. Sensor positions, true phases,
frequencies and generating equations are not arguments of any fit function.
Local PCA is performed in the original window space: no global truncation
is assumed to preserve the Hankel overlap identity.
"""

from __future__ import annotations

import numpy as np
from scipy import linalg, sparse
from scipy.sparse.linalg import eigsh
from sklearn.neighbors import NearestNeighbors


def windows(y, lag):
    """Time-major readings (N,m) -> time-major joint windows (N-L+1,L*m)."""
    view = np.lib.stride_tricks.sliding_window_view(np.asarray(y), lag, axis=0)
    return view.transpose(0, 2, 1).reshape(len(view), -1).copy()


def fold(h, channels, lag):
    n = len(h) + lag - 1
    total = np.zeros((n, channels))
    count = np.zeros(n)
    blocks = np.asarray(h).reshape(len(h), lag, channels)
    for ell in range(lag):
        total[ell:ell + len(h)] += blocks[:, ell]
        count[ell:ell + len(h)] += 1
    return total / count[:, None]


def local_geometry(x, neighbors=32, max_dimension=5):
    """Estimate local dimension from the median consecutive singular-value gap.

    This is a pilot heuristic, not a finite noisy observability certificate.
    All neighbours are selected by state distance, regardless of temporal lag.
    """
    distance, index = NearestNeighbors(n_neighbors=neighbors + 1).fit(x).kneighbors(x)
    distance, index = distance[:, 1:], index[:, 1:]
    size = min(max_dimension + 1, neighbors - 1, x.shape[1])
    eigenvalues = np.zeros((len(x), size))
    eigenvectors = []
    for i, js in enumerate(index):
        delta = x[js] - np.mean(x[js], axis=0)
        gram = delta @ delta.T
        val, vec = linalg.eigh(gram, subset_by_index=(len(js) - size, len(js) - 1))
        order = np.arange(size - 1, -1, -1)
        eigenvalues[i] = np.maximum(val[order], 0)
        eigenvectors.append(vec[:, order])
    singular = np.sqrt(eigenvalues)
    gaps = np.median(singular[:, :-1] / np.maximum(singular[:, 1:], 1e-12), axis=0)
    dimension = int(1 + np.argmax(gaps))
    q = np.zeros((len(x), x.shape[1], dimension))
    for i, js in enumerate(index):
        delta = x[js] - np.mean(x[js], axis=0)
        q[i] = delta.T @ (eigenvectors[i][:, :dimension] /
                          np.maximum(singular[i, :dimension], 1e-12))
    return {"Q": q, "neighbors": index, "distance": distance,
            "dimension": dimension, "median_singular_gaps": gaps,
            "median_singular_values": np.median(singular, axis=0)}


def overlap_maps(q, channels):
    maps, conditioning = [], []
    for i in range(len(q) - 1):
        shared_next = q[i + 1, :-channels]
        shared_previous = q[i, channels:]
        conditioning.append(linalg.svdvals(shared_next)[-1])
        maps.append(linalg.lstsq(shared_next, shared_previous)[0])
    return np.asarray(maps), np.asarray(conditioning)


def _block_matrix(n, r, edges, blocks):
    row, col, data = [], [], []
    for (i, j), block in zip(edges, blocks):
        for a in range(r):
            for b in range(r):
                row.append(i * r + a)
                col.append(j * r + b)
                data.append(block[a, b])
    return sparse.coo_matrix((data, (row, col)), shape=(n * r, n * r)).tocsr()


def frame_operators(q, maps, neighbors):
    n, _, r = q.shape
    # C^T C for v_(i+1) - A_i v_i. Mean-per-time normalization.
    edges, blocks = [], []
    eye = np.eye(r)
    for i, a in enumerate(maps):
        edges.extend([(i, i), (i + 1, i + 1), (i, i + 1), (i + 1, i)])
        blocks.extend([a.T @ a, eye, -a.T, -a])
    ctc = _block_matrix(n, r, edges, blocks) / max(n - 1, 1)
    # State-neighbour smoothness uses ambient vector differences, not a
    # temporally propagated frame. It therefore penalizes curvature as well;
    # lambda is decreased to check whether this bias vanishes.
    pairs = sorted({tuple(sorted((i, int(j))))
                    for i, js in enumerate(neighbors) for j in js if i != j})
    edges, blocks = [], []
    for i, j in pairs:
        cross = q[i].T @ q[j]
        edges.extend([(i, i), (j, j), (i, j), (j, i)])
        blocks.extend([eye, eye, -cross, -cross.T])
    roughness = _block_matrix(n, r, edges, blocks) / max(len(pairs), 1)
    # Normalize with respect to empirical average vector norm.
    return ctc * n, roughness * n + sparse.eye(n * r, format="csr")


def fit(y, lag=24, neighbors=32, lambdas=(0.03, 0.003, 0.0003)):
    y = np.asarray(y, dtype=float)
    mean = np.mean(y, axis=0)
    scale = np.std(y, axis=0)
    scale = np.maximum(scale, 1e-10)
    normalized = (y - mean) / scale
    x = windows(normalized, lag) / np.sqrt(lag * y.shape[1])
    geometry = local_geometry(x, neighbors=neighbors)
    q = geometry["Q"]
    r = q.shape[-1]
    maps, conditioning = overlap_maps(q, y.shape[1])
    ctc, roughness = frame_operators(q, maps, geometry["neighbors"])
    fits = []
    modes = min(max(2 * r + 2, 6), ctc.shape[0] - 2)
    for lam in lambdas:
        matrix = ctc + lam * roughness
        values, vectors = eigsh(matrix, k=modes, sigma=-1e-9, which="LM",
                               tol=1e-7, v0=np.ones(matrix.shape[0]))
        order = np.argsort(values)
        values, vectors = values[order], vectors[:, order]
        coefficients = vectors[:, :r].reshape(len(x), r, r) * np.sqrt(len(x))
        field = np.einsum("npa,nab->npb", q, coefficients)
        norms = np.linalg.svd(field, compute_uv=False)
        temporal = float(np.trace(vectors[:, :r].T @ ctc @ vectors[:, :r]))
        smooth = float(np.trace(vectors[:, :r].T @ roughness @ vectors[:, :r]))
        fits.append({"lambda": float(lam), "eigenvalues": values,
                     "field": field, "coefficients": coefficients,
                     "transport_energy": temporal, "roughness": smooth,
                     "frame_min_singular_quantiles": np.quantile(norms[:, -1], [.0, .1, .5]),
                     "frame_max_singular_quantiles": np.quantile(norms[:, 0], [.5, .9, 1.])})
    return {"X": x, "geometry": geometry, "maps": maps,
            "overlap_min_singular_quantiles": np.quantile(conditioning, [0, .1, .5]),
            "fits": fits, "mean": mean, "scale": scale, "lag": lag,
            "channels": y.shape[1]}


def encode_geometry(model, y, neighbors=None):
    """Interpolate a learned frame on new observed windows; this is not forecasting."""
    normalized = (np.asarray(y) - model["mean"]) / model["scale"]
    x = windows(normalized, model["lag"]) / np.sqrt(model["lag"] * model["channels"])
    k = model["geometry"]["neighbors"].shape[1] if neighbors is None else neighbors
    distance, index = NearestNeighbors(n_neighbors=k).fit(model["X"]).kneighbors(x)
    width = np.maximum(distance[:, -1], 1e-9)
    weights = np.exp(-(distance / width[:, None]) ** 2)
    weights /= np.sum(weights, axis=1)[:, None]
    return x, index, weights
