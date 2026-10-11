"""Small-rank SSA/DMD character proposals, with explicit diagnostic gates.

Several rotations can occupy the leading SSA space. A single leading pair
then need not be a character. This adapter diagonalizes a larger shift and
ranks its complex modes by represented trajectory energy. It is a standard
SSA/DMD readout inside the conditional-deflation prototype, not a novelty
claim or a calibrated character-identification theorem.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import lstsq
from scipy.sparse.linalg import svds


def extract_ssa_dynamic_character(signals: np.ndarray, spatial_weights: np.ndarray,
                                  window: int, noise_sigma: float | None = None,
                                  ranks=(4, 8, 16),
                                  temporal_gate_mode: str = "legacy",
                                  noise_covariance: np.ndarray | None = None) -> dict:
    y = np.asarray(signals, dtype=float)
    weights = np.asarray(spatial_weights, dtype=float)
    if (y.ndim != 2 or y.shape[0] != len(weights) or y.shape[1] <= window+3
            or window < 4 or np.any(weights <= 0) or not np.isfinite(y).all()):
        raise ValueError("finite field, positive weights and valid lag window required")
    weights = weights/weights.sum()
    q, n = y.shape
    columns = n-window+1
    centered = y-y.mean(axis=1, keepdims=True)
    row_scale = np.sqrt(weights/window)
    hankel = np.vstack([row_scale[:, None]*centered[:, lag:lag+columns]
                       for lag in range(window)])
    max_rank = min(max(ranks), min(hankel.shape)-2)
    v0 = np.random.default_rng(471).standard_normal(min(hankel.shape))
    u, singular, _ = svds(hankel, k=max_rank+1, which="LM", tol=1e-8, v0=v0)
    order = np.argsort(singular)[::-1]
    u, singular = u[:, order], singular[order]
    raw_lag1 = float(np.sum(weights[:, None]*centered[:, 1:]*centered[:, :-1])
                     /max(np.sum(weights[:, None]*centered[:, :-1]**2), 1e-15))
    lag_covariance = float(np.sum(weights[:, None]*centered[:, 1:]*centered[:, :-1])/(n-1))
    temporal_reference = {"mode": temporal_gate_mode, "lag_covariance": lag_covariance}
    if temporal_gate_mode == "legacy":
        temporal_gate = abs(raw_lag1) >= .1
        temporal_reference["correlation_threshold"] = .1
    elif temporal_gate_mode == "iid_reference":
        covariance = np.asarray(noise_covariance, dtype=float)
        if covariance.shape != (q, q) or not np.isfinite(covariance).all():
            raise ValueError("iid reference requires a finite coordinate noise covariance")
        omega = np.sqrt(weights)[:, None]*covariance*np.sqrt(weights)[None, :]
        eigenvalues = np.linalg.eigvalsh((omega+omega.T)/2)
        if eigenvalues[0] < -1e-12:
            raise ValueError("noise covariance must be positive semidefinite")
        x = np.log(2/1e-4)
        threshold = (float(np.trace(omega))/n
                     +np.sqrt(2*float(np.sum(omega**2))*x/(n-1))
                     +2*float(eigenvalues[-1])*x/(n-1))
        temporal_gate = abs(lag_covariance) > threshold
        temporal_reference.update(threshold=threshold, reference_alpha=1e-4,
            status="iid Gaussian zero-signal reference; learned residual dependence not calibrated")
    else:
        raise ValueError("unknown temporal dependence gate")
    proposals = []
    for desired_rank in ranks:
        rank = min(desired_rank, max_rank)
        subspace = u[:, :rank]
        scores = np.einsum("ij,jt->it", subspace.T, hankel, optimize=False)
        previous, following = scores[:, :-1], scores[:, 1:]
        cross = np.einsum("it,jt->ij", following, previous, optimize=False)
        gram = np.einsum("it,jt->ij", previous, previous, optimize=False)
        shift = lstsq(gram.T, cross.T, lapack_driver="gelsy")[0].T
        values, vectors = np.linalg.eig(shift)
        if np.linalg.cond(vectors) > 1e8:
            continue
        readout = np.linalg.solve(vectors, subspace.T)
        coordinates = np.einsum("ij,jt->it", readout, hankel, optimize=False)
        positive = np.flatnonzero(np.imag(values) > 1e-5)
        for mode in positive:
            value = values[mode]
            coordinate = coordinates[mode]
            radii = abs(coordinate)
            if radii.mean() <= 1e-12:
                continue
            normalized = coordinate/np.maximum(radii, 1e-15)
            unwrapped = np.unwrap(np.angle(normalized))
            times = np.arange(columns, dtype=float)
            tc = times-times.mean()
            slope = float(np.dot(tc, unwrapped-unwrapped.mean())/np.dot(tc, tc))
            phase_residual = float(np.std(unwrapped-slope*times))
            coherence = float(abs(np.mean(normalized[1:]*np.conj(normalized[:-1]))))
            radius_cv = float(radii.std()/radii.mean())
            slope_gap = float(abs(np.angle(np.exp(1j*(slope-np.angle(value))))))
            energy = float(np.vdot(vectors[:, mode], vectors[:, mode]).real
                           *np.mean(abs(coordinate)**2))
            gates = {
                "unit_modulus": bool(abs(abs(value)-1) <= .05),
                "phase_coherent": bool(coherence >= .90),
                "radius_stable": bool(radius_cv <= .60),
                "linear_phase": bool(phase_residual <= .50),
                "slope_agreement": bool(slope_gap <= .01),
                "interior_positive_frequency": bool(2*np.pi/n < slope < np.pi-2*np.pi/n),
                "raw_temporal_dependence": bool(temporal_gate),
            }
            lag_filter = readout[mode].reshape(window, q)*row_scale[None, :]
            proposals.append({"rank": rank, "mode": int(mode),
                "frequency": slope, "dmd_frequency": float(np.angle(value)),
                "represented_energy": energy,
                "phase_linear_fit_rms": phase_residual,
                "phase_step_coherence": coherence,
                "coordinate_radius_cv": radius_cv,
                "eigenvalue": [float(value.real), float(value.imag)],
                "gates": gates, "passed": all(gates.values()),
                "complex_coordinate": coordinate, "normalized_character": normalized,
                "lag_filter": lag_filter})
    public = [{key: value for key, value in proposal.items()
               if key not in ("complex_coordinate", "normalized_character", "lag_filter")}
              for proposal in proposals]
    eligible = [proposal for proposal in proposals if proposal["passed"]]
    if not eligible:
        return {"status": "unresolved", "reason": "no dynamic mode passed heuristic gates",
                "dynamic_proposals": public, "raw_lag1_correlation": raw_lag1,
                "temporal_reference": temporal_reference,
                "window": window, "hankel_columns": columns}
    selected = max(eligible, key=lambda item: item["represented_energy"])
    return {"status": "resolved", "reason": None,
        "candidate_frequency": selected["frequency"],
        "complex_coordinate": selected["complex_coordinate"],
        "lag_filter": selected["lag_filter"], "signal_mean": y.mean(axis=1),
        "normalized_character": selected["normalized_character"],
        "phase": np.angle(selected["normalized_character"]),
        "metrics": {"frequency_temporal_dmd": selected["dmd_frequency"],
            "frequency_phase_slope": selected["frequency"],
            "phase_linear_fit_rms": selected["phase_linear_fit_rms"],
            "phase_step_coherence": selected["phase_step_coherence"],
            "coordinate_radius_cv": selected["coordinate_radius_cv"],
            "represented_energy": selected["represented_energy"],
            "raw_lag1_correlation": raw_lag1,
            "temporal_reference": temporal_reference,
            "singular_values": singular.tolist(), "noise_sigma": noise_sigma},
        "gates": selected["gates"], "gate_status": "heuristic_uncalibrated",
        "dynamic_rank": selected["rank"], "dynamic_proposals": public,
        "window": window, "hankel_columns": columns}
