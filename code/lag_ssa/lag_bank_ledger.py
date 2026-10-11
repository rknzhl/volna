"""Fixed lag-bank temporal reference for the raw observed-line ledger.

Only the global iid temporal-reference branch changes from lag1 to ANY of
lags(1,2,3,4). All other SSA/DMD arithmetic, candidate gates, rank/mode bank,
ordering and readout conversion are directly copied from the frozen source.
The underlying free-line fitter remains the unchanged raw_spectral_ledger.
No integer labels, true rank, frequencies or analytic field enter this module.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import lstsq
from scipy.sparse.linalg import svds

from alternative_cover_selection_ssa import SPATIAL_MODEL, symmetric_sqrt, extract_ssa_character


TEMPORAL_LAGS = (1, 2, 3, 4)
TEMPORAL_FAMILY_ALPHA = 1e-4


def iid_lag_bank_reference(centered, weights, noise_covariance):
    """IID Gaussian zero-signal reference; not calibrated for fitted residuals.

    For lag ell, c_ell is the weighted centered lag covariance. The threshold
    contains the demeaning-bias allowance tr(Omega)/N and a two-sided Gaussian
    quadratic-form reference with family union bound alpha across fixed lags.
    For one harmonic max(abs(cos(nu)),abs(cos(2nu))) >= 1/2; a finite lag bank
    is not a universal signal detector when multiple harmonic terms cancel.
    """
    y = np.asarray(centered, dtype=float)
    weights = np.asarray(weights, dtype=float)
    covariance = np.asarray(noise_covariance, dtype=float)
    q, n = y.shape
    if covariance.shape != (q, q) or not np.isfinite(covariance).all():
        raise ValueError("iid reference requires a finite coordinate noise covariance")
    if n <= max(TEMPORAL_LAGS):
        raise ValueError("not enough samples for declared temporal lag bank")
    omega = np.sqrt(weights)[:, None]*covariance*np.sqrt(weights)[None, :]
    eigenvalues = np.linalg.eigvalsh((omega+omega.T)/2)
    if eigenvalues[0] < -1e-12:
        raise ValueError("noise covariance must be positive semidefinite")
    trace = float(np.trace(omega))
    trace_square = float(np.sum(omega**2))
    maximum = float(eigenvalues[-1])
    count = len(TEMPORAL_LAGS)
    logarithm = float(np.log(2*count/TEMPORAL_FAMILY_ALPHA))
    references = []
    for lag in TEMPORAL_LAGS:
        value = float(np.sum(weights[:, None]*y[:, lag:]*y[:, :-lag])/(n-lag))
        threshold = (trace/n + np.sqrt(2*trace_square*logarithm/(n-lag))
                     +2*maximum*logarithm/(n-lag))
        references.append(dict(lag=lag, lag_covariance=value, threshold=float(threshold),
            passed=bool(abs(value) > threshold), noise_reference_tail_alpha=TEMPORAL_FAMILY_ALPHA/count))
    passed = any(item["passed"] for item in references)
    return passed, dict(mode="iid_lag_bank_reference", original_requested_mode="iid_reference",
        declared_lags=list(TEMPORAL_LAGS), lagwise_references=references,
        raw_temporal_dependence=passed, reference_family_alpha=TEMPORAL_FAMILY_ALPHA,
        reference_logarithm=logarithm, noise_weighted_trace=trace,
        noise_weighted_trace_square=trace_square, noise_weighted_maximum_eigenvalue=maximum,
        centered_bias_allowance=trace/n, reference_bound_scope="unconditional iid Gaussian zero-signal noise only",
        status="fixed-lag family iid Gaussian noise reference; learned residual dependence not calibrated",
        learned_residual_calibrated=False, universal_harmonic_signal_detection_guarantee=False,
        change_scope="only global iid temporal reference; remaining SSA/DMD gates unchanged")


def extract_ssa_dynamic_alternatives(signals: np.ndarray, spatial_weights: np.ndarray,
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
        temporal_gate, temporal_reference = iid_lag_bank_reference(
            centered, weights, noise_covariance)
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
    def render(selected):
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
            "dynamic_rank": selected["rank"], "alternative_dynamic_mode": selected["mode"],
            "dynamic_proposals": public,
            "window": window, "hankel_columns": columns}
    seen = set()
    alternatives = []
    for proposal in sorted(eligible, key=lambda item: -item["represented_energy"]):
        identity = (proposal["rank"], proposal["mode"])
        if identity in seen:
            continue
        seen.add(identity)
        alternatives.append(render(proposal))
    result = dict(alternatives[0])
    result["candidate_alternatives"] = alternatives
    return result


def character_frontend(residual, noise_sigma, settings, sampling_model=None):
    spatial = SPATIAL_MODEL if sampling_model is None else sampling_model
    weights = np.ones(len(residual))/len(residual)
    if settings.phase_metric_mode == "field":
        inference_values = residual
        whitener = None
        inference_noise_covariance = spatial.noise_covmass(noise_sigma)
    elif settings.phase_metric_mode == "noise":
        whitener = symmetric_sqrt(spatial.noise_covmass(1.), inverse=True)
        inference_values = np.einsum("ij,jt->it", whitener, residual, optimize=False)
        inference_noise_covariance = noise_sigma**2*np.eye(len(residual))
    else:
        raise ValueError("unknown phase metric mode")
    if settings.frontend_mode == "pair":
        fitted = extract_ssa_character(inference_values, weights, settings.lag_window,
                                       noise_sigma=noise_sigma)
    elif settings.frontend_mode == "dynamic":
        fitted = extract_ssa_dynamic_alternatives(inference_values, weights, settings.lag_window,
            noise_sigma=noise_sigma, ranks=settings.dynamic_ranks,
            temporal_gate_mode=settings.temporal_gate_mode,
            noise_covariance=inference_noise_covariance)
    else:
        raise ValueError("unknown SSA frontend mode")
    alternatives = fitted.pop("candidate_alternatives", None)
    payloads = alternatives if alternatives is not None else [fitted]
    for payload in payloads:
        if whitener is not None and "lag_filter" in payload:
            payload["lag_filter"] = np.einsum("li,ij->lj", payload["lag_filter"], whitener,
                                             optimize=False)
            payload["signal_mean"] = residual.mean(axis=1)
        payload["phase_metric_mode"] = settings.phase_metric_mode
        payload["filter_input_coordinates"] = "physical mass coordinates; whitening pulled back"
    result = dict(payloads[0])
    if alternatives is not None:
        result["candidate_alternatives"] = payloads
    return result
