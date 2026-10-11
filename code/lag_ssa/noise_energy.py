"""Prediction-energy intervals on a validation block independent of fitting.

The mean prediction is frozen given training. Time noise vectors are iid
Gaussian with known covariance. This bounds deterministic finite-time error;
turning it into a phase-completeness certificate additionally needs temporal
coverage, phase-error and missing-character visibility bounds.
"""
from __future__ import annotations

import numpy as np


def prediction_energy_interval(residual: np.ndarray, covariance: np.ndarray,
                               alpha: float = .01, look_budget: int = 1) -> dict:
    r = np.asarray(residual, dtype=float)
    sigma = np.asarray(covariance, dtype=float)
    if (r.ndim != 2 or sigma.shape != (r.shape[0], r.shape[0])
            or not np.isfinite(r).all() or not np.isfinite(sigma).all()
            or not 0 < alpha < 1 or look_budget < 1
            or not np.allclose(sigma, sigma.T)):
        raise ValueError("finite coordinate×time residual, symmetric covariance and valid error budget required")
    eigenvalues = np.linalg.eigvalsh(sigma)
    if eigenvalues[0] < -1e-12 or r.shape[1] < 1:
        raise ValueError("positive semidefinite covariance and nonempty validation record required")
    eigenvalues = np.maximum(eigenvalues, 0.)
    trace = float(eigenvalues.sum())
    trace_squared = float(np.sum(eigenvalues**2))
    operator = float(eigenvalues[-1])
    count = r.shape[1]
    observed = float(np.mean(np.sum(r*r, axis=0)))
    corrected = observed-trace
    x = float(np.log(2*look_budget/alpha))
    a = 2*np.sqrt(x/count)
    d = 2*operator
    b = 2*operator*x/count

    def root(value: float, sign: int) -> float:
        inside = trace_squared+d*value+a*a*d*d/4
        return max(0., float(value+a*a*d/2+sign*a*np.sqrt(max(inside, 0.))))

    # Lower-tail concentration has no linear b term. On its high-probability
    # event the upper root contains the true nonnegative error energy.
    upper = root(max(corrected, 0.), +1)
    lower = (0. if corrected-b <= a*np.sqrt(trace_squared)
             else root(corrected-b, -1))
    return {"corrected_validation_energy": corrected,
            "lower": lower, "upper": upper, "raw_residual_energy": observed,
            "noise_trace": trace, "noise_trace_squared": trace_squared,
            "noise_operator_norm": operator, "validation_samples": count,
            "alpha_for_all_predeclared_looks": alpha, "look_budget": look_budget,
            "assumptions": ["frozen predictor conditional on training",
                            "independent iid-in-time Gaussian validation noise",
                            "known coordinate noise covariance"],
            "phase_completeness_certified": False}


def conditional_completeness_decision(interval: dict, visibility: float,
                                     phase_error_bound: float,
                                     temporal_energy_discrepancy: float,
                                     bound_scope: str = "haar") -> dict:
    """Conditional only: phase_error_bound is prediction error in field units.

    For scope `haar`, it is L2(Haar) distance to a true factor-measurable
    predictor and discrepancy controls the actual loss. For `trajectory`, it
    is finite-record prediction discrepancy from that ideal predictor, whose
    full-phase loss has the supplied temporal coverage bound. Neither bound
    is estimated by this function.
    """
    if min(visibility, phase_error_bound, temporal_energy_discrepancy) < 0:
        raise ValueError("bounds must be nonnegative")
    if bound_scope == "haar":
        threshold = max(visibility-phase_error_bound, 0.)**2-temporal_energy_discrepancy
    elif bound_scope == "trajectory":
        ideal_norm = np.sqrt(max(visibility**2-temporal_energy_discrepancy, 0.))
        threshold = max(ideal_norm-phase_error_bound, 0.)**2
    else:
        raise ValueError("bound scope must be haar or trajectory")
    certified = visibility > phase_error_bound and threshold > interval["upper"]
    return {"status": "complete_conditional_on_supplied_bounds" if certified else "unresolved",
            "visibility_lower_bound": visibility,
            "phase_error_bound": phase_error_bound,
            "temporal_energy_discrepancy": temporal_energy_discrepancy,
            "bound_scope": bound_scope,
            "threshold": threshold,
            "upper_validation_error_energy": interval["upper"],
            "bounds_inferred_from_data": False,
            "requires_correct_factor_membership": True}
