"""Joint raw-data sinusoid fit and local frequency uncertainty.

The Fisher covariance below is a local asymptotic approximation for a
correct finite sinusoid model with independent, equal-variance Gaussian
sensor/time noise. It does not bound bias from omitted infinite harmonics,
post-selection error, or label switching near unresolved frequencies.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import lstsq
from scipy.optimize import minimize


def sinusoid_design(frequencies: np.ndarray, times: np.ndarray) -> np.ndarray:
    angles = times[:, None] * np.asarray(frequencies)[None, :]
    return np.column_stack((np.ones(len(times)), np.cos(angles), np.sin(angles)))


def fit_joint_frequencies(observed: np.ndarray, starting: np.ndarray,
                          half_width_cycles: float = 1.0) -> dict:
    """Variable projection on raw samples; all spatial amplitudes are free.

    The frequency search is local to the supplied resolved peaks. This is
    neither peak detection nor a global frequency/lattice estimator.
    """
    y = np.asarray(observed, dtype=float)
    initial = np.asarray(starting, dtype=float).reshape(-1)
    if y.ndim != 2 or not len(initial) or not np.isfinite(y).all():
        raise ValueError("finite (sensor,time) data and resolved peaks required")
    q, n = y.shape
    p = len(initial)
    if n <= 2*p+1 or np.any((initial <= 0) | (initial >= np.pi)):
        raise ValueError("need interior positive frequencies and enough samples")
    times = np.arange(n, dtype=float) - (n-1)/2
    target = y.T

    def profile(frequencies: np.ndarray):
        x = sinusoid_design(frequencies, times)
        coefficients, _, linear_rank, _ = lstsq(x, target, lapack_driver="gelsy")
        if linear_rank < x.shape[1]:
            raise np.linalg.LinAlgError("sinusoid design is rank deficient")
        fitted = np.einsum("nj,ji->ni", x, coefficients, optimize=False)
        residual = target-fitted
        cosine, sine = coefficients[1:1+p], coefficients[1+p:]
        angle = times[:, None]*frequencies[None, :]
        derivatives = times[:, None, None]*(
            -np.sin(angle)[:, :, None]*cosine[None, :, :]
            +np.cos(angle)[:, :, None]*sine[None, :, :])
        return x, coefficients, residual, derivatives

    def objective(displacement: np.ndarray):
        _, _, residual, derivatives = profile(initial+displacement/n)
        rss = float(np.sum(residual**2))
        gradient = -2*np.einsum("ni,nji->j", residual, derivatives, optimize=False)/n
        return rss/(q*n), gradient/(q*n)

    width = 2*np.pi*half_width_cycles
    bounds = [(max(-width, (1e-8-value)*n),
               min(width, (np.pi-1e-8-value)*n)) for value in initial]
    result = minimize(objective, np.zeros(p), jac=True, method="L-BFGS-B",
                      bounds=bounds,
                      options={"ftol": 1e-13, "gtol": 1e-10, "maxiter": 200})
    frequencies = initial+result.x/n
    x, coefficients, residual, derivatives = profile(frequencies)
    # Residualize every nonlinear derivative against all linear amplitudes.
    flattened = derivatives.reshape(n, p*q)
    projected_coefficients = lstsq(x, flattened, lapack_driver="gelsy")[0]
    projected = np.einsum("nj,jk->nk", x, projected_coefficients, optimize=False)
    efficient = (flattened-projected).reshape(n, p, q)
    information_without_variance = np.einsum("nji,nki->jk", efficient, efficient,
                                            optimize=False)
    dof = q*n-q*(2*p+1)-p
    variance = float(np.sum(residual**2)/dof)
    condition = float(np.linalg.cond(information_without_variance))
    covariance = variance*np.linalg.inv(information_without_variance)
    if not (np.isfinite(covariance).all() and np.min(np.linalg.eigvalsh(covariance)) > 0):
        raise np.linalg.LinAlgError("frequency covariance is not positive definite")
    score = np.einsum("ni,nji->j", residual, derivatives, optimize=False)
    local_step = np.linalg.solve(information_without_variance, score)
    stationary_distance_squared = float(
        local_step @ np.linalg.solve(covariance, local_step))
    return {"frequencies": frequencies, "covariance": covariance,
            "frequency_standard_errors": np.sqrt(np.diag(covariance)),
            "noise_variance_estimate": variance,
            "rss": float(np.sum(residual**2)), "degrees_of_freedom": dof,
            "information_condition": condition,
            "optimizer_success": bool(result.success),
            "optimizer_message": str(result.message),
            "stationary_score_distance_squared": stationary_distance_squared,
            "stationary_in_frequency_uncertainty": stationary_distance_squared < 1e-4,
            "iterations": int(result.nit), "function_evaluations": int(result.nfev),
            "coefficients": coefficients, "residual": residual.T,
            "times_center": (n-1)/2,
            "covariance_status": "local_asymptotic_correct_finite_iid_model_only"}
