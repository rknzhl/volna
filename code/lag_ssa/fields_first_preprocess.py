"""Train-only masked MSSA conditioning and a fixed polynomial-annihilating split."""
from __future__ import annotations

import hashlib
import numpy as np
from scipy import linalg
from scipy.signal import savgol_filter
from sklearn.utils.extmath import randomized_svd


def _windows(y, lag):
    z = np.lib.stride_tricks.sliding_window_view(y, lag, axis=0)
    return z.transpose(0, 2, 1).reshape(len(y) - lag + 1, -1).T.copy()


def _fold(h, channels, lag):
    count = h.shape[1] + lag - 1
    out = np.zeros((count, channels)); copies = np.zeros(count)
    blocks = h.T.reshape(h.shape[1], lag, channels)
    for ell in range(lag):
        out[ell:ell + len(blocks)] += blocks[:, ell]
        copies[ell:ell + len(blocks)] += 1
    return out / copies[:, None]


def raw_mask_to_hankel(observed, lag):
    """Every copy of the same raw scalar has exactly the same observed flag."""
    return _windows(np.asarray(observed, dtype=float), lag).astype(bool)


def _svd(h, rank, seed):
    rank = min(int(rank), min(h.shape))
    if rank == 0:
        return np.empty((h.shape[0], 0)), np.empty(0), np.empty((0, h.shape[1]))
    if min(h.shape) <= 64 or rank + 8 >= min(h.shape):
        u, s, vt = linalg.svd(h, full_matrices=False, lapack_driver="gesdd")
        return u[:, :rank], s[:rank], vt[:rank]
    # Some macOS BLAS builds leave spurious floating-point flags after finite
    # matrix products. Validate the actual result rather than trusting flags.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        result = randomized_svd(h, n_components=rank, n_oversamples=8, n_iter=2,
                                random_state=int(seed), flip_sign=True)
    if not all(np.isfinite(value).all() for value in result):
        raise FloatingPointError("nonfinite randomized SVD result")
    return result


def _polynomial(times, center, scale, degree):
    t = (np.asarray(times, dtype=float) - center) / scale
    return np.vander(t, N=int(degree) + 1, increasing=True)


def _fit_trend(y, observed, degree, times):
    center = float((times[0] + times[-1]) / 2)
    tscale = max(float((times[-1] - times[0]) / 2), 1.)
    design = _polynomial(times, center, tscale, degree)
    coef = np.empty((degree + 1, y.shape[1]))
    for c in range(y.shape[1]):
        take = observed[:, c]
        if take.sum() < degree + 1:
            raise ValueError("insufficient unmasked values for polynomial trend")
        coef[:, c] = linalg.lstsq(design[take], y[take, c])[0]
    trend = np.einsum("nt,tc->nc", design, coef, optimize=False)
    residual = np.zeros_like(y)
    residual[observed] = (y - trend)[observed]
    channel_scale = np.sqrt(np.sum(residual ** 2, axis=0) / observed.sum(axis=0))
    floor = max(float(np.max(channel_scale)) * 1e-12, 1e-12)
    channel_scale[channel_scale <= floor] = 1.
    return dict(degree=int(degree), coefficients=coef, center=center,
                time_scale=tscale, channel_scale=channel_scale), trend


def _masked_family(y, observed, lag, maxrank, degree, iterations, seed):
    """Predictions for all ranks after one fixed maximal-rank completion.

    This function never reads held-out values in fitting. The final predictions
    are rank-projected/folded estimates, not the noisy observed-value clamps.
    """
    times = np.arange(len(y), dtype=float)
    parameters, trend = _fit_trend(y, observed, degree, times)
    residual = np.zeros_like(y)
    residual[observed] = ((y - trend) / parameters["channel_scale"])[observed]
    work = residual.copy()
    for step in range(int(iterations)):
        h = _windows(work, lag)
        u, s, vt = _svd(h, maxrank, seed + step)
        projected = np.einsum("ak,k,kb->ab", u, s, vt, optimize=True)
        prediction = _fold(projected, y.shape[1], lag)
        work = np.where(observed, residual, prediction)
    h = _windows(work, lag)
    u, s, vt = _svd(h, maxrank, seed + int(iterations))
    # Successive outer products give every nested rank without repeated SVDs.
    projected = np.zeros_like(h)
    predictions = []
    for rank in range(len(s) + 1):
        if rank:
            projected += s[rank - 1] * np.outer(u[:, rank - 1], vt[rank - 1])
        estimate = _fold(projected, y.shape[1], lag)
        predictions.append(trend + estimate * parameters["channel_scale"])
    return predictions, parameters


def fit_preprocess(Y, lag=32, maxrank=24, trend_degrees=(0, 1, 2),
                   cv_folds=2, holdout_fraction=0.1, iterations=10, seed=1729):
    """Select degree/rank by masked raw measurement MSE on training data only.

    Default candidates are every integer rank 0..min(24,m*lag,K), including odd
    ranks and the zero signal. All trend, scale and initialization statistics in
    each fold use unmasked raw values only. One fixed maximal-rank completion is
    followed by all nested rank projections. Numerical ties prefer lower degree,
    then lower rank. The final subspace is refit on all training observations.
    """
    y = np.asarray(Y, dtype=float)
    if y.ndim != 2 or not np.isfinite(y).all():
        raise ValueError("Y must be a finite time-by-sensor matrix")
    n, m = y.shape; lag = int(lag)
    if n < 5 or m < 1 or not 1 <= lag <= n or int(cv_folds) < 1:
        raise ValueError("invalid sequence length, lag, or fold count")
    if int(iterations) < 0:
        raise ValueError("completion iterations must be nonnegative")
    if not 0 < holdout_fraction < .5:
        raise ValueError("holdout_fraction must lie between zero and 0.5")
    degrees = tuple(sorted(set(int(q) for q in trend_degrees)))
    if not degrees or min(degrees) < 0 or max(degrees) > 2:
        raise ValueError("trend degrees must be a nonempty subset of {0,1,2}")
    kmax = min(int(maxrank), m * lag, n - lag + 1)
    if kmax < 0:
        raise ValueError("maxrank must be nonnegative")
    rng = np.random.default_rng(seed)
    masks = []
    for _ in range(int(cv_folds)):
        observed = rng.random(y.shape) >= holdout_fraction
        for c in range(m):
            if observed[:, c].all():
                observed[int(rng.integers(n)), c] = False
            if observed[:, c].sum() < max(degrees) + 1:
                raise ValueError("mask leaves too few training observations")
        masks.append(observed)
    losses = {(q, rank): [] for q in degrees for rank in range(kmax + 1)}
    weights = []
    for fold, observed in enumerate(masks):
        heldout = ~observed; weights.append(int(heldout.sum()))
        for degree in degrees:
            predictions, _ = _masked_family(y, observed, lag, kmax, degree,
                iterations, seed + 1000 * fold + 100 * degree)
            for rank, prediction in enumerate(predictions):
                losses[(degree, rank)].append(float(np.mean((prediction[heldout] - y[heldout]) ** 2)))
    candidates = [dict(degree=q, rank=k, cv_mse_raw=float(np.average(value, weights=weights)),
                       fold_mse_raw=value) for (q, k), value in losses.items()]
    best_loss = min(item["cv_mse_raw"] for item in candidates)
    tie = 64 * np.finfo(float).eps * max(float(np.mean(y ** 2)), 1e-12)
    eligible = [item for item in candidates if item["cv_mse_raw"] <= best_loss + tie]
    selected = min(eligible, key=lambda item: (item["degree"], item["rank"]))
    degree, rank = selected["degree"], selected["rank"]
    parameters, trend = _fit_trend(y, np.ones_like(y, dtype=bool), degree,
                                   np.arange(n, dtype=float))
    normalized = (y - trend) / parameters["channel_scale"]
    u, singular, _ = _svd(_windows(normalized, lag), rank, seed + 99991)
    return dict(lag=lag, channels=m, train_count=n, rank=rank, trend=parameters,
        U=u, singular_values=singular, seed=int(seed),
        cv=dict(folds=int(cv_folds), holdout_fraction=float(holdout_fraction),
            completion_iterations=int(iterations), completion_rank=kmax,
            completion_rule="maximal-rank masked completion followed by nested rank projections",
            mask_hashes=[hashlib.sha256(mask.tobytes()).hexdigest() for mask in masks],
            heldout_counts=weights, candidates=candidates, selected=selected,
            numerical_tie_mse=tie,
            selection_rule="minimum held-out raw scalar MSE; numerical ties prefer lower degree then rank"),
        scope=dict(no_phases_or_clean_values_used=True,
            trend_separation_identifiable=False,
            residual_geometry_note="low total reading error does not certify a stationary or torus-preserving residual; trend-degree CV can remove genuine low-frequency content",
            masked_risk_identity_requires="independent zero-mean measurement noise at held-out raw entries",
            trend_assumption="one global polynomial of the selected degree over train and query times",
            test_mode="conditioning on new observed windows, not forecasting; no query SVD or trend refit",
            rank_note="rank is the truncated window-subspace rank; folding enforces Hankel consistency but can raise matrix rank",
            boundary_note="finite-block folding has boundary-specific filters; trim lag-1 samples per side before field fitting"))


def transform_preprocess(model, Y, start_index=0):
    """Apply frozen trend, scaling and signal subspace to new observed readings."""
    y = np.asarray(Y, dtype=float)
    if y.ndim != 2 or y.shape[1] != model["channels"] or not np.isfinite(y).all():
        raise ValueError("Y must be finite and have the fitted sensor count")
    lag = model["lag"]
    if len(y) < lag:
        raise ValueError("query block shorter than denoising lag")
    times = np.arange(len(y), dtype=float) + float(start_index)
    parameters = model["trend"]
    trend = np.einsum("nt,tc->nc", _polynomial(times, parameters["center"],
        parameters["time_scale"], parameters["degree"]), parameters["coefficients"], optimize=False)
    normalized = (y - trend) / parameters["channel_scale"]
    h = _windows(normalized, lag); u = model["U"]
    coefficients = np.einsum("ak,ab->kb", u, h, optimize=True)
    projected = np.einsum("ak,kb->ab", u, coefficients, optimize=True)
    residual = _fold(projected, y.shape[1], lag) * parameters["channel_scale"]
    valid = np.zeros(len(y), dtype=bool)
    if len(y) >= 2 * lag - 1:
        valid[lag - 1:len(y) - lag + 1] = True
    return dict(residual=residual, trend=trend, denoised=trend + residual,
                valid_mask=valid, time_index=times, boundary_trim_each_side=lag - 1,
                rank=model["rank"], trend_degree=parameters["degree"])


def fit_transform_train(Y, **kwargs):
    model = fit_preprocess(Y, **kwargs)
    return model, transform_preprocess(model, Y, start_index=0)


def highpass_decompose(model, Y, start_index=0, window=129, polyorder=2):
    """Frozen MSSA conditioning followed by one fixed symmetric FIR split.

    Requires a mean-only preprocessing model, fitted with trend_degrees=(0,).
    The slow component is the Savitzky--Golay lowpass of the denoised observed
    readings; it is not an identified physical trend. The returned residual is
    the complementary highpass. Both maps commute with time shifts on the
    reported valid interior. Polynomial input components of degree at most
    polyorder vanish from that residual on the same interior, up to roundoff.

    window/polyorder must be kept fixed between training and query blocks. No
    parameters are fitted to the new block. This is observed reconstruction,
    not forecasting. Essential phase characters can still be attenuated or
    removed by either filter; delay embedding must be checked downstream.
    """
    if model["trend"]["degree"] != 0:
        raise ValueError("highpass_decompose requires trend_degrees=(0,) at fit")
    window = int(window); polyorder = int(polyorder)
    if window < 3 or window % 2 != 1 or not 0 <= polyorder < window:
        raise ValueError("window must be odd >=3, with 0 <= polyorder < window")
    conditioned = transform_preprocess(model, Y, start_index=start_index)
    denoised = conditioned["denoised"]
    if len(denoised) < window:
        raise ValueError("query block shorter than the fixed smoothing window")
    # Edge interpolation is deliberately excluded by valid_mask. In its valid
    # interior this is exactly the fixed symmetric convolution kernel.
    slow = savgol_filter(denoised, window_length=window, polyorder=polyorder,
                         axis=0, mode="interp")
    trim = model["lag"] - 1 + (window - 1) // 2
    valid = np.zeros(len(denoised), dtype=bool)
    if len(denoised) >= 2 * trim + 1:
        valid[trim:len(denoised)-trim] = True
    return dict(slow=slow, residual=denoised-slow, denoised=denoised,
        valid_mask=valid, time_index=conditioned["time_index"],
        boundary_trim_each_side=trim,
        filter=dict(kind="symmetric_savitzky_golay", window=window,
                    polyorder=polyorder, fixed_between_train_and_query=True),
        scope=dict(slow_is_identified_physical_trend=False,
            polynomial_annihilation_degree=polyorder,
            shift_equivariance_region="valid_mask interior only",
            phase_preservation_condition="surviving filtered characters must still give an observable delay embedding",
            test_mode="conditioning on the new observed block, not forecasting"))
