"""Full local linear Gaussian prediction with shared frequency coordinates.

Xtrain[N,p] is the common real amplitude design at every raw sensor.
Dtrain[N,q,r] is the derivative of each sensor mean with respect to the SAME
r frequency tangent coordinates. Arrays at new times use the same parameter
gauge/centering. Residuals are relative to the supplied old nonlinear fit.

For the frozen LOCAL LINEAR model, iid known raw noise variance, flat parameter
prior and resolved full information rank, this is the exact posterior mean and
Gaussian predictive density. It is not the posterior of the original nonlinear
sinusoidal decoder, an integrated group evidence, or a calibrated predictive
density when noise/template assumptions or the local approximation fail.
No data archive, benchmark, true phase or physical rank is read by this module.
"""

import math

import numpy as np
from scipy.linalg import cho_factor, cho_solve


INFORMATION_RCOND = 1e-12


def _real_finite(value, name, dimension):
    array = np.asarray(value)
    if array.ndim != dimension or np.iscomplexobj(array):
        raise ValueError(name+" must be a real array of the required dimension")
    array = np.asarray(array, dtype=float)
    if not np.isfinite(array).all():raise ValueError(name+" contains nonfinite values")
    return array


def _spd(matrix, name, reference_scale=None):
    symmetric = (matrix+matrix.T)/2
    values = np.linalg.eigvalsh(symmetric)
    if not len(values):raise ValueError("empty SPD solver should be handled explicitly")
    resolved_scale = max(float(values[-1]),float(reference_scale)) if reference_scale is not None else float(values[-1])
    if (not np.isfinite(values).all() or values[-1] <= 0
            or values[0] <= INFORMATION_RCOND*resolved_scale):
        raise np.linalg.LinAlgError(name+" unresolved positive-definite information; no truncation or ridge used")
    factor = cho_factor(symmetric, lower=True, check_finite=True)
    logdet = float(2*np.sum(np.log(np.diag(factor[0]))))
    return factor, logdet, dict(minimum_eigenvalue=float(values[0]), maximum_eigenvalue=float(values[-1]),
        condition=float(values[-1]/values[0]), required_relative_eigenvalue=INFORMATION_RCOND,
        numerical_information_reference_scale=resolved_scale,
        full_rank_resolved=True, pseudoinverse_or_eigenvalue_clipping=False)


def _solve(factor, right):
    shape = right.shape
    answer = cho_solve(factor, right.reshape(shape[0], -1), check_finite=True)
    return answer.reshape(shape)


def score_local_predictive(Xtrain, Dtrain, residual_train, Xnew, Dnew, residual_new, sigma):
    """Return exact LOCAL posterior shifts and a normalized raw predictive score.

    Both sensor-specific amplitude updates and shared-frequency updates use
    the nonzero TRAIN residual. No stationarity of the original fit is assumed.
    Frequency directions with unresolved information fail closed; setting their
    variance to zero via a pseudoinverse would not be a proper flat-prior model.
    """
    x, new_x = _real_finite(Xtrain,"Xtrain",2), _real_finite(Xnew,"Xnew",2)
    derivative, new_derivative = _real_finite(Dtrain,"Dtrain",3), _real_finite(Dnew,"Dnew",3)
    error, new_error = _real_finite(residual_train,"residual_train",2), _real_finite(residual_new,"residual_new",2)
    n, p = x.shape
    m = len(new_x)
    if (n < 1 or p < 1 or m < 1 or new_x.shape[1] != p or error.shape[0] != n
            or new_error.shape[0] != m or error.shape[1] < 1 or new_error.shape[1] != error.shape[1]):
        raise ValueError("matching positive raw sample/sensor/amplitude dimensions required")
    q, r = error.shape[1], derivative.shape[2]
    if derivative.shape != (n,q,r) or new_derivative.shape != (m,q,r):
        raise ValueError("shared frequency derivative shape must be [samples,sensors,r]")
    if not np.isfinite(sigma) or sigma <= 0:raise ValueError("known positive raw Gaussian sigma required")
    variance = float(sigma)**2

    gram = np.einsum("np,nk->pk", x, x, optimize=False)
    g_factor, g_logdet, g_audit = _spd(gram,"amplitude Gram G")
    gram_new = np.einsum("mp,mk->pk", new_x, new_x, optimize=False)
    gplus_factor, gplus_logdet, gplus_audit = _spd(gram+gram_new,"amplitude Gplus")
    amplitude_rhs = np.einsum("np,nq->pq", x, error, optimize=False)
    ols_update = _solve(g_factor, amplitude_rhs)
    amplitude_only_shift = np.einsum("mp,pq->mq", new_x, ols_update, optimize=False)

    def amplitude_predictive_inverse(value):
        if value.ndim == 2:
            right = np.einsum("mp,mq->pq", new_x, value, optimize=False)
            return value-np.einsum("mp,pq->mq", new_x, _solve(gplus_factor,right), optimize=False)
        right = np.einsum("mp,mqr->pqr", new_x, value, optimize=False)
        return value-np.einsum("mp,pqr->mqr", new_x, _solve(gplus_factor,right), optimize=False)

    if r:
        projection = _solve(g_factor, np.einsum("np,nqr->pqr", x, derivative, optimize=False))
        efficient_train = derivative-np.einsum("np,pqr->nqr", x, projection, optimize=False)
        efficient_new = new_derivative-np.einsum("mp,pqr->mqr", new_x, projection, optimize=False)
        information = np.einsum("nqr,nqs->rs", efficient_train, efficient_train, optimize=False)
        # Compare against the unprojected derivative scale as well: a sole
        # direction D=X T can leave a tiny positive ROUNDING residual J. Its
        # scalar S has condition one, which alone would falsely resolve it.
        full_derivative_information = np.einsum("nqr,nqs->rs", derivative, derivative, optimize=False)
        derivative_scale = float(np.max(np.linalg.eigvalsh(full_derivative_information)))
        s_factor, s_logdet, s_audit = _spd(information,"efficient shared-frequency information S",derivative_scale)
        frequency_rhs = np.einsum("nqr,nq->r", efficient_train, error, optimize=False)
        delta_frequency = _solve(s_factor, frequency_rhs)
        covariance_frequency = variance*_solve(s_factor, np.eye(r))
        covariance_coefficient_frequency = -np.einsum("pqr,rs->pqs", projection, covariance_frequency, optimize=False)
        coefficient_update = ols_update-np.einsum("pqr,r->pq", projection, delta_frequency, optimize=False)
        frequency_shift = np.einsum("mqr,r->mq", efficient_new, delta_frequency, optimize=False)
        inverse_new_derivative = amplitude_predictive_inverse(efficient_new)
        frequency_inflation = np.einsum("mqr,mqs->rs", efficient_new, inverse_new_derivative, optimize=False)
        splus_factor, splus_logdet, splus_audit = _spd(information+frequency_inflation,"frequency Splus")
    else:
        projection = np.empty((p,q,0))
        efficient_train, efficient_new = np.empty((n,q,0)), np.empty((m,q,0))
        information, covariance_frequency = np.empty((0,0)), np.empty((0,0))
        covariance_coefficient_frequency = np.empty((p,q,0))
        delta_frequency, frequency_rhs = np.empty(0), np.empty(0)
        coefficient_update, frequency_shift = ols_update.copy(), np.zeros((m,q))
        s_logdet, splus_logdet = 0., 0.
        s_audit = splus_audit = dict(dimension=0, inverse_not_needed=True, full_rank_resolved=True)
    shift = amplitude_only_shift+frequency_shift
    shifted_error = new_error-shift
    inverse_error = amplitude_predictive_inverse(shifted_error)
    base_quadratic = float(np.einsum("mq,mq->", shifted_error, inverse_error, optimize=False))
    frequency_quadratic = 0.
    if r:
        rhs = np.einsum("mqr,mq->r", efficient_new, inverse_error, optimize=False)
        frequency_quadratic = float(np.dot(rhs, _solve(splus_factor,rhs)))
    standardized_quadratic = (base_quadratic-frequency_quadratic)/variance
    if not np.isfinite(standardized_quadratic) or standardized_quadratic < 0:
        raise FloatingPointError("unresolved nonnegative predictive quadratic; no clipping performed")
    logdet_inflation = q*(gplus_logdet-g_logdet)+splus_logdet-s_logdet
    observations = m*q
    negative_log_density = .5*(observations*math.log(2*math.pi*variance)
                                +logdet_inflation+standardized_quadratic)
    train_projection_energy = float(np.einsum("pq,pq->", amplitude_rhs, ols_update, optimize=False))
    if r:train_projection_energy += float(np.dot(frequency_rhs,delta_frequency))
    result = dict(predicted_shift=shift,
        posterior_coefficient_mean_shift=coefficient_update,
        posterior_frequency_mean_shift=delta_frequency,
        posterior_frequency_covariance=covariance_frequency,
        posterior_coefficient_frequency_covariance=covariance_coefficient_frequency,
        coefficient_only_prediction_shift=amplitude_only_shift, frequency_prediction_shift=frequency_shift,
        negative_log_density=float(negative_log_density),
        negative_log_density_per_observation=float(negative_log_density/observations),
        logdet_inflation=float(logdet_inflation), standardized_quadratic=float(standardized_quadratic),
        point_mse_original=float(np.mean(new_error**2)), point_mse_shifted=float(np.mean(shifted_error**2)),
        mean_stationarity=dict(amplitude_normal_equation_norm=float(np.linalg.norm(amplitude_rhs)),
            efficient_frequency_normal_equation_norm=float(np.linalg.norm(frequency_rhs)),
            maximum_frequency_mean_shift=float(np.max(abs(delta_frequency))) if r else 0.,
            maximum_coefficient_mean_shift=float(np.max(abs(coefficient_update))),
            train_linear_projection_energy=train_projection_energy,
            train_standardized_linear_projection_energy=train_projection_energy/variance,
            exact_old_ML_stationarity_assumed=False, nonzero_train_residual_mean_correction_included=True),
        rank_diagnostics=dict(G=g_audit, Gplus=gplus_audit, S=s_audit, Splus=splus_audit),
        shape=dict(train_samples=n, new_samples=m, sensors=q, amplitude_terms=p,
                   shared_frequency_directions=r, joint_parameter_dimension=p*q+r,
                   raw_new_observations=observations),
        supplied_raw_sigma=float(sigma), raw_noise_variance_known_assumed=True,
        dense_new_covariance_constructed=False, full_amplitude_frequency_Schur_complement=True,
        no_pseudoinverse_truncation_or_ridge=True,
        normalized_density_for_frozen_local_linear_model=True,
        original_nonlinear_or_group_Bayesian_evidence=False,
        finite_winding_frequency_cell_and_mode_conditioned=True,
        estimated_sigma_template_tail_and_linearization_not_calibrated_here=True)
    for key,value in result.items():
        if isinstance(value,np.ndarray) and not np.isfinite(value).all():raise FloatingPointError("nonfinite local posterior output: "+key)
    return result


def _dense_reference(x,d,error,new_x,new_d,new_error,sigma):
    """Independent JOINT posterior inversion, used only by tiny analytic tests."""
    n,q,r = d.shape
    m,p = new_x.shape
    amplitude = np.kron(np.eye(q),x)
    new_amplitude = np.kron(np.eye(q),new_x)
    joint = np.column_stack((amplitude,d.transpose(1,0,2).reshape(n*q,r)))
    new_joint = np.column_stack((new_amplitude,new_d.transpose(1,0,2).reshape(m*q,r)))
    gram = np.einsum("np,nk->pk",joint,joint,optimize=False)
    inverse = np.linalg.solve(gram,np.eye(len(gram)))
    update = np.linalg.solve(gram,np.einsum("np,n->p",joint,error.T.ravel(),optimize=False))
    covariance = sigma**2*(np.eye(m*q)+np.einsum("mp,pk,nk->mn",new_joint,inverse,new_joint,optimize=False))
    shift = np.einsum("mp,p->m",new_joint,update,optimize=False)
    residual = new_error.T.ravel()-shift
    sign,logdet = np.linalg.slogdet(covariance)
    assert sign == 1
    quadratic = float(np.dot(residual,np.linalg.solve(covariance,residual)))
    return dict(predicted_shift=shift.reshape(q,m).T,
        posterior_coefficient_mean_shift=update[:p*q].reshape(q,p).T,
        posterior_frequency_mean_shift=update[p*q:],
        posterior_frequency_covariance=sigma**2*inverse[p*q:,p*q:],
        posterior_coefficient_frequency_covariance=(sigma**2*inverse[:p*q,p*q:]).reshape(q,p,r).transpose(1,0,2),
        standardized_quadratic=quadratic, logdet_inflation=float(logdet-m*q*math.log(sigma**2)),
        negative_log_density=.5*(m*q*math.log(2*math.pi)+logdet+quadratic))


def analytic_self_test():
    rng = np.random.default_rng(510237)
    n,m,p,q,r,sigma = 11,7,3,2,2,.37
    x,new_x = rng.normal(size=(n,p)),rng.normal(size=(m,p))
    x[:,0],new_x[:,0] = 1,1
    d,new_d = rng.normal(size=(n,q,r)),rng.normal(size=(m,q,r))
    error,new_error = rng.normal(size=(n,q)),rng.normal(size=(m,q))
    fields = ("predicted_shift","posterior_coefficient_mean_shift","posterior_frequency_mean_shift",
              "posterior_frequency_covariance","posterior_coefficient_frequency_covariance",
              "standardized_quadratic","logdet_inflation","negative_log_density")
    result = score_local_predictive(x,d,error,new_x,new_d,new_error,sigma)
    reference = _dense_reference(x,d,error,new_x,new_d,new_error,sigma)
    for field in fields:np.testing.assert_allclose(result[field],reference[field],rtol=3e-12,atol=3e-12)
    assert np.linalg.norm(result["posterior_frequency_mean_shift"]) > 1e-3
    transform = np.asarray([[1.3,.4],[-.2,.8]])
    changed = score_local_predictive(x,np.einsum("nqr,rs->nqs",d,transform,optimize=False),error,
        new_x,np.einsum("mqr,rs->mqs",new_d,transform,optimize=False),new_error,sigma)
    for field in ("predicted_shift","posterior_coefficient_mean_shift","negative_log_density",
                  "standardized_quadratic","logdet_inflation"):
        np.testing.assert_allclose(result[field],changed[field],rtol=3e-12,atol=3e-12)
    inverse_transform = np.linalg.inv(transform)
    np.testing.assert_allclose(changed["posterior_frequency_mean_shift"],
        np.einsum("rs,s->r",inverse_transform,result["posterior_frequency_mean_shift"],optimize=False),rtol=3e-12,atol=3e-12)
    np.testing.assert_allclose(changed["posterior_frequency_covariance"],
        np.einsum("rp,pk,sk->rs",inverse_transform,result["posterior_frequency_covariance"],inverse_transform,optimize=False),
        rtol=3e-12,atol=3e-12)
    zero,zero_new = np.empty((n,q,0)),np.empty((m,q,0))
    result_zero = score_local_predictive(x,zero,error,new_x,zero_new,new_error,sigma)
    reference_zero = _dense_reference(x,zero,error,new_x,zero_new,new_error,sigma)
    for field in fields:np.testing.assert_allclose(result_zero[field],reference_zero[field],rtol=3e-12,atol=3e-12)
    # A frequency derivative contained in the amplitude span has an improper
    # flat-prior frequency posterior and must fail, never acquire zero variance.
    collinear = np.einsum("np,pqr->nqr",x,np.ones((p,q,1)),optimize=False)
    try:
        score_local_predictive(x,collinear,error,new_x,np.zeros((m,q,1)),new_error,sigma)
    except np.linalg.LinAlgError:
        rank_failure = True
    else:
        raise AssertionError("unresolved shared frequency direction was silently accepted")
    return dict(status="passed", dense_nonstationary_joint_checks=len(fields),
        frequency_basis_invariance_checks=7, dense_r0_checks=len(fields),
        unresolved_frequency_information_rejected=rank_failure,
        no_benchmark_public_or_truth_archives_read=True)


if __name__ == "__main__":
    print(analytic_self_test())
