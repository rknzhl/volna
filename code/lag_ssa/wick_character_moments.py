"""Noise-corrected measured coefficient products of degree at most eight.

Inputs are real linear readouts beta=[all cos coefficients, all sin
coefficients] and their FULL correlated real Gaussian noise covariance Sigma.
No coordinate is normalized, no polynomial is clipped, and no clock phase,
data archive, profile fitter or physical rank is an input.

For fixed bank and known covariance, the subset recurrence computes
exp(-Sigma:grad^2/2) p_a exactly as a finite polynomial (up to roundoff).
Its conditional mean equals p_a(mu). The character/unit-circle interpretation
and the variance bound additionally require noiseless complex coordinates
mu_cj+i*mu_sj to have modulus one. An estimated covariance or learned-template
error does not make the returned reference bounds a calibrated confidence
interval. Polynomial cancellation and finite-record near resonances remain.
"""

from math import comb, factorial, sqrt

import numpy as np


MAXIMUM_DEGREE = 8
COVARIANCE_RELATIVE_TOLERANCE = 1e-10
COVARIANCE_ABSOLUTE_TOLERANCE = 1e-12


def _validated_factors(beta, covariance, relation):
    value = np.asarray(beta)
    if (value.ndim != 2 or value.shape[0] < 1 or value.shape[1] % 2
            or value.shape[1] == 0 or np.iscomplexobj(value)):
        raise ValueError("real beta(samples,2J), layout allcos/allsin, required")
    value = np.asarray(value, dtype=float)
    if not np.isfinite(value).all():raise ValueError("finite measured coefficient readouts required")
    dimension = value.shape[1]//2
    indices = list(relation)
    if (len(indices) != dimension
            or any(isinstance(item, (bool, np.bool_)) or not isinstance(item, (int, np.integer)) for item in indices)):
        raise ValueError("relation must contain exactly J integer coordinates")
    integers = [int(item) for item in indices]
    degree = sum(abs(item) for item in integers)
    if degree > MAXIMUM_DEGREE:raise ValueError("only l1 degree <=8 is supported")
    sigma = np.asarray(covariance)
    if sigma.shape != (2*dimension, 2*dimension) or np.iscomplexobj(sigma):
        raise ValueError("full real Sigma(2J,2J) required")
    sigma = np.asarray(sigma, dtype=float)
    if not np.isfinite(sigma).all():raise ValueError("finite full covariance required")
    if not np.allclose(sigma, sigma.T, rtol=COVARIANCE_RELATIVE_TOLERANCE,
                       atol=COVARIANCE_ABSOLUTE_TOLERANCE):
        raise ValueError("real Gaussian covariance must be symmetric")
    symmetry_defect = float(np.max(abs(sigma-sigma.T)))
    sigma = (sigma+sigma.T)/2
    eigenvalues = np.linalg.eigvalsh(sigma)
    scale = max(float(np.max(abs(eigenvalues))), COVARIANCE_ABSOLUTE_TOLERANCE)
    if eigenvalues[0] < -COVARIANCE_RELATIVE_TOLERANCE*scale-COVARIANCE_ABSOLUTE_TOLERANCE:
        raise ValueError("real Gaussian covariance must be positive semidefinite")
    factors, labels = np.zeros((degree, 2*dimension), dtype=complex), []
    position = 0
    for coordinate, power in enumerate(integers):
        sign = 1 if power >= 0 else -1
        for _ in range(abs(power)):
            factors[position, coordinate] = 1
            factors[position, dimension+coordinate] = 1j*sign
            labels.append(dict(coordinate=coordinate, sign=sign))
            position += 1
    audit = dict(real_coordinate_layout="allcos,allsin", degree=degree, relation=integers,
        repeated_factor_positions=labels, covariance_symmetry_absolute_defect=symmetry_defect,
        covariance_minimum_eigenvalue=float(eigenvalues[0]),
        covariance_symmetric_part_used=True, covariance_eigenvalues_not_clipped=True,
        gaussian_covariance_is_supplied_not_estimated_here=True)
    return value, sigma, factors, audit


def _wick_subset_products(linear_values, pair_covariance):
    """Position-labelled matchings, including every repeated-factor pairing.

    Pick i=min(S): F(S)=ell_i F(S\i)-sum_{j in S\i} Cij F(S\{i,j}).
    Cij=E eta_i eta_j is BILINEAR, without complex conjugation.
    """
    samples, degree = linear_values.shape
    values = np.empty((1 << degree, samples), dtype=complex)
    values[0] = 1
    for mask in range(1, 1 << degree):
        first_bit = mask & -mask
        first = first_bit.bit_length()-1
        remaining = mask ^ first_bit
        answer = linear_values[:, first]*values[remaining]
        other_bits = remaining
        while other_bits:
            bit = other_bits & -other_bits
            second = bit.bit_length()-1
            answer = answer-pair_covariance[first, second]*values[remaining ^ bit]
            other_bits ^= bit
        values[mask] = answer
    if not np.isfinite(values).all():raise FloatingPointError("nonfinite Wick polynomial; no clipping performed")
    return values[-1].copy()


def _nonempty_cross_matching_bound(absolute_cross_covariance):
    """Sum all nonempty partial bipartite matchings, a positive rook DP.

    The two labelled factor copies have no internal contractions after Wick
    correction. Cross contractions are E eta_i conjugate(eta_j). Unpaired
    noiseless factors have modulus one. Taking absolute values therefore
    bounds the conditional complex variance uniformly over their phases.
    """
    degree = len(absolute_cross_covariance)
    weights = np.zeros(1 << degree, dtype=float)
    weights[0] = 1.
    for first in range(degree):
        previous, current = weights, weights.copy()  # unmatched left vertex
        for mask, weight in enumerate(previous):
            if not weight:continue
            for second in range(degree):
                bit = 1 << second
                if not mask & bit:
                    current[mask | bit] += weight*absolute_cross_covariance[first, second]
        weights = current
    bound = float(np.sum(weights[1:]))  # omit empty matching without subtractive cancellation
    if not np.isfinite(bound):raise FloatingPointError("nonfinite matching variance bound")
    return bound


def wick_character_moments(beta, Sigma, a):
    """Return raw/corrected products and the CONDITIONAL unit-mean variance B.

    The corrected arrays are unbiased for the polynomial of the actual
    noiseless linear readout, not automatically for a physical character.
    A unit gauge q must be learned/frozen independently of these heldout
    samples for 2-2Re(conj(q)*corrected_mean) to be an unbiased average
    squared unit-character departure. It may numerically fall below zero;
    that value is preserved. Fitting q on this mean introduces selection bias.
    """
    value, covariance, factors, audit = _validated_factors(beta, Sigma, a)
    linear = np.einsum("nj,dj->nd", value, factors, optimize=False)
    contractions = np.einsum("ij,jk,lk->il", factors, covariance, factors, optimize=False)
    cross = np.einsum("ij,jk,lk->il", factors, covariance, factors.conj(), optimize=False)
    raw = np.prod(linear, axis=1)
    corrected = _wick_subset_products(linear, contractions)
    if not np.isfinite(raw).all():raise FloatingPointError("nonfinite raw polynomial; no clipping performed")
    bound = _nonempty_cross_matching_bound(abs(cross))
    return dict(raw_products=raw, corrected_products=corrected,
        raw_mean=complex(np.mean(raw)), corrected_mean=complex(np.mean(corrected)),
        pointwise_variance_upper=bound, sample_count=len(value), audit=audit,
        variance_bound_scope="uniform conditional E|H-EH|^2; fixed bank, known covariance, unit noiseless complex coordinates",
        products_normalized=False, products_or_means_clipped=False,
        calibrated_confidence_interval=False, physical_character_or_relation_certified=False,
        constant_departure_requires_independently_frozen_unit_gauge=True)


def average_noise_bound(pointwise_variance_upper, samples, window_length):
    """Conservative variance/STD of the mean for contiguous overlapping windows.

    With frozen independent TRAIN bank, iid raw Gaussian measurement noise,
    and window length L, noise errors at separation >=L are independent.
    Cauchy-Schwarz bounds each remaining covariance by B. Phases may be
    deterministic and dependent: the target is THIS finite-record average.
    This does not test population Haar coverage or exact integer relations.
    """
    if (isinstance(samples, (bool, np.bool_)) or isinstance(window_length, (bool, np.bool_))
            or not isinstance(samples, (int, np.integer)) or not isinstance(window_length, (int, np.integer))
            or samples < 1 or window_length < 1):
        raise ValueError("positive integer sample count and window length required")
    bound = float(pointwise_variance_upper)
    if not np.isfinite(bound) or bound < 0:raise ValueError("finite nonnegative pointwise variance bound required")
    count, length = int(samples), int(window_length)
    overlap = min(length-1, count-1)
    pair_count = count*(1+2*overlap)-overlap*(overlap+1)
    factor = pair_count/(count*count)
    variance = bound*factor
    standard_deviation = sqrt(variance)
    return dict(pointwise_variance_upper=bound, sample_count=count, window_length=length,
        maximum_dependent_start_separation=overlap, counted_diagonal_and_ordered_overlap_pairs=pair_count,
        averaging_factor=factor, average_variance_upper=variance,
        average_standard_deviation_upper=standard_deviation,
        fixed_unit_gauge_departure_standard_deviation_upper=2*standard_deviation,
        formula="B*(M+2*sum_{h=1}^{min(L-1,M-1)}(M-h))/M^2",
        variance_target="noise around this deterministic finite-record mean, not randomness of orbit phases",
        conditional_fixed_bank_known_covariance_unit_means=True,
        estimated_covariance_does_not_give_calibrated_CI=True,
        orbit_iid_or_Haar_distribution_required=False,
        future_or_global_relation_certified=False)


def character_moment(beta, covariance, row):
    """Compact runner interface, same unnormalized polynomial and assumptions."""
    result = wick_character_moments(beta, covariance, row)
    return dict(raw=result["raw_products"], corrected=result["corrected_products"],
                pointwise_variance_upper_bound=result["pointwise_variance_upper"],
                audit=result["audit"], calibrated_confidence_interval=False,
                unit_noiseless_readout_condition_required=True)


def overlapping_mean_variance_bound(pointwise_variance_upper_bound, samples, window_length):
    """Return the conservative CONDITIONAL VARIANCE bound, not its square root."""
    return average_noise_bound(pointwise_variance_upper_bound, samples, window_length)["average_variance_upper"]


def analytic_self_test():
    """Independent explicit polynomial/variance identities; no data or fitting."""
    beta = np.asarray([[.7, -.3, .2, .8], [-1.2, .4, .9, -.2], [.1, .6, -.5, .3]])
    root = np.asarray([[.8, .1, -.2, .3], [.2, .9, .4, -.1], [.1, -.3, .6, .2], [.4, .2, -.1, .7]])
    covariance = np.einsum("ij,kj->ik", root, root, optimize=False)
    first, second = beta[:, 0]+1j*beta[:, 2], beta[:, 1]+1j*beta[:, 3]
    hermitian = covariance[0,1]+covariance[2,3]+1j*(covariance[2,1]-covariance[0,3])
    bilinear = covariance[0,1]-covariance[2,3]+1j*(covariance[0,3]+covariance[2,1])
    first_square = covariance[0,0]-covariance[2,2]+2j*covariance[0,2]
    np.testing.assert_allclose(wick_character_moments(beta, covariance, [1,-1])["corrected_products"],
                               first*second.conj()-hermitian, rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(wick_character_moments(beta, covariance, [1,1])["corrected_products"],
                               first*second-bilinear, rtol=2e-14, atol=2e-14)
    expected = first**2*second.conj()-first_square*second.conj()-2*hermitian*first
    np.testing.assert_allclose(wick_character_moments(beta, covariance, [2,-1])["corrected_products"],
                               expected, rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(wick_character_moments(beta, covariance, [-2,1])["corrected_products"],
                               expected.conj(), rtol=2e-14, atol=2e-14)
    variance_checks = 0
    for degree in range(1, MAXIMUM_DEGREE+1):
        for variance in (0., .005, .1, 1., 2.):
            single_beta = np.asarray([[.4, -.7], [-.2, .8]])
            result = wick_character_moments(single_beta, np.eye(2)*(variance/2), [degree])
            reference = sum(comb(degree,k)**2*factorial(k)*variance**k for k in range(1, degree+1))
            np.testing.assert_allclose(result["pointwise_variance_upper"], reference, rtol=2e-14, atol=2e-14)
            np.testing.assert_allclose(result["corrected_products"], result["raw_products"], rtol=2e-14, atol=2e-14)
            variance_checks += 1
    phase = np.asarray([[.2,-.7,1.3], [2.1,.8,-.3], [-1.4,2.6,.1]])
    relation = np.asarray([2,-1,3], dtype=int)
    readout = np.column_stack((np.cos(phase), np.sin(phase)))
    result = wick_character_moments(readout, np.zeros((6,6)), relation)
    np.testing.assert_allclose(result["corrected_products"], np.exp(1j*np.einsum("nj,j->n",phase,relation)),
                               rtol=2e-14, atol=2e-14)
    assert result["pointwise_variance_upper"] == 0
    np.testing.assert_allclose(average_noise_bound(7., 19, 1)["average_variance_upper"], 7./19)
    np.testing.assert_allclose(average_noise_bound(7., 19, 128)["average_variance_upper"], 7.)
    np.testing.assert_allclose(average_noise_bound(7., 1, 128)["average_variance_upper"], 7.)
    return dict(status="passed", explicit_correlated_low_degree_checks=4,
                circular_single_coordinate_variance_checks=variance_checks,
                noiseless_character_check=1, overlap_boundary_checks=3,
                no_benchmark_public_or_truth_archives_read=True)


if __name__ == "__main__":
    print(analytic_self_test())
