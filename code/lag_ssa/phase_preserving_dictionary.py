"""TRAIN coefficient-noise pruning with all acquired phase anchors retained.

This operates on a fixed discovered rotation/group and an already selected
real Fourier dictionary. It neither discovers phases nor changes its ridge.
The noise ratio is a conditional coefficient-energy proxy, not a calibrated
test and not the energy of a mode over a finite, nonorthogonal time window.
"""

import numpy as np
from scipy.linalg import solve
from sympy import Matrix
from sympy.matrices.normalforms import hermite_normal_form


THRESHOLDS = (0., 1., 2., 4.)


def row_multiple(value, relation):
    """Exact membership in the UNSATURATED lattice Z*relation."""
    value = np.asarray(value, dtype=int)
    if relation is None:
        return 0 if not np.any(value) else None
    row = np.asarray(relation, dtype=int).reshape(-1)
    indices = np.flatnonzero(row)
    if not len(indices):
        raise ValueError("nonzero relation required")
    first = int(indices[0])
    if int(value[first]) % int(row[first]):
        return None
    multiple = int(value[first]) // int(row[first])
    return multiple if np.array_equal(value, multiple*row) else None


def anchor_mapping(labels, acquired_labels, phase_products, relation=None):
    labels = np.asarray(labels, dtype=int)
    acquired = np.asarray(acquired_labels, dtype=int)
    products = np.asarray(phase_products, dtype=int)
    rank = labels.shape[1]
    if acquired.ndim != 2 or acquired.shape[1] != rank or products.shape != (rank, len(acquired)):
        raise ValueError("acquired A and integer product B must use the selected phase basis")
    identity = products.astype(object) @ acquired.astype(object)
    valid = np.array_equal(identity, np.eye(rank, dtype=int))
    mapping = [];protected = set();missing = []
    for number, anchor in enumerate(acquired):
        constant = row_multiple(anchor, relation)
        if constant is not None:
            mapping.append(dict(acquired_index=number, acquired_label=anchor.tolist(),
                                constant_on_presented_group=True, relation_multiple=constant))
            continue
        found = None
        for index, label in enumerate(labels):
            for sign in (1, -1):
                multiple = row_multiple(anchor-sign*label, relation)
                if multiple is not None:
                    found = dict(acquired_index=number, acquired_label=anchor.tolist(),
                        dictionary_index=index, dictionary_label=label.tolist(),
                        conjugacy_sign=sign, relation_multiple=multiple,
                        constant_on_presented_group=False)
                    break
            if found is not None:
                break
        if found is None:
            missing.append(number)
            mapping.append(dict(acquired_index=number, acquired_label=anchor.tolist(),
                                status="missing_from_frozen_dictionary"))
        else:
            mapping.append(found);protected.add(found["dictionary_index"])
    return dict(mapping=mapping, protected_indices=sorted(protected),
        missing_acquired_indices=missing, exact_integer_BA_identity=bool(valid),
        acquired_labels=acquired.tolist(), phase_products=products.tolist(),
        relation_lattice_saturated=False, missing_anchors_added=False)


def lattice_index(labels, relation=None):
    rows = np.asarray(labels, dtype=int)
    if relation is not None:
        rows = np.vstack((rows, np.asarray(relation, dtype=int).reshape(1, -1)))
    basis = hermite_normal_form(Matrix(rows.T.tolist()))
    return abs(int(basis.det())) if basis.cols == rows.shape[1] else None


def _design(times, frequencies, labels, nyquist):
    angle = np.asarray(times)[:, None]*np.einsum("ij,j->i", labels, frequencies,
                                               optimize=False)[None, :]
    cosine, sine = np.cos(angle), np.sin(angle)
    sine[:, nyquist] = 0.
    return np.column_stack((np.ones(len(times)), cosine, sine))


def fixed_refit(raw_train, times, frequencies, labels, temporal_ridge,
                mass_sqrt, multiplicities=None, nyquist=None):
    """Full finite Gram ridge solve and conditional raw coefficient covariance."""
    values = np.asarray(raw_train, dtype=float)
    times = np.asarray(times, dtype=float)
    labels = np.asarray(labels, dtype=int)
    count = len(labels)
    if values.ndim != 2 or values.shape[1] != len(times):
        raise ValueError("raw sensor-by-time TRAIN observations required")
    if temporal_ridge < 0:
        raise ValueError("nonnegative frozen temporal ridge required")
    multiplicity = np.ones(count) if multiplicities is None else np.asarray(multiplicities, dtype=float)
    nyquist = np.zeros(count, dtype=bool) if nyquist is None else np.asarray(nyquist, dtype=bool)
    if multiplicity.shape != (count,) or nyquist.shape != (count,) or np.any(multiplicity <= 0):
        raise ValueError("one positive multiplicity and Nyquist flag per retained line required")
    full = _design(times, np.asarray(frequencies), labels, nyquist)
    active = np.r_[0, np.arange(1, 1+count), 1+count+np.flatnonzero(~nyquist)]
    design = full[:, active]
    penalty = np.r_[0., 1/multiplicity, 1/multiplicity[~nyquist]]
    gram = np.einsum("tn,tm->nm", design, design, optimize=False)/len(times)
    hessian = gram+temporal_ridge*np.diag(penalty)
    right = np.einsum("tn,qt->nq", design, values, optimize=False)/len(times)
    fitted = solve(hessian, right, assume_a="pos")
    inverse = solve(hessian, np.eye(len(active)), assume_a="pos")
    covariance_left = np.einsum("ij,jk->ik", inverse, gram, optimize=False)
    covariance_active = np.einsum("ij,kj->ik", covariance_left, inverse,
                                  optimize=False)/len(times)
    coefficients = np.zeros((1+2*count, values.shape[0]));coefficients[active] = fitted
    covariance = np.zeros((1+2*count, 1+2*count))
    covariance[np.ix_(active, active)] = covariance_active
    coefficient_mass = np.einsum("nq,iq->ni", coefficients, mass_sqrt, optimize=False)
    return dict(coefficients_sensor=coefficients, coefficients_mass=coefficient_mass,
        coefficient_covariance_unit_raw_noise=covariance,
        temporal_gram_condition=float(np.linalg.cond(gram)),
        ridge_hessian_condition=float(np.linalg.cond(hessian)),
        active_design_columns=active.tolist(), multiplicities=multiplicity,
        nyquist=nyquist, full_finite_temporal_gram=True)


def prepare_dictionary_candidates(raw_train, times, frequencies, labels,
        baseline_coefficients_mass, temporal_ridge, mass_sqrt, sigma_train,
        acquired_labels, phase_products, relation=None, multiplicities=None,
        nyquist=None, thresholds=THRESHOLDS):
    """Four fixed-rotation TRAIN fits, with exact integer anchor protection.

    A caller supplies an independent TRAIN conditioning guard and conditional
    held-lag VAL comparison. This function never reads VAL or evaluation data.
    """
    labels = np.asarray(labels, dtype=int)
    frequencies = np.asarray(frequencies, dtype=float)
    mass_sqrt = np.asarray(mass_sqrt, dtype=float)
    count = len(labels)
    if count < 1 or labels.shape[1] != len(frequencies):
        raise ValueError("nonempty fixed learned character dictionary required")
    if not np.isfinite(sigma_train) or sigma_train < 0:
        raise ValueError("nonnegative TRAIN-only raw noise estimate required")
    multiplicity = np.ones(count) if multiplicities is None else np.asarray(multiplicities)
    nyquist = np.zeros(count, dtype=bool) if nyquist is None else np.asarray(nyquist, dtype=bool)
    anchors = anchor_mapping(labels, acquired_labels, phase_products, relation)
    baseline = fixed_refit(raw_train, times, frequencies, labels, temporal_ridge,
                           mass_sqrt, multiplicity, nyquist)
    difference = float(np.max(abs(baseline["coefficients_mass"]-baseline_coefficients_mass)))
    scale = max(1., float(np.max(abs(baseline_coefficients_mass))))
    if difference > 1e-7*scale:
        raise ArithmeticError("threshold0 TRAIN refit differs from the frozen selected model")
    covariance = baseline["coefficient_covariance_unit_raw_noise"]
    mass = baseline["coefficients_mass"]
    energy = np.sum(mass[1:1+count]**2+mass[1+count:]**2, axis=1)
    mass_trace = float(np.sum(mass_sqrt**2))
    expected = sigma_train**2*mass_trace*(np.diag(covariance)[1:1+count]
                                         +np.diag(covariance)[1+count:])
    proxy_available = sigma_train > 0
    if proxy_available and np.any(expected <= 0):
        raise ValueError("strictly positive conditional pair-noise expectations required")
    ratio = energy/expected if proxy_available else None
    protected = np.zeros(count, dtype=bool);protected[anchors["protected_indices"]] = True
    candidates = []
    for threshold in thresholds:
        retained = (np.flatnonzero((ratio >= threshold) | protected)
                    if threshold and proxy_available else np.arange(count))
        fit = baseline if len(retained) == count else fixed_refit(raw_train, times, frequencies,
            labels[retained], temporal_ridge, mass_sqrt, multiplicity[retained], nyquist[retained])
        index = lattice_index(labels[retained], relation)
        complete_anchors = not anchors["missing_acquired_indices"] and anchors["exact_integer_BA_identity"]
        candidates.append(dict(threshold=float(threshold), retained_indices=retained.tolist(),
            labels=labels[retained], frequencies=frequencies.copy(),
            coefficients_mass=fit["coefficients_mass"], coefficients_sensor=fit["coefficients_sensor"],
            relation=None if relation is None else np.asarray(relation, dtype=int).copy(),
            quotient_multiplicities=multiplicity[retained].tolist(),
            nyquist_cosine_only_labels=labels[retained][nyquist[retained]].tolist(),
            retained_count=len(retained), original_count=count, character_lattice_index=index,
            all_original_observed_anchors_preserved=bool(complete_anchors),
            anchor_preserving_pruning_eligible=bool(complete_anchors and index == 1),
            temporal_gram_condition=fit["temporal_gram_condition"],
            ridge_hessian_condition=fit["ridge_hessian_condition"]))
    return dict(candidates=candidates, metadata=dict(declared_thresholds=list(thresholds),
        coefficient_mass_energy=energy.tolist(), conditional_expected_coefficient_noise=expected.tolist(),
        conditional_coefficient_noise_ratio=ratio.tolist() if proxy_available else [None]*count,
        noise_proxy_available=proxy_available,
        zero_noise_rule="retain complete frozen dictionary; no noise-ratio pruning",
        noise_estimate_raw_sigma_train=float(sigma_train),
        mass_matrix_trace=mass_trace, acquired_anchor_protection=anchors,
        baseline_refit_max_absolute_difference=difference, baseline_refit_comparison_scale=scale,
        ridge_scale_frozen=float(temporal_ridge), rotation_and_group_unchanged=True,
        finite_temporal_gram_not_diagonalized=True, threshold_ratios_recomputed_after_pruning=False,
        coefficient_energy_is_finite_time_mode_energy=False, calibrated_noise_test_claimed=False,
        raw_noise_model="conditional iid sensor-time noise; supplied TRAIN estimate",
        conditional_on_previously_validation_selected_dictionary=True))
