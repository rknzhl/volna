"""Numerical witness conditioning on a zero-gauge torus or one-relation group.

For G={theta: R theta=0 mod 2*pi}, use the UNSATURATED relation lattice.
Every original acquired row A_i must be a retained dictionary character,
up to sign and an integer multiple of R, or be constant on G.  Exact
integer identities A=T A_eff+D R and B A=I imply
B_eff A_eff+C R=I, with B_eff=B T and C=B D.

On G, wrap(theta-phi)=wrap(B_eff wrap(A_eff(theta-phi))).  Thus, if the
actual witness lag columns W survive the orthogonal complement of ALL
nuisance columns N with singular value sigma>0,
 d_T(theta,phi) <= pi*||B_eff||_2/(2*sigma) * ||Q.T(Htheta-Hphi)||_2.
Order-two characters have a genuine cosine and identically zero sine;
constant characters have no columns in a difference of two templates.

The API returns a thin Q and diagnostics, never an ambient projector.
Numerical rank decisions and nuisance leakage are explicitly reported;
the result is not a validated floating-point certificate, true-field
confidence bound, tail bound, phase-rank proof, or rationality certificate.
Only no relation or ONE nonzero integer relation is currently supported.
"""

from math import gcd

import numpy as np
from scipy.linalg import svd
from sympy import Matrix, eye


ROTATION_RESIDUE_TOLERANCE = 1e-8


def _integers(value, name, ndim=2):
    original = np.asarray(value)
    try:
        result = np.asarray(original, dtype=np.int64)
    except (ValueError, TypeError, OverflowError) as error:
        raise ValueError(name+" must contain finite representable integers") from error
    if result.ndim != ndim or not np.array_equal(original, result):
        raise ValueError(name+" must be an exact integer array of dimension "+str(ndim))
    return result


def _relation(value, rank):
    if value is None:
        return np.empty((0, rank), dtype=np.int64)
    original = np.asarray(value)
    if original.ndim == 1:
        original = original[None, :]
    result = _integers(original, "relation")
    if result.shape != (1, rank) or not np.any(result):
        raise ValueError("only one nonzero unsaturated integer relation row is supported")
    return result


def _multiple(value, relation):
    """Exact membership in Z*R, or in {0} when no relation is supplied."""
    row = [int(x) for x in value]
    if not len(relation):
        return 0 if not any(row) else None
    generator = [int(x) for x in relation[0]]
    index = next(index for index, x in enumerate(generator) if x)
    quotient, remainder = divmod(row[index], generator[index])
    if remainder or any(x != quotient*y for x, y in zip(row, generator)):
        return None
    return quotient


def character_order(label, relation=None):
    """Return the exact finite character order, or None for infinite order."""
    row = _integers(label, "character label", ndim=1)
    relation = _relation(relation, len(row))
    if _multiple(row, relation) is not None:
        return 1
    if not len(relation):
        return None
    generator = [int(x) for x in relation[0]]
    divisor = 0
    for value in generator:
        divisor = gcd(divisor, abs(value))
    primitive = np.asarray([value//divisor for value in generator], dtype=np.int64)[None, :]
    coefficient = _multiple(row, primitive)
    return None if coefficient is None else divisor//gcd(divisor, abs(coefficient))


def _matrix(array):
    rows, columns = array.shape
    return Matrix(rows, columns, [int(x) for x in array.reshape(-1)])


def _as_array(matrix):
    return np.asarray(matrix.tolist(), dtype=np.int64).reshape(matrix.rows, matrix.cols)


def quotient_witnesses(labels, acquired_labels, phase_products, relation=None):
    """Match ALL original anchors and return exact quotient identities.

    The input dictionary must already merge aliases and conjugate aliases.
    Its constant rows are ignored.  A missing nonconstant acquired class or
    an ambiguous retained class raises ValueError; no new anchors are learnt.
    """
    labels = _integers(labels, "labels")
    acquired = _integers(acquired_labels, "acquired_labels")
    products = _integers(phase_products, "phase_products")
    rank = labels.shape[1]
    if rank < 1 or acquired.shape[1] != rank or products.shape != (rank, len(acquired)):
        raise ValueError("matching labels, original acquired rows and r-by-p products required")
    relation = _relation(relation, rank)
    if _matrix(products)*_matrix(acquired) != eye(rank):
        raise ValueError("phase_products @ acquired_labels must equal I exactly")
    orders = [character_order(row, relation[0] if len(relation) else None) for row in labels]
    active = [index for index, order in enumerate(orders) if order != 1]
    for position, index in enumerate(active):
        for other in active[:position]:
            if (_multiple(labels[index]-labels[other], relation) is not None or
                    _multiple(labels[index]+labels[other], relation) is not None):
                raise ValueError("dictionary contains unmerged quotient/conjugate aliases")
    matched = []
    unique = []
    for row in acquired:
        constant = _multiple(row, relation)
        if constant is not None:
            matched.append((None, 0, constant))
            continue
        matches = []
        for index in active:
            # For order two both signs represent the same character; prefer
            # the positive sign consistently and retain its cosine only.
            for sign in (1, -1):
                multiple = _multiple(row-sign*labels[index], relation)
                if multiple is not None:
                    matches.append((index, sign, multiple))
                    break
        if len(matches) != 1:
            raise ValueError("missing or ambiguous acquired anchor quotient class: "+str(row.tolist()))
        match = matches[0]
        if match[0] not in unique:
            unique.append(match[0])
        matched.append(match)
    effective = labels[unique].reshape(len(unique), rank)
    transform = np.zeros((len(acquired), len(unique)), dtype=np.int64)
    offsets = np.zeros((len(acquired), len(relation)), dtype=np.int64)
    for row_index, (index, sign, multiple) in enumerate(matched):
        if index is not None:
            transform[row_index, unique.index(index)] = sign
        if len(relation):
            offsets[row_index, 0] = multiple
    if _matrix(acquired) != _matrix(transform)*_matrix(effective)+_matrix(offsets)*_matrix(relation):
        raise ArithmeticError("exact acquired-to-quotient identity failed")
    effective_products = _as_array(_matrix(products)*_matrix(transform))
    relation_products = _as_array(_matrix(products)*_matrix(offsets))
    if _matrix(effective_products)*_matrix(effective)+_matrix(relation_products)*_matrix(relation) != eye(rank):
        raise ArithmeticError("B_eff A_eff + C R must equal I exactly")
    return {
        "indices": np.asarray(unique, dtype=int), "orders": orders,
        "effective_labels": effective, "T": transform, "D": offsets,
        "B_eff": effective_products, "C": relation_products, "relation": relation,
        "matches": [{"original_acquired_label": row.tolist(),
            "dictionary_index": index, "orientation": sign,
            "relation_multiple": multiple, "constant_on_presented_group": index is None}
            for row, (index, sign, multiple) in zip(acquired, matched)]}


def conditioning(labels, coefficients_mass, alpha, acquired_labels, phase_products,
                 relation=None, lag_count=128, rcond=1e-10):
    """Return (Q_thin, information) for the same fixed mass metric and anchors.

    Rows of coefficients_mass are [mean, all cosines, all sines].  Alpha is
    radians/sample.  relation=None is the full torus.  For a compact model
    the rotation must already satisfy the relation numerically; this helper
    neither snaps it nor chooses a phase origin.  The compact gauge is zero.
    Compare sigma values only with unchanged anchors, alpha, relation, lag
    count and supplied mass metric. Missing anchors fail explicitly; rank
    degeneracy returns guard_sigma_available=False.
    """
    labels = _integers(labels, "labels")
    coefficients = np.asarray(coefficients_mass, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    count, rank = labels.shape
    if (coefficients.ndim != 2 or coefficients.shape[0] != 1+2*count or
            coefficients.shape[1] < 1 or not np.isfinite(coefficients).all() or
            alpha.shape != (rank,) or not np.isfinite(alpha).all() or
            not isinstance(lag_count, (int, np.integer)) or lag_count < 1 or
            not np.isfinite(rcond) or not 0 < rcond < 1):
        raise ValueError("finite matching coefficients/rotation, positive lag count and 0<rcond<1 required")
    matching = quotient_witnesses(labels, acquired_labels, phase_products, relation)
    relation_array = matching["relation"]
    residue = (float(abs(np.angle(np.exp(1j*np.dot(relation_array[0], alpha)))))
               if len(relation_array) else 0.)
    if residue > ROTATION_RESIDUE_TOLERANCE:
        raise ValueError("compact-group rotation must be snapped before conditioning")
    frequency = np.einsum("ij,j->i", labels, alpha, optimize=False)
    angles = frequency[:, None]*np.arange(lag_count)[None, :]
    cosines, sines = np.cos(angles), np.sin(angles)
    a, b = coefficients[1:1+count], coefficients[1+count:]
    first = a[:, :, None]*cosines[:, None, :]+b[:, :, None]*sines[:, None, :]
    second = -a[:, :, None]*sines[:, None, :]+b[:, :, None]*cosines[:, None, :]
    full = np.concatenate((first, second), axis=0).reshape(
        2*count, coefficients.shape[1]*lag_count).T/np.sqrt(lag_count)
    if not np.isfinite(full).all():
        raise ValueError("nonfinite numerical lag bank")
    orders = matching["orders"]
    active_columns = [index for index, order in enumerate(orders) if order != 1]
    active_columns += [count+index for index, order in enumerate(orders) if order not in (1, 2)]
    indices = matching["indices"]
    witness_columns = list(indices)+[count+index for index in indices if orders[index] != 2]
    nuisance_columns = [index for index in active_columns if index not in witness_columns]
    witness = full[:, witness_columns]
    nuisance = full[:, nuisance_columns]
    ambient = full.shape[0]
    if nuisance.shape[1]:
        # Amplitude does not change an exact nuisance span. Normalization
        # prevents a weak nonzero profile from disappearing merely because
        # a different nuisance column has large amplitude. Scale before the
        # norm to avoid underflow for very weak, but nonzero, columns.
        scales = np.max(abs(nuisance), axis=0)
        nonzero = scales > 0
        scaled = nuisance[:, nonzero]/scales[nonzero][None, :]
        unit_norms = np.sqrt(np.einsum("ij,ij->j", scaled, scaled, optimize=False))
        nuisance_norms = np.zeros(nuisance.shape[1])
        nuisance_norms[nonzero] = scales[nonzero]*unit_norms
        normalized = scaled/unit_norms[None, :]
        if normalized.shape[1]:
            nuisance_u, nuisance_s, _ = svd(normalized, full_matrices=False, lapack_driver="gesdd")
            nuisance_threshold = rcond*float(nuisance_s[0])
            keep = nuisance_s > nuisance_threshold
            nuisance_basis = nuisance_u[:, keep]
        else:
            nuisance_s = np.empty(0);nuisance_threshold = 0.;keep = np.zeros(0, dtype=bool)
            nuisance_basis = np.empty((ambient, 0))
        residual = witness-np.einsum("ij,jk->ik", nuisance_basis,
            np.einsum("ji,jk->ik", nuisance_basis, witness, optimize=False), optimize=False)
        nuisance_rank = int(keep.sum())
    else:
        nuisance_s = np.empty(0);nuisance_threshold = 0.;nuisance_rank = 0
        nuisance_norms = np.empty(0);nonzero = np.zeros(0, dtype=bool)
        residual = witness.copy()
    if witness.shape[1]:
        q_all, projected_s, _ = svd(residual, full_matrices=False, lapack_driver="gesdd")
        witness_s = svd(witness, compute_uv=False, lapack_driver="gesdd")
        threshold = rcond*max(float(projected_s[0]), float(witness_s[0]))
        projected_rank = int(np.sum(projected_s > threshold))
        enough_rows = ambient >= witness.shape[1]
        enough_complement = ambient-nuisance_rank >= witness.shape[1]
        full_rank = bool(enough_rows and enough_complement and projected_rank == witness.shape[1])
        q = q_all[:, :projected_rank]
        raw_minimum = float(projected_s[-1])
        sigma = raw_minimum if enough_rows else 0.
        status = ("resolved_numerical_learned_template_conditioning" if full_rank else
                  "unresolved_insufficient_witness_rows" if not enough_rows else
                  "unresolved_projected_witness_rank")
        left_norm = float(np.linalg.norm(matching["B_eff"], ord=2))
        constant = float(np.pi*left_norm/(2*sigma)) if full_rank else None
    else:
        q = np.empty((ambient, 0));projected_s = np.empty(0);raw_minimum = None
        threshold = 0.;projected_rank = 0;enough_rows = True;enough_complement = True
        full_rank = True;sigma = None;left_norm = 0.;constant = 0.
        status = "resolved_trivial_presented_group"
    leakage = np.einsum("ji,jk->ik", q, nuisance, optimize=False)
    leakage_max = float(np.max(abs(leakage))) if leakage.size else 0.
    leakage_norm = float(np.linalg.norm(leakage, ord=2)) if leakage.size else 0.
    orthogonality = (float(np.max(abs(np.einsum("ji,jk->ik", q, q, optimize=False)-np.eye(q.shape[1]))))
                     if q.shape[1] else 0.)
    information = {
        "conditioning_status": status, "guard_sigma_available": bool(full_rank and witness.shape[1]),
        "sigma0_numerical": sigma, "raw_thin_svd_minimum": raw_minimum,
        "witness_full_column_rank_at_declared_tolerance": full_rank,
        "effective_witness_labels": matching["effective_labels"].tolist(),
        "effective_integer_left_inverse": matching["B_eff"].tolist(),
        "integer_relation_products_C": matching["C"].tolist(),
        "acquired_to_quotient_T": matching["T"].tolist(),
        "acquired_relation_offsets_D": matching["D"].tolist(),
        "original_acquired_labels": np.asarray(acquired_labels).tolist(),
        "original_phase_products": np.asarray(phase_products).tolist(),
        "original_BA_equals_I_exact": True, "acquired_A_equals_TAeff_plus_DR_exact": True,
        "Beff_Aeff_plus_CR_equals_I_exact": True, "integer_left_inverse_exact": True,
        "relation": relation_array.tolist(), "relation_lattice_saturated": False,
        "supported_relation_rank": len(relation_array), "group_phase_origin": 0.,
        "rotation": alpha.tolist(), "rotation_relation_residue_rad": residue,
        "rotation_relation_numeric_tolerance": ROTATION_RESIDUE_TOLERANCE,
        "witness_dictionary_indices": indices.tolist(),
        "witness_character_orders": [orders[index] for index in indices],
        "infinite_character_order_encoded_as_null": True,
        "original_anchor_matches": matching["matches"],
        "constant_dictionary_indices": [index for index, order in enumerate(orders) if order == 1],
        "order_two_dictionary_indices": [index for index, order in enumerate(orders) if order == 2],
        "omitted_order_two_sine_indices": [count+index for index, order in enumerate(orders) if order == 2],
        "witness_real_column_indices": [int(x) for x in witness_columns],
        "nuisance_real_column_indices": [int(x) for x in nuisance_columns],
        "witness_real_columns": witness.shape[1], "nuisance_real_columns": nuisance.shape[1],
        "uniform_lag_count": int(lag_count), "mass_sensor_count": coefficients.shape[1],
        "witness_bank_ambient_rows": ambient, "ambient_rows_at_least_witness_columns": enough_rows,
        "nuisance_complement_dimension_at_least_witness_columns": enough_complement,
        "nuisance_numerical_rank": nuisance_rank, "nuisance_svd_rcond": float(rcond),
        "nuisance_columns_normalized_before_svd": True,
        "nuisance_column_norms_before_normalization": nuisance_norms.tolist(),
        "exactly_zero_numerical_nuisance_column_indices": [int(nuisance_columns[index])
            for index in np.flatnonzero(~nonzero)],
        "normalized_nuisance_singular_values": nuisance_s.tolist(),
        "discarded_positive_numerical_nuisance_singular_values": int(np.sum(
            (nuisance_s > 0)&(nuisance_s <= nuisance_threshold))),
        "nuisance_singular_threshold": nuisance_threshold,
        "projected_witness_numerical_rank": projected_rank,
        "projected_witness_singular_threshold": threshold,
        "projected_witness_singular_values": projected_s.tolist(),
        "projected_witness_maximum_singular": float(projected_s[0]) if len(projected_s) else None,
        "projected_witness_condition": (float(projected_s[0]/sigma) if sigma is not None and sigma > 0 else None),
        "integer_left_inverse_operator_norm": left_norm,
        "phase_bound_constant_rad_per_mass_window_norm": constant,
        "q_orthogonality_max_absolute_defect": orthogonality,
        "q_nuisance_annihilation_max_absolute_defect": leakage_max,
        "q_nuisance_annihilation_operator_norm": leakage_norm,
        "numerical_projection_residual_frobenius_norm": float(np.linalg.norm(residual)),
        "floating_point_bound_validated": False, "exact_group_identification_certified": False,
        "true_phase_rank_certified": False, "true_field_or_model_tail_bound_available": False,
        "metric_scope": "supplied mass coordinates; guard comparison requires the same fixed mass metric",
        "phase_metric": "Euclidean norm of coordinate-wise principal phase differences in the fixed ambient basis",
        "scope": "sufficient learned-template inverse bound on the presented zero-gauge group only"}
    return q, information
