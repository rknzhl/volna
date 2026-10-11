"""Independent arithmetic, template-bound and degeneracy audit of the helper.

No inference/field source is modified. Random phases test the actual Fourier
field at every lag, rather than reconstructing the helper's bank. This audit
does not assess a true field, recovery rank or exact sampled orbit closure.
"""

from project_paths import RESULTS_DIR

import hashlib
import json
from pathlib import Path

import numpy as np

from group_witness_conditioning import conditioning, character_order
from hankel_torus_projection import HankelTorusProjector, ProjectionSettings
from phase_witness_conditioning import witness_conditioning


HERE = Path(__file__).resolve().parent
HELPER = HERE/"group_witness_conditioning.py"
OUTPUT = RESULTS_DIR / 'group_witness_conditioning_audit.json'
FROZEN_HELPER_SHA256 = "dc6f9fbcbadb23bfd03a164808431d8a7adbafb180075e67a5491705ef5a46e3"


def fingerprint():
    return hashlib.sha256(HELPER.read_bytes()).hexdigest()


def field_window(labels, coefficients, alpha, phase, lag_count):
    """Independent direct F(theta+alpha*l) evaluation, sensor-major stacking."""
    phase = np.asarray(phase)
    count = len(labels)
    shifted = phase[None, :]+np.arange(lag_count)[:, None]*alpha[None, :]
    angles = np.einsum("jr,lr->lj", labels, shifted, optimize=False)
    design = np.column_stack((np.ones(lag_count), np.cos(angles), np.sin(angles)))
    values = np.einsum("lj,jq->ql", design, coefficients, optimize=False)
    return values.reshape(-1)/np.sqrt(lag_count)


def phases_on_group(rng, rank, relation, samples):
    points = rng.uniform(-np.pi, np.pi, size=(samples, rank))
    if relation is None:
        return points
    row = np.asarray(relation)
    if rank == 2 and np.array_equal(row, [50, 0]):
        points[:, 0] = 2*np.pi*rng.integers(0, 50, size=samples)/50
    elif rank == 2 and np.array_equal(row, [7, 7]):
        points[:, 1] = 2*np.pi*rng.integers(0, 7, size=samples)/7-points[:, 0]
    elif rank == 2 and np.array_equal(row, [1, 1]):
        points[:, 1] = -points[:, 0]
    elif rank == 2 and np.array_equal(row, [2, 0]):
        points[:, 0] = np.pi*rng.integers(0, 2, size=samples)
    else:
        raise ValueError("audit phase sampler supports these declared toy groups only")
    return points


def check_bound(name, labels, acquired, products, alpha, relation, rng):
    coefficients = rng.normal(size=(1+2*len(labels), 4))
    q, information = conditioning(labels, coefficients, alpha, acquired, products, relation)
    assert information["guard_sigma_available"]
    assert information["Beff_Aeff_plus_CR_equals_I_exact"]
    first = phases_on_group(rng, len(alpha), relation, 256)
    second = phases_on_group(rng, len(alpha), relation, 256)
    ratios = []
    maximum_violation = 0.
    for theta, phi in zip(first, second):
        difference = field_window(labels, coefficients, alpha, theta, 128)-field_window(
            labels, coefficients, alpha, phi, 128)
        projected = np.einsum("ji,j->i", q, difference, optimize=False)
        distance = float(np.linalg.norm(np.angle(np.exp(1j*(theta-phi)))))
        bound = information["phase_bound_constant_rad_per_mass_window_norm"]*float(np.linalg.norm(projected))
        maximum_violation = max(maximum_violation, distance-bound)
        if bound > 0:
            ratios.append(distance/bound)
    assert maximum_violation <= 1e-9
    record = {"name": name, "sigma0": information["sigma0_numerical"],
        "K_bound": information["phase_bound_constant_rad_per_mass_window_norm"],
        "witness_indices": information["witness_dictionary_indices"],
        "witness_orders": information["witness_character_orders"],
        "witness_real_columns": information["witness_real_columns"],
        "B_eff": information["effective_integer_left_inverse"],
        "C": information["integer_relation_products_C"],
        "T": information["acquired_to_quotient_T"], "D": information["acquired_relation_offsets_D"],
        "exact_identity_checks_passed": True, "random_phase_pairs": len(first),
        "maximum_distance_divided_by_bound": max(ratios),
        "maximum_positive_bound_violation": maximum_violation,
        "q_nuisance_leakage_max": information["q_nuisance_annihilation_max_absolute_defect"]}
    return record, coefficients, information


def expect_failure(function, text):
    try:
        function()
    except ValueError as error:
        assert text in str(error)
        return str(error)
    raise AssertionError("expected explicit failure: "+text)


def main():
    if OUTPUT.exists():
        raise FileExistsError("preserve the existing independent guard audit")
    assert fingerprint() == FROZEN_HELPER_SHA256
    rng = np.random.default_rng(202610094)
    identity = np.eye(2, dtype=int)
    base_labels = np.asarray([[1, 0], [0, 1], [1, 1], [2, 0]])
    records = []
    record, coefficients, information = check_bound("pure_torus_matches_old_helper", base_labels,
        identity, identity, np.asarray([np.sqrt(2)/11, np.sqrt(3)/7]), None, rng)
    old = HankelTorusProjector(base_labels, coefficients, np.asarray([np.sqrt(2)/11, np.sqrt(3)/7]), ProjectionSettings())
    _, old_info = witness_conditioning(old, identity, identity)
    sigma_difference = abs(information["sigma0_numerical"]-old_info["sigma0_numerical"])
    assert sigma_difference < 1e-11
    record["sigma_difference_from_old_pure_torus_helper"] = sigma_difference
    records.append(record)

    acquired50 = np.asarray([[1, 0], [0, 1], [51, 0], [50, 0]])
    products50 = np.asarray([[1, 0, 0, 0], [0, 1, 0, 0]])
    record, _, information = check_bound("Z50_times_circle_unsaturated_alias_and_constant",
        base_labels, acquired50, products50, np.asarray([2*np.pi*3/50, np.sqrt(2)/5]), [50, 0], rng)
    assert information["witness_character_orders"] == [50, None]
    assert information["acquired_relation_offsets_D"] == [[0], [0], [1], [1]]
    records.append(record)

    acquired7 = np.asarray([[1, 0], [0, 1], [1, 1], [8, 7], [7, 7]])
    products7 = np.asarray([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]])
    record, _, information = check_bound("mixed_Z7_times_circle_with_three_anchor_classes",
        base_labels, acquired7, products7, np.asarray([.27, 2*np.pi/7-.27]), [7, 7], rng)
    assert information["witness_character_orders"] == [None, None, 7]
    assert character_order([1, 1], [7, 7]) == 7
    records.append(record)

    record, _, information = check_bound("primitive_relation_signed_quotient_anchor",
        np.asarray([[0, 1], [0, 2]]), identity, identity, np.asarray([.27, -.27]), [1, 1], rng)
    assert information["acquired_to_quotient_T"] == [[-1], [1]]
    assert information["integer_relation_products_C"] == [[1], [0]]
    records.append(record)

    order2_labels = np.asarray([[1, 0], [0, 1], [1, 1], [0, 2]])
    record, _, information = check_bound("Z2_times_circle_cosine_witness_sine_removed",
        order2_labels, identity, identity, np.asarray([np.pi, np.sqrt(3)/7]), [2, 0], rng)
    assert information["witness_real_columns"] == 3
    assert information["witness_character_orders"] == [2, None]
    assert 0 in information["witness_real_column_indices"]
    assert len(order2_labels) not in information["witness_real_column_indices"]
    records.append(record)

    acquired_cover2 = np.asarray([[2, 0], [0, 3], [1, 1]])
    products_cover2 = np.asarray([[2, 1, -3], [1, 1, -2]])
    labels_cover2 = np.vstack((acquired_cover2, [[1, 0], [0, 1], [2, 1]]))
    record, _, _ = check_bound("r2_primitive_cover_three_original_anchors", labels_cover2,
        acquired_cover2, products_cover2, np.asarray([np.sqrt(2)/11, np.sqrt(3)/7]), None, rng)
    records.append(record)

    acquired_cover3 = np.asarray([[1, 1, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    products_cover3 = np.asarray([[0, -1, 0, 1], [0, 0, -1, 1], [-1, 0, 0, 1]])
    labels_cover3 = np.vstack((acquired_cover3, [[2, 0, 0], [0, 2, 0], [0, 0, 2]]))
    record, _, _ = check_bound("r3_primitive_cover_four_original_anchors", labels_cover3,
        acquired_cover3, products_cover3, np.asarray([np.sqrt(2)/11, np.sqrt(3)/7, np.sqrt(5)/5]), None, rng)
    records.append(record)

    failure = expect_failure(lambda: conditioning(np.asarray([[3, 0], [0, 1]]),
        np.ones((5, 4)), np.asarray([2*np.pi*3/50, .37]), identity, identity, [50, 0]),
        "missing or ambiguous acquired anchor")
    records.append({"name": "missing_measured_anchor_fails_even_if_other_characters_generate_G", "explicit_failure": failure})

    _, insufficient = conditioning(np.asarray([[1, 0], [0, 1]]), np.ones((5, 1)),
        np.asarray([.23, .37]), identity, identity, lag_count=1)
    assert not insufficient["guard_sigma_available"]
    assert not insufficient["witness_full_column_rank_at_declared_tolerance"]
    assert insufficient["sigma0_numerical"] == 0.
    records.append({"name": "thin_svd_rows_less_than_witness_columns_fails_closed",
        "status": insufficient["conditioning_status"], "thin_minimum": insufficient["raw_thin_svd_minimum"],
        "sigma0": insufficient["sigma0_numerical"]})

    weak_coefficients = np.zeros((7, 3))
    weak_coefficients[1, 0] = 1.;weak_coefficients[4, 1] = 1.
    weak_coefficients[2, 0] = 1e-14;weak_coefficients[5, 1] = 1e-14
    weak_coefficients[3, 2] = 1.
    _, weak = conditioning(np.asarray([[1], [2], [3]]), weak_coefficients, np.asarray([.23]),
        np.asarray([[1]]), np.asarray([[1]]), lag_count=1)
    assert not weak["guard_sigma_available"]
    assert weak["nuisance_numerical_rank"] == 3
    records.append({"name": "weak_nonzero_nuisance_columns_preserve_their_span",
        "status": weak["conditioning_status"], "nuisance_rank": weak["nuisance_numerical_rank"],
        "normalized_before_svd": weak["nuisance_columns_normalized_before_svd"],
        "input_nuisance_norms": weak["nuisance_column_norms_before_normalization"]})

    _, trivial = conditioning(np.empty((0, 1), dtype=int), np.ones((1, 2)), np.asarray([2*np.pi]),
        np.asarray([[1]]), np.asarray([[1]]), [1])
    assert trivial["conditioning_status"] == "resolved_trivial_presented_group"
    assert trivial["phase_bound_constant_rad_per_mass_window_norm"] == 0.
    assert not trivial["guard_sigma_available"]
    records.append({"name": "trivial_group_has_zero_phase_bound_and_no_sigma_guard",
        "status": trivial["conditioning_status"], "K_bound": 0.})

    unsnapped = expect_failure(lambda: conditioning(base_labels, np.ones((9, 4)),
        np.asarray([.1, .37]), identity, identity, [50, 0]), "must be snapped")
    records.append({"name": "unsnapped_rotation_rejected", "explicit_failure": unsnapped})
    assert fingerprint() == FROZEN_HELPER_SHA256
    payload = {"status": "passed", "checks": len(records), "rows": records,
        "helper_source": str(HELPER), "helper_sha256": fingerprint(),
        "helper_unchanged_from_declared_freeze": True, "old_sources_edited": False,
        "direct_field_random_pair_checks": 7*256, "floating_point_certificate_claimed": False,
        "true_field_phase_rank_or_rationality_claimed": False}
    OUTPUT.write_text(json.dumps(payload, indent=2, allow_nan=False)+"\n")
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
