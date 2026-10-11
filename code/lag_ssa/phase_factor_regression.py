"""Finite, data-selected Fourier regression on an inferred phase factor.

Frequencies and factor dimension must already have been learned from data.
The dictionary contains individual harmonics and bounded mixed characters,
with a sample-size budget.  Inner training holdout selects the dictionary and
ridge scale; no source law, physical rank, or true harmonic order is supplied.
This finite sieve is not an exact conditional expectation or tail certificate.
"""

from __future__ import annotations

from itertools import combinations, product

import numpy as np
from scipy.linalg import solve


HARMONIC_ORDERS = (1, 2, 4, 8, 16, 24)
MULTIPHASE_BUDGETS = ((1, 1), (2, 2), (4, 2), (4, 3), (8, 3), (16, 4), (24, 4))
EXPANDED_MULTIPHASE_BUDGETS = ((1, 1), (2, 2), (4, 4), (8, 6), (16, 8), (24, 12))
RIDGE_SCALES = (0., 1e-6, 1e-3)
MAX_MIXED_SEARCH_CHARACTERS = 4096


def _positive_compositions(total: int, size: int):
    if size == 1:
        yield (total,)
        return
    for first in range(1, total - size + 2):
        for rest in _positive_compositions(total - first, size - 1):
            yield (first,) + rest


def _gap(first: float, second: float) -> float:
    return float(abs(np.angle(np.exp(1j * (first - second)))))


def _dictionary(frequencies: np.ndarray, harmonic_order: int, mixed_order: int,
                n: int, max_lines: int, extra_labels=None) -> tuple[np.ndarray, list]:
    rank = len(frequencies)
    labels = set()
    for coordinate in range(rank):
        for harmonic in range(1, harmonic_order + 1):
            label = [0] * rank
            label[coordinate] = harmonic
            labels.add(tuple(label))
    search_count, search_exhausted = 0, False
    if rank > 1:
        # Enumerate sparse mixed characters by their l1 norm; never build an
        # exponential full hypercube in the unknown phase dimension.
        for total in range(2, mixed_order + 1):
            for size in range(2, min(rank, total) + 1):
                for active in combinations(range(rank), size):
                    for magnitudes in _positive_compositions(total, size):
                        for remaining_signs in product((-1, 1), repeat=size - 1):
                            if search_count >= MAX_MIXED_SEARCH_CHARACTERS:
                                search_exhausted = True
                                break
                            label = [0] * rank
                            for coordinate, magnitude, sign in zip(active, magnitudes, (1,) + remaining_signs):
                                label[coordinate] = sign * magnitude
                            labels.add(tuple(label))
                            search_count += 1
                        if search_exhausted:
                            break
                    if search_exhausted:
                        break
                if search_exhausted:
                    break
            if search_exhausted:
                break
    forced = set()
    if extra_labels is not None:
        for label in np.asarray(extra_labels, dtype=int).reshape(-1, rank):
            if np.any(label):
                if next(item for item in label if item) < 0:
                    label = -label
                forced.add(tuple(int(item) for item in label))
        labels.update(forced)
    ordered = sorted(labels, key=lambda row: (row not in forced, sum(abs(item) for item in row),
                                               max(abs(item) for item in row), row))
    retained, line_frequencies, omitted = [], [], []
    if search_exhausted:
        omitted.append({"reason": "mixed-character search budget exhausted",
                        "searched_characters": search_count,
                        "search_budget": MAX_MIXED_SEARCH_CHARACTERS})
    tolerance = .25 * 2 * np.pi / n
    for label in ordered:
        value = float(np.dot(label, frequencies))
        if min(_gap(value, 0.), _gap(value, np.pi)) <= tolerance:
            omitted.append({"label": list(label), "reason": "near constant or Nyquist"})
            continue
        if any(min(_gap(value, old), _gap(value, -old)) <= tolerance for old in line_frequencies):
            omitted.append({"label": list(label), "reason": "unresolved duplicate real line"})
            continue
        if len(retained) >= max_lines:
            omitted.append({"label": list(label), "reason": "sample-size dictionary cap"})
            continue
        retained.append(label)
        line_frequencies.append(value)
    return np.asarray(retained, dtype=int).reshape(-1, rank), omitted


def _design(times: np.ndarray, frequencies: np.ndarray, labels: np.ndarray) -> np.ndarray:
    lines = np.einsum("ij,j->i", labels, frequencies, optimize=False)
    angle = np.asarray(times)[:, None] * lines[None, :]
    return np.column_stack((np.ones(len(times)), np.cos(angle), np.sin(angle)))


def _ridge_coefficients(x: np.ndarray, y: np.ndarray, ridge_scale: float) -> np.ndarray:
    gram = np.einsum("ni,nj->ij", x, x, optimize=False)
    right = np.einsum("ni,nq->iq", x, y, optimize=False)
    penalty = np.eye(x.shape[1]) * ridge_scale * len(x)
    penalty[0, 0] = 0.
    return solve(gram + penalty, right, assume_a="pos", check_finite=True)


class FourierFactorModel:
    def __init__(self, frequencies, labels, coefficients, selected, selection_errors,
                 omitted, train_samples, sieve_mode="legacy"):
        self.frequencies = np.asarray(frequencies, dtype=float).copy()
        self.labels = np.asarray(labels, dtype=int).copy()
        self.coefficients = np.asarray(coefficients, dtype=float).copy()
        self.selected_k = None
        self.selected = dict(selected)
        self.selection_errors = selection_errors
        self.omitted = omitted
        self.train_samples = train_samples
        self.sieve_mode = sieve_mode

    def predict(self, times: np.ndarray, exclude_time_radius=None) -> np.ndarray:
        # Linear regression has no temporal nearest-neighbour lookup.
        design = _design(np.asarray(times, dtype=float), self.frequencies, self.labels)
        return np.einsum("nj,jq->qn", design, self.coefficients, optimize=False)

    def metadata(self) -> dict:
        return {"model": "finite regularized Fourier factor sieve",
                "frequencies": self.frequencies.tolist(), "k": None,
                "factor_dimension": len(self.frequencies),
                "selected_dictionary": self.selected,
                "character_labels": self.labels.tolist(),
                "coefficient_count_per_mass_coordinate": self.coefficients.shape[0],
                "dictionary_selection": "last28% inner training holdout",
                "train_selection_mse": self.selection_errors,
                "omitted_characters": self.omitted,
                "training_samples": self.train_samples,
                "uses_generator_truth": False,
                "general_mixed_characters_included": len(self.frequencies) > 1,
                "complete_harmonic_support_or_exact_conditional_expectation": False,
                "sieve_mode": self.sieve_mode,
                "implementation_dictionary_budget": "256 real lines maximum, inner-training samples/8 capped; 4096 mixed search characters maximum"
                    if self.sieve_mode == "expanded" else "128 real lines maximum, sample-size capped; 4096 mixed search characters maximum",
                "dictionary_budget_exhaustion_status": "unresolved_finite_sieve" if any(
                    item["reason"] in ("mixed-character search budget exhausted", "sample-size dictionary cap")
                    for item in self.omitted) else "finite_dictionary_no_tail_certificate",
                "training_prediction": "in-sample linear fit; no nearest-neighbour self reuse",
                "iid_residual_covariance_calibrated": False}


def fit_fourier_factor(times: np.ndarray, values: np.ndarray,
                       frequencies: np.ndarray, extra_labels=None,
                       sieve_mode: str = "legacy") -> FourierFactorModel:
    t = np.asarray(times, dtype=float)
    y = np.asarray(values, dtype=float)
    frequency = np.asarray(frequencies, dtype=float)
    if (y.ndim != 2 or y.shape[1] != len(t) or not np.isfinite(y).all()
            or not len(frequency) or not np.isfinite(frequency).all()):
        raise ValueError("finite field and a nonempty finite inferred phase factor required")
    split = max(100, int(.72 * len(t)))
    if split >= len(t) - 8:
        raise ValueError("not enough samples for inner training holdout")
    if sieve_mode not in ("legacy", "expanded"):
        raise ValueError("sieve_mode must be legacy or expanded")
    max_lines = min(256, split // 8) if sieve_mode == "expanded" else min(128, (split // 2 - 1) // 2)
    multiphase_budgets = EXPANDED_MULTIPHASE_BUDGETS if sieve_mode == "expanded" else MULTIPHASE_BUDGETS
    budgets = [(order, 0) for order in HARMONIC_ORDERS] if len(frequency) == 1 else multiphase_budgets
    candidates, audit, seen = [], [], set()
    for harmonic_order, mixed_order in budgets:
        labels, omitted = _dictionary(frequency, harmonic_order, mixed_order, len(t), max_lines, extra_labels)
        key = tuple(map(tuple, labels.tolist()))
        if key in seen or not len(labels):
            continue
        seen.add(key)
        design = _design(t, frequency, labels)
        for ridge_scale in RIDGE_SCALES:
            item = {"harmonic_order": harmonic_order, "mixed_l1_order": mixed_order,
                    "ridge_scale": ridge_scale, "line_count": len(labels),
                    "max_lines_sample_size_budget": max_lines,
                    "discovered_extra_character_labels": np.asarray(extra_labels, dtype=int).reshape(-1, len(frequency)).tolist()
                    if extra_labels is not None else []}
            try:
                coefficient = _ridge_coefficients(design[:split], y[:, :split].T, ridge_scale)
                prediction = np.einsum("nj,jq->qn", design[split:], coefficient, optimize=False)
                error = float(np.mean(np.sum((y[:, split:] - prediction) ** 2, axis=0)))
                if not np.isfinite(error):
                    raise FloatingPointError("nonfinite inner prediction error")
                item["inner_holdout_mass_mse"] = error
                candidates.append((error, design.shape[1], ridge_scale, labels, omitted, item))
            except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
                item["failure"] = f"{type(error).__name__}: {error}"
            audit.append(item)
    if not candidates:
        raise ValueError("no usable phase-factor dictionary")
    candidates.sort(key=lambda row: row[:3])
    _, _, ridge_scale, labels, omitted, selected = candidates[0]
    coefficients = _ridge_coefficients(_design(t, frequency, labels), y.T, ridge_scale)
    return FourierFactorModel(frequency, labels, coefficients, selected, audit, omitted, len(t), sieve_mode)


def refit_factor_dictionary(times: np.ndarray, values: np.ndarray,
                            frequencies: np.ndarray, labels: np.ndarray) -> FourierFactorModel:
    """Direct field baseline at exactly the discovered hierarchy's dictionary.

    Only the ridge coefficient is selected on an inner training suffix.
    This baseline receives learned frequencies and labels, never phase truth.
    """
    t, y = np.asarray(times), np.asarray(values)
    labels = np.asarray(labels, dtype=int)
    frequency = np.asarray(frequencies, dtype=float)
    design = _design(t, frequency, labels)
    split = max(100, int(.72 * len(t)))
    choices, audit = [], []
    for ridge in RIDGE_SCALES:
        try:
            coef = _ridge_coefficients(design[:split], y[:, :split].T, ridge)
            error = float(np.mean(np.sum((y[:, split:]-np.einsum("nj,jq->qn", design[split:], coef, optimize=False))**2, axis=0)))
            if np.isfinite(error):
                choices.append((error, ridge))
            audit.append({"ridge_scale": ridge, "inner_holdout_mass_mse": error})
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
            audit.append({"ridge_scale": ridge, "failure": str(error)})
    if not choices:
        raise ValueError("no usable direct fit at the inferred dictionary")
    _, selected_ridge = min(choices)
    coefficients = _ridge_coefficients(design, y.T, selected_ridge)
    return FourierFactorModel(frequency, labels, coefficients,
        {"ridge_scale": selected_ridge, "line_count": len(labels), "dictionary_fixed_to_learned_hierarchy": True},
        audit, [], len(t), "fixed_hierarchy_dictionary")


def phase_product(characters: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Products of measured unit-circle SSA readouts, including negative powers."""
    z = np.asarray(characters, dtype=complex)
    m = np.asarray(labels, dtype=int)
    if (z.ndim != 2 or m.ndim != 2 or m.shape[1] != z.shape[0]
            or not np.isfinite(z).all() or np.any(abs(z) < 1e-12)):
        raise ValueError("finite nonzero character rows and aligned integer labels required")
    return np.exp(1j * np.einsum("ij,jt->it", m, np.angle(z), optimize=False))


def observed_relation_check(train_characters: np.ndarray,
                            validation_characters: np.ndarray,
                            relations: np.ndarray,
                            coherence_threshold: float = .95) -> dict:
    """Check a constant phase product with its origin frozen on training.

    The threshold is a declared heuristic.  It is not a relation certificate
    and does not correct for shared residual-model or filter-estimation bias.
    Validation readouts must use training-frozen SSA filters.
    """
    train = np.asarray(train_characters, dtype=complex)
    validation = np.asarray(validation_characters, dtype=complex)
    labels = np.asarray(relations, dtype=int).reshape(-1, train.shape[0])
    train_products = phase_product(train, labels)
    validation_products = phase_product(validation, labels)
    rows = []
    for relation, tr, va in zip(labels, train_products, validation_products):
        tr_mean, va_mean = np.mean(tr), np.mean(va)
        origin = float(np.angle(tr_mean))
        aligned = va * np.exp(-1j * origin)
        train_coherence = float(abs(tr_mean))
        validation_coherence = float(abs(va_mean))
        frozen_origin_agreement = float(np.real(np.mean(aligned)))
        supported = bool(min(train_coherence, validation_coherence,
                             frozen_origin_agreement) >= coherence_threshold)
        rows.append({"relation": relation.tolist(), "train_constant_origin_rad": origin,
                     "train_coherence": train_coherence,
                     "validation_coherence": validation_coherence,
                     "validation_frozen_origin_agreement": frozen_origin_agreement,
                     "validation_frozen_origin_circular_mae_rad": float(np.mean(abs(np.angle(aligned)))),
                     "supported_by_observed_phase_products": supported})
    return {"relations": rows, "all_relations_supported": all(row["supported_by_observed_phase_products"] for row in rows),
            "relation_count": len(rows), "coherence_threshold": coherence_threshold,
            "train_columns": train.shape[1], "validation_columns": validation.shape[1],
            "origin_fitted_on_training_only": True, "validation_filters_frozen_on_training": True,
            "status": "heuristic_observed_phase_product_check_not_exact_relation_certificate",
            "exact_relation_certified": False}


def associate_observed_character_with_factor(character: np.ndarray,
                                             factor_frequencies: np.ndarray,
                                             extra_labels=None,
                                             factor_characters=None,
                                             validation_character=None,
                                             validation_factor_characters=None) -> dict:
    """Bounded association with measured factor products when supplied.

    Without factor readouts the legacy branch compares an SSA readout to
    fitted clock exponentials and is explicitly a clock diagnostic.
    """
    z = np.asarray(character, dtype=complex)
    frequencies = np.asarray(factor_frequencies, dtype=float)
    if factor_characters is not None:
        factor_characters = np.asarray(factor_characters, dtype=complex)
        count = min(len(z), factor_characters.shape[1])
        z, factor_characters = z[:count], factor_characters[:, :count]
        validation_character = np.asarray(validation_character, dtype=complex)
        validation_factor_characters = np.asarray(validation_factor_characters, dtype=complex)
        validation_count = min(len(validation_character), validation_factor_characters.shape[1])
        validation_character = validation_character[:validation_count]
        validation_factor_characters = validation_factor_characters[:, :validation_count]
    labels, omitted = _dictionary(frequencies, 24, 4, len(z), 128, extra_labels)
    times = np.arange(len(z))
    best = {"within_old_factor": False, "coherence": 0., "label": None}
    for label in labels:
        for sign in (1, -1):
            signed = sign * label
            value = float(np.dot(signed, frequencies))
            old = (np.exp(1j * value * times) if factor_characters is None else
                   phase_product(factor_characters, np.asarray([signed]))[0])
            coherence = float(abs(np.mean(z * np.conj(old))))
            if coherence > best["coherence"]:
                best.update(coherence=coherence, label=signed.tolist(),
                            frequency_rad_per_sample=float(np.angle(np.exp(1j * value))))
    if factor_characters is not None:
        if validation_character is None or validation_factor_characters is None:
            raise ValueError("observed association requires independent validation readouts")
        selected = np.asarray(best["label"], dtype=int)
        train_rows = np.vstack((z, np.asarray(factor_characters, dtype=complex)))
        validation_rows = np.vstack((validation_character, validation_factor_characters))
        check = observed_relation_check(train_rows, validation_rows,
                                         np.asarray([np.r_[1, -selected]]))
        best.update(within_old_factor=bool(check["all_relations_supported"]),
                    coherence_threshold=.95, observed_relation_check=check,
                    association_mode="observed_frozen_ssa_phase_products",
                    status="bounded observed phase-product association; heuristic, not certified")
    else:
        best.update(within_old_factor=bool(best["coherence"] >= .9), coherence_threshold=.9,
                    association_mode="legacy_fitted_clock_diagnostic",
                    status="SSA-to-fitted-clock association diagnostic; not observed phase-product verification")
    best.update(searched_real_characters=len(labels), omitted_characters=omitted)
    return best
