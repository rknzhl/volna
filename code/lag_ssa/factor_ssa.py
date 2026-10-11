"""Research prototype: residual SSA with one global phase-factor field fit.

The existing discovery/readout orchestration is reused by a scoped fitter
injection; its source stays unchanged.  Each state fits the entire observed
field once on a union of selected characters, expressed in the current
observed lattice.  Earlier fitted residual corrections are never summed.
The fit uses the existing training-selected ridge rule: it is an orthogonal
least-squares projection only when the selected ridge is zero.
"""

from __future__ import annotations

from contextlib import contextmanager
import threading

import numpy as np

import iterative_ssa as discovery
from phase_factor_regression import fit_fourier_factor, refit_factor_dictionary


_FIT_LOCK = threading.RLock()


def _key(matrix):
    matrix = np.asarray(matrix, dtype=int)
    return matrix.shape, tuple(matrix.ravel())


class GlobalFactorModel:
    def __init__(self, fit, values, times, peak_frequencies, products, components,
                 extra_labels, sieve_mode, dictionary_labels=None):
        self.fit = fit
        self.mean = np.asarray(values).mean(axis=1)
        self.times = np.asarray(times)
        self.frequencies = fit.frequencies if fit is not None else np.empty(0)
        self.model_peak_frequencies = np.asarray(peak_frequencies).copy()
        self.phase_products = tuple(np.asarray(matrix, dtype=int).copy() for matrix in products)
        self.factor_extra_labels = extra_labels
        # Compatibility for dictionary auditors; these fits are not summed.
        self.components = (None,) + tuple(components)
        self.selected_k = None
        self.conditional_model_mode = "fourier_sieve"
        self.fourier_sieve_mode = sieve_mode
        self.dictionary_labels = dictionary_labels

    def predict(self, times, exclude_time_radius=None):
        if self.fit is None:
            return np.repeat(self.mean[:, None], len(times), axis=1)
        return self.fit.predict(times)

    def predict_training(self, times):
        return self.predict(times)

    def predict_on_phase(self, phase):
        """Evaluate the fitted field at measured phase coordinates, not clocks."""
        phase = np.asarray(phase, dtype=float)
        if phase.ndim != 2 or phase.shape[0] != len(self.frequencies):
            raise ValueError("one row per current factor phase is required")
        if self.fit is None:
            return np.repeat(self.mean[:, None], phase.shape[1], axis=1)
        angles = np.einsum("ij,jt->it", self.fit.labels, phase, optimize=False)
        design = np.column_stack((np.ones(phase.shape[1]), np.cos(angles.T), np.sin(angles.T)))
        return np.einsum("nj,jq->qn", design, self.fit.coefficients, optimize=False)

    def metadata(self):
        return {"structure": "one global field fit on the current phase-factor character dictionary",
                "conditional_model_mode": "fourier_sieve", "fourier_sieve_mode": self.fourier_sieve_mode,
                "frequencies": self.frequencies.tolist(), "k": None,
                "model_peak_frequencies": self.model_peak_frequencies.tolist(),
                "discovered_extra_labels_by_factor": self.factor_extra_labels,
                "factor_products_in_original_acquired_coordinates": [matrix.tolist() for matrix in self.phase_products],
                "dictionary_selection_history": [component.metadata() for component in self.components[1:]],
                "global_dictionary": None if self.fit is None else self.fit.metadata(),
                "earlier_residual_corrections_summed": False,
                "full_field_refit_at_every_tentative_state": True,
                "orthogonal_projection_only_when_selected_ridge_zero": True,
                "training_temporal_exclusion_radius": None}


class _GlobalFitter:
    def __init__(self):
        self.lattice_by_products = {}
        self.selected_by_factor = {}
        self.retained_labels_by_factor = {}
        self.previous_model = None

    def capture_joint(self, original, *args, **kwargs):
        result = original(*args, **kwargs)
        if result["usable_lattice"]:
            lattice = result["lattice"]
            self.lattice_by_products[_key(lattice["products"])] = np.asarray(lattice["peak_labels"], dtype=int)
        return result

    def __call__(self, times, values, products, acquired_frequencies,
                 k_candidates=(32, 64), exclusion_radius=128,
                 conditional_model_mode="fourier_sieve", factor_extra_labels=None,
                 fourier_sieve_mode="legacy"):
        if conditional_model_mode != "fourier_sieve":
            raise ValueError("global-factor prototype supports the Fourier sieve only")
        extras = factor_extra_labels or [[] for _ in products]
        peaks = np.asarray(acquired_frequencies, dtype=float)
        if not products:
            model = GlobalFactorModel(None, values, times, peaks, products, [], extras, fourier_sieve_mode)
            self.previous_model = model
            return model
        current_products = np.asarray(products[-1], dtype=int)
        labels_to_basis = self.lattice_by_products.get(_key(current_products))
        if labels_to_basis is None:
            if np.array_equal(current_products, np.eye(len(peaks), dtype=int)):
                labels_to_basis = np.eye(len(peaks), dtype=int)
            else:
                raise ValueError("current factor requires the already observation-supported integer lattice")
        if not np.array_equal(np.einsum("ij,jk->ik", current_products, labels_to_basis, optimize=False),
                              np.eye(len(current_products), dtype=int)):
            raise ValueError("captured lattice does not invert the current factor products")
        frequencies = np.angle(np.exp(1j*np.einsum("ij,j->i", current_products, peaks, optimize=False)))
        reconstructed = np.einsum("ij,j->i", labels_to_basis, frequencies, optimize=False)
        if np.max(abs(np.angle(np.exp(1j*(reconstructed-peaks))))) > 1e-9:
            raise ValueError("current fitted lines are inconsistent with the supported factor basis")
        components, union = [], set()
        for matrix, extra in zip(products, extras):
            matrix = np.asarray(matrix, dtype=int)
            count = matrix.shape[1]
            factor_frequencies = np.angle(np.exp(1j*np.einsum("ij,j->i", matrix, peaks[:count], optimize=False)))
            signature = (_key(matrix), tuple(sorted(tuple(row) for row in extra)))
            if signature not in self.selected_by_factor:
                selection_target = (values if self.previous_model is None else
                                    values-self.previous_model.predict(times))
                self.selected_by_factor[signature] = fit_fourier_factor(times, selection_target, factor_frequencies,
                    extra_labels=extra, sieve_mode=fourier_sieve_mode)
            selected = self.selected_by_factor[signature]
            retained = self.retained_labels_by_factor.setdefault(_key(matrix), set())
            retained.update(tuple(row) for row in selected.labels)
            retained_labels = np.asarray(sorted(retained), dtype=int)
            component = _DictionarySelection(selected, retained_labels)
            components.append(component)
            acquired_labels = np.einsum("ij,jk->ik", retained_labels, matrix, optimize=False)
            current_labels = np.einsum("ij,jk->ik", acquired_labels, labels_to_basis[:count], optimize=False)
            for row in current_labels:
                if np.any(row):
                    if next(value for value in row if value) < 0:
                        row = -row
                    union.add(tuple(int(value) for value in row))
        dictionary = np.asarray(sorted(union), dtype=int).reshape(-1, len(frequencies))
        fit = refit_factor_dictionary(times, values, frequencies, dictionary)
        model = GlobalFactorModel(fit, values, times, peaks, products, components, extras,
                                  fourier_sieve_mode, dictionary)
        self.previous_model = model
        return model


class _DictionarySelection:
    def __init__(self, selected, labels):
        self.selected = selected
        self.labels = labels

    def metadata(self):
        return {**self.selected.metadata(),
                "selection_target": "residual of the previous global full-field fit",
                "old_selected_labels_retained": True,
                "retained_label_count": len(self.labels),
                "retained_labels": self.labels.tolist(),
                "prediction_used_in_field_model": False}


@contextmanager
def _global_fitter_scope(discovery_module=None):
    # Discovery is synchronous; the lock serializes prototype calls sharing
    # this interpreter. Independent legacy fits should use another process.
    with _FIT_LOCK:
        module = discovery if discovery_module is None else discovery_module
        original_fit = module.fit_hierarchical_model
        original_joint = module._joint_lattice
        fitter = _GlobalFitter()
        module.fit_hierarchical_model = fitter
        module._joint_lattice = lambda *args, **kwargs: fitter.capture_joint(original_joint, *args, **kwargs)
        try:
            yield
        finally:
            module.fit_hierarchical_model = original_fit
            module._joint_lattice = original_joint


def fit_factor_ssa(raw_train, raw_validation, settings, sampling_model=None,
                   noise_estimate=None, discovery_module=None):
    if settings.conditional_model_mode != "fourier_sieve" or not settings.enforce_validation_improvement:
        raise ValueError("factor SSA requires a Fourier sieve and strict held-out field improvement")
    module = discovery if discovery_module is None else discovery_module
    with _global_fitter_scope(module):
        fitted = module.fit_iterative_ssa(raw_train, raw_validation, settings,
            sampling_model=sampling_model, noise_estimate=noise_estimate)
    fitted["audit"].update(method="global_factor_residual_ssa_prototype",
        field_fit_structure="one global refit on the selected character union at every accepted/enriched state",
        discovery_module=module.__name__,
        earlier_residual_corrections_summed=False,
        discovery_and_observed_phase_gates_reused_without_changes=True)
    fitted["phase_factor_models"] = fitted["clock_models"]
    return fitted
