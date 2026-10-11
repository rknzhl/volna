"""Bounded alternative-candidate search with transactional rejection rollback.

Immutable cover utilities and every SSA/DMD equation/gate are reused.
Candidates are precisely the original passed rank/mode proposals, ordered
by descending represented trajectory energy with stable original ties.
Rejecting one candidate restores both accepted inference state and all
four mutable global-decoder caches before trying the next proposal.
No new frequency, observer window, relation threshold or truth input.
"""

from __future__ import annotations

import copy
import hashlib
import pickle

import numpy as np
from scipy.linalg import lstsq
from scipy.sparse.linalg import svds
import cover_selection_ssa as _base

Settings = _base.Settings
SPATIAL_MODEL = _base.SPATIAL_MODEL
pi = _base.pi
symmetric_sqrt = _base.symmetric_sqrt
extract_ssa_character = _base.extract_ssa_character
apply_ssa_character = _base.apply_ssa_character
aligned_character_rows = _base.aligned_character_rows
noise_from_spatial_orthogonal_component = _base.noise_from_spatial_orthogonal_component
fit_hierarchical_model = _base.fit_hierarchical_model
_joint_lattice = _base._joint_lattice
_field_from_mass = _base._field_from_mass
_observed_acquired_characters = _base._observed_acquired_characters
_refine_relation_readouts = _base._refine_relation_readouts
compact_frontend = _base.compact_frontend
refine_candidate_frequency = _base.refine_candidate_frequency
family_membership = _base.family_membership
fixed_sinusoid_association = _base.fixed_sinusoid_association
predict_mass = _base.predict_mass
predict_field = _base.predict_field


class _CandidateRejected(Exception):
    pass


_FITTER_ATTRIBUTES = ("lattice_by_products", "selected_by_factor",
                      "retained_labels_by_factor", "previous_model")


def _fitter_checkpoint():
    owner = fit_hierarchical_model
    present = [hasattr(owner, name) for name in _FITTER_ATTRIBUTES]
    if any(present) and not all(present):
        raise RuntimeError("incomplete mutable global-fitter rollback interface")
    if not any(present):
        return None
    state = copy.deepcopy({name:getattr(owner, name) for name in _FITTER_ATTRIBUTES})
    return {"owner":owner, "state":state,
            "sha256":hashlib.sha256(pickle.dumps(state, protocol=4)).hexdigest()}


def _restore_fitter(checkpoint):
    if checkpoint is None:
        return {"mutable_global_fitter_present":False}
    if fit_hierarchical_model is not checkpoint["owner"]:
        raise RuntimeError("global fitter changed during candidate transaction")
    for name, value in checkpoint["state"].items():
        setattr(checkpoint["owner"], name, value)
    restored = {name:getattr(checkpoint["owner"], name) for name in _FITTER_ATTRIBUTES}
    after = hashlib.sha256(pickle.dumps(restored, protocol=4)).hexdigest()
    if after != checkpoint["sha256"]:
        raise RuntimeError("global decoder caches did not restore exactly")
    return {"mutable_global_fitter_present":True,
            "all_four_cache_attributes_restored":list(_FITTER_ATTRIBUTES),
            "checkpoint_sha256":checkpoint["sha256"], "restored_sha256":after,
            "exact_cache_snapshot_restored":True}


def extract_ssa_dynamic_alternatives(signals: np.ndarray, spatial_weights: np.ndarray,
                                  window: int, noise_sigma: float | None = None,
                                  ranks=(4, 8, 16),
                                  temporal_gate_mode: str = "legacy",
                                  noise_covariance: np.ndarray | None = None) -> dict:
    y = np.asarray(signals, dtype=float)
    weights = np.asarray(spatial_weights, dtype=float)
    if (y.ndim != 2 or y.shape[0] != len(weights) or y.shape[1] <= window+3
            or window < 4 or np.any(weights <= 0) or not np.isfinite(y).all()):
        raise ValueError("finite field, positive weights and valid lag window required")
    weights = weights/weights.sum()
    q, n = y.shape
    columns = n-window+1
    centered = y-y.mean(axis=1, keepdims=True)
    row_scale = np.sqrt(weights/window)
    hankel = np.vstack([row_scale[:, None]*centered[:, lag:lag+columns]
                       for lag in range(window)])
    max_rank = min(max(ranks), min(hankel.shape)-2)
    v0 = np.random.default_rng(471).standard_normal(min(hankel.shape))
    u, singular, _ = svds(hankel, k=max_rank+1, which="LM", tol=1e-8, v0=v0)
    order = np.argsort(singular)[::-1]
    u, singular = u[:, order], singular[order]
    raw_lag1 = float(np.sum(weights[:, None]*centered[:, 1:]*centered[:, :-1])
                     /max(np.sum(weights[:, None]*centered[:, :-1]**2), 1e-15))
    lag_covariance = float(np.sum(weights[:, None]*centered[:, 1:]*centered[:, :-1])/(n-1))
    temporal_reference = {"mode": temporal_gate_mode, "lag_covariance": lag_covariance}
    if temporal_gate_mode == "legacy":
        temporal_gate = abs(raw_lag1) >= .1
        temporal_reference["correlation_threshold"] = .1
    elif temporal_gate_mode == "iid_reference":
        covariance = np.asarray(noise_covariance, dtype=float)
        if covariance.shape != (q, q) or not np.isfinite(covariance).all():
            raise ValueError("iid reference requires a finite coordinate noise covariance")
        omega = np.sqrt(weights)[:, None]*covariance*np.sqrt(weights)[None, :]
        eigenvalues = np.linalg.eigvalsh((omega+omega.T)/2)
        if eigenvalues[0] < -1e-12:
            raise ValueError("noise covariance must be positive semidefinite")
        x = np.log(2/1e-4)
        threshold = (float(np.trace(omega))/n
                     +np.sqrt(2*float(np.sum(omega**2))*x/(n-1))
                     +2*float(eigenvalues[-1])*x/(n-1))
        temporal_gate = abs(lag_covariance) > threshold
        temporal_reference.update(threshold=threshold, reference_alpha=1e-4,
            status="iid Gaussian zero-signal reference; learned residual dependence not calibrated")
    else:
        raise ValueError("unknown temporal dependence gate")
    proposals = []
    for desired_rank in ranks:
        rank = min(desired_rank, max_rank)
        subspace = u[:, :rank]
        scores = np.einsum("ij,jt->it", subspace.T, hankel, optimize=False)
        previous, following = scores[:, :-1], scores[:, 1:]
        cross = np.einsum("it,jt->ij", following, previous, optimize=False)
        gram = np.einsum("it,jt->ij", previous, previous, optimize=False)
        shift = lstsq(gram.T, cross.T, lapack_driver="gelsy")[0].T
        values, vectors = np.linalg.eig(shift)
        if np.linalg.cond(vectors) > 1e8:
            continue
        readout = np.linalg.solve(vectors, subspace.T)
        coordinates = np.einsum("ij,jt->it", readout, hankel, optimize=False)
        positive = np.flatnonzero(np.imag(values) > 1e-5)
        for mode in positive:
            value = values[mode]
            coordinate = coordinates[mode]
            radii = abs(coordinate)
            if radii.mean() <= 1e-12:
                continue
            normalized = coordinate/np.maximum(radii, 1e-15)
            unwrapped = np.unwrap(np.angle(normalized))
            times = np.arange(columns, dtype=float)
            tc = times-times.mean()
            slope = float(np.dot(tc, unwrapped-unwrapped.mean())/np.dot(tc, tc))
            phase_residual = float(np.std(unwrapped-slope*times))
            coherence = float(abs(np.mean(normalized[1:]*np.conj(normalized[:-1]))))
            radius_cv = float(radii.std()/radii.mean())
            slope_gap = float(abs(np.angle(np.exp(1j*(slope-np.angle(value))))))
            energy = float(np.vdot(vectors[:, mode], vectors[:, mode]).real
                           *np.mean(abs(coordinate)**2))
            gates = {
                "unit_modulus": bool(abs(abs(value)-1) <= .05),
                "phase_coherent": bool(coherence >= .90),
                "radius_stable": bool(radius_cv <= .60),
                "linear_phase": bool(phase_residual <= .50),
                "slope_agreement": bool(slope_gap <= .01),
                "interior_positive_frequency": bool(2*np.pi/n < slope < np.pi-2*np.pi/n),
                "raw_temporal_dependence": bool(temporal_gate),
            }
            lag_filter = readout[mode].reshape(window, q)*row_scale[None, :]
            proposals.append({"rank": rank, "mode": int(mode),
                "frequency": slope, "dmd_frequency": float(np.angle(value)),
                "represented_energy": energy,
                "phase_linear_fit_rms": phase_residual,
                "phase_step_coherence": coherence,
                "coordinate_radius_cv": radius_cv,
                "eigenvalue": [float(value.real), float(value.imag)],
                "gates": gates, "passed": all(gates.values()),
                "complex_coordinate": coordinate, "normalized_character": normalized,
                "lag_filter": lag_filter})
    public = [{key: value for key, value in proposal.items()
               if key not in ("complex_coordinate", "normalized_character", "lag_filter")}
              for proposal in proposals]
    eligible = [proposal for proposal in proposals if proposal["passed"]]
    if not eligible:
        return {"status": "unresolved", "reason": "no dynamic mode passed heuristic gates",
                "dynamic_proposals": public, "raw_lag1_correlation": raw_lag1,
                "temporal_reference": temporal_reference,
                "window": window, "hankel_columns": columns}
    def render(selected):
        return {"status": "resolved", "reason": None,
            "candidate_frequency": selected["frequency"],
            "complex_coordinate": selected["complex_coordinate"],
            "lag_filter": selected["lag_filter"], "signal_mean": y.mean(axis=1),
            "normalized_character": selected["normalized_character"],
            "phase": np.angle(selected["normalized_character"]),
            "metrics": {"frequency_temporal_dmd": selected["dmd_frequency"],
                "frequency_phase_slope": selected["frequency"],
                "phase_linear_fit_rms": selected["phase_linear_fit_rms"],
                "phase_step_coherence": selected["phase_step_coherence"],
                "coordinate_radius_cv": selected["coordinate_radius_cv"],
                "represented_energy": selected["represented_energy"],
                "raw_lag1_correlation": raw_lag1,
                "temporal_reference": temporal_reference,
                "singular_values": singular.tolist(), "noise_sigma": noise_sigma},
            "gates": selected["gates"], "gate_status": "heuristic_uncalibrated",
            "dynamic_rank": selected["rank"], "alternative_dynamic_mode": selected["mode"],
            "dynamic_proposals": public,
            "window": window, "hankel_columns": columns}
    seen = set()
    alternatives = []
    for proposal in sorted(eligible, key=lambda item: -item["represented_energy"]):
        identity = (proposal["rank"], proposal["mode"])
        if identity in seen:
            continue
        seen.add(identity)
        alternatives.append(render(proposal))
    result = dict(alternatives[0])
    result["candidate_alternatives"] = alternatives
    return result


def character_frontend(residual, noise_sigma, settings, sampling_model=None):
    spatial = SPATIAL_MODEL if sampling_model is None else sampling_model
    weights = np.ones(len(residual))/len(residual)
    if settings.phase_metric_mode == "field":
        inference_values = residual
        whitener = None
        inference_noise_covariance = spatial.noise_covmass(noise_sigma)
    elif settings.phase_metric_mode == "noise":
        whitener = symmetric_sqrt(spatial.noise_covmass(1.), inverse=True)
        inference_values = np.einsum("ij,jt->it", whitener, residual, optimize=False)
        inference_noise_covariance = noise_sigma**2*np.eye(len(residual))
    else:
        raise ValueError("unknown phase metric mode")
    if settings.frontend_mode == "pair":
        fitted = extract_ssa_character(inference_values, weights, settings.lag_window,
                                       noise_sigma=noise_sigma)
    elif settings.frontend_mode == "dynamic":
        fitted = extract_ssa_dynamic_alternatives(inference_values, weights, settings.lag_window,
            noise_sigma=noise_sigma, ranks=settings.dynamic_ranks,
            temporal_gate_mode=settings.temporal_gate_mode,
            noise_covariance=inference_noise_covariance)
    else:
        raise ValueError("unknown SSA frontend mode")
    alternatives = fitted.pop("candidate_alternatives", None)
    payloads = alternatives if alternatives is not None else [fitted]
    for payload in payloads:
        if whitener is not None and "lag_filter" in payload:
            payload["lag_filter"] = np.einsum("li,ij->lj", payload["lag_filter"], whitener,
                                             optimize=False)
            payload["signal_mean"] = residual.mean(axis=1)
        payload["phase_metric_mode"] = settings.phase_metric_mode
        payload["filter_input_coordinates"] = "physical mass coordinates; whitening pulled back"
    result = dict(payloads[0])
    if alternatives is not None:
        result["candidate_alternatives"] = payloads
    return result


def fit_iterative_ssa(raw_train: np.ndarray, raw_validation: np.ndarray,
                      settings: Settings, sampling_model=None,
                      noise_estimate: dict | None = None) -> dict:
    """Fit without physical truth; return serializable audit and frozen models."""
    y_train = np.asarray(raw_train, dtype=float)
    y_validation = np.asarray(raw_validation, dtype=float)
    if settings.phase_relation_mode not in ("clock_diagnostic", "observed"):
        raise ValueError("phase_relation_mode must be clock_diagnostic or observed")
    if settings.phase_readout_refinement not in ("none", "hann_inner"):
        raise ValueError("phase_readout_refinement must be none or hann_inner")
    spatial = SPATIAL_MODEL if sampling_model is None else sampling_model
    if (y_train.shape != (len(spatial.positions), settings.n_train)
            or y_validation.shape != (len(spatial.positions), settings.n_validation)):
        raise ValueError("record sizes disagree with settings")
    time_train = np.arange(settings.n_train)
    time_validation = np.arange(settings.n_train,
                                settings.n_train+settings.n_validation)
    if noise_estimate is None:
        noise = noise_from_spatial_orthogonal_component(y_train, spatial)
    else:
        noise = dict(noise_estimate)
        sigma = float(noise.get("sigma", float("nan")))
        if not np.isfinite(sigma) or sigma <= 0:
            raise ValueError("injected noise_estimate requires a positive finite sigma")
        noise.setdefault("sigma_squared", sigma**2)
        noise.setdefault("source", "injected external sensor-noise estimate")
    mass_train = spatial.to_mass_coordinates(spatial.recover(y_train))
    mass_validation = spatial.to_mass_coordinates(spatial.recover(y_validation))
    acquired = np.empty(0)
    models = []
    factor_products = []
    factor_extra_labels = []
    frozen_frontends = []
    family_readouts = []
    stages = []
    model = fit_hierarchical_model(time_train, mass_train, factor_products,
                                  acquired, settings.k_candidates, settings.lag_window,
                                  settings.conditional_model_mode,
                                  fourier_sieve_mode=settings.fourier_sieve_mode)
    models.append(model)
    final_lattice = None
    final_status = "max_steps_reached"
    for step in range(settings.max_steps):
        residual_train = mass_train-model.predict_training(time_train)
        residual_validation = mass_validation-model.predict(time_validation)
        try:
            frontend = character_frontend(residual_train, noise["sigma"], settings, spatial)
        except (FloatingPointError, np.linalg.LinAlgError, ValueError,
                RuntimeError) as error:
            stages.append({"step": step+1, "status": "ssa_frontend_exception",
                           "exception": f"{type(error).__name__}: {error}"})
            final_status = "no_new_validated_character"
            break
        if "candidate_frequency" not in frontend:
            stage = {"step":step+1, "frontend":compact_frontend(frontend),
                     "status":"no_ssa_character_candidate", "supported_alternatives":0,
                     "old_model":model.metadata(),
                     "validation_residual_mass_mse_before":float(np.mean(np.sum(residual_validation**2, axis=0)))}
            stages.append(stage)
            final_status = "no_new_validated_character"
            break
        bank = frontend.pop("candidate_alternatives", [frontend])
        accepted_candidate = False
        stop_after_acceptance = False
        for alternative_index, original_frontend in enumerate(bank):
            checkpoint = _fitter_checkpoint()
            accepted_state = (acquired.copy(), list(models), list(factor_products),
                [list(map(list, rows)) for rows in factor_extra_labels], list(frozen_frontends),
                list(family_readouts), model, final_lattice, final_status)
            candidate_old_model = model
            pending_frontends = frozen_frontends
            frontend = dict(original_frontend)
            stage = {"step": step+1, "frontend": compact_frontend(frontend),
                     "old_model": model.metadata(),
                     "validation_residual_mass_mse_before": float(np.mean(np.sum(residual_validation**2, axis=0)))}
            stage.update(alternative_index=alternative_index, supported_alternatives=len(bank),
                alternative_order="descending represented_energy; stable original rank/mode ties",
                alternative_frontend_rank=frontend.get("dynamic_rank"),
                alternative_frontend_mode=frontend.get("alternative_dynamic_mode"),
                alternative_represented_energy=frontend.get("metrics", {}).get("represented_energy"),
                alternatives_available_before_attempt=[{
                    "frequency":candidate["candidate_frequency"],
                    "dynamic_rank":candidate.get("dynamic_rank"),
                    "dynamic_mode":candidate.get("alternative_dynamic_mode"),
                    "represented_energy":candidate.get("metrics", {}).get("represented_energy")}
                    for candidate in bank])
            try:
                proposed = refine_candidate_frequency(residual_train, frontend, settings.n_train)
                candidate = proposed["frequency"]
                stage["candidate"] = proposed
                membership = family_membership(candidate, acquired, settings.max_relation_order,
                                               2*pi/settings.n_train)
                candidate_validation_character = None
                old_train_characters = old_validation_characters = None
                if settings.phase_relation_mode == "observed":
                    candidate_validation_character = apply_ssa_character(
                        frontend, residual_validation)["normalized_character"]
                    if len(frozen_frontends):
                        old_train_characters, old_validation_characters = _observed_acquired_characters(
                            frozen_frontends, models, mass_validation, time_validation)
                if final_lattice and (settings.conditional_model_mode == "fourier_sieve"
                                      or settings.phase_relation_mode == "observed"):
                    from phase_factor_regression import associate_observed_character_with_factor, phase_product
                    association_arguments = {}
                    association_extra_labels = list(factor_extra_labels[-1])
                    if settings.phase_relation_mode == "observed":
                        factor_labels = np.asarray(final_lattice["products"], dtype=int)
                        if membership["within_old_module"]:
                            proposed_factor_label = np.einsum("i,ij->j",
                                np.asarray(membership["coefficients"], dtype=int),
                                np.asarray(final_lattice["peak_labels"], dtype=int), optimize=False)
                            association_extra_labels.append(proposed_factor_label.tolist())
                            stage["frequency_proposed_factor_label_for_observed_check"] = proposed_factor_label.tolist()
                        association_arguments = {
                            "factor_characters": phase_product(old_train_characters, factor_labels),
                            "validation_factor_characters": phase_product(old_validation_characters, factor_labels),
                            "validation_character": candidate_validation_character}
                    polynomial_association = associate_observed_character_with_factor(
                        frontend["normalized_character"], model.frequencies,
                        association_extra_labels, **association_arguments)
                    stage["old_factor_phase_polynomial_association"] = polynomial_association
                    if (settings.phase_readout_refinement == "hann_inner" and settings.phase_relation_mode == "observed"
                            and membership["within_old_module"] and not polynomial_association["within_old_factor"]):
                        relations = [np.r_[1, -np.asarray(membership["coefficients"], dtype=int)]]
                        relations.extend(np.r_[0, relation] for relation in final_lattice.get("relations", []))
                        refinement = _refine_relation_readouts(frontend, candidate, frozen_frontends, acquired,
                            mass_train, mass_validation, residual_train, residual_validation,
                            np.asarray(relations), settings)
                        stage["phase_readout_refinement_attempt"] = refinement["audit"]
                        if refinement["supported"]:
                            pending_frontends = refinement["old_frontends"]
                            frontend = refinement["candidate_frontend"]
                            candidate_validation_character = refinement["candidate_validation_character"]
                            old_train_characters = refinement["old_train_characters"]
                            old_validation_characters = refinement["old_validation_characters"]
                            relation_check = refinement["audit"]["validation_observed_relation_check"]
                            verified = relation_check["relations"][0]
                            factor_label = np.einsum("i,ij->j", np.asarray(membership["coefficients"], dtype=int),
                                np.asarray(final_lattice["peak_labels"], dtype=int), optimize=False)
                            stage["old_factor_association_before_refinement"] = polynomial_association
                            polynomial_association = {"within_old_factor": True,
                                "label": factor_label.tolist(), "coherence": verified["train_coherence"],
                                "observed_relation_check": relation_check, "coherence_threshold": .95,
                                "association_mode": "observed_frozen_refined_field_phase_products",
                                "status": "bounded frequency-proposed relation supported by refined observed phase products; heuristic"}
                            stage["old_factor_phase_polynomial_association"] = polynomial_association
                            stage["frontend"] = compact_frontend(frontend)
                    if polynomial_association["within_old_factor"]:
                        old_frequency_membership = membership
                        coefficient = np.asarray(polynomial_association["label"], dtype=int) @ np.asarray(final_lattice["products"], dtype=int)
                        membership = {"within_old_module": True,
                                      "coefficients": coefficient.tolist(),
                                      "source": polynomial_association["association_mode"],
                                      "phase_polynomial_coherence": polynomial_association["coherence"],
                                      "bounded_frequency_membership": old_frequency_membership,
                                      "exact_module_membership_certified": False}
                    elif settings.phase_relation_mode == "observed" and membership["within_old_module"]:
                        # Rejection of one approximate old-factor relation does not
                        # reject the field character. Joint fitting may identify a
                        # different short relation, including a finite-cover lift.
                        stage["rejected_old_frequency_relation"] = membership
                        stage["old_relation_rejection_reason"] = "bounded frequency relation lacks observed train/validation phase-product support"
                        stage["joint_lattice_search_after_old_relation_rejection"] = True
                        membership = {"within_old_module": False,
                                      "status": "old_membership_unresolved_after_rejected_relation",
                                      "rejected_bounded_frequency_relation": membership,
                                      "exact_module_membership_certified": False}
                stage["old_module_membership"] = membership
                if membership["within_old_module"]:
                    if settings.conditional_model_mode == "fourier_sieve" and final_lattice:
                        extra = np.asarray(membership["coefficients"], dtype=int) @ np.asarray(final_lattice["peak_labels"], dtype=int)
                        tentative_extra = [list(map(list, labels)) for labels in factor_extra_labels]
                        tentative_extra[-1] = tentative_extra[-1] + [extra.tolist()]
                        stage["family_decoder_enrichment_label"] = extra.tolist()
                        try:
                            enriched_model = fit_hierarchical_model(
                                time_train, mass_train, factor_products, model.model_peak_frequencies,
                                settings.k_candidates, settings.lag_window,
                                settings.conditional_model_mode, tentative_extra, settings.fourier_sieve_mode)
                            enriched_residual = mass_validation-enriched_model.predict(time_validation)
                            enriched_mse = float(np.mean(np.sum(enriched_residual**2, axis=0)))
                            stage["validation_residual_mass_mse_after"] = enriched_mse
                            stage["validation_mse_improvement"] = stage["validation_residual_mass_mse_before"]-enriched_mse
                            stage["validation_gain_positive"] = bool(stage["validation_mse_improvement"] > 0)
                            stage["tentative_clock_model_after"] = enriched_model.metadata()
                            if stage["validation_gain_positive"]:
                                model = enriched_model
                                models[-1] = enriched_model
                                factor_extra_labels = tentative_extra
                                frozen_frontends = pending_frontends
                                if "phase_readout_refinement_attempt" in stage:
                                    family_readouts.append({"frontend": frontend,
                                        "prefix_model": candidate_old_model, "factor_label": extra.copy(),
                                        "step": step+1})
                                stage["status"] = "accepted_family_character_decoder_enrichment"
                                stage["accepted_character_count_after_enrichment"] = len(frozen_frontends)
                                stage["clock_model_after"] = model.metadata()
                                stages.append(stage)
                                accepted_candidate = True
                                break
                            stage["status"] = "unresolved_family_tail_no_decoder_improvement"
                            stage["accepted_state_rolled_back"] = True
                        except (ValueError, np.linalg.LinAlgError, FloatingPointError, RuntimeError) as error:
                            stage["status"] = "unresolved_family_decoder_failure"
                            stage["conditional_model_failure"] = f"{type(error).__name__}: {error}"
                            stage["accepted_state_rolled_back"] = True
                        stages.append(stage)
                        final_status = "unresolved_family_tail"
                        raise _CandidateRejected()
                    if (settings.conditional_model_mode == "legacy_knn"
                            and model.selected_k != 16 and len(acquired)):
                        retry_model = fit_hierarchical_model(
                            time_train, mass_train, factor_products, model.model_peak_frequencies,
                            k_candidates=(16,), exclusion_radius=settings.lag_window)
                        retry_residual = mass_train-retry_model.predict_training(time_train)
                        retry_frontend = character_frontend(retry_residual, noise["sigma"], settings, spatial)
                        if "candidate_frequency" in retry_frontend:
                            retry_proposed = refine_candidate_frequency(retry_residual,
                                                                        retry_frontend,
                                                                        settings.n_train)
                            retry_membership = family_membership(
                                retry_proposed["frequency"], acquired,
                                settings.max_relation_order, 2*pi/settings.n_train)
                            stage["family_tail_retry"] = {
                                "model": retry_model.metadata(),
                                "frontend": compact_frontend(retry_frontend),
                                "candidate": retry_proposed,
                                "membership": retry_membership}
                            if not retry_membership["within_old_module"]:
                                candidate_old_model = retry_model
                                if not settings.enforce_validation_improvement:
                                    model = retry_model
                                    models[-1] = retry_model
                                frontend = retry_frontend
                                stage["initial_frontend_before_retry"] = stage["frontend"]
                                stage["frontend"] = compact_frontend(retry_frontend)
                                candidate = retry_proposed["frequency"]
                                stage["candidate"] = retry_proposed
                                stage["old_module_membership"] = retry_membership
                                residual_validation = mass_validation-candidate_old_model.predict(time_validation)
                                stage["old_model"] = candidate_old_model.metadata()
                                stage["validation_residual_mass_mse_before"] = float(
                                    np.mean(np.sum(residual_validation**2, axis=0)))
                        else:
                            stage["family_tail_retry"] = {"frontend": compact_frontend(retry_frontend),
                                                          "status": "no_candidate_on_retry"}
                    if stage["old_module_membership"]["within_old_module"]:
                        stage["status"] = "unresolved_family_tail"
                        stages.append(stage)
                        final_status = "unresolved_family_tail"
                        raise _CandidateRejected()
                raw_validation_prediction = _field_from_mass(candidate_old_model.predict(time_validation),
                                                             spatial.positions, spatial)
                association = fixed_sinusoid_association(
                    y_validation-raw_validation_prediction, time_validation,
                    candidate, noise["sigma"])
                stage["heldout_association"] = association
                if association["chi_square_reference_p"] >= settings.association_p_gate:
                    stage["status"] = "no_new_validated_character"
                    stages.append(stage)
                    final_status = "no_new_validated_character"
                    raise _CandidateRejected()
                tentative = np.r_[acquired, candidate]
                joint = _joint_lattice(y_train, tentative, settings, noise_sigma=noise["sigma"])
                stage["joint_frequency_fit"] = joint["joint_fit"]
                stage["lattice"] = joint["lattice"]
                if settings.phase_relation_mode == "observed" and joint["usable_lattice"]:
                    from phase_factor_regression import observed_relation_check
                    candidate_train = np.asarray(frontend["normalized_character"])
                    train_rows = aligned_character_rows(list(old_train_characters) + [candidate_train]
                                    if len(frozen_frontends) else [candidate_train])
                    validation_rows = aligned_character_rows(list(old_validation_characters) + [candidate_validation_character]
                                         if len(frozen_frontends) else [candidate_validation_character])
                    relation_check = observed_relation_check(
                        train_rows, validation_rows, np.asarray(joint["lattice"].get("relations", []), dtype=int))
                    stage["proposed_lattice_observed_phase_relation_check"] = relation_check
                    if not relation_check["all_relations_supported"] and settings.phase_readout_refinement == "hann_inner":
                        relations = np.asarray(joint["lattice"].get("relations", []), dtype=int)
                        candidate_first = relations[:, np.r_[len(frozen_frontends), np.arange(len(frozen_frontends))]]
                        refinement = _refine_relation_readouts(frontend, float(joint["fitted_frequencies"][-1]),
                            frozen_frontends, np.asarray(joint["fitted_frequencies"][:-1]), mass_train, mass_validation,
                            residual_train, residual_validation, candidate_first, settings,
                            candidate_source="raw_field")
                        stage["phase_readout_refinement_attempt"] = refinement["audit"]
                        if refinement["supported"]:
                            pending_frontends = refinement["old_frontends"]
                            frontend = refinement["candidate_frontend"]
                            candidate_validation_character = refinement["candidate_validation_character"]
                            old_train_characters = refinement["old_train_characters"]
                            old_validation_characters = refinement["old_validation_characters"]
                            relation_check = observed_relation_check(
                                aligned_character_rows(list(old_train_characters) + [frontend["normalized_character"]]),
                                aligned_character_rows(list(old_validation_characters) + [candidate_validation_character]), relations)
                            stage["proposed_lattice_observed_phase_relation_check"] = relation_check
                            stage["frontend"] = compact_frontend(frontend)
                    if not relation_check["all_relations_supported"]:
                        stage["status"] = "unresolved_near_relation"
                        stage["near_relation_reason"] = "proposed lattice relation lacks observed train/validation phase-product support"
                        stage["accepted_state_rolled_back"] = True
                        stages.append(stage)
                        final_status = "unresolved_near_relation"
                        raise _CandidateRejected()
                candidate_acquired = joint["fitted_frequencies"] if joint["usable_lattice"] else tentative
                if joint["usable_lattice"]:
                    candidate_lattice = joint["lattice"]
                    products_matrix = np.asarray(candidate_lattice["products"], dtype=int)
                    model_frequencies = np.angle(np.exp(1j*np.einsum(
                        "ij,j->i", np.asarray(candidate_lattice["peak_labels"], dtype=int),
                        np.asarray(candidate_lattice["basis"], dtype=float), optimize=False)))
                else:
                    candidate_lattice = None
                    products_matrix = np.eye(len(candidate_acquired), dtype=int)
                    model_frequencies = candidate_acquired
                tentative_factor_products = factor_products + [products_matrix]
                tentative_extra_labels = factor_extra_labels + [[]]
                try:
                    candidate_model = fit_hierarchical_model(
                        time_train, mass_train, tentative_factor_products,
                        model_frequencies, settings.k_candidates, settings.lag_window,
                        settings.conditional_model_mode, tentative_extra_labels, settings.fourier_sieve_mode)
                except (ValueError, np.linalg.LinAlgError, FloatingPointError, RuntimeError) as error:
                    stage["status"] = "rejected_conditional_model_failure"
                    stage["conditional_model_failure"] = f"{type(error).__name__}: {error}"
                    stage["accepted_state_rolled_back"] = True
                    stages.append(stage)
                    final_status = "unresolved_conditional_model_failure"
                    raise _CandidateRejected()
                new_validation_residual = mass_validation-candidate_model.predict(time_validation)
                new_mse = float(np.mean(np.sum(new_validation_residual**2, axis=0)))
                stage["validation_residual_mass_mse_after"] = new_mse
                stage["validation_mse_improvement"] = (
                    stage["validation_residual_mass_mse_before"]-new_mse)
                stage["validation_gain_positive"] = bool(stage["validation_mse_improvement"] > 0)
                stage["maximum_joint_constraint_frequency_shift"] = float(np.max(abs(
                    np.angle(np.exp(1j*(model_frequencies-candidate_acquired))))))
                stage["tentative_clock_model_after"] = candidate_model.metadata()
                if settings.enforce_validation_improvement:
                    rejection_reasons = []
                    if not stage["validation_gain_positive"]:
                        rejection_reasons.append("nonpositive_validation_gain")
                    if not joint["usable_lattice"]:
                        rejection_reasons.append("unsupported_frequency_relation")
                    if rejection_reasons:
                        stage["status"] = "rejected_candidate_" + rejection_reasons[0]
                        stage["candidate_rejection_reasons"] = rejection_reasons
                        stage["accepted_state_rolled_back"] = True
                        stage["accepted_character_count_after_rejection"] = len(frozen_frontends)
                        stages.append(stage)
                        final_status = "unresolved_rejected_candidate"
                        raise _CandidateRejected()
                # Commit the whole tentative state atomically after strict gates pass.
                acquired = candidate_acquired
                final_lattice = candidate_lattice
                factor_products = tentative_factor_products
                factor_extra_labels = tentative_extra_labels
                models[-1] = candidate_old_model
                model = candidate_model
                models.append(model)
                frozen_frontends = list(pending_frontends) + [frontend]
                stage["status"] = ("accepted_character_conditional_lattice"
                                   if joint["usable_lattice"] else
                                   "accepted_character_but_unsupported_frequency_relation")
                stage["clock_model_after"] = model.metadata()
                stages.append(stage)
                accepted_candidate = True
                if not joint["usable_lattice"]:
                    final_lattice = None
                    final_status = "unsupported_frequency_relation"
                    stop_after_acceptance = True
                break
            except _CandidateRejected:
                stage["candidate_transaction_rollback"] = _restore_fitter(checkpoint)
                (acquired, models, factor_products, factor_extra_labels, frozen_frontends,
                 family_readouts, model, final_lattice, final_status) = accepted_state
                stage["accepted_inference_state_restored_before_next_alternative"] = True
                stage["next_supported_alternative_will_be_tried"] = alternative_index+1 < len(bank)
                continue
        if not accepted_candidate:
            final_status = "unresolved_supported_alternatives_exhausted"
            break
        if stop_after_acceptance:
            break
    final_relation_check = None
    if settings.phase_relation_mode == "observed" and final_lattice and frozen_frontends:
        from phase_factor_regression import observed_relation_check
        final_train_characters, final_validation_characters = _observed_acquired_characters(
            frozen_frontends, models, mass_validation, time_validation)
        final_relation_check = observed_relation_check(final_train_characters, final_validation_characters,
            np.asarray(final_lattice.get("relations", []), dtype=int))
    audit = {
        "status": final_status,
        "candidate_alternative_search": "all passed original SSA/DMD rank/mode candidates; first unchanged valid acceptance",
        "candidate_rejection_stops_whole_search": False,
        "candidate_bank_recomputed_after_accepted_state_only": True,
        "alternative_frequency_or_truth_catalog_supplied": False,
        "global_fitter_rejected_state_cache_rollback": "all four mutable caches, exact snapshot verification",
        "cover_selection_variant": "reject an unsupported old relation, then check joint lattice; joint-cover Hann verification on raw field",
        "old_relation_rejection_is_candidate_rejection": False,
        "joint_cover_observer_candidate_source": "raw_field",
        "frontend_mode": settings.frontend_mode,
        "joint_frequency_mode": settings.joint_frequency_mode,
        "phase_metric_mode": settings.phase_metric_mode,
        "temporal_gate_mode": settings.temporal_gate_mode,
        "conditional_model_mode": settings.conditional_model_mode,
        "enforce_validation_improvement": settings.enforce_validation_improvement,
        "phase_relation_mode": settings.phase_relation_mode,
        "phase_relation_support": "observed training and independently frozen validation phase products"
            if settings.phase_relation_mode == "observed" else "legacy frequency/clock association diagnostics",
        "observed_relation_threshold_is_heuristic": settings.phase_relation_mode == "observed",
        "phase_readout_refinement": settings.phase_readout_refinement,
        "fourier_sieve_mode": settings.fourier_sieve_mode,
        "maximum_phase_readout_window": max((frontend["window"] for frontend in frozen_frontends), default=settings.lag_window),
        "phase_readout_sources": [frontend.get("readout_source", "residual") for frontend in frozen_frontends],
        "final_observed_relation_check_of_returned_filters": final_relation_check,
        "strict_mode_requires_usable_frequency_lattice": settings.enforce_validation_improvement,
        "noise_estimate": noise,
        "spatial_model_class": type(spatial).__name__,
        "spatial_model": spatial.metadata() if hasattr(spatial, "metadata") else
                         {"model": "known stable polynomial spatial V"},
        "acquired_frequencies": acquired.tolist(),
        "accepted_character_count": len(frozen_frontends),
        "r_candidate_conditional": final_lattice.get("rank") if final_lattice else None,
        "completeness_status": "unresolved_no_finite_sample_certificate",
        "search_termination_is_phase_completeness_proof": False,
        "final_lattice": final_lattice,
        "final_clock_model": models[-1].metadata(),
        "stages": stages,
        "assumptions": ["explicit spatial reconstruction and mass metric",
                        "iid raw sensor noise for sigma estimator or injected noise scale",
                        "finite model Fisher covariance omits Poisson-tail bias",
                        "frequency relation gap and complete visible generators not certified",
                        "heldout association chi-square reference is diagnostic, not calibrated for model bias"],
        "conditional_mean_structure": "nested-factor residual corrections; earlier factor dimensions retained",
        "training_residual_predictions_exclude_temporal_neighbours": settings.lag_window
            if settings.conditional_model_mode == "legacy_knn" else None,
    }
    return {"audit": audit, "clock_models": models,
            "frozen_frontends": frozen_frontends,
            "acquired_frequencies": acquired,
            "mass_train": mass_train, "mass_validation": mass_validation,
            "sampling_model": spatial, "family_readouts": family_readouts}
