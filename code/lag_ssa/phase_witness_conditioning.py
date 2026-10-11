"""Numerical global inverse bound for a frozen finite-Fourier Hankel map.

Let rho_A(theta)=(cos(A theta),sin(A theta)). For integer B A=I, the
Euclidean wrapped torus distance satisfies
 d_T(theta,phi) <= (pi/2)||B||_2 ||rho_A(theta)-rho_A(phi)||_2.
If Q is orthonormal and annihilates all nuisance Fourier-Hankel columns,
and sigma=minsing(Q.T W)>0 for the witness cos/sin columns W, then
 d_T(theta,phi) <= K ||Q.T(H(theta)-H(phi))||, K=pi||B||/(2sigma).
Consequently any candidate has error at most K(r_Q+noise_Q+tail_Q), provided
the true observation admits that learned-map-plus-error representation.
There is no global optimizer assumption. This numerical script neither
validates floating point bounds nor certifies recovery of the true phase rank.
"""

from project_paths import RESULTS_DIR

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.linalg import svd

from hankel_torus_projection import HankelTorusProjector, ProjectionSettings


HERE = Path(__file__).resolve().parent
SOURCE = RESULTS_DIR / 'general_robustness_factor_cover.json'
PHASE_SOURCE = RESULTS_DIR / 'robust_strong_raw_hankel_audit.json'
OUTPUT = RESULTS_DIR / 'phase_witness_conditioning.json'
RCOND = 1e-10
GAUSSIAN_TAIL_PROBABILITY = .05


def canonical_witnesses(labels, acquired_labels, products):
    labels = np.asarray(labels, dtype=int)
    acquired = np.asarray(acquired_labels, dtype=int)
    products = np.asarray(products, dtype=int)
    if not np.array_equal(np.einsum("ij,jk->ik", products, acquired, optimize=False),
                          np.eye(acquired.shape[1], dtype=int)):
        raise ValueError("the acquired witness labels require an exact integer left inverse")
    indices, orientations = [], []
    for row in acquired:
        direct = np.flatnonzero(np.all(labels == row, axis=1))
        reverse = np.flatnonzero(np.all(labels == -row, axis=1))
        if len(direct)+len(reverse) != 1:
            raise ValueError("each acquired witness must occur uniquely up to sign in the fitted decoder dictionary")
        indices.append(int(direct[0] if len(direct) else reverse[0]))
        orientations.append(1 if len(direct) else -1)
    unique = list(dict.fromkeys(indices))
    relabeling = np.zeros((len(acquired), len(unique)), dtype=int)
    for row, (index, sign) in enumerate(zip(indices, orientations)):
        relabeling[row, unique.index(index)] = sign
    left_inverse = np.einsum("ij,jk->ik", products, relabeling, optimize=False)
    witnesses = labels[unique]
    if not np.array_equal(np.einsum("ij,jk->ik", left_inverse, witnesses, optimize=False),
                          np.eye(acquired.shape[1], dtype=int)):
        raise ArithmeticError("canonical witness reorientation lost primitivity")
    return np.asarray(unique, dtype=int), witnesses, left_inverse


def witness_conditioning(projector, acquired_labels, products):
    indices, witnesses, left_inverse = canonical_witnesses(projector.labels, acquired_labels, products)
    witness_columns = np.r_[indices, projector.count+indices]
    nuisance_columns = np.setdiff1d(np.arange(2*projector.count), witness_columns)
    full = projector.weighted_bank.T
    witness = full[:, witness_columns]
    nuisance = full[:, nuisance_columns]
    if nuisance.shape[1]:
        u, singular, _ = svd(nuisance, full_matrices=False, lapack_driver="gesdd")
        retained = singular > RCOND*singular[0]
        basis = u[:, retained]
        residual = witness-np.einsum("ij,jk->ik", basis,
            np.einsum("ji,jk->ik", basis, witness, optimize=False), optimize=False)
        nuisance_rank = int(retained.sum())
    else:
        singular = np.empty(0)
        residual = witness.copy()
        nuisance_rank = 0
    q, projected_singular, _ = svd(residual, full_matrices=False, lapack_driver="gesdd")
    sigma = float(projected_singular[-1])
    left_norm = float(np.linalg.norm(left_inverse, ord=2))
    numerical_full_rank = bool(sigma > RCOND*projected_singular[0])
    constant = float(np.pi*left_norm/(2*sigma)) if numerical_full_rank else None
    leakage = float(np.max(abs(np.einsum("ji,jk->ik", q, nuisance, optimize=False)))) if nuisance.shape[1] else 0.
    dynamic_rms = float(np.sqrt(.5*np.sum(projector.coefficients[1:]**2)))
    return q, {"effective_witness_labels": witnesses.tolist(),
        "effective_integer_left_inverse": left_inverse.tolist(),
        "integer_left_inverse_exact": True, "witness_dictionary_indices": indices.tolist(),
        "witness_real_columns": len(witness_columns), "nuisance_real_columns": len(nuisance_columns),
        "nuisance_numerical_rank": nuisance_rank, "nuisance_svd_rcond": RCOND,
        "sigma0_numerical": sigma, "projected_witness_maximum_singular": float(projected_singular[0]),
        "projected_witness_condition": float(projected_singular[0]/sigma) if sigma > 0 else None,
        "witness_full_column_rank_at_declared_tolerance": numerical_full_rank,
        "integer_left_inverse_operator_norm": left_norm,
        "phase_bound_constant_rad_per_mass_window_norm": constant,
        "learned_dynamic_field_haar_rms": dynamic_rms,
        "phase_bound_constant_per_relative_dynamic_error": None if constant is None else constant*dynamic_rms,
        "q_orthogonality_max_absolute_defect": float(np.max(abs(np.einsum("ji,jk->ik", q, q, optimize=False)-np.eye(q.shape[1])))),
        "q_nuisance_annihilation_max_absolute_defect": leakage,
        "floating_point_bound_validated": False,
        "scope": "global inverse conditioning of the learned finite-template torus only; true rank and true model tail not certified"}


def residual_and_noise_diagnostics(parameters, observed, phase, start, q, projector, noise_variance):
    count = phase.shape[1]
    mass = np.einsum("ij,jt->it", parameters["spatial_mass_sqrt"], observed[:, start:], optimize=False)
    windows = np.lib.stride_tricks.sliding_window_view(mass-projector.coefficients[0, :, None],
        projector.settings.window, axis=1)[:, :count, :]
    weighted = windows*np.sqrt(projector.weights)[None, None, :]
    raw_columns = weighted.transpose(0, 2, 1).reshape(-1, count)
    angles = np.einsum("ij,jt->it", projector.labels, phase, optimize=False)
    features = np.r_[np.cos(angles), np.sin(angles)]
    projected_raw = np.einsum("ji,jt->it", q, raw_columns, optimize=False)
    projected_model = np.einsum("ij,jt->it",
        np.einsum("ji,jk->ik", q, projector.weighted_bank.T, optimize=False), features, optimize=False)
    norms = np.linalg.norm(projected_raw-projected_model, axis=0)
    shaped = q.reshape(parameters["physical_mass"].shape[0], projector.settings.window, -1)
    covariance = noise_variance*np.einsum("qla,qr,rlb,l->ab", shaped, parameters["physical_mass"],
                                         shaped, projector.weights, optimize=False)
    trace = float(np.trace(covariance))
    largest = float(np.linalg.eigvalsh(covariance)[-1])
    gaussian_bound = np.sqrt(max(0., trace))+np.sqrt(2*max(0., largest)*np.log(1/GAUSSIAN_TAIL_PROBABILITY))
    return {"observed_projected_residual_median": float(np.median(norms)),
        "observed_projected_residual_maximum": float(np.max(norms)),
        "projected_noise_rms_under_training_iid_noise_assumption": float(np.sqrt(max(0., trace))),
        "projected_noise_norm_gaussian_upper_bound": float(gaussian_bound),
        "gaussian_per_window_tail_probability": GAUSSIAN_TAIL_PROBABILITY,
        "gaussian_assumption": "independent future Gaussian sensor noise with the training-estimated scalar variance; estimate uncertainty omitted",
        "uniform_over_windows_noise_guarantee": False,
        "true_model_tail_bound_available": False,
        "no_actual_phase_error_claim_without_tail_control": True}


def main():
    if OUTPUT.exists():
        raise FileExistsError("preserve the existing conditioning audit")
    initial = {name: hashlib.sha256((HERE/name).read_bytes()).hexdigest()
        for name in ("phase_witness_conditioning.py", "hankel_torus_projection.py")}
    source = json.loads(SOURCE.read_text())
    phase_source = json.loads(PHASE_SOURCE.read_text())
    phase_rows = {row["run"]: row for row in phase_source["rows"]}
    rows = []
    started = time.perf_counter()
    for entry, row in zip(source["manifest"], source["rows"]):
        if entry["seed"] != 202610081:
            continue
        parameter_path = Path(row["frozen_arrays"])
        if hashlib.sha256(parameter_path.read_bytes()).hexdigest() != row["frozen_arrays_sha256"]:
            raise ValueError("frozen fitted model changed")
        with np.load(parameter_path) as archive:
            parameters = {key: archive[key] for key in archive.files}
        projector = HankelTorusProjector(parameters["fit_labels"], parameters["fit_coefficients"],
            parameters["fit_frequencies"], ProjectionSettings())
        acquired_labels = row["inference"]["final_lattice"]["peak_labels"]
        q, information = witness_conditioning(projector, acquired_labels, parameters["phase_products"])
        phase_row = phase_rows[row["run"]]
        if hashlib.sha256(Path(phase_row["frozen_output"]).read_bytes()).hexdigest() != phase_row["frozen_output_sha256"]:
            raise ValueError("frozen phase estimate changed")
        with np.load(phase_row["frozen_output"]) as archive:
            phase = archive["hankel_phase"]
        if hashlib.sha256(Path(entry["input"]).read_bytes()).hexdigest() != entry["input_sha256"]:
            raise ValueError("public observations changed")
        with np.load(entry["input"]) as public:
            if set(public.files) != {"Y", "s", "t"}:
                raise ValueError("unexpected public input keys")
            observed = public["Y"]
        diagnostics = residual_and_noise_diagnostics(parameters, observed, phase, phase_row["time_start"],
            q, projector, row["noise_estimate_train_only"]["sigma_squared"])
        constant = information["phase_bound_constant_rad_per_mass_window_norm"]
        diagnostics["illustrative_median_residual_plus_gaussian_noise_phase_bound_ignoring_unknown_tail"] = (
            None if constant is None else constant*(diagnostics["observed_projected_residual_median"]+
                diagnostics["projected_noise_norm_gaussian_upper_bound"]))
        rows.append({"run": row["run"], "case": entry["case"], "learned_phase_dimension": projector.rank,
            "parameter_source": str(parameter_path), "parameter_source_sha256": row["frozen_arrays_sha256"],
            "phase_source": phase_row["frozen_output"], "public_input_sha256": entry["input_sha256"],
            "acquired_witness_labels": acquired_labels, "acquired_integer_left_inverse": parameters["phase_products"].tolist(),
            "conditioning": information, "observed_and_noise_diagnostics": diagnostics})
        print(json.dumps({"case": entry["case"], "sigma0": information["sigma0_numerical"],
            "Kbound": constant, "condition": information["projected_witness_condition"],
            "illustrative_bound_without_tail": diagnostics["illustrative_median_residual_plus_gaussian_noise_phase_bound_ignoring_unknown_tail"]}), flush=True)
    payload = {"status": "finished", "rows": rows, "source_fingerprints": initial,
        "source_fingerprints_unchanged": all(hashlib.sha256((HERE/name).read_bytes()).hexdigest() == value
            for name, value in initial.items()),
        "source_model_archive": str(SOURCE), "source_model_archive_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "source_phase_archive": str(PHASE_SOURCE), "projection_settings": asdict(ProjectionSettings()),
        "evaluation_truth_files_opened": False, "model_or_inference_parameters_changed": False,
        "proof_metric": "Euclidean norm of coordinate-wise principal phase differences on T^r",
        "bound": "d_T(theta,phi) <= pi||B||_2/(2sigma0) ||Q0^T(Htheta-Hphi)||; any candidate error <=K(projected residual+noise+model tail)",
        "numerical_estimate_not_validated_certificate": True,
        "uniform_true_model_tail_not_estimated": True, "true_rank_identification_not_certified": True,
        "novelty_claimed": False, "total_elapsed_seconds": time.perf_counter()-started}
    OUTPUT.write_text(json.dumps(payload, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
