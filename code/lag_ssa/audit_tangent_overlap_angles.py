"""Independent angle evaluation/decoder replay for frozen v1 and v2.

Does not refit angle extraction, frames, frequencies, or model dimension.
Uses independent cdist neighbours, explicit windows/folding and einsum.
Integer character matching is evaluation only and is not a winding proof.
"""

from project_paths import RESULTS_DIR

import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np
from scipy import linalg
from scipy.spatial.distance import cdist


HERE = Path(__file__).resolve().parent
OUT = RESULTS_DIR / 'tangent_overlap_pilot'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def windows(y, lag):
    return np.concatenate([y[j:len(y) - lag + 1 + j] for j in range(lag)], axis=1)


def fold(h, lag, channels):
    total = np.zeros((len(h) + lag - 1, channels))
    counts = np.zeros(len(total))
    for i, row in enumerate(h.reshape(len(h), lag, channels)):
        total[i:i + lag] += row
        counts[i:i + lag] += 1
    return total / counts[:, None]


def angular_difference(a, b):
    return float(np.max(abs(np.angle(np.exp(1j * (a - b))))))


def design(angles):
    r = angles.shape[1]
    modes = [m for m in itertools.product(range(-6, 7), repeat=r)
             if any(m) and next(a for a in m if a) > 0]
    argument = np.einsum("nr,mr->nm", angles, np.asarray(modes), optimize=False)
    return np.column_stack([np.ones(len(angles)), np.cos(argument), np.sin(argument)])


def train_match(angles, theta):
    modes = np.asarray([m for m in itertools.product(range(-4, 5), repeat=theta.shape[1]) if any(m)])
    true_chars = np.exp(1j * np.einsum("nr,mr->nm", theta, modes, optimize=False))
    corr = np.einsum("nr,nm->rm", np.exp(1j * angles), true_chars.conj(), optimize=False) / len(angles)
    best = abs(corr).argmax(axis=1)
    labels = modes[best]
    origins = np.angle(corr[np.arange(len(best)), best])
    target = np.einsum("nr,mr->nm", theta, labels, optimize=False) + origins
    error = np.angle(np.exp(1j * (angles - target)))
    return labels, origins, abs(corr[np.arange(len(best)), best]), float(np.mean(abs(error))), float(np.max(abs(error)))


def audit_case(name, version, protocol):
    suffix = "" if version == "v1" else "_v2"
    prefix = "angles_" if version == "v1" else "angles_v2_"
    method = HERE / ("tangent_overlap_angles.py" if version == "v1" else "tangent_overlap_angles_v2.py")
    with np.load(OUT / (prefix + name + ".npz")) as saved:
        learned = {k: saved[k] for k in saved.files}
    diagnostic = json.loads(str(learned["diagnostics_json"]))
    with np.load(OUT / f"{name}_inferred.npz") as saved:
        model = {k: saved[k] for k in saved.files}
    with np.load(OUT / "inputs" / f"{name}.npz") as saved:
        y = saved["Y"]
    with np.load(OUT / f"{name}{suffix}_phase_evaluated.npz") as saved:
        published_arrays = {k: saved[k] for k in saved.files}
    published = json.loads((OUT / f"{name}{suffix}_angles_evaluation.json").read_text())
    ntrain, lag = protocol["train_count"], protocol["lag"]
    channels = y.shape[1]
    characters = learned["characters"]
    angles = np.angle(characters)
    xf = windows((y[ntrain:] - model["mean"]) / model["scale"], lag) / np.sqrt(lag * channels)
    # Direct Euclidean calculation does not use the original NN estimator or BLAS.
    distance = cdist(xf, model["X"], metric="euclidean")
    index = np.argpartition(distance, 15, axis=1)[:, :16]
    chosen = np.take_along_axis(distance, index, axis=1)
    order = chosen.argsort(axis=1)
    chosen = np.take_along_axis(chosen, order, axis=1)
    index = np.take_along_axis(index, order, axis=1)
    width = np.maximum(chosen[:, -1], 1e-9)
    weights = np.exp(-(chosen / width[:, None])**2)
    weights /= weights.sum(axis=1, keepdims=True)
    value = np.einsum("nk,nkr->nr", weights, characters[index], optimize=False)
    future, radii = np.angle(value), abs(value)

    xdesign, fdesign = design(angles), design(future)
    xtrain = windows(y[:ntrain], lag)
    gram = np.einsum("ni,nj->ij", xdesign, xdesign, optimize=False)
    rhs = np.einsum("ni,np->ip", xdesign, xtrain, optimize=False)
    gram += 1e-5 * len(xdesign) * np.eye(len(gram))
    coefficient = linalg.solve(gram, rhs, assume_a="pos")
    prediction_windows = np.einsum("ni,ip->np", fdesign, coefficient, optimize=False)
    prediction = fold(prediction_windows, lag, channels)
    singular = linalg.svdvals(xdesign)

    # Open truth only after all numerical reconstruction above has finished.
    with np.load(OUT / "evaluation" / f"{name}.npz") as saved:
        theta, clean = saved["theta"], saved["clean"]
    labels, origins, coherence, train_mae, train_max = train_match(angles, theta[:len(angles)])
    target = np.einsum("nr,mr->nm", theta[ntrain:ntrain + len(future)], labels, optimize=False) + origins
    error = np.angle(np.exp(1j * (future - target)))
    phase_mae = float(np.mean(abs(error)))
    phase_max = float(np.max(abs(error)))
    nrmse = float(np.linalg.norm(prediction - clean[ntrain:]) / np.linalg.norm(clean[ntrain:]))
    determinant = int(round(np.linalg.det(labels))) if labels.shape[0] == labels.shape[1] else None
    selected = diagnostic["selected_candidate_indices"]
    reports = {row["index"]: row for row in diagnostic["joint_character_candidates"]}
    selection_rule = diagnostic["angle_basis_rule"]
    frame_index = int(learned["frame_index"])
    char_error = float(np.max(abs(abs(characters) - 1)))
    prediction_error = float(np.linalg.norm(prediction - published_arrays["prediction"]) / max(np.linalg.norm(published_arrays["prediction"]), 1e-30))
    metric_error = max(abs(phase_mae - published["future_phase_mae_rad"]),
                       abs(phase_max - published["future_phase_max_abs_rad"]),
                       abs(nrmse - published["future_clean_nrmse"]),
                       abs(train_mae - published["train_phase_evaluation"]["phase_mae_rad"]),
                       abs(train_max - published["train_phase_evaluation"]["phase_max_abs_rad"]))
    checks = {"method_source_hash_matches_archive": str(learned["source_sha256"]) == sha(method),
              "inferred_input_hash_matches_archive": str(learned["input_sha256"]) == sha(OUT / f"{name}_inferred.npz"),
              "fixed_last_frame_used": frame_index in (-1, len(model["fields"]) - 1),
              "dimension_from_inferred_frame": characters.shape[1] == model["fields"][frame_index].shape[-1],
              "characters_finite_unit_modulus": np.isfinite(characters).all() and char_error < 1e-12,
              "selected_candidates_accepted": all(reports[i]["accepted"] for i in selected),
              "train_angles_saved_correctly": angular_difference(angles, published_arrays["train_angles"]) < 1e-10,
              "future_angles_replayed": angular_difference(future, published_arrays["future_angles"]) < 1e-8,
              "decoder_reconstruction_replayed": prediction_error < 1e-8,
              "evaluation_integer_labels_replayed": np.array_equal(labels, published["train_phase_evaluation"]["winding_candidate"]),
              "evaluation_metrics_replayed": metric_error < 1e-8,
              "reported_decoder_size_and_condition_replayed": xdesign.shape[1] == published["decoder_features"] and abs(singular[-1] / singular[0] - published["decoder_relative_min_singular"]) < 1e-12,
              "declared_nonforecast_scope": published["future_is_encoding_observed_readings_not_forecast"] is True,
              "declared_character_match_not_winding_proof": published["train_phase_evaluation"]["candidate_is_finite_character_match_not_global_certificate"] is True}
    return {"case": name, "version": version, "passed": all(checks.values()), "checks": checks,
            "dimension": characters.shape[1], "selection_rule": selection_rule,
            "selected_candidate_indices": selected,
            "selected_candidate_diagnostics": [reports[i] for i in selected],
            "finite_character_match_candidate": labels.tolist(),
            "absolute_candidate_determinant": None if determinant is None else abs(determinant),
            "unimodular_candidate_only": determinant is not None and abs(determinant) == 1,
            "global_winding_or_injectivity_certified": False,
            "train_character_coherence": coherence.tolist(),
            "future_phase_mae_rad": phase_mae, "future_phase_max_abs_rad": phase_max,
            "future_clean_nrmse": nrmse, "decoder_features": xdesign.shape[1],
            "decoder_ridge": 1e-5 * len(xdesign),
            "future_character_modulus_quantiles": np.quantile(radii, [0, .1, .5]).tolist(),
            "future_angle_replay_max_rad": angular_difference(future, published_arrays["future_angles"]),
            "prediction_relative_replay_error": prediction_error,
            "score_max_absolute_replay_error": metric_error}


def main():
    start = time.perf_counter()
    protocol = json.loads((OUT / "protocol.json").read_text())
    paths = [HERE / "tangent_overlap.py", HERE / "tangent_overlap_angles.py", HERE / "tangent_overlap_angles_v2.py", HERE / "evaluate_tangent_overlap_angles.py", OUT / "protocol.json"]
    for name in protocol["cases"]:
        paths += [OUT / "inputs" / f"{name}.npz", OUT / "evaluation" / f"{name}.npz", OUT / f"{name}_inferred.npz"]
        for version in ["v1", "v2"]:
            suffix = "" if version == "v1" else "_v2"
            prefix = "angles_" if version == "v1" else "angles_v2_"
            paths += [OUT / (prefix + name + ".npz"), OUT / f"{name}{suffix}_phase_evaluated.npz", OUT / f"{name}{suffix}_angles_evaluation.json"]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise RuntimeError("Run only after both versions have finished: " + ", ".join(missing))
    before = {str(path): sha(path) for path in paths}
    rows = [audit_case(name, version, protocol) for version in ["v1", "v2"] for name in protocol["cases"]]
    after = {str(path): sha(path) for path in paths}
    output = {"status": "passed" if before == after and all(row["passed"] for row in rows) else "failed",
              "frozen_sources_and_outputs_unchanged": before == after, "hashes": before,
              "cases": rows, "elapsed_seconds": time.perf_counter() - start,
              "scope": {"inferred_frames_and_angles_refitted": False,
                        "angle_frontend_eigensolver_numerically_replayed": False,
                        "decoder_and_future_interpolation_numerically_replayed": True,
                        "truth_used_to_fit_decoder_or_future_interpolation": False,
                        "integer_labels_and_phase_origins_fitted_on_train_for_evaluation_only": True,
                        "candidate_determinant_is_not_a_global_phase_certificate": True,
                        "future_windows_start_at_train_count_and_use_observed_future": True,
                        "all_v1_failures_retained": True}}
    (OUT / "angle_audit.json").write_text(json.dumps(output, indent=2,
        default=lambda value: value.item() if isinstance(value, np.generic) else value.tolist()))
    print(json.dumps({"status": output["status"], "unchanged": before == after,
                      "cases": [{"case": row["case"], "version": row["version"], "passed": row["passed"],
                                 "failed_checks": [k for k, v in row["checks"].items() if not v]} for row in rows],
                      "elapsed_seconds": output["elapsed_seconds"]}))


if __name__ == "__main__":
    main()
