"""Independent replay of the frozen tangent-overlap pilot; no model fits.

Uses explicit windows and einsum contractions instead of the model's matrix
multiplications. Truth is used only for the already published diagnostics.
"""

from project_paths import RESULTS_DIR

from pathlib import Path
import ast
import hashlib
import json
import time

import numpy as np
from scipy import linalg


HERE = Path(__file__).resolve().parent
OUT = RESULTS_DIR / 'tangent_overlap_pilot'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def explicit_windows(y, lag):
    return np.concatenate([y[j:len(y) - lag + 1 + j] for j in range(lag)], axis=1)


def explicit_fold(h, channels, lag):
    h = h.reshape(len(h), lag, channels)
    result = np.zeros((len(h) + lag - 1, channels))
    count = np.zeros(len(result))
    for row in range(len(h)):
        result[row:row + lag] += h[row]
        count[row:row + lag] += 1
    return result / count[:, None]


def rel(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def field_value(theta, s, name):
    """Evaluator-only independent expression for the analytic benchmark."""
    a = .8 + .3 * s[:, 0]
    b = .7 + .3 * s[:, 1]
    c = .7 + .2 * np.sin(s[:, 0] + s[:, 1])
    beta = 2 * np.pi * (1.1 * s[:, 0] - .4 * s[:, 1])
    gamma = 2 * np.pi * (.3 * s[:, 0] + .9 * s[:, 1])
    mixed = 2 * np.pi * (.7 * s[:, 0] + .4 * s[:, 1])

    def p(v, rho):
        return 2 * rho * (np.cos(v) - rho) / (1 - 2 * rho * np.cos(v) + rho**2)

    if theta.shape[1] == 1:
        return a * p(theta[:, 0, None] + beta, .5) + .3 * b * p(2 * theta[:, 0, None] + gamma, .3)
    k1, k2 = (3, 2) if "cover" in name else (1, 1)
    value = a * p(k1 * theta[:, 0, None] + beta, .5) + .7 * b * p(k2 * theta[:, 1, None] + gamma, .5)
    if "cover" in name:
        return value + .15 * c * np.cos(theta[:, 0, None] + theta[:, 1, None] + mixed)
    return value + .2 * c * p(theta[:, 0, None] - theta[:, 1, None] + mixed, .3)


def baseline_replay(ytrain, yfuture, clean, lag, ranks):
    train = explicit_windows(ytrain, lag)
    center = np.mean(train, axis=0)
    centered = train - center
    covariance = np.einsum("ni,nj->ij", centered, centered, optimize=False)
    _, basis = linalg.eigh(covariance, subset_by_index=(len(covariance) - max(ranks), len(covariance) - 1))
    basis = basis[:, ::-1]
    future = explicit_windows(yfuture, lag)
    results = []
    for rank in ranks:
        score = np.einsum("np,pa->na", future - center, basis[:, :rank], optimize=False)
        hhat = np.einsum("na,pa->np", score, basis[:, :rank], optimize=False) + center
        results.append(rel(explicit_fold(hhat, ytrain.shape[1], lag), clean))
    return results


def section_replay(q, maps, neighbors, field, lam, eigenvalues):
    n, _, r = q.shape
    coeff = np.einsum("npa,npb->nab", q, field, optimize=False)
    reconstructed = np.einsum("npa,nab->npb", q, coeff, optimize=False)
    defect = coeff[1:] - np.einsum("nab,nbc->nac", maps, coeff[:-1], optimize=False)
    transport_energy = float(np.sum(defect**2) / (n - 1))
    temporal_gradient = np.zeros_like(coeff)
    temporal_gradient[:-1] -= np.einsum("nba,nbc->nac", maps, defect, optimize=False)
    temporal_gradient[1:] += defect
    temporal_gradient *= n / (n - 1)

    pairs = np.sort(np.column_stack((np.repeat(np.arange(n), neighbors.shape[1]), neighbors.ravel())), axis=1)
    pairs = np.unique(pairs[pairs[:, 0] != pairs[:, 1]], axis=0)
    edge_energy = 0.0
    rough_gradient = np.zeros_like(coeff)
    for start in range(0, len(pairs), 512):
        edge = pairs[start:start + 512]
        i, j = edge[:, 0], edge[:, 1]
        diff = field[i] - field[j]
        edge_energy += float(np.sum(diff**2))
        gi = np.einsum("epa,epb->eab", q[i], diff, optimize=False)
        gj = np.einsum("epa,epb->eab", q[j], diff, optimize=False)
        np.add.at(rough_gradient, i, gi)
        np.add.at(rough_gradient, j, -gj)
    rough_gradient *= n / len(pairs)
    rough_gradient += coeff
    roughness = edge_energy / len(pairs) + float(np.sum(coeff**2) / n)
    action = temporal_gradient + lam * rough_gradient
    expected = coeff * eigenvalues[:r][None, None, :]
    return {"tangency_relative_error": rel(reconstructed, field),
            "transport_energy": transport_energy, "roughness": roughness,
            "eigen_equation_relative_residual": rel(action, expected),
            "empirical_frame_gram": (np.einsum("npa,npb->ab", field, field, optimize=False) / n).tolist()}


def audit_case(name, protocol):
    ntrain, lag = protocol["train_count"], protocol["lag"]
    with np.load(OUT / "inputs" / f"{name}.npz") as source:
        input_keys = sorted(source.files)
        y = source["Y"]
    with np.load(OUT / f"{name}_inferred.npz") as archive:
        saved = {key: archive[key] for key in archive.files}
    reported = json.loads((OUT / f"{name}.json").read_text())
    with np.load(OUT / "evaluation" / f"{name}.npz") as archive:
        truth = {key: archive[key] for key in archive.files}
    mean, scale = np.mean(y[:ntrain], axis=0), np.maximum(np.std(y[:ntrain], axis=0), 1e-10)
    x = explicit_windows((y[:ntrain] - mean) / scale, lag) / np.sqrt(lag * y.shape[1])
    q, maps, neighbors = saved["Q"], saved["maps"], saved["neighbors"]
    orthogonality = np.einsum("npa,npb->nab", q, q, optimize=False)
    ortho_error = float(np.max(np.abs(orthogonality - np.eye(q.shape[-1]))))

    replay_maps, singular, overlap_residual = [], [], []
    for i in range(len(maps)):
        left, right = q[i + 1, :-y.shape[1]], q[i, y.shape[1]:]
        u, values, vt = linalg.svd(left, full_matrices=False)
        projected = np.einsum("pa,pb->ab", u, right, optimize=False) / values[:, None]
        replay = np.einsum("ab,ac->bc", vt, projected, optimize=False)
        replay_maps.append(replay)
        singular.append(values[-1])
        overlap_residual.append(rel(np.einsum("pa,ab->pb", left, replay, optimize=False), right))
    replay_maps = np.asarray(replay_maps)

    d = np.stack([explicit_windows(truth["derivative"][:ntrain, :, j] / scale, lag)
                  for j in range(truth["theta"].shape[1])], axis=-1) / np.sqrt(lag * y.shape[1])
    coeff = np.einsum("npa,npb->nab", q, d, optimize=False)
    projected = np.einsum("npa,nab->npb", q, coeff, optimize=False)
    transfer = np.einsum("nab,nbc->nac", replay_maps, coeff[:-1], optimize=False)
    tangent_error, transfer_error = rel(projected, d), rel(transfer, coeff[1:])
    frame_rows = []
    for field, row in zip(saved["fields"], reported["frame_evaluation"]["frames"]):
        gram = np.einsum("npa,npb->ab", field, field, optimize=False)
        rhs = np.einsum("npa,npb->ab", field, d, optimize=False)
        gauge = np.linalg.solve(gram, rhs)
        estimate = np.einsum("npa,ab->npb", field, gauge, optimize=False)
        metrics = section_replay(q, replay_maps, neighbors, field, row["lambda"], np.asarray(row["eigenvalues"]))
        metrics.update({"lambda": row["lambda"], "frame_relative_error_after_global_basis": rel(estimate, d),
                        "global_basis_evaluation_only": gauge.tolist(),
                        "reported_frame_error_absolute_difference": abs(rel(estimate, d) - row["frame_relative_error_after_global_basis"]),
                        "reported_transport_energy_absolute_difference": abs(metrics["transport_energy"] - row["transport_energy"]),
                        "reported_roughness_absolute_difference": abs(metrics["roughness"] - row["roughness"])})
        frame_rows.append(metrics)

    checkpoints = np.unique(np.linspace(0, len(x) - 1, 9, dtype=int))
    neighbor_errors = []
    for i in checkpoints:
        distance2 = np.sum((x - x[i])**2, axis=1)
        order = np.argsort(distance2)
        reference = order[order != i][:protocol["neighbors"]]
        neighbor_errors.append(len(set(reference) ^ set(neighbors[i])))
    times = np.unique(np.linspace(0, len(y) - 1, 13, dtype=int))
    theta = truth["theta"][times]
    clean_error = rel(field_value(theta, truth["sensors"], name), truth["clean"][times])
    finite_derivative = []
    step = 1e-6
    for j in range(theta.shape[1]):
        plus, minus = theta.copy(), theta.copy()
        plus[:, j] += step
        minus[:, j] -= step
        finite_derivative.append((field_value(plus, truth["sensors"], name) - field_value(minus, truth["sensors"], name)) / (2 * step))
    derivative_error = rel(np.stack(finite_derivative, axis=-1), truth["derivative"][times])
    mssa = baseline_replay(y[:ntrain], y[ntrain:], truth["clean"][ntrain:], lag, protocol["mssa_ranks"])
    mssa_difference = max(abs(value - row["future_clean_nrmse"]) for value, row in zip(mssa, reported["mssa"]))
    checks = {"input_has_only_Y": input_keys == ["Y"],
              "train_only_mean_scale_and_windows": max(float(np.max(np.abs(mean - saved["mean"]))), float(np.max(np.abs(scale - saved["scale"]))), float(np.max(np.abs(x - saved["X"])))) < 1e-12,
              "Q_orthonormal": ortho_error < 1e-9,
              "overlap_maps_replay": rel(replay_maps, maps) < 1e-9,
              "neighbors_sample_replay": max(neighbor_errors) == 0,
              "analytic_truth_and_derivative_consistent": clean_error < 1e-12 and derivative_error < 1e-5,
              "evaluation_metrics_replay": abs(tangent_error - reported["frame_evaluation"]["local_tangent_relative_error"]) < 1e-10 and abs(transfer_error - reported["frame_evaluation"]["overlap_transfer_relative_error"]) < 1e-10 and max(row["reported_frame_error_absolute_difference"] for row in frame_rows) < 1e-10,
              "section_eigen_equation_and_energy_replay": max(row["eigen_equation_relative_residual"] for row in frame_rows) < 1e-5 and max(row["reported_transport_energy_absolute_difference"] for row in frame_rows) < 1e-9 and max(row["reported_roughness_absolute_difference"] for row in frame_rows) < 1e-8,
              "MSSA_future_coding_replay": mssa_difference < 1e-9}
    return {"case": name, "passed": all(checks.values()), "checks": checks,
            "estimated_dimension": q.shape[-1], "true_dimension_evaluation_only": d.shape[-1],
            "Q_max_orthonormal_error": ortho_error,
            "overlap_maps_relative_replay_error": rel(replay_maps, maps),
            "overlap_map_fit_residual_quantiles": np.quantile(overlap_residual, [0, .5, 1]).tolist(),
            "overlap_min_singular_quantiles_replay": np.quantile(singular, [0, .1, .5]).tolist(),
            "true_derivative_finite_difference_relative_error": derivative_error,
            "local_tangent_relative_error": tangent_error,
            "overlap_transfer_relative_error": transfer_error, "frames": frame_rows,
            "MSSA_future_clean_nrmse_replay": mssa,
            "MSSA_max_absolute_replay_difference": mssa_difference,
            "train_window_start_indices": [0, len(x) - 1],
            "train_window_end_indices": [lag - 1, ntrain - 1],
            "future_MSSA_raw_indices": [ntrain, len(y) - 1]}


def main():
    start = time.perf_counter()
    protocol = json.loads((OUT / "protocol.json").read_text())
    paths = [HERE / "tangent_overlap.py", HERE / "run_tangent_overlap_pilot.py", OUT / "protocol.json"]
    for name in protocol["cases"]:
        paths += [OUT / "inputs" / f"{name}.npz", OUT / "evaluation" / f"{name}.npz",
                  OUT / f"{name}_inferred.npz", OUT / f"{name}.json"]
    before = {str(path): sha(path) for path in paths}
    source_tree = ast.parse((HERE / "tangent_overlap.py").read_text())
    fit_function = next(node for node in source_tree.body if isinstance(node, ast.FunctionDef) and node.name == "fit")
    fit_arguments = [node.arg for node in fit_function.args.args]
    rows = [audit_case(name, protocol) for name in protocol["cases"]]
    after = {str(path): sha(path) for path in paths}
    output = {"status": "passed" if all(row["passed"] for row in rows) and before == after else "failed",
              "fit_arguments": fit_arguments,
              "frozen_sources_and_archives_unchanged": before == after,
              "hashes": before, "cases": rows, "elapsed_seconds": time.perf_counter() - start,
              "scope": {"sensor_coordinates_used_in_fit": False,
                        "true_phases_dimension_frequencies_used_in_fit": False,
                        "new_method_evaluated_on_training_frame_only": True,
                        "new_method_future_phase_decoder_or_forecast_evaluated": False,
                        "MSSA_metric": "offline coding of observed future block, not forecasting",
                        "roughness_is_a_verified_Sobolev_consistent_estimator": False,
                        "global_basis_alignment_is_a_real_frame_gauge_not_integer_phase_gauge": True}}
    (OUT / "audit.json").write_text(json.dumps(output, indent=2))
    print(json.dumps({"status": output["status"], "unchanged": before == after,
                      "cases": [{"case": row["case"], "passed": row["passed"], "failed_checks": [k for k, v in row["checks"].items() if not v]} for row in rows],
                      "elapsed_seconds": output["elapsed_seconds"]}))


if __name__ == "__main__":
    main()
