"""One fixed TRAIN-only residual detection audit for the three weak02 runs.

No discovery, relation, phase gate, or frozen estimator is modified. A finite
frequency grid and three geometry-only spatial subspaces are declared here.
The old learned Fourier span is projected out of both the data and candidate
cos/sin functions. Chi-square/Bonferroni values are conditional iid references,
not calibrated guarantees for a data-selected, finite approximation to F.
All scan outputs are frozen before evaluation arrays are opened.
"""

from project_paths import RESULTS_DIR

from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.linalg import qr
from scipy.linalg.blas import dgemm
from scipy.stats import chi2, ncx2


HERE = Path(__file__).resolve().parent
INPUT_RESULTS = RESULTS_DIR / 'general_robustness_factor_cover.json'
OUTPUT = RESULTS_DIR / 'weak_phase_detection_audit.json'
ROOT = RESULTS_DIR / 'weak_phase_detection_audit'
TRAIN = 2400
WINDOW = 128
OVERSAMPLING = 4
ALPHA = .01
SUBSPACE_SIZES = (1, 3, "all")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def product(first, second):
    return dgemm(1., np.asarray(first, order="F"), np.asarray(second, order="F"))


def graph_basis(positions):
    """Euclidean-orthonormal graph modes preserve iid sensor noise variance."""
    distance2 = np.sum((positions[:, None]-positions[None, :])**2, axis=2)
    bandwidth2 = float(np.median(distance2[np.triu_indices(len(positions), 1)]))
    adjacency = np.exp(-distance2/bandwidth2)
    np.fill_diagonal(adjacency, 0.)
    laplacian = np.diag(adjacency.sum(axis=1))-adjacency
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    # The constant mode is imposed exactly, independent of eigensolver signs.
    eigenvectors[:, 0] = 1/np.sqrt(len(positions))
    return eigenvectors, eigenvalues, bandwidth2


def reference_scan(y, old_frequencies, sigma, spatial_basis):
    """Finite-grid unknown-profile GLRT after joint temporal nuisance removal."""
    count = y.shape[1]
    times = np.arange(count)
    design = np.column_stack((np.ones(count),
        np.cos(times[:, None]*old_frequencies[None, :]),
        np.sin(times[:, None]*old_frequencies[None, :])))
    temporal, upper = qr(design, mode="economic")
    if np.min(abs(np.diag(upper))) < 1e-8:
        raise ValueError("old temporal nuisance design is numerically singular")
    residual = y-product(product(y, temporal), temporal.T)
    transform_size = OVERSAMPLING*count
    bins = np.arange(1, transform_size//2)
    frequencies = 2*np.pi*bins/transform_size
    nuisance_fft = np.fft.rfft(temporal.T, n=transform_size, axis=1)[:, bins]
    nuisance_cos, nuisance_sin = nuisance_fft.real, -nuisance_fft.imag
    twice = np.fft.fft(np.ones(count), n=transform_size)[(2*bins) % transform_size]
    cc = (count+twice.real)/2-np.sum(nuisance_cos**2, axis=0)
    ss = (count-twice.real)/2-np.sum(nuisance_sin**2, axis=0)
    cs = -twice.imag/2-np.sum(nuisance_cos*nuisance_sin, axis=0)
    determinant = cc*ss-cs*cs
    usable = determinant > 1e-9*count*count
    curves, summaries = [], []
    total_tests = len(bins)*len(SUBSPACE_SIZES)
    for spatial_size in SUBSPACE_SIZES:
        size = len(y) if spatial_size == "all" else spatial_size
        z = product(spatial_basis[:, :size].T, residual)
        transform = np.fft.rfft(z, n=transform_size, axis=1)[:, bins]
        bcos, bsin = transform.real, -transform.imag
        statistic = np.zeros(len(bins))
        numerator = np.sum(ss[None, :]*bcos*bcos-2*cs[None, :]*bcos*bsin
                           +cc[None, :]*bsin*bsin, axis=0)
        statistic[usable] = numerator[usable]/determinant[usable]/sigma**2
        # At a fixed frequency the added free coefficients number is 2*size.
        degrees = 2*size
        p_single = chi2.sf(statistic, degrees)
        p_family = np.minimum(1., total_tests*p_single)
        selected = int(np.argmax(statistic))
        threshold = float(chi2.isf(ALPHA/total_tests, degrees))
        top = np.argsort(-statistic, kind="stable")[:5]
        curves.append(statistic)
        summaries.append({"spatial_dimension": int(size), "degrees_of_freedom": degrees,
            "selected_frequency": float(frequencies[selected]),
            "maximum_statistic": float(statistic[selected]),
            "bonferroni_threshold": threshold,
            "selected_fixed_frequency_p_reference": float(p_single[selected]),
            "all_grid_and_subspaces_p_reference": float(p_family[selected]),
            "detected_at_declared_alpha_reference": bool(statistic[selected] >= threshold),
            "top5": [{"frequency": float(frequencies[index]),
                      "statistic": float(statistic[index]),
                      "all_grid_and_subspaces_p_reference": float(p_family[index])}
                     for index in top]})
    return {"frequencies": frequencies, "statistic": np.asarray(curves),
            "temporal_basis": temporal, "summaries": summaries,
            "residual": residual, "grid_points": len(bins),
            "total_tests": total_tests, "usable_grid_points": int(np.sum(usable))}


def fit_only(row, entry):
    with np.load(entry["input"]) as data:
        y, positions = data["Y"][:, :TRAIN], data["s"]
    with np.load(row["frozen_arrays"]) as frozen:
        old_frequency = np.einsum("ij,j->i", frozen["fit_labels"],
                                 frozen["fit_frequencies"], optimize=False)
    sigma = float(row["noise_estimate_train_only"]["sigma"])
    spatial, eigenvalues, bandwidth2 = graph_basis(positions)
    scan = reference_scan(y, old_frequency, sigma, spatial)
    path = ROOT/(row["run"]+".npz")
    np.savez_compressed(path, frequencies=scan["frequencies"], statistic=scan["statistic"],
                        spatial_basis=spatial, old_frequencies=old_frequency,
                        temporal_basis=scan["temporal_basis"])
    frontend = row["inference"]["stages"][-1]["frontend"]
    proposals = frontend["dynamic_proposals"]
    gates = Counter(name for proposal in proposals for name, passed in proposal["gates"].items()
                    if not passed)
    return {"run": row["run"], "source_input": entry["input"],
        "source_input_sha256": digest(entry["input"]),
        "source_parameters": row["frozen_arrays"],
        "source_parameters_sha256": digest(row["frozen_arrays"]),
        "frozen_scan": str(path), "frozen_scan_sha256": digest(path),
        "sigma_train_only": sigma, "sensor_count": len(y),
        "old_temporal_design_dimension": 1+2*len(old_frequency),
        "graph_eigenvalues": eigenvalues.tolist(), "graph_bandwidth_squared": bandwidth2,
        "inference_phase_rank": row["inference"]["r_candidate_conditional"],
        "frontend_status": frontend["status"], "frontend_reason": frontend["reason"],
        "proposals": len(proposals), "failed_gate_counts": dict(gates),
        "minimum_phase_linear_fit_rms_radians": min(p["phase_linear_fit_rms"] for p in proposals),
        "raw_lag1_correlation": frontend["raw_lag1_correlation"],
        "temporal_reference": frontend["temporal_reference"],
        "grid_points": scan["grid_points"], "total_tests": scan["total_tests"],
        "usable_grid_points": scan["usable_grid_points"], "scan": scan["summaries"],
        "provisional_detection_status": "detected_but_phase_state_unresolved" if
             any(s["detected_at_declared_alpha_reference"] for s in scan["summaries"]) else
             "no_extra_frequency_detected_at_declared_reference_level",
        "new_phase_or_rank_forced": False}


def score_only(row, entry):
    """Use known weak component only to interpret already frozen diagnostics."""
    with np.load(row["frozen_scan"]) as frozen:
        spatial, temporal = frozen["spatial_basis"], frozen["temporal_basis"]
        frequency_grid, curves = frozen["frequencies"], frozen["statistic"]
    with np.load(entry["evaluation"]) as truth:
        weak_group = 1
        index = np.flatnonzero(truth["catalog_groups"] == weak_group)
        mode = truth["catalog_modes"][index]
        coefficient = truth["catalog_coefficients"][index]
        phases = truth["theta"][:, :TRAIN]
        weak_frequency = np.einsum("ij,j->i", mode, truth["alpha"], optimize=False)
        angle = np.einsum("ij,jt->it", mode, phases, optimize=False)
        weak = 2*np.einsum("iq,it->qt", coefficient, np.exp(1j*angle), optimize=False).real
        derivative = 2*np.einsum("iq,it->qt", coefficient, 1j*np.exp(1j*angle), optimize=False).real
    sigma = row["sigma_train_only"]
    projected_weak = weak-product(product(weak, temporal), temporal.T)
    frequency = float(weak_frequency[0])
    nearest = int(np.argmin(abs(frequency_grid-frequency)))
    records = []
    for which, summary in enumerate(row["scan"]):
        size = summary["spatial_dimension"]
        signal = product(spatial[:, :size].T, projected_weak)
        noncentrality = float(np.sum(signal**2)/sigma**2)
        # This power is an optimistic known-frequency reference. A grid search
        # at Bonferroni threshold has less power when the line is off-grid.
        records.append({"spatial_dimension": size,
            "ideal_train_noncentrality_truth_only": noncentrality,
            "known_frequency_power_at_search_threshold_truth_only": float(ncx2.sf(
                summary["bonferroni_threshold"], summary["degrees_of_freedom"], noncentrality)),
            "nearest_frozen_grid_frequency": float(frequency_grid[nearest]),
            "nearest_grid_statistic_truth_only": float(curves[which, nearest]),
            "nearest_grid_global_p_reference_truth_only": float(min(1., row["total_tests"]*
                chi2.sf(curves[which, nearest], summary["degrees_of_freedom"])))})
    information = np.sum(derivative**2, axis=0)/sigma**2
    window_information = np.convolve(information, np.ones(WINDOW), mode="valid")
    row["posthoc_weak_component"] = {"generator_frequency_truth_only": frequency,
        "noise_estimate_and_geometry_unchanged": True,
        "scans_at_known_weak_component": records,
        "ideal_train_coherent_snr_truth_only": float(np.sqrt(np.sum(weak**2)/sigma**2)),
        "ideal_known_profile_known_frequency_L128_phase_fisher_information": {
            "minimum": float(window_information.min()), "median": float(np.median(window_information)),
            "maximum": float(window_information.max())},
        "ideal_known_profile_known_frequency_L128_local_phase_CR_std_radians": {
            "minimum": float(np.min(1/np.sqrt(window_information))),
            "median": float(np.median(1/np.sqrt(window_information))),
            "maximum": float(np.max(1/np.sqrt(window_information)))},
        "fisher_interpretation": "local best-case bound under known weak spatial profile/frequency, iid Gaussian noise; does not certify unbiased phase estimator or global identifiability"}


def main():
    started = time.perf_counter()
    ROOT.mkdir(exist_ok=True)
    source_hash = digest(__file__)
    original = json.loads(INPUT_RESULTS.read_text())
    entries = {entry["run"]: entry for entry in original["manifest"]}
    selected = [row for row in original["rows"] if row["case"] == "r2_clustered_boundary_weak02"]
    payload = {"status": "running", "source_sha256": source_hash,
        "settings": {"train": TRAIN, "window": WINDOW, "oversampling": OVERSAMPLING,
            "alpha": ALPHA, "spatial_subspaces": SUBSPACE_SIZES,
            "frequency_grid": "2*pi*j/(4*TRAIN), j=1,...,2*TRAIN-1",
            "graph": "fully connected Gaussian Euclidean-distance graph; median squared-distance bandwidth; unnormalized Laplacian"},
        "conditional_null": "fixed learned old Fourier span + iid Gaussian raw-sensor noise",
        "calibration_status": "conditional iid reference only: same-TRAIN nuisance-frequency/dictionary selection and plug-in noise variance not calibrated; no guarantee for arbitrary infinite F",
        "multiple_search_correction": "Bonferroni over all predeclared frequency bins and all three spatial subspaces; independence between searches not assumed",
        "primary_references": ["https://arxiv.org/abs/0711.0330", "https://www.robertnz.net/pdf/nuisance2.pdf"],
        "rows": [], "evaluation_files_opened": False}
    for original_row in selected:
        payload["rows"].append(fit_only(original_row, entries[original_row["run"]]))
    payload["all_scans_frozen_before_truth_open"] = True
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
    for row in payload["rows"]:
        score_only(row, entries[row["run"]])
    payload["evaluation_files_opened"] = True
    payload["source_sha256_unchanged"] = source_hash == digest(__file__)
    payload["all_frozen_arrays_hashes_unchanged"] = all(
        digest(row["frozen_scan"]) == row["frozen_scan_sha256"] for row in payload["rows"])
    payload["status"] = "finished"
    payload["elapsed_seconds"] = time.perf_counter()-started
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
    print(json.dumps({"status": payload["status"], "elapsed_seconds": payload["elapsed_seconds"],
        "rows": [{"run": row["run"], "scan_p_reference": [s["all_grid_and_subspaces_p_reference"]
            for s in row["scan"]], "status": row["provisional_detection_status"],
            "ideal_weak_snr": row["posthoc_weak_component"]["ideal_train_coherent_snr_truth_only"],
            "phase_CR_std_median": row["posthoc_weak_component"]["ideal_known_profile_known_frequency_L128_local_phase_CR_std_radians"]["median"]}
            for row in payload["rows"]]}, indent=2))


if __name__ == "__main__":
    main()
