"""Sufficient-rank raw MSSA control for the normalized conditional assay.

Replays only the deterministic normalized held-out field and reuses every
candidate and selected Fourier order from conditional_ssa_normalized.json.
No first-family model, candidate, or noise threshold is re-estimated here.
"""

from __future__ import annotations

from project_paths import RESULTS_DIR

import hashlib
import json
from pathlib import Path

import numpy as np

from method import circular_mae, trajectory_matrix
from run_conditional_ssa import (
    AREA, CASES, DESIGN, L, N_TEST, N_TRAIN, Q, SENSORS, TIME_TEST,
    component_fields,
)


OUT = RESULTS_DIR
SOURCE = OUT/"conditional_ssa_normalized.json"
DESTINATION = OUT/"conditional_rank_control.json"
CASE_BY_NAME = {case.name: case for case in CASES}


def cached_spatial_ssa(field: np.ndarray, weights: np.ndarray) -> dict:
    h = trajectory_matrix(field, weights, DESIGN)
    centered = h-h.mean(axis=1, keepdims=True)
    covariance = np.einsum("ij,kj->ik", centered[:, :-1], centered[:, :-1],
                           optimize=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if not np.isfinite(eigenvalues).all():
        raise FloatingPointError("nonfinite SSA covariance spectrum")
    return {"centered": centered, "eigenvalues": eigenvalues,
            "eigenvectors": eigenvectors}


def dmd_from_cache(cache: dict, frequency: float, requested_rank: int) -> dict:
    centered = cache["centered"]
    eigenvalues = cache["eigenvalues"]
    eigenvectors = cache["eigenvectors"]
    usable = int(np.sum(eigenvalues > max(0, eigenvalues[-1])*1e-10))
    rank = min(requested_rank, usable)
    if rank < 2:
        raise ValueError("fewer than two numerically usable SSA directions")
    basis = eigenvectors[:, -rank:][:, ::-1]
    coordinates = np.einsum("ij,ik->jk", basis, centered, optimize=False)
    z0, z1 = coordinates[:, :-1], coordinates[:, 1:]
    cross = np.einsum("ij,kj->ik", z1, z0, optimize=False)
    gram = np.einsum("ij,kj->ik", z0, z0, optimize=False)
    shift = np.einsum("ij,jk->ik", cross, np.linalg.pinv(gram), optimize=False)
    if not np.isfinite(shift).all():
        raise FloatingPointError("nonfinite projected DMD shift")
    values, vectors = np.linalg.eig(shift)
    choice = int(np.argmin(abs(values-np.exp(1j*frequency))))
    phase = np.angle(np.linalg.solve(vectors, coordinates)[choice])
    return {"phase": phase, "rank_used": rank,
            "eigenfrequency": float(np.angle(values[choice])),
            "frequency_error": float(abs(np.angle(values[choice]
                                                 * np.exp(-1j*frequency)))),
            "leading_variance_fraction":
                float(np.sum(eigenvalues[-rank:])/np.sum(eigenvalues))}


def replay_test_field(case_name: str, noise: float, seed: int) -> tuple[np.ndarray, dict]:
    case = CASE_BY_NAME[case_name]
    times = np.arange(N_TRAIN+N_TEST)
    strong, weak, meta = component_fields(case, times, normalize_strong=True)
    clean = strong+weak
    scale = np.sqrt(np.mean(np.sum(AREA[:, None]*clean**2, axis=0)))
    observed = clean+noise*scale*np.random.default_rng(seed).standard_normal(clean.shape)
    return observed[:, N_TRAIN:], meta


def score_variant(cache: dict, candidate: float, truth: np.ndarray,
                  requested_rank: int) -> dict:
    try:
        readout = dmd_from_cache(cache, candidate, requested_rank)
        return {"phase_mae_rad": circular_mae(readout["phase"][None, :],
                                               truth[None, :]),
                "rank_requested": requested_rank,
                "rank_used": readout["rank_used"],
                "eigenfrequency": readout["eigenfrequency"],
                "frequency_error": readout["frequency_error"],
                "leading_variance_fraction": readout["leading_variance_fraction"]}
    except (FloatingPointError, np.linalg.LinAlgError, ValueError) as error:
        return {"rank_requested": requested_rank,
                "failure": f"{type(error).__name__}: {error}"}


def evaluate_row(source_row: dict) -> dict:
    result = {"case": source_row["case"], "noise": source_row["noise"],
              "seed": source_row["seed"], "source_status": source_row["status"]}
    if source_row["status"] != "weak_candidate_scored":
        result["status"] = "no_shared_weak_candidate"
        return result
    candidate = source_row["residual_screen"]["candidate"]["frequency"]
    selected_order = source_row["first_family_fit"]["order"]
    adaptive = min(2*selected_order+2, 128)
    field, meta = replay_test_field(result["case"], result["noise"], result["seed"])
    target_name = source_row["candidate_audit_after_inference"]["matched_name"]
    target = next(item for item in meta["targets"] if item["name"] == target_name)
    truth = target["phase"][N_TRAIN:N_TRAIN+N_TEST-L+1]
    equal_cache = cached_spatial_ssa(field, np.full(Q, 1/Q))
    spatial_cache = cached_spatial_ssa(field, AREA)
    result.update({
        "status": "scored", "candidate_from_normalized_assay": candidate,
        "target_name_for_scoring_only": target_name,
        "selected_family_order_from_normalized_assay": selected_order,
        "adaptive_rank": adaptive,
        "scores": {
            "raw_classical_mssa_rank8": score_variant(equal_cache, candidate, truth, 8),
            "raw_classical_mssa_rank16": score_variant(equal_cache, candidate, truth, 16),
            "raw_classical_mssa_adaptive": score_variant(equal_cache, candidate, truth, adaptive),
            "raw_spatial_mssa_adaptive": score_variant(spatial_cache, candidate, truth, adaptive),
        },
    })
    old = source_row["scores_on_common_candidate"]
    result["source_scores_on_same_candidate"] = {
        name: {"phase_mae_rad": old[name].get("phase_mae_rad"),
               "frequency_error": old[name].get("frequency_error")}
        for name in ("raw_classical_mssa_rank4", "raw_spatial_mssa_rank4",
                     "family_residual_ssa_rank2", "direct_spectrum")}
    # Exact replay of rank-4 is a useful check that field regeneration,
    # covariance, candidate, and phase truth all match the source assay.
    replay_rank4 = score_variant(equal_cache, candidate, truth, 4)
    source_rank4 = old["raw_classical_mssa_rank4"]["phase_mae_rad"]
    result["rank4_replay_absolute_mae_difference"] = abs(
        replay_rank4["phase_mae_rad"]-source_rank4)
    return result


def summarize(rows: list[dict]) -> list[dict]:
    methods = ("raw_classical_mssa_rank4", "raw_classical_mssa_rank8",
               "raw_classical_mssa_rank16", "raw_classical_mssa_adaptive",
               "raw_spatial_mssa_adaptive", "family_residual_ssa_rank2",
               "direct_spectrum")
    result = []
    for case in CASES:
        for noise in (.03, .08, .20):
            group = [row for row in rows if row["case"] == case.name and row["noise"] == noise]
            scored = [row for row in group if row["status"] == "scored"]
            medians = {}
            failures = {}
            for method in methods:
                values = []
                for row in scored:
                    score = (row["scores"].get(method) if method in row["scores"]
                             else row["source_scores_on_same_candidate"].get(method))
                    if score is not None and score.get("phase_mae_rad") is not None:
                        values.append(score["phase_mae_rad"])
                medians[method] = float(np.median(values)) if values else None
                failures[method] = (len(group)-sum(value <= .3 for value in values)
                                    if case.kind != "negative" else None)
            result.append({"case": case.name, "noise": noise,
                           "trials": len(group), "common_weak_candidate": len(scored),
                           "selected_order_counts": {str(order): sum(row["selected_family_order_from_normalized_assay"] == order
                                                              for row in scored)
                                                     for order in sorted({row["selected_family_order_from_normalized_assay"]
                                                                          for row in scored})},
                           "median_mae_on_detected_weak_only": medians,
                           "positive_failures_all_trials_including_candidate_misses": failures,
                           "negative_false_positive_candidates":
                               sum(row["source_status"] == "wrong_residual_candidate"
                                   for row in group) if case.kind == "negative" else None})
    return result


def main() -> None:
    source = json.loads(SOURCE.read_text())
    if not source.get("normalized_strong") or len(source["rows"]) != 120:
        raise ValueError("expected complete normalized 120-row source assay")
    rows = []
    for index, source_row in enumerate(source["rows"]):
        row = evaluate_row(source_row)
        rows.append(row)
        if (index+1) % 12 == 0:
            print(f"{index+1}/{len(source['rows'])}: {row['case']} {row['noise']}", flush=True)
    differences = [row["rank4_replay_absolute_mae_difference"]
                   for row in rows if row["status"] == "scored"]
    maximum_difference = max(differences) if differences else None
    if maximum_difference is None or maximum_difference > 1e-8:
        raise AssertionError(f"rank-4 replay did not match source: {maximum_difference}")
    payload = {
        "description": "Rank-control replay on normalized held-out fields using identical L=32 trajectory matrices and shared test candidate from the source conditional assay.",
        "source_assay": str(SOURCE),
        "source_assay_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "candidate_policy": "Candidate and BIC-selected K are imported unchanged from the normalized assay; all raw variants receive the same candidate. Truth target name is consulted only after the source candidate was selected.",
        "replay_policy": "Regenerate the same normalized field and seeded noise, then use the same test block. One covariance eigendecomposition per row and spatial weighting is reused across ranks.",
        "ranks": {"fixed": [4, 8, 16], "adaptive": "min(2*K_selected+2,128)"},
        "lag_window": L, "train_samples": N_TRAIN, "test_samples": N_TEST,
        "sensor_coordinates": SENSORS.tolist(), "voronoi_weights": AREA.tolist(),
        "rows": rows, "summary": summarize(rows),
        "maximum_rank4_replay_absolute_mae_difference": maximum_difference,
    }
    DESTINATION.write_text(json.dumps(payload, indent=2)+"\n")
    print("saved", DESTINATION, "max rank4 replay difference", maximum_difference, flush=True)
    print("noise .08 main cases:", flush=True)
    for row in payload["summary"]:
        if row["noise"] == .08 and row["case"] in (
            "a_finite_strong_r2", "b_poisson70_r2", "c_poisson88_r2",
            "d_poisson70_mixed_r2"):
            print(row["case"], row["median_mae_on_detected_weak_only"], flush=True)


if __name__ == "__main__":
    main()
