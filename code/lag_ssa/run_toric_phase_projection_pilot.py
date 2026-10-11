"""Frozen toric phase-denoising pilot on six cover runs and cos2/cos3.

All origins, weights and multistart projections finish and are saved before
evaluation archives are opened. Learned integer labels/frequencies only enter
fitting. True phase states are used exclusively in the subsequent assessment.
"""

from __future__ import annotations

from project_paths import RESULTS_DIR

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial import cKDTree

from toric_phase_projection import fit_toric_phase_projector, wrap


HERE = Path(__file__).resolve().parent
RESULTS = RESULTS_DIR
OUTPUT = RESULTS / "toric_phase_projection_pilot.json"
FROZEN = RESULTS / "toric_phase_projection_pilot"
COVER_CASES = ("r2_primitive_no_fundamentals", "r3_random_smooth_no_fundamentals")
WEIGHT_RATIO_CAP = 100.


def fit_row(run: str, case: str, seed, source_path: Path, inference: dict,
            future_start: int, scoring_kind: str) -> dict:
    started = time.perf_counter()
    lattice = inference["final_lattice"]
    labels = np.asarray(lattice["peak_labels"], dtype=int)
    products = np.asarray(lattice["products"], dtype=int)
    frequency = np.asarray(lattice["basis"], dtype=float)
    with np.load(source_path, allow_pickle=False) as source:
        train = np.exp(1j*source["train_phase"])
        future = np.exp(1j*source["future_phase"])
    projector = fit_toric_phase_projector(train, labels, products, frequency, WEIGHT_RATIO_CAP)
    train_projection = projector.project(train)
    future_projection = projector.project(future)
    destination = FROZEN / (run+".npz")
    np.savez_compressed(destination,
        train_product_phase=train_projection["product_phase"],
        future_product_phase=future_projection["product_phase"],
        train_projected_phase=train_projection["phase"],
        future_projected_phase=future_projection["phase"],
        future_product_cost=future_projection["product_weighted_character_cost"],
        future_projection_cost=future_projection["projection_weighted_character_cost"])
    return {"run": run, "case": case, "seed": seed, "scoring_kind": scoring_kind,
        "source_coordinates": str(source_path), "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "opened_source_phase_keys": ["train_phase", "future_phase"],
        "future_start": future_start, "inferred_toric_model": projector.metadata(),
        "train_projection_diagnostics": train_projection["diagnostics"],
        "future_projection_diagnostics": future_projection["diagnostics"],
        "future_mean_weighted_cost_product": float(np.mean(future_projection["product_weighted_character_cost"])),
        "future_mean_weighted_cost_projection": float(np.mean(future_projection["projection_weighted_character_cost"])),
        "frozen_phase_arrays": str(destination), "frozen_phase_arrays_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "fit_frozen_utc": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": time.perf_counter()-started}


def phase_assessment(train: np.ndarray, future: np.ndarray,
                     true_train: np.ndarray, true_future: np.ndarray) -> dict:
    origin = np.angle(np.mean(np.exp(1j*(train-true_train)), axis=1))
    aligned = wrap(future-origin[:, None])
    error = abs(wrap(aligned-true_future))
    # Directly test empirical state collapse within the sampled future cloud.
    embedding = np.concatenate((np.cos(aligned), np.sin(aligned)), axis=0).T
    pairs = np.asarray(list(cKDTree(embedding).query_pairs(.1)), dtype=int).reshape(-1, 2)
    if len(pairs):
        true_chord = np.sqrt(np.sum(abs(np.exp(1j*true_future[:, pairs[:, 0]])-
                                             np.exp(1j*true_future[:, pairs[:, 1]]))**2, axis=0))
        false_pairs = int(np.sum(true_chord > .5))
    else:
        false_pairs = 0
    _, estimated_neighbors = cKDTree(embedding).query(embedding, k=2)
    neighbor = estimated_neighbors[:, 1]
    true_embedding = np.concatenate((np.cos(true_future), np.sin(true_future)), axis=0).T
    true_nearest_distance = cKDTree(true_embedding).query(true_embedding, k=2)[0][:, 1]
    selected_true_distance = np.linalg.norm(true_embedding-true_embedding[neighbor], axis=1)
    ratio = selected_true_distance/np.maximum(true_nearest_distance, 1e-12)
    return {"future_train_origin_phase_mae_rad": float(np.mean(error)),
            "future_phase_mae_per_coordinate_rad": np.mean(error, axis=1).tolist(),
            "future_phase_rmse_rad": float(np.sqrt(np.mean(error**2))),
            "future_phase_max_error_rad": float(np.max(error)),
            "train_origin_truth_only_rad": origin.tolist(),
            "empirical_distinct_state_check": {
                "estimated_near_pair_chord_threshold": .1, "true_far_pair_chord_threshold": .5,
                "estimated_near_pairs": len(pairs), "false_near_pairs": false_pairs,
                "false_near_pair_fraction": false_pairs/len(pairs) if len(pairs) else None,
                "nearest_neighbor_states_checked": future.shape[1],
                "estimated_neighbor_true_distance_p95": float(np.quantile(selected_true_distance, .95)),
                "true_distance_to_chosen_neighbor_over_best_true_neighbor_median": float(np.median(ratio)),
                "distant_wrong_neighbor_count": int(np.sum((ratio > 2) & (selected_true_distance > .5))),
                "empirical_pair_check_is_global_injectivity_proof": False}}


def main() -> None:
    FROZEN.mkdir(exist_ok=True)
    cover_path = RESULTS / "general_robustness_cover.json"
    final_path = RESULTS / "final_phase_factor_assay_fixtures.json"
    cover = json.loads(cover_path.read_text())
    final = json.loads(final_path.read_text())
    rows = []
    for source_row in cover["rows"]:
        if source_row["case"] not in COVER_CASES:
            continue
        row = fit_row(source_row["run"], source_row["case"], source_row["seed"],
                      Path(source_row["frozen_arrays"]), source_row["inference"],
                      cover["settings"]["n_train"]+cover["settings"]["n_validation"], "general_cover")
        rows.append(row)
        print("frozen "+row["run"], flush=True)
    cos_row = next(row for row in final["rows"] if row["case"] == "r1_poisson2_cos3")
    rows.append(fit_row("r1_poisson2_cos3", "r1_poisson2_cos3", cos_row["data_seed"],
                       RESULTS / "final_phase_factor_assay_fixtures_r1_poisson2_cos3_coordinates.npz",
                       cos_row["inference"], cos_row["settings"]["n_train"]+cos_row["settings"]["n_validation"],
                       "original_fixture"))
    fits_finished = datetime.now(timezone.utc).isoformat()
    # All seven origins/weights/projections are frozen. Begin truth scoring.
    manifest = {entry["run"]: entry for entry in cover["manifest"]}
    sources = {row["run"]: row for row in cover["rows"]}
    for row in rows:
        if row["scoring_kind"] == "general_cover":
            with np.load(manifest[row["run"]]["evaluation"], allow_pickle=False) as evaluation:
                true_theta = evaluation["theta"]
            # Existing post-fit catalog match maps the inferred basis to truth.
            physical_basis = np.asarray(sources[row["run"]]["module_truth_only"]["phase_basis_in_physical_coordinates"], dtype=int)
        else:
            from iterative_fields import generate_field, SENSORS
            n = cos_row["settings"]["n_train"]+cos_row["settings"]["n_validation"]+cos_row["settings"]["n_future"]
            generated = generate_field(row["case"], np.arange(n), SENSORS, n_train=cos_row["settings"]["n_train"])
            true_theta = generated["theta"]
            physical_basis = np.asarray(cos_row["inference"]["final_lattice"]["products"], dtype=int) @ np.asarray(cos_row["acquired_labels_truth_only"], dtype=int)
        with np.load(row["frozen_phase_arrays"], allow_pickle=False) as arrays:
            train_count = arrays["train_product_phase"].shape[1]
            future_count = arrays["future_product_phase"].shape[1]
            true_train = np.einsum("ij,jt->it", physical_basis, true_theta[:, :train_count], optimize=False)
            start = row["future_start"]
            true_future = np.einsum("ij,jt->it", physical_basis, true_theta[:, start:start+future_count], optimize=False)
            score = {name: phase_assessment(arrays["train_"+name+"_phase"], arrays["future_"+name+"_phase"], true_train, true_future)
                     for name in ("product", "projected")}
        row["assessment_truth_only"] = score
        row["phase_mae_projection_over_product_ratio"] = score["projected"]["future_train_origin_phase_mae_rad"] / score["product"]["future_train_origin_phase_mae_rad"]
        print(json.dumps({"run": row["run"], "product_mae": score["product"]["future_train_origin_phase_mae_rad"],
                          "projection_mae": score["projected"]["future_train_origin_phase_mae_rad"]}), flush=True)
    payload = {"status": "finished", "rows": rows,
        "all_fits_frozen_before_truth_scoring_utc": fits_finished,
        "fit_input_scope": "observed SSA phase arrays and learned integer label/frequency basis only",
        "future_clock_used_as_output_or_initialization": False,
        "weight_ratio_cap_fixed_before_cases": WEIGHT_RATIO_CAP,
        "new_ssa_candidate_thresholds_or_dictionary_changed": False,
        "global_closest_point_or_statistical_injectivity_certified": False,
        "novelty_claimed": False,
        "summary": {"cases": len(rows), "phase_mae_improved": sum(row["phase_mae_projection_over_product_ratio"] < 1 for row in rows),
                    "median_phase_mae_ratio": float(np.median([row["phase_mae_projection_over_product_ratio"] for row in rows])),
                    "false_near_pairs_projected_total": sum(row["assessment_truth_only"]["projected"]["empirical_distinct_state_check"]["false_near_pairs"] for row in rows),
                    "nearest_neighbor_states_checked": sum(row["assessment_truth_only"]["projected"]["empirical_distinct_state_check"]["nearest_neighbor_states_checked"] for row in rows),
                    "distant_wrong_neighbors_projected_total": sum(row["assessment_truth_only"]["projected"]["empirical_distinct_state_check"]["distant_wrong_neighbor_count"] for row in rows),
                    "selected_future_gradient_norm_max": max(row["future_projection_diagnostics"]["selected_solution_gradient_norm_max"] for row in rows),
                    "near_tied_distinct_branch_count_total": sum(row["future_projection_diagnostics"]["near_tied_distinct_branch_count"] for row in rows)},
        "source_sha256": {name: hashlib.sha256((HERE/name).read_bytes()).hexdigest()
                          for name in ("toric_phase_projection.py", "run_toric_phase_projection_pilot.py")}}
    OUTPUT.write_text(json.dumps(payload, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"output": str(OUTPUT), "summary": payload["summary"]}), flush=True)


if __name__ == "__main__":
    main()
