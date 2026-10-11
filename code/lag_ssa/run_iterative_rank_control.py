"""Frozen MSSA rank controls for the saved iterative spatial-field pilot.

Targets and integer products come from the saved inference, never from truth.
Every variant uses one cached rank-64 randomized SVD; smaller ranks reuse its
prefix. DMD and the linear future readout are fitted only to observed training.
Physical character labels, phase errors, and ideal rank budgets are scored
after all fits. This is a finite pilot, not a completeness or novelty claim.
"""

from __future__ import annotations

from project_paths import RESULTS_DIR

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from iterative_fields import AREA, generate_field, score_frequency
from method import LagDesign, trajectory_matrix
from spatial_field import SPATIAL_MODEL


OUT = RESULTS_DIR
PILOT = OUT / "iterative_ssa_pilot.json"
DESTINATION = OUT / "iterative_rank_control.json"
RANKS = (4, 8, 16, 32, 64)
DEFAULT_CASES = ("r2_poisson_dominant_weak", "r2_partial_cover")


def _serializable(value):
    if isinstance(value, np.ndarray):
        return _serializable(value.tolist())
    if isinstance(value, np.generic):
        return _serializable(value.item())
    if isinstance(value, dict):
        return {str(key): _serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serializable(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _save(payload: dict, destination: Path) -> None:
    destination.parent.mkdir(exist_ok=True)
    payload["last_update_utc"] = datetime.now(timezone.utc).isoformat()
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(_serializable(payload), indent=2,
                                    allow_nan=False) + "\n")
    temporary.replace(destination)


def _gap(first, second):
    return abs(np.angle(np.exp(1j*(np.asarray(first)-np.asarray(second)))))


def _mm(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Avoid stale macOS BLAS floating-point flags in small projections."""
    result = np.einsum("ij,jk->ik", first, second, optimize=False)
    if not np.isfinite(result).all():
        raise FloatingPointError("nonfinite matrix product")
    return result


def _trajectory(observed: np.ndarray, variant: str, window: int) -> np.ndarray:
    design = LagDesign(np.arange(window), np.ones(window)/window, 0., window-1)
    if variant == "raw_area_weighted_mssa":
        return trajectory_matrix(observed, AREA, design)
    mass = SPATIAL_MODEL.to_mass_coordinates(SPATIAL_MODEL.recover(observed))
    return trajectory_matrix(mass, np.ones(mass.shape[0]), design)


def _cached_svd(centered: np.ndarray, maximum_rank: int = 64) -> dict:
    """One deterministic decomposition, shared by all ranks of this variant."""
    started = time.monotonic()
    x0, x1 = centered[:, :-1], centered[:, 1:]
    sketch_rank = min(min(x0.shape), maximum_rank+16)
    if min(x0.shape) <= 256:
        u, singular, vh = np.linalg.svd(x0, full_matrices=False)
        kind = "full"
    else:
        omega = np.random.default_rng(101).standard_normal((x0.shape[1], sketch_rank))
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            q, _ = np.linalg.qr(x0@omega, mode="reduced")
            for _ in range(2):
                q, _ = np.linalg.qr(x0@(x0.T@q), mode="reduced")
            small = q.T@x0
        if not (np.isfinite(q).all() and np.isfinite(small).all()):
            raise FloatingPointError("randomized SVD produced nonfinite arrays")
        small_u, singular, vh = np.linalg.svd(small, full_matrices=False)
        u = _mm(q, small_u)
        kind = "randomized_two_power_iterations"
    usable = min(maximum_rank, int(np.sum(singular > singular[0]*1e-10)))
    u, singular, vh = u[:, :usable], singular[:usable], vh[:usable]
    return {"u": u, "singular": singular, "vh": vh,
            "ux1": _mm(u.T, x1), "kind": kind, "decomposition_count": 1,
            "maximum_cached_rank": usable, "sketch_rank": sketch_rank,
            "elapsed_seconds": time.monotonic()-started}


def _frozen_dmd(cache: dict, centered: np.ndarray, future: np.ndarray,
                center: np.ndarray, targets: np.ndarray, rank: int,
                frequency_gate: float) -> dict:
    rank = min(rank, cache["maximum_cached_rank"])
    u = cache["u"][:, :rank]
    shift = (_mm(cache["ux1"][:rank], cache["vh"][:rank].T)
             / cache["singular"][None, :rank])
    eigenvalues, eigenvectors = np.linalg.eig(shift)
    full_readout = np.linalg.solve(eigenvectors, u.T)
    used, selected = set(), []
    for frequency in targets:
        choices = np.argsort(abs(eigenvalues-np.exp(1j*frequency)))
        index = next(int(choice) for choice in choices if int(choice) not in used)
        used.add(index)
        selected.append(index)
    chosen = eigenvalues[selected]
    readout = full_readout[selected]
    train_coordinates = _mm(readout, centered)
    future_coordinates = _mm(readout, future-center)
    angular_gaps = _gap(np.angle(chosen), targets)
    # Explicit diagnostic gate; raw numbers are retained because this is not
    # a calibrated statistical test of the presence of a Koopman character.
    near_unit = abs(abs(chosen)-1) <= .1
    return {"rank": rank, "train_phase": np.angle(train_coordinates),
            "future_phase": np.angle(future_coordinates),
            "selected_eigenvalues": np.column_stack((chosen.real, chosen.imag)),
            "selected_eigenfrequency_rad_per_sample": np.angle(chosen),
            "selected_eigenvalue_radius": abs(chosen),
            "selected_target_angular_gap_rad_per_sample": angular_gaps,
            "selected_target_complex_gap": abs(chosen-np.exp(1j*targets)),
            "nearest_any_eigenfrequency_target_gap_rad_per_sample":
                np.min(_gap(np.angle(eigenvalues)[:, None], targets[None, :]), axis=0),
            "target_mode_present_diagnostic": (angular_gaps <= frequency_gate)&near_unit,
            "eigenvector_condition": float(np.linalg.cond(eigenvectors)),
            "future_radius_cv_by_character":
                abs(future_coordinates).std(axis=1)
                / np.maximum(abs(future_coordinates).mean(axis=1), 1e-15),
            "all_eigenfrequencies_rad_per_sample": np.angle(eigenvalues),
            "all_eigenvalue_radii": abs(eigenvalues),
            "future_used_in_fit": False}


def _phase_scores(train: np.ndarray, future: np.ndarray,
                  true_train: np.ndarray, true_future: np.ndarray) -> dict:
    origins = np.angle(np.mean(np.exp(1j*(train-true_train)), axis=1))
    delta = future-true_future
    aligned = abs(np.angle(np.exp(1j*(delta-origins[:, None]))))
    future_origin = np.angle(np.mean(np.exp(1j*delta), axis=1))
    free = abs(np.angle(np.exp(1j*(delta-future_origin[:, None]))))
    return {"future_train_origin_aligned_mae_rad": float(aligned.mean()),
            "future_train_origin_aligned_mae_by_coordinate_rad": aligned.mean(axis=1),
            "future_origin_free_mae_rad": float(free.mean()),
            "future_origin_free_mae_by_coordinate_rad": free.mean(axis=1),
            "train_origin_shifts_truth_only_rad": origins,
            "future_columns": future.shape[1]}


def _ideal_budget(rho: float, epsilon: float, stronger_extra_pairs: int = 0) -> dict:
    count = 0
    while 2*(1-rho*rho)*rho**(2*count) > epsilon*epsilon:
        count += 1
    return {"rho": rho, "epsilon": epsilon,
            "poisson_pairs_stronger_than_weak": count,
            "stronger_extra_pairs": stronger_extra_pairs,
            "minimum_ideal_rank_including_weak_pair": 2*(count+stronger_extra_pairs+1),
            "weak_complex_mode_eigenvalue": epsilon*epsilon/4,
            "scope": "infinite normalized lag covariance, unit sampling-norm profiles, distinct temporal frequencies; not a finite-window recovery bound"}


def evaluate(pilot_row: dict) -> dict:
    started = time.monotonic()
    case = pilot_row["case"]
    settings = pilot_row["settings"]
    n_train = settings["n_train"]
    future_start = n_train+settings["n_validation"]
    count = future_start+settings["n_future"]
    window = settings["lag_window"]
    targets = np.asarray(pilot_row["inference"]["acquired_frequencies"])
    if not len(targets):
        raise ValueError(f"{case}: saved inference has no acquired targets")
    products = np.asarray(pilot_row["inference"]["final_lattice"]["products"], dtype=int)
    generated = generate_field(case, np.arange(count), n_train=n_train)
    clean = generated["clean"]
    noise = pilot_row["relative_sensor_noise"]
    sigma = noise*float(np.sqrt(np.mean(np.sum(AREA[:, None]*clean[:, :n_train]**2, axis=0))))
    observed = clean+sigma*np.random.default_rng(pilot_row["seed"]).standard_normal(clean.shape)
    raw_train = observed[:, :n_train]
    raw_future = observed[:, future_start:]
    frequency_gate = 2*np.pi/n_train
    fitted = {}
    cache_audit = {}
    for variant in ("raw_area_weighted_mssa", "spatial_physical_L2_mssa"):
        train = _trajectory(raw_train, variant, window)
        future = _trajectory(raw_future, variant, window)
        center = train.mean(axis=1, keepdims=True)
        centered = train-center
        cache = _cached_svd(centered)
        cache_audit[variant] = {key: cache[key] for key in
                               ("kind", "decomposition_count", "maximum_cached_rank",
                                "sketch_rank", "elapsed_seconds")}
        cache_audit[variant]["singular_values_rank_1_to_64"] = cache["singular"]
        fitted[variant] = [_frozen_dmd(cache, centered, future, center, targets,
                                     rank, frequency_gate) for rank in RANKS]

    # Truth enters only here, after every training fit and frozen future readout.
    scores = [score_frequency(case, float(frequency), frequency_gate,
                              max_harmonic=12, n_train=n_train) for frequency in targets]
    labels = []
    true_frequencies = []
    for score in scores:
        distinct = {tuple(match["mode"]) for match in score["matches"]}
        if len(distinct) != 1:
            raise ValueError(f"{case}: target label is ambiguous or absent in truth scoring")
        labels.append(next(iter(distinct)))
        true_frequencies.append(score["matches"][0]["positive_frequency"])
    labels = np.asarray(labels, dtype=int)
    true_frequencies = np.asarray(true_frequencies)
    physical_basis = _mm(products, labels)
    train_count = n_train-window+1
    future_count = settings["n_future"]-window+1
    true_train = _mm(labels, generated["theta"][:, :train_count])
    true_future = _mm(labels, generated["theta"][:, future_start:future_start+future_count])
    target_clock_train = targets[:, None]*np.arange(train_count)[None, :]
    target_clock_future = targets[:, None]*np.arange(future_start, future_start+future_count)[None, :]
    for variant, rows in fitted.items():
        for row in rows:
            train_phase = row.pop("train_phase")
            future_phase = row.pop("future_phase")
            row["character_phase_scores_truth_only"] = _phase_scores(
                train_phase, future_phase, true_train, true_future)
            row["inferred_product_phase_scores_truth_only"] = _phase_scores(
                _mm(products, train_phase), _mm(products, future_phase),
                _mm(products, true_train), _mm(products, true_future))
            eigenfrequency = row["selected_eigenfrequency_rad_per_sample"]
            row["selected_eigenfrequency_truth_gap_rad_per_sample"] = _gap(eigenfrequency, true_frequencies)
            row["selected_eigenmode_labels_truth_only"] = [
                score_frequency(case, float(frequency), frequency_gate,
                                max_harmonic=12, n_train=n_train)
                for frequency in eigenfrequency]
    budgets = []
    if case == "r2_poisson_dominant_weak":
        budgets = [_ideal_budget(.7, .08)]
    elif case == "r2_partial_cover":
        budgets = [_ideal_budget(.7, .2), _ideal_budget(.7, .08, stronger_extra_pairs=1)]
    return {"case": case, "seed": pilot_row["seed"], "settings": settings,
            "relative_sensor_noise": noise, "applied_sigma_truth_only": sigma,
            "acquired_target_frequencies_from_saved_inference": targets,
            "inferred_integer_products": products,
            "physical_target_labels_truth_only": labels,
            "physical_basis_labels_truth_only": physical_basis,
            "inferred_target_truth_frequency_bias_rad_per_sample": targets-true_frequencies,
            "target_clock_character_scores_truth_only": _phase_scores(
                target_clock_train, target_clock_future, true_train, true_future),
            "target_clock_product_scores_truth_only": _phase_scores(
                _mm(products, target_clock_train), _mm(products, target_clock_future),
                _mm(products, true_train), _mm(products, true_future)),
            "target_mode_diagnostic_gate": {
                "angular_gap_max_rad_per_sample": frequency_gate,
                "eigenvalue_radius_distance_from_one_max": .1,
                "meaning": "descriptive presence gate, not a calibrated significance test"},
            "cached_svd_audit": cache_audit, "variants": fitted,
            "saved_original_classical_baseline": pilot_row.get("classical_baseline"),
            "saved_original_phase_comparison": pilot_row.get("phase_comparison"),
            "ideal_rank_budgets_truth_only": budgets,
            "run_status": "finished", "elapsed_seconds": time.monotonic()-started}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", default=list(DEFAULT_CASES))
    parser.add_argument("--pilot", type=Path, default=PILOT)
    parser.add_argument("--output", type=Path, default=DESTINATION)
    args = parser.parse_args()
    source_text = args.pilot.read_text()
    pilot = json.loads(source_text)
    available = {row["case"]: row for row in pilot["rows"] if row.get("run_status") == "finished"}
    missing = [case for case in args.cases if case not in available]
    if missing:
        raise ValueError(f"No finished saved inference for: {missing}")
    payload = {"description": "frozen raw and spatial-preprocessed MSSA/DMD rank controls",
               "ranks": RANKS, "pilot_source": str(args.pilot),
               "pilot_source_sha256": hashlib.sha256(source_text.encode()).hexdigest(),
               "selected_case_names": args.cases,
               "future_used_in_fit": False, "targets_fitted_from_truth": False,
               "svd_prefix_reused_across_ranks": True,
               "rank_selected_by_future_or_truth": False,
               "ideal_reference_examples_truth_only": [_ideal_budget(.7, .08), _ideal_budget(.88, .08)],
               "control_run_complete": False, "rows": []}
    _save(payload, args.output)
    for case in args.cases:
        row = evaluate(available[case])
        payload["rows"].append(row)
        _save(payload, args.output)
        brief = {variant: [{"rank": item["rank"],
                            "mode_present": item["target_mode_present_diagnostic"].tolist(),
                            "future_mae": item["inferred_product_phase_scores_truth_only"][
                                "future_train_origin_aligned_mae_rad"]} for item in items]
                 for variant, items in row["variants"].items()}
        print(json.dumps({"case": case, "results": brief}), flush=True)
    payload["control_run_complete"] = True
    _save(payload, args.output)


if __name__ == "__main__":
    main()
