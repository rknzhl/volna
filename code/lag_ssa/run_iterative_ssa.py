"""Saved exploratory assay of iterative SSA on one spatial field.

Inference receives noisy training/validation records and the declared spatial
model. Physical labels, ranks, and the infinite-support truth catalog are used
only after inference returns. Future character coordinates use frozen filters.
The pilot reports failures; completing its case loop is not a method guarantee.
"""

from __future__ import annotations

from project_paths import RESULTS_DIR

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
from itertools import combinations
import json
from math import gcd
from pathlib import Path
import re
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sympy import Matrix
from sympy.matrices.normalforms import hermite_normal_form

from iterative_fields import AREA, CASES, CASE_BY_NAME, SENSORS, generate_field, score_frequency
from iterative_ssa import Settings, fit_clock_model, fit_iterative_ssa, predict_field
from method import LagDesign, circular_mae, torus_xyz, trajectory_matrix
from spatial_field import SPATIAL_MODEL
from ssa_character import apply_ssa_character


OUT = RESULTS_DIR
DESTINATION = OUT / "iterative_ssa_pilot.json"
RUN_VARIANT = "baseline"
PLOTS_ENABLED = True
NOISE = .08
SEED = 54000
HELDOUT_POSITIONS = np.random.default_rng(54099).uniform(.03, .97, (100, 2))
SETTINGS = Settings(n_train=3200, n_validation=1600, n_future=1600,
                    lag_window=128)


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


def _stamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _source_fingerprints() -> dict:
    """Hash local modules actually loaded, including dispatched helpers."""
    directory = Path(__file__).resolve().parent
    paths = {Path(__file__).resolve()}
    for module in tuple(sys.modules.values()):
        location = getattr(module, "__file__", None)
        if location is not None:
            path = Path(location).resolve()
            if path.parent == directory and path.suffix == ".py":
                paths.add(path)
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(paths)}


def _resume_settings(settings: dict) -> dict:
    """Normalize archived baseline defaults and JSON tuple/list conversion."""
    normalized = dict(settings)
    normalized.setdefault("frontend_mode", "pair")
    normalized.setdefault("joint_frequency_mode", "raw")
    normalized.setdefault("dynamic_ranks", [4, 8, 16])
    normalized.setdefault("phase_metric_mode", "field")
    normalized.setdefault("temporal_gate_mode", "legacy")
    return _serializable(normalized)


def _save(payload: dict) -> None:
    OUT.mkdir(exist_ok=True)
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    payload["last_update_utc"] = _stamp()
    temporary = DESTINATION.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(_serializable(payload), indent=2,
                                    allow_nan=False) + "\n")
    temporary.replace(DESTINATION)


def _products(phases: np.ndarray, products: np.ndarray) -> np.ndarray:
    return np.angle(np.exp(1j*np.einsum("ij,jt->it", products, phases,
                                       optimize=False)))


def _origin_calibrate(phases: np.ndarray, frequencies: np.ndarray,
                      times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference = np.asarray(frequencies)[:, None]*np.asarray(times)[None, :]
    origins = np.angle(np.mean(np.exp(1j*(phases-reference)), axis=1))
    return np.angle(np.exp(1j*(phases-origins[:, None]))), origins


def _phase_scores(train: np.ndarray, future: np.ndarray,
                  true_train: np.ndarray, true_future: np.ndarray) -> tuple[dict, np.ndarray]:
    """Truth-only origin calibration from train plus origin-free future error."""
    origins = np.angle(np.mean(np.exp(1j*(train-true_train)), axis=1))
    aligned = np.angle(np.exp(1j*(future-origins[:, None])))
    errors = abs(np.angle(np.exp(1j*(aligned-true_future))))
    return {"future_origin_free_circular_mae_rad": circular_mae(future, true_future),
            "future_train_origin_aligned_mae_rad": float(errors.mean()),
            "future_train_origin_aligned_mae_by_coordinate_rad": errors.mean(axis=1).tolist(),
            "train_origin_shifts_truth_only_rad": origins.tolist(),
            "future_columns": future.shape[1]}, aligned


def _signature(labels: np.ndarray, ambient_rank: int) -> dict:
    """Post-inference physical integer module signature."""
    matrix = np.asarray(labels, dtype=int)
    if not len(matrix):
        return {"rank": 0, "basis": [], "index": None}
    rank = int(Matrix(matrix).rank())
    basis = [[int(item) for item in row]
             for row in hermite_normal_form(Matrix(matrix).T).T.tolist()]
    index = None
    if rank == ambient_rank:
        index = 0
        for rows in combinations(matrix, ambient_rank):
            index = gcd(index, abs(int(Matrix(rows).det())))
    return {"rank": rank, "basis": basis, "index": index}


def _frozen_classical_mssa(raw_train: np.ndarray, raw_future: np.ndarray,
                            frequencies: np.ndarray, window: int) -> dict:
    """Frozen version of method.dmd_phase_coordinates's classical MSSA fit.

    Its public function returns phases but no reusable projection. This adapter
    follows the same deterministic randomized SVD, reduced shift, and target
    matching, retaining the trained linear map for a future independent record.
    """
    rank = min(2*len(frequencies)+8, 64)
    design = LagDesign(np.arange(window), np.ones(window)/window, 0., window-1)
    train = trajectory_matrix(raw_train, AREA, design)
    future = trajectory_matrix(raw_future, AREA, design)
    center = train.mean(axis=1, keepdims=True)
    centered = train-center
    x0, x1 = centered[:, :-1], centered[:, 1:]
    if min(x0.shape) <= 256:
        u, singular, vh = np.linalg.svd(x0, full_matrices=False)
    else:
        sketch_rank = min(min(x0.shape), rank+16)
        omega = np.random.default_rng(101).standard_normal((x0.shape[1], sketch_rank))
        # method.py suppresses stale floating-point flags from macOS BLAS;
        # explicitly check the actual arrays before using the decomposition.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            q, _ = np.linalg.qr(x0@omega, mode="reduced")
            for _ in range(2):
                q, _ = np.linalg.qr(x0@(x0.T@q), mode="reduced")
            small = q.T@x0
        if not (np.isfinite(q).all() and np.isfinite(small).all()):
            raise FloatingPointError("classical randomized SVD produced nonfinite arrays")
        small_u, singular, vh = np.linalg.svd(small, full_matrices=False)
        u = np.einsum("ij,jk->ik", q, small_u, optimize=False)
    usable = int(np.sum(singular > singular[0]*1e-10))
    rank = min(rank, usable)
    u, singular, vh = u[:, :rank], singular[:rank], vh[:rank]
    ux1 = np.einsum("ij,jk->ik", u.T, x1, optimize=False)
    shift = np.einsum("ij,jk->ik", ux1, vh.T, optimize=False)/singular[None, :]
    eigenvalues, eigenvectors = np.linalg.eig(shift)
    readout = np.linalg.solve(eigenvectors, u.T)
    used, indices = set(), []
    for frequency in frequencies:
        candidates = np.argsort(abs(eigenvalues-np.exp(1j*frequency)))
        selected = next(int(candidate) for candidate in candidates
                        if int(candidate) not in used)
        used.add(selected)
        indices.append(selected)
    readout = readout[indices]
    train_coordinates = np.einsum("ij,jt->it", readout, centered, optimize=False)
    future_coordinates = np.einsum("ij,jt->it", readout, future-center, optimize=False)
    return {"train_phase": np.angle(train_coordinates),
            "future_phase": np.angle(future_coordinates),
            "rank": rank, "target_frequencies": frequencies.tolist(),
            "selected_eigenvalues": [[float(item.real), float(item.imag)]
                                     for item in eigenvalues[indices]],
            "selected_frequency_errors": abs(np.angle(np.exp(1j*(
                np.angle(eigenvalues[indices])-frequencies)))).tolist(),
            "future_fit": False, "window": window,
            "method": "method.py deterministic classical weighted raw MSSA/DMD, frozen projection"}


def _read_frozen_iterative(fitted: dict, raw_future: np.ndarray,
                           settings: Settings) -> dict:
    times = np.arange(settings.n_train+settings.n_validation,
                      settings.n_train+settings.n_validation+settings.n_future)
    mass = SPATIAL_MODEL.to_mass_coordinates(SPATIAL_MODEL.recover(raw_future))
    frequencies = np.asarray(fitted["acquired_frequencies"])
    train_phases, future_phases, qualities, origins = [], [], [], []
    for index, frontend in enumerate(fitted["frozen_frontends"]):
        old_model = fitted["clock_models"][index]
        residual = mass-old_model.predict(times)
        readout = apply_ssa_character(frontend, residual)
        train = np.asarray(frontend["phase"])
        calibrated, zero = _origin_calibrate(train[None, :], frequencies[index:index+1],
                                              np.arange(len(train)))
        future = np.angle(np.exp(1j*(np.asarray(readout["phase"])-zero[0])))
        train_phases.append(calibrated[0])
        future_phases.append(future)
        origins.append(float(zero[0]))
        radius = np.asarray(readout["radii"])
        normalized = np.asarray(readout["normalized_character"])
        qualities.append({"character": index+1,
                          "train_frontend_status": frontend.get("status"),
                          "future_radius_cv": float(radius.std()/max(radius.mean(), 1e-15)),
                          "future_phase_step_coherence": float(abs(np.mean(
                              normalized[1:]*np.conj(normalized[:-1])))),
                          "future_columns": len(future),
                          "frozen_filter": True,
                          "old_model_index": index})
    return {"train_phase": np.asarray(train_phases),
            "future_phase": np.asarray(future_phases),
            "train_origins_to_inferred_clock_rad": origins,
            "qualities": qualities}


def _field_scores(prediction: np.ndarray, clean: np.ndarray,
                  weights: np.ndarray | None = None) -> dict:
    squared = (prediction-clean)**2
    norm = clean**2
    if weights is None:
        mse, signal = float(squared.mean()), float(norm.mean())
    else:
        mse = float(np.mean(np.sum(weights[:, None]*squared, axis=0)))
        signal = float(np.mean(np.sum(weights[:, None]*norm, axis=0)))
    return {"rmse": float(np.sqrt(mse)),
            "relative_rmse": float(np.sqrt(mse/max(signal, 1e-30))),
            "clean_signal_rms": float(np.sqrt(signal))}


def _torus_plot(case_name: str, truth: np.ndarray, iterative: np.ndarray,
                 clock: np.ndarray) -> str:
    prefix = "iterative_ssa" if RUN_VARIANT == "baseline" else f"iterative_ssa_{RUN_VARIANT}"
    destination = OUT/f"{prefix}_{case_name}_imposed_torus.png"
    fig = plt.figure(figsize=(12.5, 4.5))
    for index, (phase, title) in enumerate(((truth, "Truth character phases"),
                                           (iterative, "Frozen iterative SSA"),
                                           (clock, "Inferred pure clock"))):
        ax = fig.add_subplot(1, 3, index+1, projection="3d")
        coordinates = torus_xyz(phase)
        ax.scatter(*coordinates.T, c=np.arange(len(coordinates)), s=3,
                   cmap="viridis", alpha=.6)
        ax.set(xlim=(-2.8, 2.8), ylim=(-2.8, 2.8), zlim=(-.8, .8), title=title,
               xticks=(-2, 0, 2), yticks=(-2, 0, 2), zticks=(-.7, 0, .7))
        ax.tick_params(labelsize=8, pad=1)
        ax.set_box_aspect((1, 1, .35))
        ax.view_init(27, 40)
    fig.suptitle(f"{case_name}: imposed standard torus chart (R=2, a=0.7)\n"
                 "Same fixed chart; no field or delay embedding geometry is inferred", fontsize=10)
    fig.tight_layout(pad=3)
    fig.savefig(destination, dpi=160)
    plt.close(fig)
    return str(destination)


def evaluate(case_name: str, settings: Settings | None = None) -> dict:
    settings = SETTINGS if settings is None else settings
    start = time.monotonic()
    times = np.arange(settings.n_train+settings.n_validation+settings.n_future)
    generated = generate_field(case_name, times, n_train=settings.n_train)
    clean = generated["clean"]
    sigma = NOISE*float(np.sqrt(np.mean(np.sum(AREA[:, None]*clean[:, :settings.n_train]**2,
                                               axis=0))))
    observed = clean+sigma*np.random.default_rng(SEED).standard_normal(clean.shape)
    train_end = settings.n_train
    future_start = train_end+settings.n_validation
    raw_train, raw_validation, raw_future = (observed[:, :train_end],
        observed[:, train_end:future_start], observed[:, future_start:])
    # Nothing from generated other than clean sensor observations enters this fit.
    fitted = fit_iterative_ssa(raw_train, raw_validation, settings)
    frequencies = np.asarray(fitted["acquired_frequencies"])
    iterative = (_read_frozen_iterative(fitted, raw_future, settings)
                 if len(frequencies) else None)
    classical = (_frozen_classical_mssa(raw_train, raw_future, frequencies,
                                      settings.lag_window)
                 if len(frequencies) else None)

    # Begin truth-only scoring after inference and frozen baseline fitting.
    fixture = CASE_BY_NAME[case_name]
    true_lattice = generated["observable_lattice"]
    row = {"case": case_name, "seed": SEED, "relative_sensor_noise": NOISE,
           "variant_label": RUN_VARIANT,
           "inference_source_sha256": _source_fingerprints(),
           "settings": settings.__dict__, "run_status": "finished",
           "inference": fitted["audit"],
           "applied_noise_sigma_truth_only": sigma,
           "physical_phase_rank_truth_only": fixture.phase_rank,
           "observable_lattice_truth_only": {key: true_lattice[key]
                                             for key in ("rank", "basis", "index", "scope")},
           "known_spatial_model_valid_truth_only": not fixture.violates_stable_spatial_model,
           "phase_coordinate_kind": "frozen observed SSA characters, products from inferred lattice",
           "finite_pilot_is_completeness_proof": False}
    tolerance = 2*np.pi/settings.n_train
    scores = [score_frequency(case_name, float(frequency), tolerance,
                              max_harmonic=12, n_train=settings.n_train)
              for frequency in frequencies]
    row["acquired_frequency_labels_truth_only"] = scores
    labels = []
    for score in scores:
        # Multiple source groups may contain the same physical character.
        distinct = {tuple(match["mode"]) for match in score["matches"]}
        labels.append(next(iter(distinct)) if len(distinct) == 1 else None)
    labels_available = bool(len(labels) and all(label is not None for label in labels))
    row["unique_physical_labels_truth_only"] = labels_available
    basis = fitted["audit"].get("final_lattice")
    products = np.asarray(basis["products"], dtype=int) if basis else None
    candidate_rank = fitted["audit"].get("r_candidate_conditional")
    row["r_candidate_matches_observable_truth_only"] = candidate_rank == true_lattice["rank"]
    row["r_candidate_matches_physical_truth_only"] = candidate_rank == fixture.phase_rank
    module_complete = False
    coordinate_relation_valid = False
    phase_data = None
    if labels_available:
        integer_labels = np.asarray(labels, dtype=int)
        acquired_module = _signature(integer_labels, fixture.phase_rank)
        row["acquired_physical_module_truth_only"] = acquired_module
        module_complete = (acquired_module["rank"] == true_lattice["rank"]
                           and acquired_module["basis"] == true_lattice["basis"])
        if products is not None:
            physical_basis = products@integer_labels
            coordinate_relation_valid = np.array_equal(
                np.asarray(basis["peak_labels"], dtype=int)@physical_basis, integer_labels)
            row["physical_basis_labels_truth_only"] = physical_basis.tolist()
            row["inferred_integer_relation_matches_physical_labels_truth_only"] = coordinate_relation_valid
            count = iterative["future_phase"].shape[1]
            train_count = iterative["train_phase"].shape[1]
            true_train = np.einsum("ij,jt->it", physical_basis,
                                   generated["theta"][:, :train_count], optimize=False)
            true_future = np.einsum("ij,jt->it", physical_basis,
                                    generated["theta"][:, future_start:future_start+count], optimize=False)
            itr_train = _products(iterative["train_phase"], products)
            itr_future = _products(iterative["future_phase"], products)
            classical_train, classical_origins = _origin_calibrate(
                classical["train_phase"], frequencies, np.arange(train_count))
            classical_future = np.angle(np.exp(1j*(classical["future_phase"]
                                                   -classical_origins[:, None])))
            mssa_train = _products(classical_train, products)
            mssa_future = _products(classical_future, products)
            basis_frequencies = np.asarray(basis["basis"])
            clock_train = basis_frequencies[:, None]*np.arange(train_count)[None, :]
            clock_future = basis_frequencies[:, None]*np.arange(future_start, future_start+count)[None, :]
            phase_scores, aligned_iterative = _phase_scores(itr_train, itr_future, true_train, true_future)
            mssa_scores, _ = _phase_scores(mssa_train, mssa_future, true_train, true_future)
            clock_scores, aligned_clock = _phase_scores(clock_train, clock_future, true_train, true_future)
            row["phase_comparison"] = {"iterative_frozen_ssa": phase_scores,
                                       "classical_raw_mssa_dmd": mssa_scores,
                                       "direct_inferred_clock": clock_scores,
                                       "B_products_inferred_only": products.tolist(),
                                       "A_physical_basis_truth_only": physical_basis.tolist(),
                                       "same_inferred_acquired_frequencies_all_methods": True,
                                       "relation_valid_for_physical_phase_coordinates": coordinate_relation_valid,
                                       "origins_fitted_from_training_only": True}
            phase_data = (np.angle(np.exp(1j*true_future)), aligned_iterative, aligned_clock)
    row["acquired_module_complete_truth_only"] = module_complete
    row["torus_geometry_recovered"] = False
    if iterative is not None:
        row["frozen_iterative_readout"] = {key: iterative[key] for key in
                                           ("train_origins_to_inferred_clock_rad", "qualities")}
        row["classical_baseline"] = {key: value for key, value in classical.items()
                                      if key not in ("train_phase", "future_phase")}
    if not labels_available or products is None:
        row["phase_comparison_status"] = "unavailable_no_unambiguous_physical_lattice_map"
    elif not coordinate_relation_valid:
        row["phase_comparison_status"] = "invalid_physical_lattice_relation_diagnostic_scores_only"
    elif candidate_rank != true_lattice["rank"]:
        row["phase_comparison_status"] = "wrong_observable_rank_diagnostic_character_scores_only"
    elif not module_complete:
        row["phase_comparison_status"] = "partial_observable_module_scores_only_no_complete_torus"
    else:
        row["phase_comparison_status"] = "complete_observable_module_in_this_truth_fixture"
    future_times = times[future_start:]
    heldout = generate_field(case_name, future_times, HELDOUT_POSITIONS,
                            n_train=settings.n_train)["clean"]
    final_model = fitted["clock_models"][-1]
    field_sensors = predict_field(final_model, future_times, SENSORS)
    field_heldout = predict_field(final_model, future_times, HELDOUT_POSITIONS)
    row["field_future"] = {"iterative_hierarchy_sensor": _field_scores(
        field_sensors, clean[:, future_start:], AREA),
        "iterative_hierarchy_100_unseen_spatial_points": _field_scores(field_heldout, heldout),
        "unseen_positions_count": len(HELDOUT_POSITIONS),
        "decoder_assumes_known_degree_two_spatial_basis": True,
        "hidden_profile_in_negative_case_cannot_be_reconstructed_from_sensors": fixture.violates_stable_spatial_model}
    if basis:
        direct_model = fit_clock_model(np.arange(settings.n_train), fitted["mass_train"],
                                       np.asarray(basis["basis"]), settings.k_candidates)
        row["field_future"]["direct_all_phase_knn_sensor"] = _field_scores(
            predict_field(direct_model, future_times, SENSORS), clean[:, future_start:], AREA)
        row["field_future"]["direct_all_phase_knn_100_unseen_spatial_points"] = _field_scores(
            predict_field(direct_model, future_times, HELDOUT_POSITIONS), heldout)
    if (candidate_rank == 2 and true_lattice["rank"] == 2 and module_complete
            and coordinate_relation_valid and phase_data is not None):
        if PLOTS_ENABLED:
            row["imposed_torus_chart_plot"] = _torus_plot(case_name, *phase_data)
            row["imposed_torus_chart_status"] = "visualization_of_scored_phases_only_no_geometry_certificate"
        else:
            row["imposed_torus_chart_status"] = "omitted_plotting_disabled_complete_scored_module"
    else:
        row["imposed_torus_chart_status"] = "omitted_incomplete_or_wrong_rank_or_unmapped_physical_lattice"
    row["elapsed_seconds"] = time.monotonic()-start
    return row


def main() -> None:
    global DESTINATION, RUN_VARIANT, SETTINGS, SEED, NOISE, PLOTS_ENABLED
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", choices=tuple(CASE_BY_NAME),
                        default=[case.name for case in CASES])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--variant", default="baseline",
                        help="output/audit label only; nonbaseline labels get separate JSON and plot names")
    parser.add_argument("--output", type=Path,
                        help="JSON output path; defaults to pilot JSON or a variant-specific JSON")
    parser.add_argument("--frontend", choices=("pair", "dynamic"), default="pair",
                        help="SSA character frontend mapped to Settings.frontend_mode")
    parser.add_argument("--frequency-fit", choices=("raw", "harmonic_nuisance"), default="raw",
                        help="joint frequency fit mapped to Settings.joint_frequency_mode")
    parser.add_argument("--phase-metric", choices=("field", "noise"), default="field",
                        help="SSA extraction metric mapped to Settings.phase_metric_mode")
    parser.add_argument("--temporal-gate", choices=("legacy", "iid_reference"), default="legacy",
                        help="temporal dependence diagnostic mapped to Settings.temporal_gate_mode")
    parser.add_argument("--seed", type=int, default=SEED, help="sensor noise generator seed")
    parser.add_argument("--noise", type=float, default=NOISE,
                        help="noise sigma relative to clean training sensor RMS")
    parser.add_argument("--window", type=int, default=SETTINGS.lag_window,
                        help="SSA and classical MSSA lag window")
    parser.add_argument("--no-plots", action="store_true",
                        help="skip imposed torus chart figures for robustness repeats")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9][a-z0-9_\-]*", args.variant):
        parser.error("--variant must contain lowercase letters, digits, underscores, or hyphens")
    RUN_VARIANT = args.variant
    PLOTS_ENABLED = not args.no_plots
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    if not np.isfinite(args.noise) or args.noise <= 0:
        parser.error("--noise must be finite and positive")
    if not 4 <= args.window < min(SETTINGS.n_train, SETTINGS.n_future)-3:
        parser.error("--window must be at least four and leave at least four columns per record")
    SEED, NOISE = args.seed, args.noise
    SETTINGS = replace(SETTINGS, frontend_mode=args.frontend,
                       joint_frequency_mode=args.frequency_fit,
                       phase_metric_mode=args.phase_metric,
                       temporal_gate_mode=args.temporal_gate, lag_window=args.window)
    if args.output is not None:
        DESTINATION = args.output.expanduser().resolve()
    elif RUN_VARIANT != "baseline":
        DESTINATION = OUT/f"iterative_ssa_{RUN_VARIANT}.json"
    rows = []
    if args.resume and DESTINATION.exists():
        previous = json.loads(DESTINATION.read_text())
        if previous.get("variant_label", "baseline") != RUN_VARIANT:
            parser.error("resume output has a different variant label; use a separate output path")
        if _resume_settings(previous.get("settings", {})) != _resume_settings(SETTINGS.__dict__):
            parser.error("resume settings differ from this invocation; use matching settings or a separate output path")
        if previous.get("seed") != SEED:
            parser.error("resume seed differs from this invocation; use the saved seed or a separate output path")
        if previous.get("relative_sensor_noise") != NOISE:
            parser.error("resume noise differs from this invocation; use the saved noise or a separate output path")
        rows = previous.get("rows", [])
    payload = {"description": "exploratory iterative known-spatial-V SSA pilot; finite budgets and uncalibrated gates",
               "pilot_run_complete": False, "method_guarantee_established": False,
               "expected_case_names": [case.name for case in CASES],
               "selected_case_names_this_invocation": args.cases,
               "settings": SETTINGS.__dict__, "relative_sensor_noise": NOISE,
               "variant_label": RUN_VARIANT,
               "plots_enabled": PLOTS_ENABLED,
               "variant_label_selects_output_not_inference_implementation": True,
               "frontend_mode": args.frontend, "joint_frequency_mode": args.frequency_fit,
               "phase_metric_mode": args.phase_metric,
               "temporal_gate_mode": args.temporal_gate,
               "source_sha256_at_invocation": _source_fingerprints(),
               "seed": SEED, "rows": rows,
               "truth_supplied_to_inference": False,
               "future_used_in_fit": False,
               "infinite_families_generated_by_closed_kernel": True,
               "classical_targets_are_estimated_frequencies_only": True,
               "spatial_assumption": "known six-dimensional degree-two polynomial V, 12 fixed sensors",
               "progress": {"state": "starting", "started_utc": _stamp()}}
    _save(payload)
    finished = {row["case"] for row in rows if row.get("run_status") == "finished"}
    for case_name in args.cases:
        if args.resume and case_name in finished:
            continue
        rows[:] = [row for row in rows if row["case"] != case_name]
        payload["progress"] = {"state": "running_case", "case": case_name,
                               "started_utc": _stamp(), "finished_cases": len(finished)}
        _save(payload)
        print(f"START {case_name}", flush=True)
        try:
            row = evaluate(case_name, SETTINGS)
        except Exception as error:
            row = {"case": case_name, "run_status": "exception",
                   "exception": f"{type(error).__name__}: {error}"}
        rows.append(row)
        if row["run_status"] == "finished":
            finished.add(case_name)
        payload["progress"] = {"state": "case_saved", "case": case_name,
                               "finished_cases": len(finished)}
        _save(payload)
        audit = row.get("inference", {})
        print(f"SAVED {case_name} run={row['run_status']} status={audit.get('status')} "
              f"p={audit.get('accepted_character_count')} r={audit.get('r_candidate_conditional')} "
              f"module_complete={row.get('acquired_module_complete_truth_only')} "
              f"elapsed={row.get('elapsed_seconds', 0):.1f}s", flush=True)
    payload["pilot_run_complete"] = finished == set(CASE_BY_NAME)
    payload["progress"] = {"state": "invocation_finished",
                           "finished_cases": len(finished),
                           "exception_cases": sum(row.get("run_status") == "exception" for row in rows)}
    _save(payload)
    print(str(DESTINATION), flush=True)


if __name__ == "__main__":
    main()
