"""Public-data adapter and protocol for a single geometric-front-end probe.

This module reads only the six immutable Y,s,t observation
archives.  It does not import a generator, evaluator, spectral ledger,
fitted factor or physical phase label.  --prepare freezes the protocol only;
--run requires that exact protocol and is intended for execution after the
independent source/math gate.  Every attempt freezes before any later audit.
"""

from __future__ import annotations

from project_paths import RESULTS_DIR

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback
from typing import Any

import numpy as np

from nodal_spatial import NodalSpatialModel
from geometric_phase_frontend import (
    SETTINGS as FRONTEND_SETTINGS, fit_hankel_topology, frozen_arrays,
    library_fingerprints,
)
from geometric_phase_encoding import encode_hankel_blocks


HERE = Path(__file__).resolve().parent
PUBLIC_ROOT = RESULTS_DIR / 'fresh_phase_pipeline/inputs'
ROOT = RESULTS_DIR / 'geometric_phase_frontend'
OUTPUT = ROOT.with_suffix(".json")
PROTOCOL_PATH = RESULTS_DIR / 'geometric_phase_frontend_protocol.json'
RUN_NAMES = (
    "fresh_r2_primitive_cover6", "fresh_r3_infinite_weak",
    "fresh_r3_nofund_cover3", "fresh_r2_irrational_travelling_1d",
    "fresh_weak_phase06", "fresh_weak_phase008",
)
SOURCES = ("run_geometric_phase_frontend.py", "nodal_spatial.py",
           "geometric_phase_frontend.py", "geometric_phase_encoding.py",
           "phase_lift_average.py", "cohomology_primitivity.py")


@dataclass(frozen=True)
class AdapterSettings:
    n_train: int = 2400
    n_validation: int = 1200
    n_future: int = 1200
    lag_window: int = 128
    window_normalization: str = "uniform 1/sqrt(L)"
    vector_order: str = "lag-major, sensor-minor"
    heldout_context: str = "127 immediately preceding observed samples"
    coordinate_orientation: str = "coordinate dimension by window column"


SETTINGS = AdapterSettings()


def digest(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def serializable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return serializable(value.tolist())
    if isinstance(value, np.generic):
        return serializable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: serializable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [serializable(item) for item in value]
    return value


def write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(serializable(payload), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def fingerprints() -> dict:
    return {name: digest(HERE / name) for name in SOURCES}


def _public_path(run: str) -> Path:
    if run not in RUN_NAMES:
        raise ValueError("only the predeclared six public observation archives are eligible")
    return (PUBLIC_ROOT / (run + ".npz")).resolve(strict=True)


def _read_public(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"Y", "s", "t"}:
            raise ValueError("exactly public Y,s,t keys are required")
        y, positions, times = (np.asarray(archive[key], dtype=float).copy()
                               for key in ("Y", "s", "t"))
    total = SETTINGS.n_train + SETTINGS.n_validation + SETTINGS.n_future
    if (y.ndim != 2 or positions.ndim != 2 or times.shape != (total,)
            or y.shape != (len(positions), total)):
        raise ValueError("public sensor/time array shapes changed")
    if not all(np.isfinite(value).all() for value in (y, positions, times)):
        raise ValueError("finite public observations, positions and times are required")
    if not np.array_equal(times, np.arange(total)):
        raise ValueError("the frozen six cases use t=0,...,4799 in sample units")
    return y, positions, times


def public_manifest() -> list[dict]:
    """Metadata comes from observations, not old result/generation manifests."""
    rows = []
    for run in RUN_NAMES:
        path = _public_path(run)
        y, positions, times = _read_public(path)
        rows.append(dict(
            run=run, input=str(path), input_sha256=digest(path),
            public_keys=["Y", "s", "t"], Y_shape=list(y.shape),
            s_shape=list(positions.shape), t_shape=list(times.shape),
            sensor_count=len(positions), spatial_dimension=positions.shape[1],
            first_time=float(times[0]), last_time=float(times[-1]),
            sample_spacing=float(times[1] - times[0]), time_units="sample index",
            n_train=SETTINGS.n_train, n_validation=SETTINGS.n_validation,
            n_future=SETTINGS.n_future,
        ))
    return rows


def load_public_case(entry: dict) -> dict:
    """Load only a verified public archive; case identity stays outside fitting."""
    path = _public_path(entry["run"])
    if path != Path(entry["input"]).resolve(strict=True):
        raise ValueError("entry is not the corresponding predeclared public archive")
    if digest(path) != entry["input_sha256"]:
        raise ValueError("public observations changed after protocol preparation")
    y, positions, times = _read_public(path)
    spatial = NodalSpatialModel.from_positions(positions)
    mean = np.mean(y[:, :SETTINGS.n_train], axis=1)
    case = dict(
        Y=y, sensors=positions, times=times,
        mean_sensor_train=mean,
        physical_mass=spatial.physical_mass.copy(),
        mass_sqrt=spatial.mass_sqrt.copy(),
        spatial_metadata=spatial.metadata(),
    )
    # Frozen observer inputs are not writable model state.
    for value in case.values():
        if isinstance(value, np.ndarray):
            value.setflags(write=False)
    return case


def _hankel_block(case: dict, first_endpoint: int, stop_endpoint: int) -> dict:
    lag_count = SETTINGS.lag_window
    first_start = first_endpoint - lag_count + 1
    if first_start < 0 or stop_endpoint > case["Y"].shape[1] or stop_endpoint <= first_endpoint:
        raise ValueError("one or more complete observed windows are required")
    observed = case["Y"][:, first_start:stop_endpoint]
    mass = np.einsum("ij,jt->it", case["mass_sqrt"],
                     observed - case["mean_sensor_train"][:, None], optimize=False)
    count = stop_endpoint - first_endpoint
    hankel = np.asfortranarray(np.vstack(
        [mass[:, lag:lag + count] for lag in range(lag_count)]) / np.sqrt(lag_count))
    starts = np.arange(first_start, first_start + count)
    endpoints = starts + lag_count - 1
    return dict(H=hankel, window_start_index=starts, window_end_index=endpoints,
                time_at_window_start=case["times"][starts].copy(),
                time_at_window_end=case["times"][endpoints].copy())


def build_hankel_blocks(case: dict) -> dict:
    """Same geometry-only full mass metric and layout as weighted MSSA.

    Training columns are entirely inside TRAIN.  Each held-out endpoint uses
    its 128 trailing observations, including preceding observed context.
    Future windows are encoding inputs and are never fitter inputs.
    """
    train = SETTINGS.n_train
    cutoff = train + SETTINGS.n_validation
    total = cutoff + SETTINGS.n_future
    return dict(
        train=_hankel_block(case, SETTINGS.lag_window - 1, train),
        validation=_hankel_block(case, train, cutoff),
        future=_hankel_block(case, cutoff, total),
    )


def protocol() -> dict:
    """Freeze sources/settings only after the independent launch gate.

    Agreed root API (features by columns):
      fit_hankel_topology(H_train, settings) -> model and fit diagnostics
      transform_hankel_topology(model, H_block) -> phi and coverage diagnostics

    The learner receives H_train only.  Validation/future H are passed solely
    to the frozen transform.  Coordinate units remain the front-end's declared
    native units; this adapter does not invent a frequency or phase gauge.
    """
    return dict(
        status="prepared_not_run", created_at=datetime.now(timezone.utc).isoformat(),
        adapter_settings=asdict(SETTINGS), frontend_settings=asdict(FRONTEND_SETTINGS),
        frontend_api_agreed=True, frontend_settings_frozen=True,
        frontend_api=dict(module="geometric_phase_frontend.py",
                          fit="fit_hankel_topology(H_train, settings)",
                          transform="transform_hankel_topology(model, H_block)",
                          input_orientation="features by columns",
                          output_orientation="phi coordinate dimension by window column",
                          fit_receives_validation_or_future=False),
        frontend_sources=list(SOURCES),
        source_fingerprints=fingerprints(), library_fingerprints=library_fingerprints(),
        manifest=public_manifest(),
        public_input_scope="only noisy Y, known sensor coordinates s, observed sample times t",
        input_case_names_not_given_to_learner=True,
        known_phase_count_or_frequencies_given_to_inference=False,
        cached_factor_or_spectral_ledger_given_to_inference=False,
        generator_or_evaluator_loaded=False, field_fit_executed=False,
        source_math_review_required_before_run=True,
        future_values_given_to_fitter=False, all_six_attempts_and_outputs_before_truth_required=True,
        sources_and_public_inputs_verified_before_and_after_future_run_required=True,
        prior_case_inspection=True, independent_external_test=False,
        geometric_frontend_viability_or_novelty_established=False,
        spatial_metric="full exact P1 finite-element mass on the sensor convex hull; no PDE supplied",
        training_centering="one sensorwise mean over Y[:,0:2400], reused for every block",
        windows=dict(train_endpoint_range=[127, 2399], validation_endpoint_range=[2400, 3599],
                     future_endpoint_range=[3600, 4799],
                     column_counts=dict(train=2273, validation=1200, future=1200)),
        requested_output_keys=["initial_train_coordinates", "train_coordinates",
                               "validation_coordinates", "future_coordinates",
                               "window_start_index", "window_end_index", "diagnostics",
                               "closed_integer_cocycles", "finite_complex_edges",
                               "landmark_distances", "streamed_triangle_count_and_closure_defect",
                               "graph_cycle_period_matrix",
                               "finite_complex_cohomology_certificate"],
        coordinate_labels_are_algorithmic_not_true_phases=True,
        frontend_settings_and_library_source_review_required_before_protocol_freeze=True,
        decoder_or_MSSA_comparison_included=False,
        unknown_rank_is_a_heuristic_topological_diagnostic_not_certified_true_rank=True,
        finite_complex_basis_condition=("connected finite Rips complex; selected integer cocycles "
            "are closed on every stored triangle; column HNF of graph-cycle periods is identity; "
            "active H1 bar count over F47 equals selected class count"),
        finite_complex_certificate_scope=("full primitive free H1 basis of the stored finite complex "
            "only, when every stated condition passes; torsion may remain"),
        mod_prime_classes_alone_are_not_an_integral_primitivity_certificate=True,
        finite_complex_homotopy_equivalence_to_true_field_manifold_certified=False,
        primitive_true_phase_basis_certified=False,
        overlapping_window_columns_are_not_independent=True,
        phase_units="radians modulo 2*pi; no physical phase origin supplied",
        raw_support_of_initial_coordinates=SETTINGS.lag_window,
        raw_support_of_encoded_coordinates=SETTINGS.lag_window + FRONTEND_SETTINGS.averaging_window - 1,
        phase_lift_assumptions_certified_from_finite_graph=False,
        nonfinite_diagnostics_serialized_as_null=True,
    )


def _window_arrays(blocks: dict) -> dict:
    arrays = {}
    for name, block in blocks.items():
        for key in ("window_start_index", "window_end_index",
                    "time_at_window_start", "time_at_window_end"):
            arrays[name + "_" + key] = np.asarray(block[key]).copy()
    return arrays


def _coordinate_counts(arrays: dict, prefix: str = "") -> dict:
    counts = {}
    for name in ("train", "validation", "future"):
        value = arrays.get(prefix + name + "_coordinates")
        counts[name] = (int(np.sum(np.isfinite(value).all(axis=0)))
                        if value is not None and value.ndim == 2 and value.shape[0] else 0)
    return counts


def fit_entry(entry: dict) -> dict:
    """Fit TRAIN topology/lift; held-out blocks use only frozen transforms."""
    started = time.perf_counter()
    case = load_public_case(entry)
    blocks = build_hankel_blocks(case)
    # Neither the run name nor VAL/FUT values enter the topology learner.
    model = fit_hankel_topology(blocks["train"]["H"], FRONTEND_SETTINGS)
    timestamps = _window_arrays(blocks)
    parameters = frozen_arrays(model)
    parameters.update(
        physical_mass=case["physical_mass"], mass_sqrt=case["mass_sqrt"],
        mean_sensor_train=case["mean_sensor_train"], sensors=case["sensors"],
        public_sample_times=case["times"], lag_window=np.asarray(SETTINGS.lag_window),
        averaging_window=np.asarray(FRONTEND_SETTINGS.averaging_window), **timestamps,
    )
    parameter_path = ROOT / "parameters" / (entry["run"] + ".npz")
    np.savez_compressed(parameter_path, **parameters)
    parameter_hash = digest(parameter_path)
    row = dict(
        run=entry["run"], run_status="parameters_frozen_encoding_pending",
        frozen_parameters=str(parameter_path), frozen_parameters_sha256=parameter_hash,
        frontend_diagnostics=model["diagnostics"],
        h1_diagram=model["h1_diagram"], selected_coordinate_count=model["diagnostics"].get("selected_rank"),
        spatial_metadata=case["spatial_metadata"],
        block_shapes={name: list(block["H"].shape) for name, block in blocks.items()},
        temporal_fit_inputs="TRAIN Hankel columns only; full geometry-only mass metric",
        evaluator_loaded=False, cached_models_loaded=False,
        true_phase_rank_or_field_law_received=False,
    )
    try:
        encoded = encode_hankel_blocks(model, blocks, FRONTEND_SETTINGS)
        arrays = {**encoded["arrays"], **timestamps}
        output_path = ROOT / "encoded" / (entry["run"] + ".npz")
        np.savez_compressed(output_path, **arrays)
        if digest(parameter_path) != parameter_hash:
            raise ValueError("encoding mutated frozen topology parameters")
        row.update(
            run_status="fitted_and_frozen", encoding_diagnostics=encoded["diagnostics"],
            frozen_output=str(output_path), frozen_output_sha256=digest(output_path),
            initial_coordinate_valid_counts=_coordinate_counts(arrays, "initial_"),
            encoded_coordinate_valid_counts=_coordinate_counts(arrays),
            validation_and_future_topology_recomputed=False,
            no_field_decoder_or_MSSA_comparison=True,
        )
    except Exception as error:
        row.update(run_status="encoding_exception_parameters_preserved",
                   exception=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
    row["elapsed_seconds"] = time.perf_counter() - started
    return row


def _verify_prepared(prepared: dict, current: dict) -> None:
    for key in ("adapter_settings", "frontend_settings", "source_fingerprints",
                "library_fingerprints", "manifest"):
        if prepared[key] != serializable(current[key]):
            raise ValueError(key + " changed after protocol preparation")


def _verify_finished(payload: dict) -> None:
    if fingerprints() != payload["source_fingerprints"]:
        raise ValueError("inference source changed during the six attempts")
    if library_fingerprints() != payload["library_fingerprints"]:
        raise ValueError("inference library version/source changed during the six attempts")
    for entry in payload["manifest"]:
        if digest(entry["input"]) != entry["input_sha256"]:
            raise ValueError("public observations changed during the six attempts")
    for row in payload["rows"]:
        for key in ("frozen_parameters", "frozen_output"):
            if key in row and digest(row[key]) != row[key + "_sha256"]:
                raise ValueError("a saved inference archive changed")
    if digest(PROTOCOL_PATH) != payload["protocol_sha256"]:
        raise ValueError("prepared protocol changed during the six attempts")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true", help="freeze reviewed protocol only; no field fit")
    action.add_argument("--run", action="store_true", help="run only after independent source/math gate")
    arguments = parser.parse_args()
    if ROOT.exists() or OUTPUT.exists():
        raise FileExistsError("preserve every existing geometric-front-end artifact")
    declaration = protocol()
    if arguments.prepare:
        if PROTOCOL_PATH.exists():
            raise FileExistsError("preserve the existing prepared geometric-front-end protocol")
        write_json(PROTOCOL_PATH, declaration)
        print("Protocol prepared; no field fitted: " + str(PROTOCOL_PATH), flush=True)
        return
    if not PROTOCOL_PATH.exists():
        raise FileNotFoundError("prepare and review the protocol before --run")
    prepared = json.loads(PROTOCOL_PATH.read_text())
    _verify_prepared(prepared, declaration)
    for directory in ("parameters", "encoded"):
        (ROOT / directory).mkdir(parents=True, exist_ok=False)
    started, rows = time.perf_counter(), []
    payload = dict(prepared, status="running", rows=rows, protocol_file=str(PROTOCOL_PATH),
                   protocol_sha256=digest(PROTOCOL_PATH), field_fit_executed=True,
                   evaluation_files_opened=False,
                   all_six_attempts_and_outputs_frozen_before_truth=False)
    write_json(OUTPUT, payload)
    for entry in prepared["manifest"]:
        print("starting geometric phase frontend " + entry["run"], flush=True)
        try:
            row = fit_entry(entry)
        except Exception as error:
            row = dict(run=entry["run"], run_status="fit_exception",
                       exception=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        rows.append(row)
        write_json(OUTPUT, payload)
        print(json.dumps(serializable(dict(run=row["run"], status=row["run_status"],
                         frontend=row.get("frontend_diagnostics", {}).get("status"),
                         encoding=row.get("encoding_diagnostics", {}).get("status"),
                         counts=row.get("encoded_coordinate_valid_counts"),
                         exception=row.get("exception"), seconds=row.get("elapsed_seconds"))),
                         allow_nan=False), flush=True)
    _verify_finished(payload)
    frozen = [row for row in rows if row["run_status"] == "fitted_and_frozen"]
    payload.update(
        status="finished_inference_only", all_six_attempts_and_outputs_frozen_before_truth=True,
        inference_finished_at=datetime.now(timezone.utc).isoformat(),
        sources_unchanged=True, libraries_unchanged=True, public_inputs_unchanged=True,
        frozen_archives_unchanged=True, protocol_unchanged=True, evaluation_files_opened=False,
        total_elapsed_seconds=time.perf_counter() - started,
        summary=dict(
            requested_runs=6, attempted_runs=len(rows), frozen_runs=len(frozen),
            exceptions=sum("exception" in row for row in rows),
            initial_coordinate_models=sum(row.get("frontend_diagnostics", {}).get("status") ==
                                          "conditional_finite_complex_coordinates" for row in rows),
            finite_complex_full_primitive_basis_count=sum(bool(row.get("frontend_diagnostics", {})
                .get("finite_complex_certificate", {}).get("full_primitive_free_cohomology_basis_certified"))
                for row in rows),
            selected_coordinate_counts={row["run"]: row.get("selected_coordinate_count") for row in rows},
            encoded_valid_counts={row["run"]: row.get("encoded_coordinate_valid_counts") for row in rows},
            true_phase_count_or_faithfulness_scored=False,
            complete_field_reconstruction_goal_achieved=False,
        ),
    )
    write_json(OUTPUT, payload)
    print(json.dumps(serializable(payload["summary"]), allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
