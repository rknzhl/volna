"""Post-freeze PHYSICAL winding/phase audit of the geometric frontend.

Evaluation only: true phases, catalogue and analytic Poisson tails are NEVER
passed to inference. --prepare records sources/input hashes without numerical
evaluation; --run requires that exact reviewed protocol. No field/model refit.
Stable primitive winding is a necessary degree test, not global injectivity.
"""

from project_paths import RESULTS_DIR

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import traceback

import numpy as np
from sympy import Matrix

from generate_fresh_phase_fields import CASES, terms
from geometric_phase_frontend import library_fingerprints, transform_hankel_topology


HERE = Path(__file__).resolve().parent
INFERENCE = RESULTS_DIR / 'geometric_phase_frontend.json'
EVALUATION_ROOT = RESULTS_DIR / 'fresh_phase_pipeline/evaluation'
ROOT = RESULTS_DIR / 'geometric_phase_frontend_evaluation'
OUTPUT = ROOT.with_suffix(".json")
PROTOCOL = RESULTS_DIR / 'geometric_phase_frontend_evaluation_protocol.json'
TAU = 2 * np.pi


@dataclass(frozen=True)
class AuditSettings:
    loop_resolutions: tuple = (512, 1024)
    loop_base_train_endpoint: int = 127
    field_check_count: int = 12
    field_check_relative_tolerance: float = 2e-11
    maximum_loop_principal_step: float = np.pi / 2
    integer_winding_tolerance: float = 1e-6
    loop_closure_tolerance: float = 2e-7
    gauge: str = "TRAIN-only circular mean of coordinate minus physical B*theta at window END"


SETTINGS = AuditSettings()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def serializable(value):
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


def write_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(serializable(payload), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_archive(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _verify_inference(source):
    if (source["status"] != "finished_inference_only"
            or not source["all_six_attempts_and_outputs_frozen_before_truth"]
            or source["evaluation_files_opened"] or len(source["rows"]) != 6):
        raise ValueError("all six inference attempts must finish/freeze before evaluation")
    protected = {str(INFERENCE): digest(INFERENCE),
                 source["protocol_file"]: source["protocol_sha256"]}
    for name, expected in source["source_fingerprints"].items():
        protected[str(HERE / name)] = expected
    if library_fingerprints() != source["library_fingerprints"]:
        raise ValueError("inference library changed since the field fits")
    for entry in source["manifest"]:
        protected[entry["input"]] = entry["input_sha256"]
    for row in source["rows"]:
        for key in ("frozen_parameters", "frozen_output"):
            if key in row:
                protected[row[key]] = row[key + "_sha256"]
    for path, expected in protected.items():
        if digest(path) != expected:
            raise ValueError("frozen inference source/input/output changed: " + path)
    return protected


def preparation():
    source = json.loads(INFERENCE.read_text())
    protected = _verify_inference(source)
    evaluation_hashes = {str(EVALUATION_ROOT / (row["run"] + ".npz")):
                         digest(EVALUATION_ROOT / (row["run"] + ".npz")) for row in source["rows"]}
    return dict(status="prepared_not_run", created_at=datetime.now(timezone.utc).isoformat(),
        settings=asdict(SETTINGS), evaluation_source_sha256=digest(__file__),
        evaluation_generator_source_sha256=digest(HERE / "generate_fresh_phase_fields.py"),
        protected_inference_hashes=protected, evaluation_input_hashes=evaluation_hashes,
        inference_libraries=source["library_fingerprints"], numerical_evaluation_executed=False,
        evaluation_archives_loaded_during_preparation=False,
        physical_information_used_only_after_all_six_inference_frozen=True,
        inference_sources_or_outputs_modified=False, no_refit_or_threshold_changes=True,
        loop_reference="TRAIN theta at endpoint127; vary one physical angle through a full2pi turn",
        loop_windows="128 past lags at each analytic physical endpoint; original frozen mass/mean",
        full_loop_coverage_required=True, unresolved_loops_never_filled=True,
        winding_is_numerical_not_a_continuous_loop_or_global_injectivity_proof=True,
        physical_B_uses_only_declared_clean_loops_not_future_truth=True,
        gauge_training_data_only=True, canonical_scores_require_square_unimodular_B=True,
        finite_complex_certificate_is_not_physical_primitivity=True,
        future_scores_are_masked_encoding_scores_not_forecasts=True,
        decoder_MSSA_or_full_SSA_reconstruction_scored=False)


def field_at_phases(specification, truth, theta):
    """Exact evaluation-only field at arbitrary physical phase vectors.

    Poisson first coefficient has positive amplitude/profile in these frozen
    generator cases: c1=A*sqrt((1-rho^2)/2)*exp(-i spatial_phase).
    Thus the exact infinite-tail field is |c1|/rho * (PoissonKernel-1).
    No finite Fourier catalogue truncation is used for a Poisson group.
    """
    theta = np.asarray(theta, dtype=float)
    if theta.ndim != 2 or theta.shape[0] != int(truth["rank"]) or not np.isfinite(theta).all():
        raise ValueError("finite physical rank-by-state phase array required")
    modes, coefficients, groups = (truth[key] for key in
                                    ("catalog_modes", "catalog_coefficients", "catalog_groups"))
    result = np.zeros((coefficients.shape[1], theta.shape[1]))
    descriptors = terms(specification)
    if set(map(int, np.unique(groups))) != set(range(len(descriptors))):
        raise ValueError("evaluation catalogue/group descriptors disagree")
    for group, (family, mode, amplitude, rho) in enumerate(descriptors):
        indices = np.flatnonzero(groups == group)
        fundamental = [index for index in indices if np.array_equal(modes[index], mode)]
        if len(fundamental) != 1:
            raise ValueError("exact first group character is absent/ambiguous in evaluation catalogue")
        coefficient = coefficients[fundamental[0]]
        angle = np.asarray(mode) @ theta
        if family == "cos":
            if len(indices) != 1:
                raise ValueError("finite cosine group has unexpected catalogue entries")
            result += 2 * np.real(coefficient[:, None] * np.exp(1j * angle)[None, :])
        elif family == "poisson":
            if not 0 < rho < 1 or amplitude <= 0:
                raise ValueError("this exact evaluator requires the frozen positive Poisson amplitudes")
            profile_phase = angle[None, :] + np.angle(coefficient)[:, None]
            kernel = (1 - rho * rho) / (1 - 2 * rho * np.cos(profile_phase) + rho * rho)
            result += np.abs(coefficient)[:, None] / rho * (kernel - 1)
        else:
            raise ValueError("unsupported evaluation-only group family")
    if not np.isfinite(result).all():
        raise FloatingPointError("nonfinite analytic evaluation field")
    return result


def physical_hankel(specification, truth, parameters, endpoint_theta):
    length = int(parameters["lag_window"])
    count = endpoint_theta.shape[1]
    phases = (endpoint_theta[:, None, :]
              + truth["alpha"][:, None, None] * (np.arange(length) - length + 1)[None, :, None])
    clean = field_at_phases(specification, truth, phases.reshape(phases.shape[0], -1))
    clean = clean.reshape(clean.shape[0], length, count)
    mass = np.einsum("ij,jlm->ilm", parameters["mass_sqrt"],
                     clean - parameters["mean_sensor_train"][:, None, None], optimize=False)
    return mass.transpose(1, 0, 2).reshape(length * mass.shape[0], count) / np.sqrt(length)


def frozen_model(parameters):
    if "train_phases" not in parameters:
        return None
    keys = ("landmark_columns", "edges", "edge_indices", "harmonics", "integrals",
            "change_basis", "train_phases")
    model = {key: parameters[key] for key in keys}
    model["cover_radius"] = float(parameters["cover_radius"])
    return model


def loop_winding(phi, covered):
    resolved = bool(np.all(covered) and np.isfinite(phi).all())
    result = dict(status="unresolved_loop_coverage", covered_count=int(np.sum(covered)),
                  total_points=len(covered), complete_coverage=resolved)
    if not resolved:
        return result
    steps = np.angle(np.exp(1j * np.diff(phi, axis=1)))
    maximum = float(np.max(np.abs(steps)))
    # Unwrapping is ONLY on the fine analytic physical loop, never time data.
    lifted = np.unwrap(phi, axis=1)
    values = (lifted[:, -1] - lifted[:, 0]) / TAU
    rounded = np.rint(values).astype(np.int64)
    defect = float(np.max(np.abs(values - rounded)))
    closure = float(np.max(np.abs(np.angle(np.exp(1j * (phi[:, -1] - phi[:, 0]))))))
    stable = (maximum < SETTINGS.maximum_loop_principal_step
              and defect <= SETTINGS.integer_winding_tolerance
              and closure <= SETTINGS.loop_closure_tolerance)
    result.update(status="numerically_resolved_loop" if stable else "unresolved_loop_sampling",
                  maximum_principal_step=maximum, winding_real=values, winding_integer=rounded,
                  winding_integer_defect=defect, closed_loop_phase_defect=closure)
    return result


def phase_score(coordinates, endpoints, truth, B, origin=None, inverse=None):
    coordinates = np.asarray(coordinates)
    target = B @ truth["theta"][:, endpoints]
    valid = np.isfinite(coordinates).all(axis=0)
    if not np.any(valid):
        return dict(status="unresolved_no_valid_phase_columns", valid_count=0,
                    total_count=len(endpoints)), None
    if origin is None:
        moment = np.mean(np.exp(1j * (coordinates[:, valid] - target[:, valid])), axis=1)
        origin = np.angle(moment)
        coherence = np.abs(moment)
    else:
        coherence = None
    aligned = coordinates[:, valid] - origin[:, None]
    character_error = np.angle(np.exp(1j * (aligned - target[:, valid])))
    result = dict(status="scored_valid_columns_only", valid_count=int(np.sum(valid)),
                  total_count=len(endpoints), character_phase_MAE=float(np.mean(np.abs(character_error))),
                  character_phase_MAE_by_coordinate=np.mean(np.abs(character_error), axis=1),
                  TRAIN_gauge=origin, TRAIN_gauge_resultant=coherence,
                  physical_origin_claimed=False, missing_columns_not_filled=True)
    if inverse is not None:
        physical = inverse @ aligned
        errors = np.angle(np.exp(1j * (physical - truth["theta"][:, endpoints[valid]])))
        result.update(canonical_physical_phase_MAE=float(np.mean(np.abs(errors))),
                      canonical_physical_phase_MAE_by_coordinate=np.mean(np.abs(errors), axis=1))
    return result, origin


def evaluate_row(row, source, protocol):
    run = row["run"]
    specification = next(item for item in CASES if item["name"] == run)
    evaluation_path = str(EVALUATION_ROOT / (run + ".npz"))
    truth = load_archive(evaluation_path)
    rank = int(truth["rank"])
    audit = dict(run=run, inference_status=row["run_status"], physical_rank_truth_only=rank,
        selected_coordinate_count=row.get("selected_coordinate_count"),
        physical_information_given_to_inference=False, physical_winding_status="unresolved_no_frozen_map",
        numerical_primitive_physical_degree=False, global_map_injectivity_proved=False,
        finite_complex_certificate=row.get("frontend_diagnostics", {}).get("finite_complex_certificate"))
    check_times = np.unique(np.linspace(0, truth["clean"].shape[1] - 1,
                                        SETTINGS.field_check_count).astype(int))
    check = field_at_phases(specification, truth, truth["theta"][:, check_times])
    defect = float(np.max(np.abs(check - truth["clean"][:, check_times])))
    scale = max(1.0, float(np.max(np.abs(truth["clean"][:, check_times]))))
    audit.update(arbitrary_phase_field_validation_times=check_times,
                 arbitrary_phase_field_validation_max_error=defect)
    if defect > SETTINGS.field_check_relative_tolerance * scale:
        raise ValueError("arbitrary-phase analytic evaluator does not reproduce frozen clean field")
    if "frozen_parameters" not in row:
        return audit
    parameters = load_archive(row["frozen_parameters"])
    model = frozen_model(parameters)
    if model is None:
        return audit
    learned_rank = model["train_phases"].shape[0]
    audit["selected_coordinate_count"] = learned_rank
    audit["coordinate_count_matches_physical_rank"] = learned_rank == rank
    base = truth["theta"][:, SETTINGS.loop_base_train_endpoint].copy()
    loops, diagnostics, degrees = {}, [], []
    for resolution in SETTINGS.loop_resolutions:
        angles = np.linspace(0.0, TAU, resolution + 1)
        columns, per_loop = [], []
        for physical_axis in range(rank):
            phases = np.repeat(base[:, None], resolution + 1, axis=1)
            phases[physical_axis] += angles
            H = physical_hankel(specification, truth, parameters, phases)
            transformed = transform_hankel_topology(model, H)
            phi, covered = transformed["phi"], transformed["covered"]
            prefix = f"resolution{resolution}_physical_axis{physical_axis}"
            loops[prefix + "_angles"] = angles
            loops[prefix + "_coordinates"] = phi
            loops[prefix + "_covered"] = covered
            loops[prefix + "_nearest_landmark_distance"] = transformed["nearest_landmark_distance"]
            loop = loop_winding(phi, covered)
            loop["physical_axis"] = physical_axis
            per_loop.append(loop)
            if loop["status"] == "numerically_resolved_loop":
                columns.append(loop["winding_integer"])
        resolved = len(columns) == rank
        B = np.column_stack(columns) if resolved else None
        degrees.append(B)
        diagnostics.append(dict(resolution=resolution, loops=per_loop, B=B,
                                all_physical_loops_resolved=resolved))
    stable = (all(degree is not None for degree in degrees)
              and all(np.array_equal(degrees[0], degree) for degree in degrees[1:]))
    audit.update(physical_loop_diagnostics=diagnostics,
                 physical_winding_status="numerically_stable_two_resolution_degree" if stable
                                          else "unresolved_loop_coverage_or_degree",
                 loop_unwrap_used_only_for_evaluation=True)
    if stable:
        B = degrees[0]
        loops["physical_winding_B"] = B
        determinant = int(Matrix(B.tolist()).det()) if B.shape == (rank, rank) else None
        unimodular = determinant is not None and abs(determinant) == 1
        inverse = np.asarray(Matrix(B.tolist()).inv().tolist(), dtype=np.int64) if unimodular else None
        audit.update(physical_winding_B=B, physical_winding_determinant=determinant,
                     numerical_primitive_physical_degree=unimodular,
                     numerical_degree_test_is_not_global_nofold_test=True)
        if "frozen_output" in row:
            output = load_archive(row["frozen_output"])
            scores = {}
            for label, prefix in (("initial", "initial_"), ("averaged", "")):
                train_key, future_key = prefix + "train_coordinates", prefix + "future_coordinates"
                if train_key not in output or future_key not in output:
                    scores[label] = dict(status="unresolved_no_frozen_phase_output")
                    continue
                train_score, origin = phase_score(output[train_key], output["train_window_end_index"],
                                                  truth, B, inverse=inverse)
                if origin is None:
                    future_score = dict(status="unresolved_no_TRAIN_gauge")
                else:
                    future_score, _ = phase_score(output[future_key], output["future_window_end_index"],
                                                  truth, B, origin=origin, inverse=inverse)
                    loops[label + "_TRAIN_gauge"] = origin
                scores[label] = dict(TRAIN=train_score, FUTURE=future_score,
                    B_selected_from_declared_TRAIN_based_physical_loops=True,
                    FUTURE_used_for_B_or_origin=False)
            audit["phase_scores"] = scores
    loop_path = ROOT / (run + "__physical_loops.npz")
    np.savez_compressed(loop_path, **loops)
    audit.update(loop_archive=str(loop_path), loop_archive_sha256=digest(loop_path))
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true")
    action.add_argument("--run", action="store_true")
    args = parser.parse_args()
    current = preparation()
    if args.prepare:
        if PROTOCOL.exists() or ROOT.exists() or OUTPUT.exists():
            raise FileExistsError("preserve all previous evaluation artifacts")
        write_json(PROTOCOL, current)
        print("Evaluation protocol prepared; no numerical evaluation executed: " + str(PROTOCOL), flush=True)
        return
    if not PROTOCOL.exists() or ROOT.exists() or OUTPUT.exists():
        raise FileExistsError("exact prepared protocol and new output paths required")
    prepared = json.loads(PROTOCOL.read_text())
    for key in ("settings", "evaluation_source_sha256", "evaluation_generator_source_sha256",
                "protected_inference_hashes", "evaluation_input_hashes", "inference_libraries"):
        if prepared[key] != serializable(current[key]):
            raise ValueError("evaluation protocol/source/input changed: " + key)
    ROOT.mkdir()
    protocol_hash = digest(PROTOCOL)
    source = json.loads(INFERENCE.read_text())
    rows = []
    for row in source["rows"]:
        print("evaluating frozen physical winding " + row["run"], flush=True)
        try:
            audit = evaluate_row(row, source, prepared)
        except Exception as error:
            audit = dict(run=row["run"], status="evaluation_exception", exception=f"{type(error).__name__}: {error}",
                         traceback=traceback.format_exc(), numerical_primitive_physical_degree=False)
        rows.append(audit)
        print(json.dumps(serializable(dict(run=audit["run"], rhat=audit.get("selected_coordinate_count"),
                         rtrue=audit.get("physical_rank_truth_only"), degree=audit.get("physical_winding_B"),
                         status=audit.get("physical_winding_status", audit.get("status")),
                         primitive=audit.get("numerical_primitive_physical_degree"))), allow_nan=False), flush=True)
    _verify_inference(source)
    for path, expected in {**prepared["protected_inference_hashes"],
                           **prepared["evaluation_input_hashes"]}.items():
        if digest(path) != expected:
            raise ValueError("protected inference/evaluation source or archive changed during audit")
    if digest(PROTOCOL) != protocol_hash or digest(__file__) != prepared["evaluation_source_sha256"]:
        raise ValueError("evaluation protocol/source changed during audit")
    if digest(HERE / "generate_fresh_phase_fields.py") != prepared["evaluation_generator_source_sha256"]:
        raise ValueError("evaluation-only analytic generator source changed during audit")
    for row in rows:
        if "loop_archive" in row and digest(row["loop_archive"]) != row["loop_archive_sha256"]:
            raise ValueError("saved physical loop output changed")
    result = dict(prepared, status="finished_postfreeze_evaluation", protocol_file=str(PROTOCOL),
        protocol_sha256=protocol_hash, numerical_evaluation_executed=True, rows=rows,
        inference_sources_inputs_outputs_unchanged=True, evaluation_inputs_unchanged=True,
        loop_archives_unchanged=True, no_refit_or_threshold_changes=True,
        summary=dict(attempted_runs=len(rows), evaluation_exceptions=sum("exception" in row for row in rows),
            numerically_resolved_physical_degrees=sum(row.get("physical_winding_status") ==
                "numerically_stable_two_resolution_degree" for row in rows),
            numerical_primitive_physical_degree_count=sum(bool(row.get("numerical_primitive_physical_degree")) for row in rows),
            global_phase_injectivity_proved=False, general_new_SSA_method_validated=False,
            decoder_or_MSSA_scored=False))
    write_json(OUTPUT, result)
    print(json.dumps(serializable(result["summary"]), allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
