"""One paired coverage extension of the frozen overlap/angle pilot.

Only the cover-witness field is fitted. The first 4096 observed rows are
copied verbatim from the original pilot; the sequence is extended to 8192.
Inference opens only Y. Generated evaluator archives are never opened here.
No statistical setting, fit source, or angle source is changed.
"""

from project_paths import RESULTS_DIR

from pathlib import Path
import hashlib
import json
import time
import warnings

import numpy as np

import run_tangent_overlap_pilot as pilot
from tangent_overlap_angles_v2 import extract_tangent_overlap_angles


HERE = Path(__file__).resolve().parent
ORIGINAL = RESULTS_DIR / 'tangent_overlap_pilot'
OUT = RESULTS_DIR / 'tangent_overlap_coverage8192'
CASE = "two_phase_cover_witness"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def run():
    inferred_path = OUT / f"{CASE}_inferred.npz"
    angles_path = OUT / f"angles_v2_{CASE}.npz"
    if inferred_path.exists() or angles_path.exists():
        raise FileExistsError("Coverage outputs already exist; no frozen result is overwritten")
    sources = [HERE / "tangent_overlap.py", HERE / "tangent_overlap_angles_v2.py",
               HERE / "run_tangent_overlap_pilot.py", Path(__file__).resolve()]
    hashes = {path.name: sha(path) for path in sources}
    original_path = ORIGINAL / "inputs" / f"{CASE}.npz"
    original_hash = sha(original_path)
    with np.load(original_path) as data:
        if set(data.files) != {"Y"}:
            raise ValueError("Original inference archive must contain only Y")
        original_y = data["Y"].copy()
    if original_y.shape != (4096, 12):
        raise ValueError("This paired extension requires the original 4096-by-12 pilot")

    pilot.OUT = OUT
    pilot.generate(count=8192, channels=12)
    input_path = OUT / "inputs" / f"{CASE}.npz"
    with np.load(input_path) as data:
        if set(data.files) != {"Y"}:
            raise ValueError("Extended inference archive must contain only Y")
        y = data["Y"].copy()
    y[:len(original_y)] = original_y
    np.savez_compressed(input_path, Y=y)
    if not np.array_equal(y[:4096], original_y):
        raise ArithmeticError("Paired observation prefix is not identical")
    protocol = json.loads((OUT / "protocol.json").read_text())
    protocol.update(active_fit_cases=[CASE], output_scope="one coverage probe",
        source_sha256=hashes, witness_input_sha256=sha(input_path),
        original_pilot_input_sha256=original_hash,
        paired_prefix_observation_count=4096,
        paired_prefix_array_sha256=array_sha(original_y),
        paired_prefix_identical=True,
        tail="newly generated observations under the unchanged generator",
        selected_frame_index=-1, selected_lambda=.0003,
        angle_module="tangent_overlap_angles_v2.py", angle_low_modes=24,
        angle_modulus_cv_limit=.20, angle_projected_derivative_residual_limit=.30,
        angle_independent_sine_min=.20,
        angle_max_states=5000,
        max_states_change_is_resource_bound_only=True,
        no_parameter_selection_from_results=True,
        prepared_before_inference=True,
        generated_evaluation_archives_opened_by_this_runner=False)
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2))
    started = time.perf_counter()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        model = pilot.fit(y[:protocol["train_count"]], lag=protocol["lag"],
            neighbors=protocol["neighbors"], lambdas=protocol["lambdas"])
    fit_seconds = time.perf_counter() - started
    np.savez_compressed(inferred_path, X=model["X"], Q=model["geometry"]["Q"],
        neighbors=model["geometry"]["neighbors"], maps=model["maps"],
        fields=np.stack([row["field"] for row in model["fits"]]),
        mean=model["mean"], scale=model["scale"])
    started = time.perf_counter()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        result = extract_tangent_overlap_angles(model["X"], model["fits"][-1]["field"],
            model["geometry"]["neighbors"], neighbour_count=32, mode_count=24,
            max_states=5000)
    angle_seconds = time.perf_counter() - started
    phases = result["phases"]
    characters = result["characters"]
    if phases is None:
        phases = np.empty((len(model["X"]), 0))
        characters = np.empty((len(model["X"]), 0), dtype=complex)
    np.savez_compressed(angles_path, phases=phases, characters=characters,
        selected_weights=result.get("selected_weights", np.empty((0, 0))),
        diagnostics_json=np.asarray(json.dumps(result["diagnostics"])),
        source_sha256=np.asarray(hashes["tangent_overlap_angles_v2.py"]),
        input_sha256=np.asarray(sha(inferred_path)), frame_index=np.asarray(-1))
    if hashes != {path.name: sha(path) for path in sources}:
        raise ArithmeticError("Frozen inference sources changed during the coverage probe")
    if sha(original_path) != original_hash or sha(input_path) != protocol["witness_input_sha256"]:
        raise ArithmeticError("Inference inputs changed during the coverage probe")
    status = dict(case=CASE, status="inference_finished_without_truth_evaluation",
        count=len(y), train_count=protocol["train_count"], window_states=len(model["X"]),
        estimated_dimension=model["geometry"]["dimension"],
        fit_seconds=fit_seconds, angle_seconds=angle_seconds,
        angle_status=result["diagnostics"]["status"],
        selected_candidate_indices=result["diagnostics"].get("selected_candidate_indices"),
        selected_directional_weights=result["diagnostics"].get("selected_directional_weights"),
        source_hashes_unchanged=True, input_hashes_unchanged=True,
        paired_prefix_identical=True, evaluation_archives_opened=False)
    (OUT / "run_status.json").write_text(json.dumps(status, indent=2))
    print(json.dumps(status), flush=True)
    return status


if __name__ == "__main__":
    run()
