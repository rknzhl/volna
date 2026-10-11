"""Independent FFT discovery followed by the identical nonlinear Hankel observer.

Six public fields, first declared noise seed each. No frequencies, labels or
coefficients from the SSA fits are supplied to the independent fit. Every fit
and observed-window output is frozen before evaluation files are opened.
"""

from project_paths import RESULTS_DIR

import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np

import cover_selection_ssa as cover
import run_general_robustness_suite as suite
import run_factor_cover_suite as combined
import run_hankel_torus_projection as hankel
from run_spectral_frontend_ablation import spectral_frontend


HERE = Path(__file__).resolve().parent
ROOT = RESULTS_DIR / 'fft_hankel_torus_ablation_v2'
OUTPUT = ROOT.with_suffix(".json")
DECLARED_FRONTEND_WINDOW = 128
NAMES = (
    "run_fft_hankel_torus_ablation.py", "run_spectral_frontend_ablation.py",
    "run_general_robustness_suite.py", "run_factor_cover_suite.py",
    "cover_selection_ssa.py", "factor_ssa.py", "phase_factor_regression.py",
    "phase_readout_refinement.py", "ssa_character.py", "nodal_spatial.py",
    "hankel_torus_projection.py", "run_hankel_torus_projection.py",
    "toric_phase_projection.py",
)


def hashes():
    return {name: hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in NAMES}


def save(payload):
    temporary = OUTPUT.with_suffix(".tmp")
    temporary.write_text(json.dumps(suite.serializable(payload), indent=2, allow_nan=False)+"\n")
    temporary.replace(OUTPUT)


def matched_scores(row, entry, reference):
    """Post-freeze scoring on exactly the same held-out time starts."""
    independent = row["projection"]
    count = min(independent["common_time_start_count"], reference["common_time_start_count"])
    start = independent["time_start"]
    if start != reference["time_start"]:
        raise ValueError("comparison must share the same window-start origin")
    with np.load(entry["evaluation"]) as truth:
        target = truth["clean"][:, start:start+count]
    with np.load(independent["frozen_output"]) as predicted:
        fft_prediction = predicted["hankel_prediction"][:, :count]
    with np.load(reference["frozen_output"]) as predicted:
        ssa_prediction = predicted["hankel_prediction"][:, :count]
    denominator = np.linalg.norm(target)
    return {"time_start_count": count,
        "independent_fft_hankel_sensor_nrmse": float(np.linalg.norm(fft_prediction-target)/denominator),
        "ssa_hankel_sensor_nrmse": float(np.linalg.norm(ssa_prediction-target)/denominator),
        "same_observed_windows_and_projection_settings": True,
        "discovery_and_learned_dictionary_independent": True}


def main():
    if OUTPUT.exists():
        raise FileExistsError("preserve the existing declared ablation")
    for folder in (ROOT/"frozen", ROOT/"projected"):
        folder.mkdir(parents=True, exist_ok=True)
    original = json.loads((RESULTS_DIR / 'general_robustness_suite.json').read_text())
    manifest = [entry for entry in original["manifest"] if entry["seed"] == suite.NOISE_SEEDS[0]]
    initial = hashes()
    rows = []
    started = time.perf_counter()
    payload = {"status": "running", "rows": rows, "manifest": manifest,
        "source_fingerprints": initial, "settings": original["settings"],
        "declared_fft_frontend_window": DECLARED_FRONTEND_WINDOW,
        "input_scope": "Y,s,t only", "SSA_parameters_supplied": False,
        "identical_factor_selection_and_global_refit": True,
        "identical_hankel_projection_settings": True,
        "truth_opening_scope": "after all six fits and observed-window predictions freeze",
        "evaluation_files_opened": False, "between_case_tuning": False}
    save(payload)
    old_frontend = cover.character_frontend
    old_suite_fit, old_baselines = suite.fit_iterative_ssa, suite.BASELINE_CASES
    old_root, old_output = suite.ROOT, suite.OUTPUT
    try:
        cover.character_frontend = spectral_frontend(DECLARED_FRONTEND_WINDOW)
        suite.fit_iterative_ssa = combined.fit_combined
        suite.BASELINE_CASES = set()
        suite.ROOT, suite.OUTPUT = ROOT, OUTPUT
        for entry in manifest:
            print("starting independent FFT/Hankel "+entry["run"], flush=True)
            try:
                row = suite.fit_public_inputs(entry)
                combined.freeze_phase_decoder(row, combined.LAST_FIT)
                row["projection"] = hankel.evaluate_observations(entry, row, ROOT/"projected")
            except Exception as error:
                row = {"run": entry["run"], "case": entry["case"], "seed": entry["seed"],
                    "run_status": "exception", "exception": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc()}
            rows.append(row)
            save(payload)
            print(json.dumps({"run": entry["run"], "status": row["run_status"],
                "rank": row.get("inference", {}).get("r_candidate_conditional"),
                "elapsed": row.get("elapsed_seconds"), "exception": row.get("exception")}), flush=True)
    finally:
        cover.character_frontend = old_frontend
        suite.fit_iterative_ssa, suite.BASELINE_CASES = old_suite_fit, old_baselines
        suite.ROOT, suite.OUTPUT = old_root, old_output
    payload["all_fits_and_outputs_frozen_before_truth_open"] = True
    payload["source_fingerprints_unchanged_before_scoring"] = hashes() == initial
    save(payload)
    reference_path = RESULTS_DIR / 'hankel_torus_projection.json'
    reference = json.loads(reference_path.read_text())
    if reference["status"] != "finished":
        raise RuntimeError("freeze succeeded; SSA reference still running, resume scoring separately")
    reference_rows = {row["run"]: row for row in reference["rows"]}
    for row, entry in zip(rows, manifest):
        if row["run_status"] == "exception":
            continue
        suite.score_frozen_row(row, entry)
        combined.score_phase_decoder(row, entry)
        hankel.score(row["projection"], entry)
        row["matched_hankel_comparison"] = matched_scores(row, entry, reference_rows[entry["run"]])
    finished = [row for row in rows if row["run_status"] == "finished"]
    payload.update(status="finished", evaluation_files_opened=True,
        source_fingerprints_unchanged=hashes() == initial,
        total_elapsed_seconds=time.perf_counter()-started,
        summary={"requested_runs": len(manifest), "finished_runs": len(finished),
            "exceptions": len(rows)-len(finished),
            "correct_module_rank_relations": sum(row["module_truth_only"].get("correct_module_rank_relations", False) for row in finished),
            "median_independent_fft_hankel_nrmse": float(np.median([
                row["matched_hankel_comparison"]["independent_fft_hankel_sensor_nrmse"] for row in finished])) if finished else None,
            "median_ssa_hankel_nrmse": float(np.median([
                row["matched_hankel_comparison"]["ssa_hankel_sensor_nrmse"] for row in finished])) if finished else None})
    save(payload)
    print(json.dumps(payload["summary"]), flush=True)


if __name__ == "__main__":
    main()
