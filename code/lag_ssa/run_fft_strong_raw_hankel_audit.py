"""Independent FFT models encoded with the same raw-128 Hankel observer.

Reuse six already frozen independently fitted FFT parameter archives; no
dictionary, frequency, profile, discovery or optimizer is refitted.  All raw
strong-character initializers and predictions freeze before truth scoring.
The comparison uses the same 1073 future window starts as the matched SSA run.
"""

from project_paths import RESULTS_DIR

import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np

import run_strong_raw_hankel_audit as matched


HERE = Path(__file__).resolve().parent
SOURCE = RESULTS_DIR / 'fft_hankel_torus_ablation_v2.json'
BASELINE = RESULTS_DIR / 'strong_raw_hankel_audit.json'
ROOT = RESULTS_DIR / 'fft_strong_raw_hankel_audit'
OUTPUT = ROOT.with_suffix(".json")


def hashes():
    result = matched.hashes()
    result[Path(__file__).name] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return result


def save(payload):
    temporary = OUTPUT.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(OUTPUT)


def main():
    if OUTPUT.exists():
        raise FileExistsError("preserve the existing matched FFT variant")
    source = json.loads(SOURCE.read_text())
    if source["status"] != "finished" or not source["source_fingerprints_unchanged"]:
        raise ValueError("complete source-consistent independent FFT assay required")
    source_rows = {row["run"]: row for row in source["rows"]}
    manifest = source["manifest"]
    ROOT.mkdir(parents=True, exist_ok=True)
    # Isolated process-local output override; no frozen helper source is edited.
    matched.ROOT = ROOT
    initial = hashes()
    rows = []
    started = time.perf_counter()
    payload = {
        "status": "running", "rows": rows, "manifest": manifest,
        "source_parameter_archive": str(SOURCE),
        "source_parameter_archive_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "source_fingerprints": initial,
        "projection_settings": matched.SETTINGS.__dict__,
        "independent_fft_discovery_and_parameters_refitted": False,
        "raw_seed_helper_identical_to_matched_ssa_run": True,
        "nonlinear_and_free_linear_total_context": "identical raw 128-sample window",
        "future_absolute_time_used_as_seed": False,
        "future_prefix_clock_subtraction_used": False,
        "truth_opening_scope": "after all six matched-context FFT observer outputs freeze",
        "evaluation_files_opened": False,
        "same_start_count_expected_from_declared_records": 1073,
    }
    save(payload)
    for entry in manifest:
        print("starting FFT strong-raw128 " + entry["run"], flush=True)
        try:
            row = matched.freeze(entry, source_rows[entry["run"]])
            if row["window_count"] != 1073:
                raise ValueError("unexpected future record/window count")
        except Exception as error:
            row = {"run": entry["run"], "case": entry["case"], "seed": entry["seed"],
                   "run_status": "exception", "exception": f"{type(error).__name__}: {error}",
                   "traceback": traceback.format_exc()}
        rows.append(row)
        save(payload)
        print(json.dumps({"run": row["run"], "status": row["run_status"],
                          "elapsed": row.get("elapsed_seconds"), "exception": row.get("exception")}), flush=True)
    payload["all_outputs_frozen_before_truth_open"] = True
    payload["source_fingerprints_unchanged_before_scoring"] = hashes() == initial
    save(payload)
    # Begin scoring only after every requested observation-only output freezes.
    for row, entry in zip(rows, manifest):
        if row["run_status"] == "exception":
            continue
        matched.score(row, entry, source_rows[entry["run"]])
    baseline = json.loads(BASELINE.read_text())
    baseline_rows = {row["run"]: row for row in baseline["rows"]}
    baseline_manifest = {entry["run"]: entry for entry in baseline["manifest"]}
    for row, entry in zip(rows, manifest):
        if row["run_status"] != "finished":
            continue
        other = baseline_rows[row["run"]]
        if (row["time_start"] != other["time_start"] or row["window_count"] != other["window_count"]
                or entry["input_sha256"] != baseline_manifest[row["run"]]["input_sha256"]):
            raise ValueError("FFT/SSA comparison must use identical inputs and all time starts")
        with np.load(row["frozen_output"]) as this_output, np.load(other["frozen_output"]) as other_output:
            if not np.array_equal(this_output["time_starts"], other_output["time_starts"]):
                raise ValueError("frozen time starts differ")
            relative_difference = float(np.linalg.norm(this_output["hankel_prediction"]-other_output["hankel_prediction"])
                                        / max(np.linalg.norm(other_output["hankel_prediction"]), 1e-30))
        row["matched_ssa_comparison"] = {
            "source": str(BASELINE), "same_public_input_sha256": True,
            "same_all_1073_future_window_starts": True,
            "same_raw_128_sample_context": True,
            "ssa_hankel_sensor_nrmse": other["sensor_nrmse"]["hankel"],
            "fft_hankel_sensor_nrmse": row["sensor_nrmse"]["hankel"],
            "fft_over_ssa_hankel_error_ratio": row["sensor_nrmse"]["hankel"]/other["sensor_nrmse"]["hankel"],
            "relative_hankel_prediction_difference": relative_difference,
            "ssa_phase_truth_only": other.get("phase_truth_only"),
            "module_comparison_truth_only": {
                "fft_complete": source_rows[row["run"]]["module_truth_only"].get("correct_module_rank_relations", False),
                "ssa_complete": other.get("phase_truth_only", {}).get("physical_module_complete", False)}}
        print(json.dumps({"run": row["run"], "fft_nrmse": row["sensor_nrmse"]["hankel"],
                          "ssa_nrmse": other["sensor_nrmse"]["hankel"],
                          "ratio": row["matched_ssa_comparison"]["fft_over_ssa_hankel_error_ratio"]}), flush=True)
    finished = [row for row in rows if row["run_status"] == "finished"]
    payload.update(status="finished", evaluation_files_opened=True,
        source_fingerprints_unchanged=hashes() == initial,
        matched_ssa_archive_sha256=hashlib.sha256(BASELINE.read_bytes()).hexdigest(),
        total_elapsed_seconds=time.perf_counter()-started,
        summary={"requested_runs": len(manifest), "finished_runs": len(finished),
            "exceptions": len(rows)-len(finished),
            "fft_hankel_sensor_nrmse_median": float(np.median([row["sensor_nrmse"]["hankel"] for row in finished])) if finished else None,
            "matched_ssa_hankel_sensor_nrmse_median": float(np.median([row["matched_ssa_comparison"]["ssa_hankel_sensor_nrmse"] for row in finished])) if finished else None,
            "max_absolute_nrmse_difference": max(abs(row["sensor_nrmse"]["hankel"]-row["matched_ssa_comparison"]["ssa_hankel_sensor_nrmse"]) for row in finished) if finished else None,
            "all_six_same_total_raw_context": all(row["matched_ssa_comparison"]["same_raw_128_sample_context"] for row in finished),
            "discovery_module_frontend_necessity_established": False})
    save(payload)
    print(json.dumps({"output": str(OUTPUT), "summary": payload["summary"]}), flush=True)


if __name__ == "__main__":
    main()
