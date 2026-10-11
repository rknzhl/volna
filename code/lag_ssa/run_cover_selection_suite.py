"""Fixed-input rerun of the 18-run suite with isolated cover-selection fixes.

The saved public inputs, seeds, spatial model, settings and truth-only scoring
are identical. No archived core file is edited. Smoke executes case4/case6 at
seed1; the full mode freezes its source and all forecasts before truth reads.
"""

from __future__ import annotations

from project_paths import RESULTS_DIR

import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback

import cover_selection_ssa
import run_general_robustness_suite as suite


HERE = Path(__file__).resolve().parent


def hashes():
    result = suite.hashes()
    for name in ("cover_selection_ssa.py", "run_cover_selection_suite.py"):
        result[name] = hashlib.sha256((HERE/name).read_bytes()).hexdigest()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    prefix = "general_robustness_cover_smoke" if args.smoke else "general_robustness_cover"
    suite.OUTPUT = RESULTS_DIR/(prefix+".json")
    suite.ROOT = RESULTS_DIR/prefix
    suite.fit_iterative_ssa = cover_selection_ssa.fit_iterative_ssa
    if suite.OUTPUT.exists():
        raise FileExistsError("preserve previous variant results")
    (suite.ROOT/"frozen").mkdir(parents=True, exist_ok=True)
    baseline_path = RESULTS_DIR / 'general_robustness_suite.json'
    baseline = json.loads(baseline_path.read_text())
    manifest = baseline["manifest"]
    if args.smoke:
        selected = {suite.CASES[3]["name"], suite.CASES[5]["name"]}
        manifest = [entry for entry in manifest if entry["case"] in selected and entry["seed"] == suite.NOISE_SEEDS[0]]
    for entry in manifest:
        if hashlib.sha256(Path(entry["input"]).read_bytes()).hexdigest() != entry["input_sha256"]:
            raise RuntimeError("public input changed")
    initial_hashes = hashes()
    started = time.perf_counter()
    rows = []
    payload = {"status": "running", "rows": rows, "manifest": manifest,
               "settings": baseline["settings"], "same_frozen_inputs_and_settings_as_baseline": True,
               "baseline_result": str(baseline_path),
               "variant_changes": ["reject an unsupported old relation rather than halt candidate search",
                   "joint-cover Hann observer reads the original field, including candidate; TRAIN-inner selection, threshold .95"],
               "core_files_edited": False, "source_fingerprints": initial_hashes,
               "input_scope": "Y,s,t public keys only", "evaluation_files_opened": False,
               "truth_opening_scope": "after every requested inference and baseline is frozen",
               "settings_change_after_truth": False, "smoke": args.smoke}
    suite.save(payload)
    for entry in manifest:
        print("starting cover variant "+entry["run"], flush=True)
        try:
            row = suite.fit_public_inputs(entry)
        except Exception as error:
            row = {"run": entry["run"], "case": entry["case"], "seed": entry["seed"],
                   "run_status": "exception", "exception": str(error), "traceback": traceback.format_exc()}
        rows.append(row)
        suite.save(payload)
        print(json.dumps({"run": entry["run"], "status": row.get("inference",{}).get("status",row["run_status"]),
                          "rank": row.get("inference",{}).get("r_candidate_conditional"),
                          "maximum_window": row.get("inference",{}).get("maximum_phase_readout_window"),
                          "elapsed": row.get("elapsed_seconds"), "exception": row.get("exception")}), flush=True)
    payload["all_fits_finished_before_truth_opening"] = True
    payload["source_fingerprints_unchanged_before_scoring"] = hashes()==initial_hashes
    suite.save(payload)
    previous = {row["run"]: row for row in baseline["rows"]}
    for row, entry in zip(rows, manifest):
        if row["run_status"] == "exception":
            continue
        suite.score_frozen_row(row, entry)
        old = previous[row["run"]]
        row["comparison_with_frozen_baseline"] = {
            "old_sensor_nrmse": old["future_field"]["future_sensor_nrmse"],
            "new_sensor_nrmse": row["future_field"]["future_sensor_nrmse"],
            "old_full_module_rank_relations": old["module_truth_only"].get("correct_module_rank_relations"),
            "new_full_module_rank_relations": row["module_truth_only"].get("correct_module_rank_relations")}
    finished = [row for row in rows if row["run_status"]=="finished"]
    scored = [row for row in finished if row["module_truth_only"]["status"]=="scored_unique_bounded_catalog"]
    payload.update(status="finished", invocation_finished=True, evaluation_files_opened=True,
        source_fingerprints_unchanged=hashes()==initial_hashes, total_elapsed_seconds=time.perf_counter()-started,
        summary={"requested_runs": len(manifest), "finished_runs": len(finished), "exceptions": len(rows)-len(finished),
                 "unique_catalog_module_scored_runs": len(scored),
                 "correct_module_rank_relations": sum(row["module_truth_only"]["correct_module_rank_relations"] for row in scored),
                 "correct_rank": sum(row["inference"]["r_candidate_conditional"]==row["physical_phase_rank_truth_only"] for row in finished)})
    suite.save(payload)
    print(json.dumps(payload["summary"]), flush=True)


if __name__ == "__main__":
    main()
