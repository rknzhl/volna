"""Verify saved controls against current runs without fitting any estimator."""
import sys
sys.dont_write_bytecode = True

import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code/lag_ssa"))
from fields_first_sensor_cases import generate_variant


BASE = Path(__file__).resolve().parent
SOURCE_MAP = {row["original"]: BASE / row["archived"] for row in json.loads((BASE / "source_manifest.json").read_text())}

def read(relative):
    return json.loads(SOURCE_MAP[relative].read_text())


def digest_array(value, dtype):
    return hashlib.sha256(np.asarray(value, dtype=dtype).tobytes()).hexdigest()


def signed_permutation(matrix):
    value = np.asarray(matrix)
    return bool(value.ndim == 2 and value.shape[0] == value.shape[1]
                and np.all(np.isin(value, (-1, 0, 1)))
                and np.all(np.abs(value).sum(axis=0) == 1)
                and np.all(np.abs(value).sum(axis=1) == 1))


def verify(name, cache_name):
    current_path = f"results/fields_first_curved_noise_fix_2026_10_22/regression/{name}.json"
    baseline_path = f"results/fields_first_debug_2026_10_22/baselines/{cache_name}"
    old_path = f"results/fields_first_debug_2026_10_22/validation/{name}.json"
    current, baseline, old = read(current_path), read(baseline_path), read(old_path)
    config = {key: current[key] for key in
              ("case", "seed", "noise", "trend", "phase_mode", "rate_profile")}
    # Deterministic observations only: no field, persistence, or baseline fit.
    observations, _ = generate_variant(n=current["ntrain"] + current["ntest"],
                                       train_count=current["ntrain"], **config)
    evaluation = baseline["evaluation"]
    report = current["transform"]
    snapshot_indices = np.arange(report["raw_time_first"], report["raw_time_last"] + 1)
    window_indices = np.arange(report["raw_time_first"],
                               report["raw_time_first"] + report["retained_window_count"])
    hashes = dict(
        train_observations_sha256=digest_array(observations[:current["ntrain"]], "<f8"),
        query_observations_sha256=digest_array(observations[current["ntrain"]:], "<f8"),
        snapshot_indices_sha256=digest_array(snapshot_indices, "<i8"),
        window_indices_sha256=digest_array(window_indices, "<i8"))
    checks = {key + "_matches_saved_baseline": value == evaluation[key]
              for key, value in hashes.items()}
    checks.update(
        configuration_matches=all(current[key] == baseline["configuration"][key]
                                  for key in config),
        split_matches=all(current[key] == baseline["configuration"][key]
                          for key in ("ntrain", "ntest")),
        selected_lag_matches=current["fit"]["selected_lag"] == evaluation["window_lag"],
        preprocessing_matches_old_source=current["fit"]["preprocessing"] == old["fit"]["preprocessing"],
        current_phase_gauge_is_signed_permutation=signed_permutation(
            current["train_phase_evaluation"]["winding_candidate"]))
    current_source_checks = {filename: hashlib.sha256((ROOT / "code/lag_ssa" / filename)
                             .read_bytes()).hexdigest() == expected
                             for filename, expected in current["fit"]["provenance"].items()}
    checks["all_current_model_sources_match_run"] = all(current_source_checks.values())
    dependency_checks = {}
    for filename, expected in baseline["comparison_version"]["code_sha256"].items():
        actual = hashlib.sha256((ROOT / "code/lag_ssa" / filename).read_bytes()).hexdigest()
        dependency_checks[filename] = dict(saved_sha256=expected, current_sha256=actual,
                                           whole_file_matches=actual == expected)
    checks["generator_source_matches_saved_baseline"] = dependency_checks[
        "fields_first_sensor_cases.py"]["whole_file_matches"]
    checks["linear_baseline_source_and_evaluator_match"] = all(
        dependency_checks[name]["whole_file_matches"] for name in
        ("compare_fields_first_baselines.py", "fields_first_preprocess.py",
         "evaluate_tangent_overlap_angles.py", "tangent_overlap.py"))
    package_checks = {package: importlib.metadata.version(package) == version
                      for package, version in baseline["comparison_version"]["package_versions"].items()}
    checks["baseline_packages_match"] = all(package_checks.values())
    rows = {row["method"]: row for row in baseline["rows"]}
    checks["mssa_matches_current_saved_preprocessing_error"] = (
        rows["mssa"]["reconstruction_rmse"] == current["preprocessed_against_clean_total"]
        ["train_channel_std_normalized_rmse"])
    metrics = dict(
        current_sensor_nrmse=current["reconstruction_against_clean_total"]
        ["train_channel_std_normalized_rmse"],
        current_phase_mae_rad=current["test_phase_evaluation"]["mean_absolute_rad"],
        ssa_sensor_nrmse=rows["ssa"]["reconstruction_rmse"],
        mssa_sensor_nrmse=rows["mssa"]["reconstruction_rmse"])
    phase_gauges = {"current": current["train_phase_evaluation"]["winding_candidate"]}
    for method in ("circular", "toroidal"):
        metrics[method + "_phase_mae_rad"] = rows[method]["phase_mae_rad"]
        phase_gauges[method] = rows[method]["train_phase_evaluation"]["winding_candidate"]
        checks[method + "_phase_gauge_is_signed_permutation"] = signed_permutation(phase_gauges[method])

    audit_dir = "results/fields_first_debug_2026_10_22/phase_advantage_audit/"
    if name == "scalar_noise10":
        cdf_path = audit_dir + "circular_cdf_diagnostic.json"
        controls = next(row for row in read(cdf_path)["results"] if row["name"] == name)
        selected_controls = {"circular_cdf": controls["circular_plus_train_cdf"]}
    else:
        cdf_path = audit_dir + "multifactor_marginal_cdf_diagnostic.json"
        controls = next(row for row in read(cdf_path)["results"] if row["name"] == name)
        selected_controls = {"circular_cdf": controls["methods"]["circular_marginal_cdf"],
                             "toroidal_cdf": controls["methods"]["toroidal_marginal_cdf"]}
    checks["cdf_inputs_and_indices_match"] = all(controls[key] == value for key, value in hashes.items())
    checks["cdf_lag_matches"] = controls["lag"] == current["fit"]["selected_lag"]
    for method, row in selected_controls.items():
        metrics[method + "_phase_mae_rad"] = row["phase_mae_rad"]
        phase_gauges[method] = row["train_phase_evaluation"]["winding_candidate"]
        checks[method + "_phase_gauge_is_signed_permutation"] = signed_permutation(phase_gauges[method])
    assert all(checks.values()), (name, checks)
    return dict(name=name, current_run=current_path, saved_baseline=baseline_path,
                saved_cdf_controls=cdf_path, configuration=baseline["configuration"],
                selected_lag=current["fit"]["selected_lag"],
                snapshot_count=report["retained_snapshot_count"],
                window_count=report["retained_window_count"],
                raw_time_first=report["raw_time_first"], raw_time_last=report["raw_time_last"],
                hashes=hashes, checks=checks, current_source_checks=current_source_checks,
                baseline_dependency_checks=dependency_checks, package_checks=package_checks,
                phase_winding_matrices=phase_gauges, metrics=metrics)


if __name__ == "__main__":
    output = dict(
        scope="Cached controls on identical inputs and timestamps versus current frozen fields-first runs.",
        input_hash_provenance=("Current run JSONs store generator configuration, not original input hashes. "
            "The input hashes below are regenerated from that configuration with the unchanged "
            "generator source matching the baseline cache. Saved preprocessing configuration and "
            "its MSSA reconstruction error also agree exactly."),
        estimator_fits_performed=False, persistence_jobs_performed=False,
        baseline_dependency_policy=("Frozen controls retain their recorded dependency versions. "
            "fields_first_periods.py changed; its full-file hash is not silently relabelled as current. "
            "The comparator uses SETTINGS and the legacy _select_prefix(..., preferred_count=None) "
            "rule; added geometry-guided branches do not update the saved controls."),
        phase_metric_policy="All compared winding matrices are signed permutations; coordinate-averaged wrapped MAE is comparable after frozen TRAIN origin alignment.",
        metrics_units="Sensor NRMSE is a fraction of clean TRAIN channel standard deviation; phase MAE is radians.",
        curved_case_controls="No matching saved SSA/CC/TC baseline cache; only the current run's MSSA preprocessing error is available.",
        records=[verify("scalar_noise10", "scalar_noise10_c62b9edc.baselines.json"),
                 verify("two_waves", "two_waves_82055dfd.baselines.json")])
    destination = BASE / "baseline_verification.json"
    destination.write_text(json.dumps(output, indent=2) + "\n")
    print(destination)
    for record in output["records"]:
        print(record["name"], json.dumps(record["metrics"], sort_keys=True))
