"""Same18 public inputs: verified cover discovery plus one global factor fit.

Freeze both clock forecasts and Fhat evaluated at measured SSA/Hann phases
before any truth file is opened.  Phase origins use training readouts only.
"""

from project_paths import RESULTS_DIR

import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np

import cover_selection_ssa as cover
import run_general_robustness_suite as suite
from factor_ssa import fit_factor_ssa


HERE = Path(__file__).resolve().parent
LAST_FIT = None


def fit_combined(*args, **kwargs):
    global LAST_FIT
    LAST_FIT = fit_factor_ssa(*args, **kwargs, discovery_module=cover)
    return LAST_FIT


def hashes():
    result = suite.hashes()
    for name in ("cover_selection_ssa.py", "factor_ssa.py", "run_factor_cover_suite.py"):
        result[name] = hashlib.sha256((HERE/name).read_bytes()).hexdigest()
    return result


def freeze_phase_decoder(row, fitted):
    path = Path(row["frozen_arrays"])
    with np.load(path) as archive:
        arrays = {key: archive[key] for key in archive.files}
    fronts = fitted["frozen_frontends"]
    acquired = np.asarray(fitted["acquired_frequencies"])
    origins = np.asarray([np.angle(np.mean(np.exp(1j*(front["phase"]-frequency*np.arange(len(front["phase"]))))))
                          for front, frequency in zip(fronts, acquired)])
    lattice = fitted["audit"]["final_lattice"]
    if lattice is None:
        raise ValueError("observed phase decoder requires a supported current lattice")
    products = np.asarray(lattice["products"], dtype=int)
    measured_phase = np.einsum("ij,jt->it", products,
        arrays["future_phase"]-origins[:, None], optimize=False)
    model = fitted["clock_models"][-1]
    mass_prediction = model.predict_on_phase(measured_phase)
    spatial = fitted["sampling_model"]
    mass_to_sensor = np.linalg.solve(spatial.mass_sqrt.T, spatial.design.T).T
    sensor_prediction = np.einsum("ij,jt->it", mass_to_sensor, mass_prediction, optimize=False)
    count = sensor_prediction.shape[1]
    arrays.update(observed_phase_decoder_prediction=sensor_prediction,
        clock_prediction_at_observed_phase_times=arrays["prediction"][:, :count],
        observed_factor_phase=measured_phase,
        phase_origin_train_only=origins, phase_products=products,
        acquired_frequencies=acquired, fit_frequencies=model.fit.frequencies,
        fit_labels=model.fit.labels, fit_coefficients=model.fit.coefficients,
        spatial_mass_sqrt=spatial.mass_sqrt, mass_to_sensor=mass_to_sensor,
        readout_windows=np.asarray([front["window"] for front in fronts]),
        readout_sources=np.asarray([front.get("readout_source", "residual") for front in fronts]),
        observed_phase_time_indices=np.arange(suite.TRAIN+suite.VALIDATION,
                                             suite.TRAIN+suite.VALIDATION+count))
    if not all(np.isfinite(value).all() for key,value in arrays.items()
               if np.asarray(value).dtype.kind in "fc"):
        raise FloatingPointError("nonfinite observed-phase decoder output")
    np.savez_compressed(path, **arrays)
    row["frozen_arrays_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    row["phase_decoder_scope"] = {
        "phase_source": "actual held-out SSA/Hann readouts; no clock phases substituted",
        "phase_origins": "training-only mean of readout phase minus acquired frequency times sample index",
        "phase_products": products.tolist(), "future_common_columns": count,
        "input_window_reconstruction_not_blind_forecast": True,
        "all_parameters_and_phase_outputs_frozen_before_truth_open": True}


def score_phase_decoder(row, entry):
    with np.load(entry["evaluation"]) as truth:
        clean = truth["clean"][:, suite.TRAIN+suite.VALIDATION:]
    with np.load(row["frozen_arrays"]) as output:
        prediction = output["observed_phase_decoder_prediction"]
        clock = output["clock_prediction_at_observed_phase_times"]
        mass = output["physical_mass"]
    target = clean[:, :prediction.shape[1]]
    denom = float(np.sum(target**2))
    mass_denom = float(np.einsum("it,ij,jt->",target,mass,target,optimize=False))
    error = prediction-target
    row["observed_phase_decoder_future_field"] = {
        "sensor_nrmse": float(np.sqrt(np.sum(error**2)/denom)),
        "mass_nrmse": float(np.sqrt(np.einsum("it,ij,jt->",error,mass,error,optimize=False)/mass_denom)),
        "clock_forecast_sensor_nrmse_on_identical_time_starts": float(np.sqrt(np.sum((clock-target)**2)/denom)),
        "same_time_start_count": prediction.shape[1],
        "evaluation_scope": "state reconstruction from untouched measured delay windows at existing sensors"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    prefix = "general_robustness_factor_cover_smoke" if args.smoke else "general_robustness_factor_cover"
    suite.OUTPUT = RESULTS_DIR/(prefix+".json")
    suite.ROOT = RESULTS_DIR/prefix
    suite.fit_iterative_ssa = fit_combined
    if suite.OUTPUT.exists():
        raise FileExistsError("preserve existing variant outputs")
    (suite.ROOT/"frozen").mkdir(parents=True,exist_ok=True)
    original_path = RESULTS_DIR / 'general_robustness_suite.json'
    original = json.loads(original_path.read_text())
    previous = {row["run"]:row for row in json.loads((RESULTS_DIR / 'general_robustness_cover.json').read_text())["rows"]}
    manifest = original["manifest"]
    if args.smoke:
        names = {suite.CASES[3]["name"],suite.CASES[5]["name"]}
        manifest = [entry for entry in manifest if entry["case"] in names and entry["seed"]==suite.NOISE_SEEDS[0]]
    for entry in manifest:
        if hashlib.sha256(Path(entry["input"]).read_bytes()).hexdigest()!=entry["input_sha256"]:
            raise RuntimeError("public input changed")
    initial = hashes()
    rows = []
    started=time.perf_counter()
    payload={"status":"running","rows":rows,"manifest":manifest,"source_fingerprints":initial,
        "settings":original["settings"],"same_inputs_settings_and_gates_as_verified_cover_suite":True,
        "method":"observed-cover discovery plus one global field refit on the selected character union",
        "phase_composition_scope":"Fhat(observedphase) frozen alongside Fhat(inferredclock)",
        "truth_opening_scope":"after all requested fits, baselines, readouts and both field predictions are frozen",
        "smoke":args.smoke,"evaluation_files_opened":False}
    suite.save(payload)
    for entry in manifest:
        print("starting combined "+entry["run"],flush=True)
        try:
            row=suite.fit_public_inputs(entry)
            freeze_phase_decoder(row,LAST_FIT)
        except Exception as error:
            row={"run":entry["run"],"case":entry["case"],"seed":entry["seed"],"run_status":"exception",
                "exception":f"{type(error).__name__}: {error}","traceback":traceback.format_exc()}
        rows.append(row);suite.save(payload)
        print(json.dumps({"run":entry["run"],"status":row.get("inference",{}).get("status",row["run_status"]),
            "rank":row.get("inference",{}).get("r_candidate_conditional"),"exception":row.get("exception")}),flush=True)
    payload["all_fits_and_phase_compositions_frozen_before_truth_open"]=True
    payload["source_fingerprints_unchanged_before_scoring"]=hashes()==initial
    suite.save(payload)
    for row,entry in zip(rows,manifest):
        if row["run_status"]=="exception":continue
        suite.score_frozen_row(row,entry)
        score_phase_decoder(row,entry)
        old=previous[row["run"]]
        row["comparison_with_verified_hierarchy_cover"]={
            "old_sensor_nrmse":old["future_field"]["future_sensor_nrmse"],
            "global_sensor_nrmse":row["future_field"]["future_sensor_nrmse"],
            "old_full_module_rank_relations":old["module_truth_only"].get("correct_module_rank_relations"),
            "global_full_module_rank_relations":row["module_truth_only"].get("correct_module_rank_relations")}
    finished=[row for row in rows if row["run_status"]=="finished"]
    payload.update(status="finished",invocation_finished=True,evaluation_files_opened=True,
        source_fingerprints_unchanged=hashes()==initial,total_elapsed_seconds=time.perf_counter()-started,
        summary={"requested_runs":len(manifest),"finished_runs":len(finished),"exceptions":len(rows)-len(finished),
            "correct_module_rank_relations":sum(row["module_truth_only"].get("correct_module_rank_relations",False) for row in finished),
            "correct_rank":sum(row["inference"]["r_candidate_conditional"]==row["physical_phase_rank_truth_only"] for row in finished),
            "observed_phase_decoder_sensor_nrmse_median":float(np.median([row["observed_phase_decoder_future_field"]["sensor_nrmse"] for row in finished])) if finished else None})
    suite.save(payload)
    print(json.dumps(payload["summary"]),flush=True)


if __name__=="__main__":main()
