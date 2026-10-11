"""Frozen18-state audit of actual Hankel folding onto the learned torus.

The window, optimizer and linear-ablation tolerance are declared here before
scoring.  Every nonlinear, free-linear and toric-only output is frozen before
the evaluation archives are opened.  No discovery settings are changed.
"""

from project_paths import RESULTS_DIR

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np

from hankel_torus_projection import HankelTorusProjector,ProjectionSettings
from toric_phase_projection import fit_toric_phase_projector


HERE=Path(__file__).resolve().parent
SOURCE=RESULTS_DIR / 'general_robustness_factor_cover.json'
SETTINGS=ProjectionSettings()
TORIC_WEIGHT_RATIO_CAP=100.


def hashes():
    return {name:hashlib.sha256((HERE/name).read_bytes()).hexdigest()
            for name in ("hankel_torus_projection.py","run_hankel_torus_projection.py","toric_phase_projection.py")}


def save(payload,destination):
    temporary=destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload,indent=2,allow_nan=False)+"\n")
    temporary.replace(destination)


def evaluate_observations(entry,source_row,directory):
    started=time.perf_counter()
    input_path=Path(entry["input"]);source_path=Path(source_row["frozen_arrays"])
    if hashlib.sha256(input_path.read_bytes()).hexdigest()!=entry["input_sha256"]:
        raise ValueError("public input changed")
    if hashlib.sha256(source_path.read_bytes()).hexdigest()!=source_row["frozen_arrays_sha256"]:
        raise ValueError("frozen fitted model changed")
    with np.load(input_path) as public:
        if set(public.files)!={"Y","s","t"}:raise ValueError("unexpected input keys")
        observed=public["Y"]
    with np.load(source_path) as archive:
        source={key:archive[key] for key in archive.files}
    future_start=observed.shape[1]-source["prediction"].shape[1]
    future=observed[:,future_start:]
    mass=np.einsum("ij,jt->it",source["spatial_mass_sqrt"],future,optimize=False)
    seed=source["observed_factor_phase"]
    count=min(seed.shape[1],mass.shape[1]-SETTINGS.window+1)
    seed=seed[:,:count]
    observer=HankelTorusProjector(source["fit_labels"],source["fit_coefficients"],source["fit_frequencies"],SETTINGS)
    numerical_check=observer.compression_check(mass,seed[:,0]+.07*np.arange(1,seed.shape[0]+1))
    if numerical_check["cost_absolute_difference"]>1e-8 or numerical_check["analytic_gradient_max_difference_from_finite_difference"]>1e-4:
        raise ArithmeticError("exact compressed cost or gradient check failed")
    projected=observer.project(mass,seed)
    free=observer.free_linear_projection(mass,count)
    labels=np.asarray(source_row["inference"]["final_lattice"]["peak_labels"],dtype=int)
    toric=fit_toric_phase_projector(np.exp(1j*source["train_phase"]),labels,source["phase_products"],
                                  source["fit_frequencies"],TORIC_WEIGHT_RATIO_CAP)
    toric_output=toric.project(np.exp(1j*source["future_phase"][:,:count]))
    angles=np.einsum("ij,jt->it",source["fit_labels"],toric_output["phase"],optimize=False)
    features=np.column_stack((np.ones(count),np.cos(angles.T),np.sin(angles.T)))
    toric_mass=np.einsum("nj,jq->qn",features,source["fit_coefficients"],optimize=False)
    mapping=source["mass_to_sensor"]
    arrays={"hankel_phase":projected["phase"],"observed_product_phase":seed,"toric_phase":toric_output["phase"],
        "hankel_prediction":np.einsum("ij,jt->it",mapping,projected["mass_field"],optimize=False),
        "free_linear_prediction":np.einsum("ij,jt->it",mapping,free["mass_field"],optimize=False),
        "toric_prediction":np.einsum("ij,jt->it",mapping,toric_mass,optimize=False),
        "observed_product_prediction":source["observed_phase_decoder_prediction"][:,:count],
        "clock_prediction":source["prediction"][:,:count],
        "free_linear_coefficients":free["free_coefficients"],
        "time_starts":np.arange(future_start,future_start+count),
        "hankel_window_cost":np.asarray([item["final_cost"] for item in projected["diagnostics"]]),
        "seed_window_cost":np.asarray([item["seed_cost"] for item in projected["diagnostics"]]),
        "selected_start":np.asarray([item["selected_start"] for item in projected["diagnostics"]])}
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise FloatingPointError("nonfinite frozen field or phase output")
    destination=directory/(entry["run"]+".npz")
    np.savez_compressed(destination,**arrays)
    info=projected["diagnostics"]
    return {"run":entry["run"],"case":entry["case"],"seed":entry["seed"],"run_status":"fitted_and_frozen",
        "frozen_output":str(destination),"frozen_output_sha256":hashlib.sha256(destination.read_bytes()).hexdigest(),
        "source_frozen_parameters":str(source_path),"source_frozen_sha256":source_row["frozen_arrays_sha256"],
        "public_input_sha256":entry["input_sha256"],"time_start":future_start,"common_time_start_count":count,
        "settings":asdict(SETTINGS),"multistart":projected["multistart"],"compression_check":numerical_check,
        "optimizer":{"cost_exceeds_product_seed_count":sum(not item["cost_not_above_product_seed"] for item in info),
            "selected_nonzero_deck_start_count":sum(item["selected_start"]!=0 for item in info),
            "gradient_converged_count":sum(item["converged_gradient"] for item in info),
            "final_gradient_max":max(item["final_gradient_max"] for item in info),
            "iterations_median":float(np.median([item["iterations"] for item in info])),
            "iterations_max":max(item["iterations"] for item in info),
            "finite_multistart_global_optimum_certified":False},
        "free_linear":free["metadata"],"toric":toric.metadata(),"toric_optimizer":toric_output["diagnostics"],
        "future_clock_used_as_initialization_or_output":False,"source_decoder_and_discovery_settings_changed":False,
        "evaluation_scope":"reconstruction of the field at window starts from untouched observed delay windows",
        "elapsed_seconds":time.perf_counter()-started}


def score(row,entry):
    with np.load(entry["evaluation"]) as evaluation:
        clean=evaluation["clean"][:,row["time_start"]:row["time_start"]+row["common_time_start_count"]]
    with np.load(row["frozen_output"]) as output:
        scores={name:float(np.linalg.norm(output[name+"_prediction"]-clean)/np.linalg.norm(clean))
                for name in ("hankel","free_linear","toric","observed_product","clock")}
    row.update(run_status="finished",future_sensor_nrmse=scores,
               hankel_over_observed_product_error_ratio=scores["hankel"]/scores["observed_product"],
               hankel_over_free_linear_error_ratio=scores["hankel"]/scores["free_linear"])


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--smoke",action="store_true");args=parser.parse_args()
    prefix="hankel_torus_projection_smoke" if args.smoke else "hankel_torus_projection"
    directory=RESULTS_DIR/prefix;destination=RESULTS_DIR/(prefix+".json")
    if destination.exists():raise FileExistsError("preserve the existing variant")
    directory.mkdir(exist_ok=True)
    source=json.loads(SOURCE.read_text());source_rows={row["run"]:row for row in source["rows"]};manifest=source["manifest"]
    if args.smoke:
        names={"r1_travelling_infinite","r2_primitive_no_fundamentals","r3_random_smooth_no_fundamentals"}
        manifest=[entry for entry in manifest if entry["case"] in names and entry["seed"]==202610081]
    initial=hashes();rows=[];started=time.perf_counter()
    payload={"status":"running","rows":rows,"manifest":manifest,"source_fingerprints":initial,"settings":asdict(SETTINGS),
        "toric_weight_ratio_cap":TORIC_WEIGHT_RATIO_CAP,"source_parameter_archive":str(SOURCE),
        "truth_opening_scope":"after all requested nonlinear, free-linear, toric and product outputs are frozen",
        "future_clock_not_used_as_phase_seed":True,"no_discovery_or_decoder_fitting_changes":True,
        "no_frequency_orthogonality_approximation":True,"evaluation_files_opened":False,"novelty_claimed":False}
    save(payload,destination)
    for entry in manifest:
        print("starting Hankel torus "+entry["run"],flush=True)
        try:row=evaluate_observations(entry,source_rows[entry["run"]],directory)
        except Exception as error:row={"run":entry["run"],"case":entry["case"],"seed":entry["seed"],"run_status":"exception",
            "exception":f"{type(error).__name__}: {error}","traceback":traceback.format_exc()}
        rows.append(row);save(payload,destination)
        print(json.dumps({"run":entry["run"],"status":row["run_status"],"elapsed":row.get("elapsed_seconds"),
                          "multistart":row.get("multistart"),"exception":row.get("exception")}),flush=True)
    payload["all_outputs_frozen_before_truth_open"]=True;payload["source_fingerprints_unchanged_before_scoring"]=hashes()==initial
    save(payload,destination)
    for row,entry in zip(rows,manifest):
        if row["run_status"]=="exception":continue
        score(row,entry);print(json.dumps({"run":row["run"],"sensor_nrmse":row["future_sensor_nrmse"]}),flush=True)
    finished=[row for row in rows if row["run_status"]=="finished"]
    payload.update(status="finished",invocation_finished=True,evaluation_files_opened=True,source_fingerprints_unchanged=hashes()==initial,
        total_elapsed_seconds=time.perf_counter()-started,summary={"requested_runs":len(manifest),"finished_runs":len(finished),
        "exceptions":len(rows)-len(finished),"hankel_improved_vs_product":sum(row["hankel_over_observed_product_error_ratio"]<1 for row in finished),
        "hankel_improved_vs_free_linear":sum(row["hankel_over_free_linear_error_ratio"]<1 for row in finished),
        "hankel_sensor_nrmse_median":float(np.median([row["future_sensor_nrmse"]["hankel"] for row in finished])) if finished else None})
    save(payload,destination);print(json.dumps(payload["summary"]),flush=True)


if __name__=="__main__":main()
