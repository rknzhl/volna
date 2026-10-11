"""Six fixed first-seed MSSA encoding comparisons with identical future windows.

Only public Y,s,t and the existing training-noise estimator enter inference.
All six selected rank models and encoded outputs are frozen before scoring.
Reported errors are measured at exactly the Hankel-torus comparison starts.
"""

from __future__ import annotations

from project_paths import RESULTS_DIR

import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np

from mssa_column_encoding import fit_mssa_column_encoder
from nodal_spatial import NodalSpatialModel
from run_actual_wave_iterative import noise_from_training_high_band


HERE=Path(__file__).resolve().parent
OUTPUT=RESULTS_DIR / 'mssa_column_encoding_audit.json'
DIRECTORY=RESULTS_DIR / 'mssa_column_encoding_audit'
SOURCE=RESULTS_DIR / 'general_robustness_factor_cover.json'
HANKEL_SOURCE=RESULTS_DIR / 'hankel_torus_projection.json'
WINDOW=128
SEED=202610081


def hashes():
    return {name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in
            ("mssa_column_encoding.py","run_mssa_column_encoding_audit.py","nodal_spatial.py","run_actual_wave_iterative.py")}


def save(payload):
    tmp=OUTPUT.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False))
    tmp.replace(OUTPUT)


def main():
    if OUTPUT.exists():
        raise FileExistsError("preserve previous audit output")
    DIRECTORY.mkdir(exist_ok=True)
    source=json.loads(SOURCE.read_text())
    # Comparison scores present in saved archives are never passed to fitting.
    manifest=[entry for entry in source["manifest"] if entry["seed"]==SEED]
    public_hankel_scope=[{key:row[key] for key in ("run","time_start","common_time_start_count","settings")}
                        for row in json.loads(HANKEL_SOURCE.read_text())["rows"] if row["seed"]==SEED]
    hankel_scope={row["run"]:row for row in public_hankel_scope}
    initial=hashes();started=time.perf_counter();rows=[]
    train_count=source["settings"]["n_train"]
    validation_count=source["settings"]["n_validation"]
    payload={"status":"running","rows":rows,"manifest":manifest,"source_fingerprints":initial,
             "source_public_data_manifest":str(SOURCE),"matched_hankel_comparison":str(HANKEL_SOURCE),
             "rank_selection_uses_future_or_clean_signal":False,
             "source_discovery_frequencies_or_phase_rank_given_to_MSSA":False,
             "same_window":WINDOW,"same_noise_seed":SEED,"input_scope":"public Y,s,t only",
             "truth_opening_scope":"after all six future-window encodings and rank models are frozen",
             "evaluation_files_opened":False}
    save(payload)
    for entry in manifest:
        run=entry["run"];print("starting MSSA column "+run,flush=True)
        case_start=time.perf_counter()
        try:
            path=Path(entry["input"])
            if hashlib.sha256(path.read_bytes()).hexdigest()!=entry["input_sha256"]:
                raise ValueError("public input changed")
            with np.load(path) as public:
                if set(public.files)!={"Y","s","t"}:
                    raise ValueError("unexpected input keys")
                y,positions=public["Y"],public["s"]
            spatial=NodalSpatialModel.from_positions(positions)
            training=y[:,:train_count]
            validation=y[:,train_count:train_count+validation_count]
            noise=noise_from_training_high_band(training)
            observer=fit_mssa_column_encoder(training,validation,spatial.mass_sqrt,noise["sigma"],WINDOW)
            scope=hankel_scope[run]
            if scope["settings"]["window"]!=WINDOW:
                raise ValueError("Hankel comparison uses a different window")
            future_start=train_count+validation_count
            if scope["time_start"]!=future_start:
                raise ValueError("Hankel comparison time starts differ")
            encoded=observer.encode(y[:,future_start:])
            count=scope["common_time_start_count"]
            prediction=encoded["prediction"][:,:count]
            output=DIRECTORY/(run+".npz")
            np.savez_compressed(output,prediction=prediction,mean=observer.mean,basis=observer.basis,
                                selected_rank=np.asarray(observer.selected_rank),mass_sqrt=observer.mass_sqrt,
                                singular_values=observer.singular_values,
                                time_starts=np.arange(future_start,future_start+count))
            row={"run":run,"case":entry["case"],"seed":SEED,"run_status":"fitted_and_frozen",
                 "encoder":observer.metadata(),"noise_estimate_train_only":noise,
                 "time_start":future_start,"common_time_start_count":count,
                 "frozen_output":str(output),"frozen_output_sha256":hashlib.sha256(output.read_bytes()).hexdigest(),
                 "public_input_sha256":entry["input_sha256"],"prediction_all_finite":bool(np.isfinite(prediction).all()),
                 "comparison_information_scope":"same raw observed L128 future windows; lag0 values at common starts",
                 "elapsed_seconds":time.perf_counter()-case_start}
        except Exception as error:
            row={"run":run,"case":entry["case"],"seed":SEED,"run_status":"exception",
                 "exception":str(error),"traceback":traceback.format_exc()}
        rows.append(row);save(payload)
        print(json.dumps({"run":run,"status":row["run_status"],"selected_rank":row.get("encoder",{}).get("selected_rank"),
                          "elapsed":row.get("elapsed_seconds"),"exception":row.get("exception")}),flush=True)
    payload["all_outputs_frozen_before_truth_opening"]=True
    payload["source_fingerprints_unchanged_before_scoring"]=hashes()==initial
    save(payload)
    comparison={row["run"]:row for row in json.loads(HANKEL_SOURCE.read_text())["rows"]}
    for row,entry in zip(rows,manifest):
        if row["run_status"]=="exception":
            continue
        with np.load(entry["evaluation"]) as evaluation:
            clean=evaluation["clean"][:,row["time_start"]:row["time_start"]+row["common_time_start_count"]]
        with np.load(row["frozen_output"]) as output:
            prediction=output["prediction"]
        error=float(np.linalg.norm(prediction-clean)/np.linalg.norm(clean))
        before=comparison[row["run"]]
        row.update(run_status="finished",future_sensor_nrmse=error,
            comparison_same_observed_future_windows={**before["future_sensor_nrmse"],"classical_MSSA_column_SURE":error})
        print(json.dumps({"case":row["case"],"rank":row["encoder"]["selected_rank"],"NRMSE":error}),flush=True)
    payload.update(status="finished",invocation_finished=True,evaluation_files_opened=True,
                   source_fingerprints_unchanged=hashes()==initial,total_elapsed_seconds=time.perf_counter()-started,
                   summary={"requested_runs":len(manifest),"finished_runs":sum(row["run_status"]=="finished" for row in rows),
                            "exceptions":sum(row["run_status"]=="exception" for row in rows)})
    save(payload)


if __name__=="__main__":
    main()
