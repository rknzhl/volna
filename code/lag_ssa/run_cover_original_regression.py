"""Regression gate of isolated cover selection on original11 waves+5 fixtures.

All16 phase-factor models and future forecasts are fitted/frozen before the
existing evaluation drivers first open scoring truth. Evaluation reuses those
cached models after checking the exact public train/validation inputs.
Archive/core/final sources remain unchanged and the final settings are fixed.
"""

from __future__ import annotations

from project_paths import RESULTS_DIR

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np

import cover_selection_ssa as cover
import final_phase_factor_assay as final
from iterative_ssa import Settings
from nodal_spatial import NodalSpatialModel
import run_actual_wave_iterative as wave
import run_actual_wave_revised as revised


HERE = Path(__file__).resolve().parent
OUTPUT = RESULTS_DIR / 'cover_original_regression.json'


def hashes():
    result = final.fingerprints()
    for name in ("cover_selection_ssa.py", "run_cover_original_regression.py"):
        result[name] = hashlib.sha256((HERE/name).read_bytes()).hexdigest()
    return result


def train_key(train, validation):
    digest = hashlib.sha256()
    digest.update(train.tobytes())
    digest.update(validation.tobytes())
    return digest.hexdigest()


def save(payload):
    wave.save(payload, OUTPUT)


def main():
    if OUTPUT.exists():
        raise FileExistsError("preserve prior regression output")
    start = time.perf_counter()
    initial_hashes = hashes()
    cache, frozen = {}, []
    rows = []
    payload = {"status": "fitting", "rows": rows, "frozen_models": frozen,
               "source_fingerprints": initial_hashes, "input_scope": "Y,s,t only",
               "case_scope": "all11 original wave scenarios and all5 final fixtures",
               "settings_rule": "same frozen final_phase_factor_assay.fixed_settings",
               "phase_factor_truth_opening_scope": "after all16 models and saved future forecasts are frozen",
               "core_or_final_sources_edited": False, "between_case_truth_adaptation": False}
    save(payload)
    fixture_settings = final.fixed_settings(Settings(n_train=3200, n_validation=1600, n_future=1600,
        lag_window=128, frontend_mode="dynamic", dynamic_ranks=(4,8,16),
        joint_frequency_mode="harmonic_nuisance", phase_metric_mode="field",
        temporal_gate_mode="iid_reference", k_candidates=(16,32)))
    inputs = []
    for name in wave.CASES:
        path = wave.DATA/"inputs"/(name+".npz")
        with np.load(path) as public:
            if set(public.files)!={"Y","s","t"}:
                raise ValueError("unexpected wave input key")
            y, positions = public["Y"], public["s"]
        inputs.append(("wave",name,y,positions,final.fixed_settings(final.ORIGINAL_SETTINGS(y.shape[1]))))
    for name in revised.FIXTURES:
        y, positions, _, _ = revised.fixture_inputs(name)
        inputs.append(("fixture",name,y,positions,fixture_settings))
    for kind,name,y,positions,settings in inputs:
        print("freezing cover regression "+name,flush=True)
        cutoff = settings.n_train+settings.n_validation
        train, validation = y[:,:settings.n_train],y[:,settings.n_train:cutoff]
        spatial = NodalSpatialModel.from_positions(positions)
        noise = wave.noise_from_training_high_band(train)
        try:
            fitted = cover.fit_iterative_ssa(train, validation, settings, sampling_model=spatial, noise_estimate=noise)
            prediction = cover.predict_field(fitted["clock_models"][-1],np.arange(cutoff,y.shape[1]),positions,spatial)
            if not np.isfinite(prediction).all():
                raise FloatingPointError("nonfinite frozen prediction")
            key = train_key(train,validation)
            cache[key] = (asdict(settings),fitted)
            path = RESULTS_DIR/("cover_original_regression_"+name+"_frozen.npz")
            np.savez_compressed(path,prediction=prediction,acquired_frequencies=fitted["acquired_frequencies"])
            frozen.append({"case":name,"kind":kind,"settings":asdict(settings),"input_key":key,
                           "prediction":str(path),"prediction_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                           "status":fitted["audit"]["status"],"rank":fitted["audit"]["r_candidate_conditional"]})
        except Exception as error:
            frozen.append({"case":name,"kind":kind,"exception":str(error),"traceback":traceback.format_exc()})
        save(payload)
    payload.update(status="scoring",all_phase_factor_fits_finished_before_truth_opening=True,
                   source_fingerprints_unchanged_before_scoring=hashes()==initial_hashes)
    save(payload)

    def frozen_fit(train,validation,settings,sampling_model=None,noise_estimate=None):
        key = train_key(train,validation)
        declared,fitted = cache[key]
        if asdict(settings)!=declared:
            raise RuntimeError("regression scoring attempted to change frozen settings")
        return fitted

    wave.fit_iterative_ssa = frozen_fit
    revised.fit_iterative_ssa = frozen_fit
    wave.settings_for_samples = lambda n: final.fixed_settings(final.ORIGINAL_SETTINGS(n))
    for kind,name,y,positions,settings in inputs:
        print("scoring frozen cover regression "+name,flush=True)
        try:
            if kind=="wave":
                row=wave.evaluate(name,wave.DATA,OUTPUT,additional_baselines=final.additional_baselines)
            else:
                row=revised.evaluate_fixture(name,OUTPUT,"observed",settings_override=final.fixed_settings,
                                              additional_baselines=final.additional_baselines)
            row["kind"]=kind
            path=RESULTS_DIR/("cover_original_regression_"+name+"_frozen.npz")
            coordinate_path=RESULTS_DIR/(OUTPUT.stem+"_"+name+"_coordinates.npz")
            with np.load(path) as frozen_archive,np.load(coordinate_path) as scored_archive:
                key="hierarchy_future_sensor_prediction" if kind=="wave" else "hierarchy_future_prediction"
                row["scoring_prediction_equals_pre_truth_frozen_forecast"]=bool(np.array_equal(frozen_archive["prediction"],scored_archive[key]))
        except Exception as error:
            row={"case":name,"kind":kind,"run_status":"exception","exception":str(error),"traceback":traceback.format_exc()}
        rows.append(row)
        save(payload)
        print(json.dumps({"case":name,"status":row.get("inference",{}).get("status",row["run_status"]),
                          "rank":row.get("inference",{}).get("r_candidate_conditional"),"exception":row.get("exception")}),flush=True)
    original_waves={row["case"]:row for row in json.loads((RESULTS_DIR / 'final_phase_factor_assay.json').read_text())["rows"]}
    original_fixtures={row["case"]:row for row in json.loads((RESULTS_DIR / 'final_phase_factor_assay_fixtures.json').read_text())["rows"]}
    for row in rows:
        if row["run_status"]=="exception":
            continue
        prior=(original_waves if row["kind"]=="wave" else original_fixtures)[row["case"]]
        key="iterative_hierarchy" if row["kind"]=="wave" else "revised_phase_family"
        row["comparison_with_saved_final"]={"old_rank":prior["inference"]["r_candidate_conditional"],
            "new_rank":row["inference"]["r_candidate_conditional"],
            "old_sensor_nrmse":prior["future_clean_sensor_field"][key]["sensor_nrmse"],
            "new_sensor_nrmse":row["future_clean_sensor_field"][key]["sensor_nrmse"],
            "old_full_module_rank_relations":prior.get("correct_observable_module_rank_relation_truth_only"),
            "new_full_module_rank_relations":row.get("correct_observable_module_rank_relation_truth_only")}
    payload.update(status="finished",invocation_finished=True,source_fingerprints_unchanged=hashes()==initial_hashes,
                   total_elapsed_seconds=time.perf_counter()-start,
                   summary={"requested_cases":16,"finished_cases":sum(row["run_status"]=="finished" for row in rows),
                            "exceptions":sum(row["run_status"]=="exception" for row in rows),
                            "same_ranks_as_saved_final":sum(row.get("comparison_with_saved_final",{}).get("old_rank")
                                ==row.get("comparison_with_saved_final",{}).get("new_rank") for row in rows if row["run_status"]=="finished"),
                            "all_scored_forecasts_equal_pre_truth_frozen_predictions":all(row.get("scoring_prediction_equals_pre_truth_frozen_forecast",False)
                                for row in rows if row["run_status"]=="finished")})
    save(payload)
    print(json.dumps(payload["summary"]),flush=True)


if __name__=="__main__":
    main()
