"""Compose the frozen field decoder with measured toric phase estimates.

All projections and predictions finish before clean fields are opened.
Neither field coefficients, candidate gates nor dictionary labels change.
This is reconstruction from observed delay windows, not blind forecasting.
"""

from project_paths import RESULTS_DIR

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from toric_phase_projection import fit_toric_phase_projector
from run_actual_wave_iterative import serializable


HERE = Path(__file__).resolve().parent
SOURCE = RESULTS_DIR / 'general_robustness_factor_cover.json'
TARGET = RESULTS_DIR / 'phase_decoder_projection_audit.json'
FROZEN = RESULTS_DIR / 'phase_decoder_projection_audit'
WEIGHT_CAP = 100.


def decode(phase, labels, coefficients, mass_to_sensor):
    angles = np.einsum("mr,rn->mn",labels,phase,optimize=False).T
    design = np.column_stack((np.ones(len(angles)),np.cos(angles),np.sin(angles)))
    mass = np.einsum("nj,jq->qn",design,coefficients,optimize=False)
    return np.einsum("sq,qn->sn",mass_to_sensor,mass,optimize=False)


def main():
    if TARGET.exists():
        raise FileExistsError("preserve the existing observed-phase audit")
    source = json.loads(SOURCE.read_text())
    if source["status"]!="finished" or not source["source_fingerprints_unchanged"]:
        raise ValueError("complete source-consistent decoder assay required")
    FROZEN.mkdir(exist_ok=True)
    started = time.perf_counter()
    hashes = {name:hashlib.sha256((HERE/name).read_bytes()).hexdigest()
              for name in ("run_phase_decoder_projection_audit.py","toric_phase_projection.py")}
    rows = []
    for row in source["rows"]:
        path = Path(row["frozen_arrays"])
        if hashlib.sha256(path.read_bytes()).hexdigest()!=row["frozen_arrays_sha256"]:
            raise ValueError("frozen decoder file changed")
        lattice = row["inference"]["final_lattice"]
        labels = np.asarray(lattice["peak_labels"],dtype=int)
        products = np.asarray(lattice["products"],dtype=int)
        frequency = np.asarray(lattice["basis"],dtype=float)
        with np.load(path,allow_pickle=False) as arrays:
            train = np.exp(1j*arrays["train_phase"])
            future = np.exp(1j*arrays["future_phase"])
            projector = fit_toric_phase_projector(train,labels,products,frequency,WEIGHT_CAP)
            if labels.shape[0]==labels.shape[1]:
                # BA=I implies that every observed point belongs to this
                # full-dimensional toric image; projection cannot denoise it.
                phase = projector.product_phases(future)
                diagnostics = {"redundant_character_count":0,
                    "projection_equals_integer_product_coordinates":True}
            else:
                projected = projector.project(future)
                phase = projected["phase"]
                diagnostics = projected["diagnostics"]
            prediction = decode(phase,arrays["fit_labels"],arrays["fit_coefficients"],
                arrays["mass_to_sensor"])
            before = arrays["observed_phase_decoder_prediction"]
            clock = arrays["clock_prediction_at_observed_phase_times"]
            mass = arrays["physical_mass"]
            sample_indices = arrays["observed_phase_time_indices"]
        if not np.isfinite(prediction).all() or prediction.shape!=before.shape:
            raise ValueError("invalid frozen composition")
        frozen = FROZEN/(row["run"]+".npz")
        np.savez_compressed(frozen,projected_phase=phase,projected_field_prediction=prediction,
            product_field_prediction=before,clock_field_prediction=clock,
            physical_mass=mass,sample_indices=sample_indices)
        rows.append(dict(run=row["run"],case=row["case"],seed=row["seed"],
            input_decoder=str(path),input_decoder_sha256=row["frozen_arrays_sha256"],
            projector=projector.metadata(),projection_diagnostics=diagnostics,
            frozen_arrays=str(frozen),frozen_sha256=hashlib.sha256(frozen.read_bytes()).hexdigest()))
        print("frozen phase-decoder "+row["run"],flush=True)
    frozen_time = datetime.now(timezone.utc).isoformat()
    # All learned origins, weights, phase estimates and field reconstructions
    # are now frozen. Clean fields are first accessed below for assessment.
    manifest = {row["run"]:row for row in source["manifest"]}
    for row in rows:
        with np.load(manifest[row["run"]]["evaluation"],allow_pickle=False) as evaluation:
            clean = evaluation["clean"]
        with np.load(row["frozen_arrays"],allow_pickle=False) as arrays:
            target = clean[:,arrays["sample_indices"]]
            denominator = np.linalg.norm(target)
            mass = arrays["physical_mass"]
            mass_denominator = np.einsum("it,ij,jt->",target,mass,target,optimize=False)
            scores = {}
            for method,key in (("product_phases","product_field_prediction"),
                               ("projected_phases","projected_field_prediction"),
                               ("inferred_clock_forecast","clock_field_prediction")):
                error = arrays[key]-target
                scores[method] = dict(sensor_nrmse=float(np.linalg.norm(error)/denominator),
                    mass_nrmse=float(np.sqrt(np.einsum("it,ij,jt->",error,mass,error,
                        optimize=False)/mass_denominator)))
        row["future_field_same_window_starts"] = scores
        row["projection_over_product_nrmse_ratio"] = scores["projected_phases"]["sensor_nrmse"]/scores["product_phases"]["sensor_nrmse"]
    current = {name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in hashes}
    redundant = [row for row in rows if len(row["projector"]["labels"])>
                 len(row["projector"]["products"])]
    payload = dict(status="finished",rows=rows,source_result=str(SOURCE),
        source_result_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        source_fingerprints=hashes,source_fingerprints_unchanged=hashes==current,
        all_compositions_frozen_before_first_truth_open_utc=frozen_time,
        future_clocks_used_for_projected_phase_output=False,
        decoder_coefficients_labels_or_selection_changed=False,
        weight_ratio_cap_fixed_before_cases=WEIGHT_CAP,
        evaluation_scope="reconstruction from independent observed delay windows at the same sensors",
        global_projection_or_injectivity_certified=False,
        summary=dict(rows=len(rows),redundant_character_cases=len(redundant),
            redundant_cases_improved=sum(row["projection_over_product_nrmse_ratio"]<1 for row in redundant),
            redundant_median_nrmse_ratio=float(np.median([
                row["projection_over_product_nrmse_ratio"] for row in redundant]))),
        elapsed_seconds=time.perf_counter()-started)
    TARGET.write_text(json.dumps(serializable(payload),ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    print(json.dumps(payload["summary"]),flush=True)


if __name__=="__main__":
    main()
