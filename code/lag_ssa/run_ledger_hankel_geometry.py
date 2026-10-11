"""Read-only geometry diagnostics for every frozen ledger group model.

Default prepares the finite protocol; --run checks its source/model hashes.
Only inference archives are loaded. No public observations, future values,
evaluation archive or physical frequency catalogue is loaded or generated.
"""

from project_paths import RESULTS_DIR

from dataclasses import asdict
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from ledger_hankel_geometry import (MATH_REFERENCE, SETTINGS, diagnose_model,
                                    lattice_pair_proposals)


HERE = Path(__file__).resolve().parent
SOURCE = RESULTS_DIR / 'ledger_group_selection.json'
OUTPUT = RESULTS_DIR / 'ledger_hankel_geometry.json'
PROTOCOL_PATH = RESULTS_DIR / 'ledger_hankel_geometry_protocol.json'
SOURCES = ("ledger_hankel_geometry.py", "run_ledger_hankel_geometry.py",
           "torus_lattice_reach_witness.py")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, payload):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def manifest():
    source = json.loads(SOURCE.read_text())
    if (source["status"] != "finished_inference_only"
            or not source["all_six_attempts_and_outputs_frozen_before_truth"]
            or source["evaluation_files_opened"] or len(source["rows"]) != 6):
        raise ValueError("six completed immutable inference-only group assays required")
    for name, expected in source["source_fingerprints"].items():
        if digest(HERE/name) != expected:
            raise ValueError("changed frozen inference source: "+name)
    models = []
    for row in source["rows"]:
        if row["run_status"] != "fitted_and_frozen" or row["evaluation_file_opened"]:
            raise ValueError("frozen inference-only model row required")
        for model in row["model_files"]:
            if digest(model["parameters"]) != model["parameters_sha256"]:
                raise ValueError("changed frozen group model archive")
            models.append(dict(run=row["run"], model_index=model["model_index"],
                model_kind=model["model_kind"], selected=model["model_index"] == row["selected_model_index"],
                parameters=model["parameters"], parameters_sha256=model["parameters_sha256"],
                presentation=model["presentation"], noise_estimate_train_only=row["noise_estimate_train_only"]))
    return models, source["source_fingerprints"]


def protocol():
    models, old_fingerprints = manifest()
    return dict(status="prepared_not_run", created_at=datetime.now(timezone.utc).isoformat(),
        source=str(SOURCE), source_sha256=digest(SOURCE),
        source_fingerprints={name:digest(HERE/name) for name in SOURCES},
        frozen_inference_source_fingerprints=old_fingerprints,
        settings=asdict(SETTINGS), math_reference=MATH_REFERENCE,
        models=models, model_count=len(models), all_selected_baselines_and_admissible_models_included=True,
        source_math_review_required_before_run=True,
        input_scope="frozen public-inference fitted frequencies/profiles/Ms/SNF and TRAIN-only estimated raw noise",
        evaluation_files_opened=False, true_rank_or_physical_catalogue_loaded=False,
        model_fitting_or_selection=False, future_observations_loaded=False,
        finite_window_algebra="all 2J real bank columns kept; exact full finite-window Gram in SVD coordinates",
        projected_noise="conditional estimated-sigma Gaussian one-window radius at alpha=.01",
        model_tail_error="e0=0 optimistic learned-template-only assumption",
        geometry_scope="sampled/curvature upper witnesses only; no reach lower bound or positive acceptance",
        integer_scope="exact modular SNF and projected-lattice displacements before bounded floating angles",
        arithmetic_scope="numerical tangent/SVD conditional; no interval or exact floating certificate",
        interpretation="can only reject the declared sufficient reach>noise test for a learned smooth embedded template",
        general_phase_recovery_necessity_or_true_group_certificate=False, novelty_claimed=False)


def validate_prepared(prepared):
    current = protocol()
    keys = ("source", "source_sha256", "source_fingerprints", "frozen_inference_source_fingerprints",
            "settings", "math_reference", "models", "model_count")
    if any(prepared[key] != current[key] for key in keys):
        raise ValueError("source/model/settings differ from prepared protocol")


def diagnose(entry):
    if digest(entry["parameters"]) != entry["parameters_sha256"]:
        raise ValueError("changed model archive")
    with np.load(entry["parameters"], allow_pickle=False) as archive:
        parameters = {name:archive[name] for name in archive.files}
    pairs, proposal_audit = lattice_pair_proposals(entry["presentation"], SETTINGS)
    result = diagnose_model(parameters, entry["presentation"],
                            float(entry["noise_estimate_train_only"]["sigma"]), SETTINGS, pairs)
    result["methods"]["exact_projected_lattice"] = proposal_audit
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    arguments = parser.parse_args()
    if not arguments.run:
        prepared = protocol()
        if OUTPUT.exists() or PROTOCOL_PATH.exists():
            raise FileExistsError("preserve existing geometry protocol/results")
        write_json(PROTOCOL_PATH, prepared)
        print(json.dumps(dict(status=prepared["status"], model_count=prepared["model_count"],
                              protocol=str(PROTOCOL_PATH), source_fingerprints=prepared["source_fingerprints"])))
        return
    prepared = json.loads(PROTOCOL_PATH.read_text())
    validate_prepared(prepared)
    if OUTPUT.exists():
        raise FileExistsError("preserve existing geometry results")
    started = time.perf_counter()
    output = dict(prepared, status="running_read_only_geometry", protocol_file=str(PROTOCOL_PATH),
                  protocol_sha256=digest(PROTOCOL_PATH), rows=[])
    write_json(OUTPUT, output)
    for index, entry in enumerate(prepared["models"]):
        tick = time.perf_counter()
        row = {key:entry[key] for key in ("run", "model_index", "model_kind", "selected",
                                         "parameters", "parameters_sha256")}
        row.update(continuous_dimension=entry["presentation"]["continuous_dimension"],
                   component_count=str(entry["presentation"]["component_count"]))
        try:
            row.update(status="diagnosed", diagnostic=diagnose(entry))
        except Exception as error:
            row.update(status="diagnostic_unavailable", exception_type=type(error).__name__,
                       exception=str(error), failure_does_not_reject_or_accept=True)
        row["elapsed_seconds"] = time.perf_counter()-tick
        output["rows"].append(row)
        write_json(OUTPUT, output)
        diagnostic = row.get("diagnostic", {})
        print(json.dumps(dict(completed=index+1, total=prepared["model_count"], run=entry["run"],
            model_index=entry["model_index"], selected=entry["selected"], status=row["status"],
            geometry_status=diagnostic.get("geometry_status"),
            ratio=diagnostic.get("reach_upper_to_projected_noise_ratio"),
            elapsed_seconds=row["elapsed_seconds"])), flush=True)
    validate_prepared(prepared)
    output.update(status="finished_read_only_geometry", source_and_model_hashes_unchanged=True,
                  finished_at=datetime.now(timezone.utc).isoformat(), total_elapsed_seconds=time.perf_counter()-started)
    diagnosed = [row for row in output["rows"] if row["status"] == "diagnosed"]
    output["summary"] = dict(models_total=len(output["rows"]), diagnosed=len(diagnosed),
        unavailable=len(output["rows"])-len(diagnosed),
        conditional_model_only_rejections=sum(row["diagnostic"]["model_only_noise_scale_rejection_numerical"] for row in diagnosed),
        selected_model_only_rejections=sum(row["selected"] and row["diagnostic"]["model_only_noise_scale_rejection_numerical"] for row in diagnosed),
        positive_acceptance_certificates=0, true_group_or_rank_certificates=0)
    write_json(OUTPUT, output)
    print(json.dumps(dict(status=output["status"], summary=output["summary"], output=str(OUTPUT))), flush=True)


if __name__ == "__main__":
    main()
