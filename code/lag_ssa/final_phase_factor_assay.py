"""Frozen law-blind phase-factor assay with declared readout/sieve budgets.

Profiles and observer windows use training only.  Phase relations retain the
.95 held-out gate.  Comparators include MSSA at128 and at the maximum returned
phase-observer window, plus a direct fit at exactly the learned dictionary.
Generator truth is first opened in the underlying drivers after all fits.
"""

from __future__ import annotations

from project_paths import RESULTS_DIR

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np

import run_actual_wave_iterative as wave_driver
import run_actual_wave_revised as revised
from iterative_ssa import predict_field
from phase_factor_regression import refit_factor_dictionary


HERE = Path(__file__).resolve().parent
OUTPUT = RESULTS_DIR / 'final_phase_factor_assay.json'
FIXTURE_OUTPUT = RESULTS_DIR / 'final_phase_factor_assay_fixtures.json'
ORIGINAL_SETTINGS = wave_driver.settings_for_samples


def fixed_settings(settings):
    return replace(settings, conditional_model_mode="fourier_sieve",
                   enforce_validation_improvement=True, phase_relation_mode="observed",
                   phase_readout_refinement="hann_inner", fourier_sieve_mode="expanded",
                   max_steps=16, max_phase_rank=4, observer_windows=(128, 256, 512))


def fingerprints():
    result = revised.fingerprints()
    for name in ("final_phase_factor_assay.py", "phase_readout_refinement.py"):
        result[name] = hashlib.sha256((HERE / name).read_bytes()).hexdigest()
    return result


def learned_dictionary(fitted):
    """Express the union of historical decoder characters in the final basis."""
    lattice = fitted["audit"]["final_lattice"]
    if lattice is None:
        raise ValueError("a supported inferred lattice is required for the direct dictionary baseline")
    final_labels = np.asarray(lattice["peak_labels"], dtype=int)
    hierarchy = fitted["clock_models"][-1]
    union = set()
    for component, products in zip(hierarchy.components[1:], hierarchy.phase_products):
        old_count = products.shape[1]
        mapped = np.einsum("ij,jk->ik", component.labels, products, optimize=False)
        mapped = np.einsum("ij,jk->ik", mapped, final_labels[:old_count], optimize=False)
        for row in mapped:
            if np.any(row):
                if next(value for value in row if value) < 0:
                    row = -row
                union.add(tuple(int(value) for value in row))
    return np.asarray(sorted(union), dtype=int).reshape(-1, lattice["rank"])


def additional_baselines(fitted, training, validation, future_times, spatial, settings):
    result = {}
    max_window = int(fitted["audit"]["maximum_phase_readout_window"])
    windows = tuple(dict.fromkeys((128, max_window)))
    for window in windows:
        if window >= training.shape[1]-2:
            continue
        recurrent = revised.recurrent_mssa(training, validation, len(future_times), window)
        result["mssa_window"+str(window)] = recurrent
    if max_window in windows:
        result["mssa_matched_returned_observer_window"] = dict(result["mssa_window"+str(max_window)],
            matched_returned_phase_window=max_window)
    labels = learned_dictionary(fitted)
    frequencies = np.asarray(fitted["audit"]["final_lattice"]["basis"])
    direct = refit_factor_dictionary(np.arange(settings.n_train), fitted["mass_train"], frequencies, labels)
    result["direct_fourier_same_learned_dictionary"] = {
        "prediction": predict_field(direct, future_times, spatial.positions, sampling_model=spatial),
        "dictionary_fit": direct.metadata(),
        "comparison_scope": "same SSA-learned frequencies and decoder character dictionary; direct amplitude refit",
        "phase_readout_record": [{"window": front["window"],
                                  "readout_source": front.get("readout_source", "residual"),
                                  "method": front.get("phase_readout_method", "SSA frozen linear readout")}
                                 for front in fitted["frozen_frontends"]],
        "SSA_role": "discover candidate field characters/frequencies; measured frozen phase readout unless refinement needed",
        "true_frequency_or_phase_rank_supplied": False}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", choices=wave_driver.CASES, default=list(wave_driver.CASES))
    parser.add_argument("--fixtures", nargs="+", choices=revised.FIXTURES, default=list(revised.FIXTURES))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--fixture-output", type=Path, default=FIXTURE_OUTPUT)
    parser.add_argument("--skip-waves", action="store_true")
    args = parser.parse_args()
    if args.output.exists() or args.fixture_output.exists():
        parser.error("a destination already exists; preserve it and use another output path")
    wave_driver.settings_for_samples = lambda n: fixed_settings(ORIGINAL_SETTINGS(n))
    hashes = fingerprints()
    started = time.perf_counter()
    scope = {"source_fingerprints": hashes, "method_settings_rule": "expanded finite sieve; observed relation gate.95; optional train-inner Hann observer;16 character/enrichment attempts; phase rank cap4",
             "input_scope": "Y,s,t only; geometry-only nodal spatial model",
             "truth_for_scoring_only": True, "between_case_adaptation_from_truth": False,
             "future_not_used_for_selection": True,
             "source_equation_or_generator_frequencies_used_in_inference": False}
    wave_rows = []
    wave_payload = dict(scope, status="running", rows=wave_rows)
    wave_driver.save(wave_payload, args.output)
    for name in (() if args.skip_waves else args.cases):
        print("starting final wave " + name, flush=True)
        try:
            row = wave_driver.evaluate(name, wave_driver.DATA, args.output, additional_baselines=additional_baselines)
        except Exception as error:
            row = {"case": name, "run_status": "exception", "exception": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()}
        wave_rows.append(row)
        wave_driver.save(wave_payload, args.output)
        print(json.dumps({"case": name, "run_status": row["run_status"], "phase_status": row.get("inference", {}).get("status"),
                          "maximum_window": row.get("inference", {}).get("maximum_phase_readout_window"),
                          "field": row.get("future_clean_sensor_field"), "exception": row.get("exception")}), flush=True)
    wave_payload.update(status="finished", invocation_finished=True,
                        summary=wave_driver.summary(wave_rows), source_fingerprints_unchanged=fingerprints()==hashes)
    wave_driver.save(wave_payload, args.output)
    fixture_rows = []
    fixture_payload = dict(scope, status="running", rows=fixture_rows)
    wave_driver.save(fixture_payload, args.fixture_output)
    for name in args.fixtures:
        print("starting final fixture " + name, flush=True)
        try:
            row = revised.evaluate_fixture(name, args.fixture_output, "observed", settings_override=fixed_settings,
                                           additional_baselines=additional_baselines)
        except Exception as error:
            row = {"case": name, "run_status": "exception", "exception": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()}
        fixture_rows.append(row)
        wave_driver.save(fixture_payload, args.fixture_output)
        print(json.dumps({"fixture": name, "run_status": row["run_status"], "phase_status": row.get("inference", {}).get("status"),
                          "rank": row.get("inference", {}).get("r_candidate_conditional"),
                          "maximum_window": row.get("inference", {}).get("maximum_phase_readout_window"),
                          "field": row.get("future_clean_sensor_field"), "exception": row.get("exception")}), flush=True)
    fixture_payload.update(status="finished", invocation_finished=True, source_fingerprints_unchanged=fingerprints()==hashes,
                           invocation_runtime_seconds=time.perf_counter()-started)
    wave_driver.save(fixture_payload, args.fixture_output)
    print(json.dumps({"wave_output": str(args.output), "fixture_output": str(args.fixture_output),
                      "wave_summary": wave_payload["summary"], "source_fingerprints_unchanged": fingerprints()==hashes}), flush=True)


if __name__ == "__main__":
    main()
