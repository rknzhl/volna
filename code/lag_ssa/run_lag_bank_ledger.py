"""Isolated six-case raw-ledger replay with a fixed temporal lag-bank gate.

Prepare the protocol by default. Fit only with --run after root/closure source
and mathematical review. All old sources/results remain immutable; only the
raw ledger's process-local proposal frontend is replaced inside try/finally.
No evaluator, old factor parameters or analytic generator is loaded here.
"""

from project_paths import RESULTS_DIR

from dataclasses import asdict
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import time
import traceback

import lag_bank_ledger as lag_bank
import raw_spectral_ledger as raw
import run_raw_spectral_ledger as base


HERE = Path(__file__).resolve().parent
ROOT = RESULTS_DIR / 'lag_bank_ledger'
OUTPUT = ROOT.with_suffix(".json")
PROTOCOL_PATH = RESULTS_DIR / 'lag_bank_ledger_protocol.json'
ORIGINAL_LEDGER = RESULTS_DIR / 'raw_spectral_ledger.json'
SOURCES = tuple(dict.fromkeys(base.SOURCES + (
    "run_raw_spectral_ledger.py", "lag_bank_ledger.py", "run_lag_bank_ledger.py", "spatial_field.py")))


def fingerprints():
    return {name: base.digest(HERE / name) for name in SOURCES}


def protocol():
    declaration = base.protocol()
    declaration.update(source_fingerprints=fingerprints(),
        temporal_reference_variant="fixed iid covariance lag bank",
        temporal_reference_lags=list(lag_bank.TEMPORAL_LAGS),
        temporal_reference_family_alpha=lag_bank.TEMPORAL_FAMILY_ALPHA,
        temporal_reference_each_lag_alpha=lag_bank.TEMPORAL_FAMILY_ALPHA / len(lag_bank.TEMPORAL_LAGS),
        temporal_reference_logarithm="log(2*K/alpha), K=4, alpha=1e-4",
        temporal_reference_decision="ANY absolute centered weighted lag covariance exceeds its declared noise reference",
        original_raw_ledger_result=str(ORIGINAL_LEDGER),
        original_raw_ledger_result_sha256=base.digest(ORIGINAL_LEDGER),
        old_ledger_results_read_into_fitting=False,
        only_algorithm_change="replace global lag1 iid temporal reference by family-adjusted lags1..4; all other SSA/DMD proposal arithmetic/gates and free-line fitting unchanged",
        unchanged_raw_free_line_settings=asdict(raw.SETTINGS),
        unconditional_iid_noise_reference_is_learned_residual_calibration=False,
        universal_harmonic_signal_detector_claimed=False,
        original_candidate_rank_mode_order_retained=True,
        source_frontend_mode="dynamic fixed ranks(4,8,16), direct static extractor copy",
        runtime_source_eval_or_generator_used=False)
    return declaration


def main(run=False):
    declaration = protocol()
    if not run:
        if PROTOCOL_PATH.exists():
            raise FileExistsError("preserve lag-bank protocol")
        base.write_json(PROTOCOL_PATH, declaration)
        print("Protocol prepared; no model fitted: " + str(PROTOCOL_PATH), flush=True)
        return
    if not PROTOCOL_PATH.exists():
        raise FileNotFoundError("prepare and independently review lag-bank protocol before --run")
    prepared = json.loads(PROTOCOL_PATH.read_text())
    if (prepared["source_fingerprints"] != declaration["source_fingerprints"] or
            prepared["manifest"] != declaration["manifest"] or
            prepared["settings"] != base.serializable(declaration["settings"]) or
            prepared["original_raw_ledger_result_sha256"] != declaration["original_raw_ledger_result_sha256"] or
            prepared["temporal_reference_lags"] != declaration["temporal_reference_lags"] or
            prepared["temporal_reference_family_alpha"] != declaration["temporal_reference_family_alpha"]):
        raise ValueError("sources/public/settings/declaration changed after protocol preparation")
    if ROOT.exists() or OUTPUT.exists():
        raise FileExistsError("preserve prior lag-bank ledger outputs")
    (ROOT / "parameters").mkdir(parents=True, exist_ok=False)
    started, rows = time.perf_counter(), []
    payload = dict(declaration, status="running", rows=rows,
        protocol_file=str(PROTOCOL_PATH), protocol_sha256=base.digest(PROTOCOL_PATH),
        evaluation_files_opened=False, all_six_attempts_and_outputs_frozen_before_truth=False,
        process_local_frontend_injection_restored=False)
    base.write_json(OUTPUT, payload)
    original_frontend, original_root = raw.character_frontend, base.ROOT
    raw.character_frontend, base.ROOT = lag_bank.character_frontend, ROOT
    try:
        for entry in declaration["manifest"]:
            print("starting fixed-lag-bank RAW observed-line ledger " + entry["run"], flush=True)
            try:
                row = base.fit_entry(entry)
                row["temporal_reference_variant"] = "fixed iid covariance lag bank(1,2,3,4)"
                row["only_temporal_reference_gate_changed"] = True
            except Exception as error:
                row = dict(run=entry["run"], run_status="exception",
                    exception=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
            rows.append(row); base.write_json(OUTPUT, payload)
            print(json.dumps(dict(run=row["run"], status=row["run_status"],
                lines=row.get("frequency_count"), termination=row.get("inference", {}).get("status"),
                seconds=row.get("elapsed_seconds"), exception=row.get("exception"))), flush=True)
    finally:
        raw.character_frontend, base.ROOT = original_frontend, original_root
    if raw.character_frontend is not original_frontend or base.ROOT != original_root:
        raise RuntimeError("process-local frontend/root scope did not restore")
    if fingerprints() != declaration["source_fingerprints"]:
        raise ValueError("source changed during lag-bank fitting")
    if any(base.digest(entry["input"]) != entry["input_sha256"] for entry in declaration["manifest"]):
        raise ValueError("public observations changed during lag-bank fitting")
    if base.digest(ORIGINAL_LEDGER) != declaration["original_raw_ledger_result_sha256"]:
        raise ValueError("original ledger output changed")
    fitted = [row for row in rows if row["run_status"] == "fitted_and_frozen"]
    if any(base.digest(row["frozen_parameters"]) != row["frozen_parameters_sha256"] for row in fitted):
        raise ValueError("frozen lag-bank parameters changed")
    payload.update(status="finished_inference_only", all_six_attempts_and_outputs_frozen_before_truth=True,
        inference_finished_at=datetime.now(timezone.utc).isoformat(), sources_unchanged=True,
        public_inputs_unchanged=True, original_ledger_output_unchanged=True, evaluation_files_opened=False,
        process_local_frontend_injection_restored=True, total_elapsed_seconds=time.perf_counter()-started,
        summary=dict(requested_runs=6, fitted_runs=len(fitted), exceptions=6-len(fitted),
            observed_line_counts={row["run"]: row["frequency_count"] for row in fitted},
            physical_phase_rank_or_true_lines_scored=False))
    base.write_json(OUTPUT, payload)
    print(json.dumps(payload["summary"]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="run only after root/closure source/math review")
    main(parser.parse_args().run)
