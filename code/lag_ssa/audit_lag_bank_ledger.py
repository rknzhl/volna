"""Post-freeze lag-bank ledger scoring with an immutable same-scope baseline.

Reuses the independently reviewed complex forecast/catalogue evaluator.
No evaluator may be opened until explicit all-six-output freeze confirmation.
The old ledger, old audit and all old fitted parameters remain unchanged.
"""

from project_paths import RESULTS_DIR

import argparse
import json
from pathlib import Path

import audit_raw_spectral_ledger as audit


HERE = Path(__file__).resolve().parent
SOURCE = RESULTS_DIR / 'lag_bank_ledger.json'
ROOT = RESULTS_DIR / 'lag_bank_ledger_audit'
OUTPUT = ROOT.with_suffix(".json")
COMPARISON = RESULTS_DIR / 'lag_bank_ledger_comparison.json'
BASELINE = RESULTS_DIR / 'raw_spectral_ledger_audit.json'


def main(confirm_all_frozen=False):
    if not confirm_all_frozen:
        raise RuntimeError("wait for explicit root/author all-six lag-bank-output freeze confirmation")
    if OUTPUT.exists() or COMPARISON.exists() or ROOT.exists():
        raise FileExistsError("preserve completed lag-bank score/comparison")
    baseline_hash = audit.digest(BASELINE)
    previous = audit.SOURCE, audit.ROOT, audit.OUTPUT
    audit.SOURCE, audit.ROOT, audit.OUTPUT = SOURCE, ROOT, OUTPUT
    try:
        audit.main(confirm_all_frozen=True)
    finally:
        audit.SOURCE, audit.ROOT, audit.OUTPUT = previous
    scored = json.loads(OUTPUT.read_text())
    baseline = json.loads(BASELINE.read_text())
    old_by_run = {row["run"]: row for row in baseline["rows"]}
    rows = []
    for row in scored["rows"]:
        if row["status"] != "finished":
            rows.append(dict(run=row["run"], status=row["status"]))
            continue
        old = old_by_run[row["run"]]
        new_modes = {tuple(item["physical_character"]) for match in row["line_matching"]
                     if match["status"] == "unique" for item in match["matches"]}
        old_modes = {tuple(item["physical_character"]) for match in old["line_matching"]
                     if match["status"] == "unique" for item in match["matches"]}
        canonical = lambda modes: {mode if next(value for value in mode if value) > 0
                                  else tuple(-value for value in mode) for mode in modes}
        new_modes, old_modes = canonical(new_modes), canonical(old_modes)
        rows.append(dict(run=row["run"], status="finished",
            old_line_count=old["line_count"], new_line_count=row["line_count"],
            old_clock_forecast_score=old["future_clock_field_score"],
            new_clock_forecast_score=row["future_clock_field_score"],
            old_complete_visible_module_from_unique_matches=old["full_visible_module_covered_by_unique_line_matches_truth_only"],
            new_complete_visible_module_from_unique_matches=row["full_visible_module_covered_by_unique_line_matches_truth_only"],
            old_missing_line_haar_mass_energy_fraction=old["missing_catalog_line_haar_mass_energy_fraction"],
            new_missing_line_haar_mass_energy_fraction=row["missing_catalog_line_haar_mass_energy_fraction"],
            newly_retained_catalog_characters_truth_only=[list(mode) for mode in sorted(new_modes - old_modes)],
            no_longer_retained_catalog_characters_truth_only=[list(mode) for mode in sorted(old_modes - new_modes)],
            unmatched_line_count=row["unmatched_line_count"], ambiguous_line_count=row["ambiguous_line_count"]))
    comparison = dict(status="finished", source=str(SOURCE), source_sha256=audit.digest(SOURCE),
        scored_source=str(OUTPUT), scored_source_sha256=audit.digest(OUTPUT),
        baseline_scored_source=str(BASELINE), baseline_scored_source_sha256=baseline_hash,
        wrapper_source_sha256=audit.digest(__file__), evaluator_source_sha256=audit.digest(audit.__file__),
        baseline_unchanged=audit.digest(BASELINE) == baseline_hash, rows=rows,
        source_fitting_change="only fixed lags1..4 family-adjusted global iid temporal reference",
        comparison_scope="same raw-input splits and independent free-line clock forecast; no observed-window MSSA comparison",
        ground_truth_given_to_fitting_or_selection=False,
        newly_retained_labels_are_postfreeze_catalogue_diagnostics=True,
        physical_rank_or_compact_group_inferred=False,
        prior_cases_previously_examined=True, independent_external_validation=False,
        learned_residual_temporal_reference_calibrated=False,
        finite_lag_bank_or_finite_ledger_completeness_guaranteed=False,
        no_model_refitted_in_scoring=True)
    COMPARISON.write_text(json.dumps(comparison, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(comparison=str(COMPARISON), rows=rows)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm-all-six-frozen", action="store_true")
    main(parser.parse_args().confirm_all_six_frozen)
