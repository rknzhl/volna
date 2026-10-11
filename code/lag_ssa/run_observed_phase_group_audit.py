"""Observed group assay on original wave SSA phases and five frozen fixtures.

The saved original assay lacks validation phase arrays. Its independent frozen
future readout is used explicitly as the heldout group gate, not as a later
untouched test set for the group estimator. No physical truth is read to fit.
Small direct-character controls exercise mixed torsion and near resonances;
these controls validate this diagnostic, not the full SSA pipeline.
"""

from __future__ import annotations

from project_paths import RESULTS_DIR

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from observed_phase_group import estimate_observed_group


HERE = Path(__file__).resolve().parent
RESULTS = RESULTS_DIR
OUTPUT = RESULTS / "observed_phase_group_audit.json"
FIXTURES = ("r2_poisson_dominant_weak", "r2_infinite_mixed_products", "r2_partial_cover",
            "r1_poisson_negative", "r1_poisson2_cos3")
OPTIONS = {"max_integer_bound": 100, "mixed_coefficient_bound": 6,
           "max_mixed_support": None, "max_primitive_directions": 4096,
           "power_proposals_per_mixed_direction": 3, "coherence_threshold": .95}


def estimate_archive(path: Path, case: str, kind: str) -> dict:
    start = time.perf_counter()
    with np.load(path, allow_pickle=False) as archive:
        training = np.exp(1j * archive["train_phase"])
        heldout = np.exp(1j * archive["future_phase"])
    inferred = estimate_observed_group(training, heldout, **OPTIONS)
    return {"case": case, "assay_kind": kind, "coordinates_npz": str(path),
            "coordinates_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "opened_keys": ["train_phase", "future_phase"],
            "heldout_gate_record": "previous assay's independently frozen future SSA readout",
            "future_used_for_group_acceptance": True,
            "further_untouched_record_available_for_group_test": False,
            "original_generator_truth_used_in_group_fit": False,
            "estimate": inferred, "elapsed_seconds": time.perf_counter()-start}


def control(name: str, frequencies: np.ndarray, expected_free: int,
            expected_finite: list, near_resonance: bool = False,
            phase_noise_sigma: float = .001, train_samples: int = 1200,
            heldout_start: int = 1800, heldout_samples: int = 600) -> dict:
    """Character generator only; the estimator receives arrays, not frequencies."""
    training_times = np.arange(train_samples)
    heldout_times = np.arange(heldout_start, heldout_start+heldout_samples)
    rng = np.random.default_rng(4028)
    phase_zero = np.linspace(.23, -.41, len(frequencies))
    training_phase = frequencies[:, None] * training_times + phase_zero[:, None]
    heldout_phase = frequencies[:, None] * heldout_times + phase_zero[:, None]
    training = np.exp(1j * (training_phase + phase_noise_sigma * rng.standard_normal(training_phase.shape)))
    heldout = np.exp(1j * (heldout_phase + phase_noise_sigma * rng.standard_normal(heldout_phase.shape)))
    inferred = estimate_observed_group(training, heldout, **OPTIONS)
    presentation = inferred["presentation"]
    # Only after inference: compare to the generating compact group.
    return {"case": name, "assay_kind": "direct character diagnostic control, not full SSA",
            "estimate": inferred,
            "generator_truth_opened_for_post_fit_assessment": {
                "frequencies_rad_per_sample": frequencies.tolist(),
                "connected_dimension": expected_free, "finite_smith_factors": expected_finite,
                "near_resonance": near_resonance, "phase_noise_sigma": phase_noise_sigma},
            "record_lengths": {"train": train_samples, "heldout_start": heldout_start,
                               "heldout": heldout_samples},
            "candidate_matches_exact_generator_group_truth_only": bool(
                presentation["candidate_connected_component_dimension"] == expected_free
                and presentation["candidate_finite_cyclic_factors"] == expected_finite)}


def main() -> None:
    rows, historical, unavailable = [], [], []
    original_prefix = "actual_wave_iterative_verified"
    for path in sorted(RESULTS.glob(original_prefix + "_wave_*_coordinates.npz")):
        case = path.stem[len(original_prefix)+1:-len("_coordinates")]
        latest = RESULTS / ("actual_wave_observed_relations_" + case + "_coordinates.npz")
        if latest.exists():
            historical.append(estimate_archive(path, case, "historical original PDE wave field; earlier frozen SSA observers"))
            path = latest
        row = estimate_archive(path, case, "original PDE wave field; current observed-gate frozen acquired SSA phases")
        rows.append(row)
        print(json.dumps({"case": case, "group": row["estimate"]["presentation"]["candidate_compact_group_symbol"],
                          "status": row["estimate"]["status"]}), flush=True)
    for case in FIXTURES:
        prefix = "actual_wave_observed_relations_fixtures"
        path = RESULTS / (prefix + "_" + case + "_coordinates.npz")
        if not path.exists():
            path = RESULTS / ("actual_wave_revised_fixtures_" + case + "_coordinates.npz")
        if not path.exists():
            unavailable.append(case)
            continue
        row = estimate_archive(path, case, "frozen SSA readouts of general harmonic field fixture")
        rows.append(row)
        print(json.dumps({"case": case, "group": row["estimate"]["presentation"]["candidate_compact_group_symbol"],
                          "status": row["estimate"]["status"]}), flush=True)
    alpha = .271*np.sqrt(2)
    four_character_frequencies = np.asarray([.17*np.sqrt(2), .21*np.sqrt(3), .13*np.sqrt(5)])
    four_character_frequencies = np.r_[four_character_frequencies, .5*four_character_frequencies.sum()]
    controls = [
        control("mixed_finite_order_7", np.asarray([alpha, -alpha+2*np.pi/7]), 1, [7]),
        control("two_coprime_finite_coordinates", np.asarray([2*np.pi/5, 2*np.pi/7]), 0, [35]),
        control("irrational_single_coordinate", np.asarray([alpha]), 1, []),
        control("detectable_near_order_50", np.asarray([2*np.pi/50+1e-5]), 1, [], True),
        control("indistinguishable_near_order_50", np.asarray([2*np.pi/50+1e-7]), 1, [], True),
        control("four_character_primitive_cover_short_record", four_character_frequencies, 3, []),
        control("four_character_primitive_cover_exact", four_character_frequencies, 3, [], phase_noise_sigma=0.,
                train_samples=6000, heldout_start=9000, heldout_samples=3000),
        control("four_character_primitive_cover_noisy", four_character_frequencies, 3, [],
                train_samples=6000, heldout_start=9000, heldout_samples=3000),
    ]
    # Generator interpretation is supplied after all original group fits.
    # Case names only distinguish the predeclared stochastic-source control.
    tone_rows = [row for row in rows if row["assay_kind"].startswith("original")
                 and row["case"] != "wave_1d_source_noise"]
    noise_rows = [row for row in rows if row["case"] == "wave_1d_source_noise"]
    payload = {
        "status": "finished", "finished_utc": datetime.now(timezone.utc).isoformat(),
        "options_fixed_before_all_cases": OPTIONS,
        "rows": rows, "historical_original_wave_rows": historical,
        "direct_character_controls": controls, "unavailable_fixtures": unavailable,
        "summary": {
            "original_tone_only_cases": len(tone_rows),
            "tone_only_T1_times_Z50_supported_candidates": sum(
                row["estimate"]["presentation"]["candidate_basis_empirically_compatible"]
                and row["estimate"]["presentation"]["candidate_connected_component_dimension"] == 1
                and row["estimate"]["presentation"]["candidate_finite_cyclic_factors"] == [50]
                for row in tone_rows),
            "noise_source_cases_with_supported_finite_factor": sum(
                bool(row["estimate"]["presentation"]["candidate_finite_cyclic_factors"]) for row in noise_rows),
            "fixture_cases": sum(row["case"] in FIXTURES for row in rows),
        },
        "input_scope": "observed training/heldout phase arrays only, converted to unit-circle characters",
        "frequency_or_lattice_truth_supplied": False,
        "exact_group_identification_or_search_completeness_established": False,
        "interpretation": {
            "original_tone_fields": "supported candidate sampled orbit group T1 x Z/50Z; continuous two-phase torus and its sampled orbit closure are different objects",
            "near_resonance": "a sufficiently small irrational detuning can pass finite noisy constant-product checks; exact torsion remains unidentifiable without additional assumptions",
            "SNF": "Smith reduction is known integer algebra; nonsaturated relation rows preserve finite cyclic factors",
            "scope": "diagnostic candidate presentation, not a proof that the whole continuous torus was reconstructed",
            "historical_three_dimensional_lens": "the earlier SSA artifact had a third acquired observer; its extra free candidate dimension is retained, not silently removed using generator truth",
        },
        "source_sha256": {name: hashlib.sha256((HERE/name).read_bytes()).hexdigest()
                          for name in ("observed_phase_group.py", "run_observed_phase_group_audit.py")},
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(OUTPUT), "summary": payload["summary"]}), flush=True)


if __name__ == "__main__":
    main()
