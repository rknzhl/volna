"""Validate integer relations using independent frozen SSA phase readouts.

Only inferred lattice labels and saved observed train/future phases enter the
diagnostics.  No forcing frequency, true phase, PDE or target rank is opened.
The constant phase reference is fitted once on training and frozen on future.
The single-coordinate power search detects approximate finite cyclic factors;
it is a bounded diagnostic and never certifies exact resonance or torsion.
"""

from __future__ import annotations

from project_paths import RESULTS_DIR

import hashlib
from itertools import product
import json
from math import gcd
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
RESULTS = RESULTS_DIR
OUTPUT = RESULTS / "character_closure_audit.json"
POWER_BUDGET = 100
TRAIN_COHERENCE_THRESHOLD = .9
SMALL_RELATION_MAX_COEFFICIENT = 6


def constant_product_score(train_phase: np.ndarray, future_phase: np.ndarray,
                           relation: np.ndarray) -> dict:
    """Fit one constant product origin on train; use untouched future phases."""
    a = np.asarray(relation, dtype=int)
    train = np.asarray(train_phase, dtype=float)
    future = np.asarray(future_phase, dtype=float)
    if (a.ndim != 1 or train.ndim != 2 or future.ndim != 2
            or len(a) != len(train) or len(a) != len(future)
            or not np.isfinite(train).all() or not np.isfinite(future).all()
            or not train.shape[1] or not future.shape[1]):
        raise ValueError("matching nonempty finite observed phases required")
    train_product = np.exp(1j * np.einsum("i,it->t", a, train, optimize=False))
    future_product = np.exp(1j * np.einsum("i,it->t", a, future, optimize=False))
    train_mean = np.mean(train_product)
    origin = float(np.angle(train_mean))
    train_error = np.angle(train_product * np.exp(-1j * origin))
    future_error = np.angle(future_product * np.exp(-1j * origin))
    return {
        "integer_relation": a.tolist(),
        "constant_origin_train_only_rad": origin,
        "train_product_coherence": float(abs(train_mean)),
        "future_product_coherence": float(abs(np.mean(future_product))),
        "train_constant_product_mae_rad": float(np.mean(abs(train_error))),
        "future_train_origin_constant_product_mae_rad": float(np.mean(abs(future_error))),
        "future_product_mean_phase_drift_rad": float(np.angle(np.mean(future_product) * np.exp(-1j * origin))),
        "future_product_chord_rmse": float(np.sqrt(np.mean(abs(future_product - np.exp(1j * origin)) ** 2))),
        "train_samples": int(train.shape[1]),
        "future_samples": int(future.shape[1]),
        "future_referenced_to_new_fit": False,
        "phase_array_kind": "independent frozen SSA readout; no clock-generated products",
    }


def power_diagnostic(train_phase: np.ndarray, future_phase: np.ndarray,
                     coordinate: int) -> dict:
    """Select q and a constant reference on train, then score future."""
    phase = np.asarray(train_phase, dtype=float)[coordinate]
    powers = np.arange(1, POWER_BUDGET + 1)
    train_coherences = abs(np.mean(np.exp(1j * powers[:, None] * phase[None, :]), axis=1))
    accepted = powers[train_coherences >= TRAIN_COHERENCE_THRESHOLD]
    # Smallest retained order avoids choosing a multiple of a proposed period.
    q = int(accepted[0]) if len(accepted) else int(powers[np.argmax(train_coherences)])
    relation = np.zeros(len(train_phase), dtype=int)
    relation[coordinate] = q
    return {
        "acquired_coordinate": coordinate,
        "power_search_budget": POWER_BUDGET,
        "coherence_threshold_train_only": TRAIN_COHERENCE_THRESHOLD,
        "candidate_selected_on_train": int(q),
        "candidate_passes_training_threshold": bool(len(accepted)),
        "selection_rule": "smallest q passing fixed .9 threshold; otherwise maximum coherence reported as rejected diagnostic",
        "train_search_coherence_by_power": train_coherences.tolist(),
        "independent_future_score": constant_product_score(train_phase, future_phase, relation),
        "exact_torsion_certified": False,
        "finite_record_near_resonance_and_phase_noise_unresolved": True,
    }


def observed_relation_search(train_phase: np.ndarray, future_phase: np.ndarray) -> dict:
    """Find small primitive relations from observed training phases only.

    This separate audit does not read the inferred frequency lattice.  A
    finite coefficient search is performed only up to three acquired phases.
    Powers in a single coordinate are handled by the separate torsion audit.
    """
    count = len(train_phase)
    if count < 2 or count > 3:
        return {"status": "not_searched", "reason": "audit dimension budget is two or three acquired phases"}
    candidates = []
    for relation in product(range(-SMALL_RELATION_MAX_COEFFICIENT,
                                  SMALL_RELATION_MAX_COEFFICIENT + 1), repeat=count):
        nonzero = [value for value in relation if value]
        if len(nonzero) < 2 or nonzero[0] < 0:
            continue
        divisor = 0
        for value in relation:
            divisor = gcd(divisor, abs(value))
        if divisor == 1:
            candidates.append(relation)
    integer_rows = np.asarray(candidates, dtype=int)
    angles = np.einsum("ai,it->at", integer_rows, train_phase, optimize=False)
    coherences = abs(np.mean(np.exp(1j * angles), axis=1))
    ranking = sorted(range(len(candidates)), key=lambda index:
                     (-coherences[index], sum(abs(item) for item in candidates[index]), candidates[index]))
    accepted = [index for index in ranking if coherences[index] >= TRAIN_COHERENCE_THRESHOLD]
    best = ranking[0]
    return {
        "status": "bounded_observed_phase_product_diagnostic",
        "source": "observed training SSA phases only; no inferred or true frequencies",
        "max_integer_coefficient": SMALL_RELATION_MAX_COEFFICIENT,
        "primitive_relation_candidates_searched": len(candidates),
        "coherence_threshold_train_only": TRAIN_COHERENCE_THRESHOLD,
        "accepted_on_train": [constant_product_score(train_phase, future_phase, integer_rows[index])
                              for index in accepted],
        "best_train_candidate_independent_future_score": constant_product_score(
            train_phase, future_phase, integer_rows[best]),
        "top_five_training_candidates": [
            {"integer_relation": candidates[index], "train_coherence": float(coherences[index])}
            for index in ranking[:5]],
        "relation_kernel_complete_or_exact_certified": False,
    }


def audit_file(path: Path, prefix: str) -> dict:
    saved = json.loads(path.read_text())
    rows = saved.get("rows", [saved])
    results, skipped = [], []
    for row in rows:
        if row.get("run_status") != "finished":
            skipped.append({"case": row.get("case"), "reason": "original assay did not finish"})
            continue
        case = row["case"]
        array_path = RESULTS / (prefix + "_" + case + "_coordinates.npz")
        if not array_path.exists():
            skipped.append({"case": case, "reason": "saved independent phases unavailable"})
            continue
        # Open only the explicitly observed keys.  Truth-bearing NPZ keys are
        # never accessed.  Frequency values are not needed for this audit.
        with np.load(array_path) as arrays:
            train = arrays["train_phase"].copy()
            future = arrays["future_phase"].copy()
        lattice = row.get("inference", {}).get("final_lattice")
        relations = lattice.get("relations", []) if lattice else []
        relation_scores = [constant_product_score(train, future, np.asarray(a, dtype=int))
                           for a in relations]
        power_scores = [power_diagnostic(train, future, coordinate)
                        for coordinate in range(len(train))]
        results.append({
            "case": case,
            "source_assay": str(path),
            "coordinates_npz": str(array_path),
            "coordinates_sha256": hashlib.sha256(array_path.read_bytes()).hexdigest(),
            "opened_array_keys": ["train_phase", "future_phase"],
            "inferred_lattice_rank": lattice.get("rank") if lattice else None,
            "acquired_character_count": len(train),
            "relation_scores": relation_scores,
            "frequency_free_observed_relation_search": observed_relation_search(train, future),
            "single_coordinate_power_diagnostics": power_scores,
            "sampled_orbit_closure_certified": False,
        })
    return {"source": str(path), "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "source_assay_finished": saved.get("invocation_finished", False),
            "rows": results, "skipped": skipped}


def main() -> None:
    sources = (
        (RESULTS / "actual_wave_iterative_verified.json", "actual_wave_iterative_verified"),
        (RESULTS / "actual_wave_revised.json", "actual_wave_revised"),
        (RESULTS / "actual_wave_revised_fixtures.json", "actual_wave_revised_fixtures"),
    )
    groups, unavailable = [], []
    for source, prefix in sources:
        if source.exists():
            groups.append(audit_file(source, prefix))
        else:
            unavailable.append(str(source))
    all_relation_scores = [score for group in groups for row in group["rows"]
                           for score in row["relation_scores"]]
    supported_powers = [
        {"source": group["source"], "case": row["case"],
         "coordinate": power["acquired_coordinate"],
         "q": power["candidate_selected_on_train"],
         "future_coherence": power["independent_future_score"]["future_product_coherence"],
         "future_origin_frozen_mae_rad": power["independent_future_score"]["future_train_origin_constant_product_mae_rad"]}
        for group in groups for row in group["rows"]
        for power in row["single_coordinate_power_diagnostics"]
        if power["candidate_passes_training_threshold"]]
    payload = {
        "status": "finished",
        "audit_input_scope": "inferred relation integers and observed frozen SSA train/future phase arrays only",
        "truth_or_forcing_frequency_opened": False,
        "origin_and_power_selected_on_training_only": True,
        "clock_generated_phase_products_used": False,
        "groups": groups,
        "unavailable_sources": unavailable,
        "summary": {
            "independent_relation_tests": len(all_relation_scores),
            "supported_single_coordinate_power_diagnostics": supported_powers,
        },
        "interpretation": {
            "wave_sampled_orbit": "q=50 approximate constant power supports a finite cyclic sampled coordinate rather than density in the full continuous two-torus",
            "no_exact_resonance_claim": "bounded noisy phase-product coherence cannot establish exact rational sampling or distinguish arbitrarily near resonance",
            "finite_closure_consequence": "if an exact constant q-th power exists in the population, that observed coordinate lies in a translate of a q-point subgroup of S1",
            "current_core_association": "associate_observed_character_with_factor currently compares an observed new character to inferred clock exp(i(m dot nu)t), rather than old observed phase products; its phase-product wording needs correction or implementation change",
        },
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(OUTPUT), "summary": payload["summary"]}, indent=2))


if __name__ == "__main__":
    main()
