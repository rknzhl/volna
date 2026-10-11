"""Conditional unknown-r character-module SSA test on one spatial field.

The same observed field supplies the spectrum, spatial SSA, classical MSSA,
and direct-spectrum readout. Generator labels are used only after inference
to score rank, observable lattice index, and phase error. A finite noisy
spectrum cannot certify that arbitrarily weak characters are absent.
"""

from __future__ import annotations

from project_paths import RESULTS_DIR

from dataclasses import dataclass
from itertools import combinations
from math import gcd
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.signal import find_peaks

from method import (
    LagDesign, circular_mae, dmd_phase_coordinates, infer_frequency_lattice,
    trajectory_matrix, voronoi_weights_grid,
)
from phase_lattice import (
    infer_character_basis, infer_character_basis_global, phase_products,
)


OUT = RESULTS_DIR
OUT.mkdir(exist_ok=True)
DESTINATION = OUT / "character_ssa.json"
N = 3200
L = 64
ALPHA = np.array([.173 * np.sqrt(2), .217 * np.sqrt(3)])
NOISE = (.08, .20, .40)
SEEDS = tuple(range(40000, 40010))
SENSOR_SEED = 1741
Q = 12
SENSORS = np.random.default_rng(SENSOR_SEED).uniform(.06, .94, (Q, 2))
AREA = voronoi_weights_grid(SENSORS, resolution=80)

# These constants are fixed before running the full noise/seed sweep.
PAD_FACTOR = 8
SEPARATION_CYCLES = 6
MAX_VISIBLE = 8
NOISE_FLOOR_MULTIPLIER = 25.0
MAX_ORDER = 6
MAX_RANK = 3
RELATION_TOLERANCE = 2 * np.pi / N
TRUTH_MATCH_TOLERANCE = 2 * np.pi / N
MAX_POISSON_LABEL_ORDER = 6
FREQUENCY_GATE = 6 * 2 * np.pi / N
RHO = .45


@dataclass(frozen=True)
class Case:
    name: str
    modes: tuple[tuple[int, ...], ...]
    amplitudes: tuple[float, ...]
    infinite: bool = False

    @property
    def dimension(self) -> int:
        return len(self.modes[0])


CASES = (
    Case("r1_2_3_no_fundamental", ((2,), (3,)), (1.0, .82)),
    Case("r1_2_4_factor_index2", ((2,), (4,)), (1.0, .72)),
    Case("r2_2_3_each_no_fundamentals",
         ((2, 0), (3, 0), (0, 2), (0, 3)), (1.0, .84, .90, .76)),
    Case("r2_index3", ((2, 1), (1, 2), (1, -1)), (1.0, .85, .73)),
    Case("r2_primitive_no_coordinate_fundamentals",
         ((2, 0), (0, 3), (1, 1), (1, -1)), (1.0, .80, .74, .68)),
    Case("r1_poisson_2_3", ((2,), (3,)), (1.0, .82), True),
    Case("r2_poisson_index3", ((2, 1), (1, 2), (1, -1)),
         (1.0, .85, .73), True),
    Case("r2_poisson_primitive",
         ((2, 0), (0, 3), (1, 1), (1, -1)),
         (1.0, .80, .74, .68), True),
)


def spatial_profiles(count: int) -> np.ndarray:
    """Smooth, nonconstant profiles sampled at the irregular 2D sensors."""
    x, y = SENSORS.T
    profiles = []
    for j in range(count):
        profile = (1.0 + .24 * np.cos(2 * np.pi * ((j % 3 + 1) * x + .7 * y))
                   + .16 * np.sin(2 * np.pi * (x - (j % 2 + 1) * y)))
        profile /= np.sqrt(np.sum(AREA * profile * profile))
        profiles.append(profile)
    return np.stack(profiles, axis=1)


def poisson(angle: np.ndarray) -> np.ndarray:
    return (1 - RHO**2) / (1 - 2 * RHO * np.cos(angle) + RHO**2) - 1


def clean_field(case: Case, alpha: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    origin = np.array([.23, 1.08])[:case.dimension]
    phases = origin[:, None] + alpha[:case.dimension, None] * np.arange(N)[None, :]
    profiles = spatial_profiles(len(case.modes))
    signals = np.zeros((Q, N))
    for j, (mode, amplitude) in enumerate(zip(case.modes, case.amplitudes)):
        angle = np.einsum("i,ij->j", np.asarray(mode), phases, optimize=False) + .21 * j
        temporal = poisson(angle) if case.infinite else np.cos(angle)
        signals += amplitude * profiles[:, j, None] * temporal[None, :]
    return signals, phases


def weighted_peak_power(observed: np.ndarray) -> tuple[np.ndarray, int, float]:
    centered = observed - observed.mean(axis=1, keepdims=True)
    nfft = 1 << int(np.ceil(np.log2(PAD_FACTOR * observed.shape[1])))
    transformed = np.fft.rfft(centered * np.hanning(observed.shape[1]), n=nfft, axis=1)
    power = np.sum(AREA[:, None] * abs(transformed)**2, axis=0)
    frequencies = 2 * np.pi * np.arange(len(power)) / nfft
    background = power[(frequencies > .03) & (frequencies < np.pi - .03)]
    return power, nfft, float(np.median(background))


def refine_peak(observed: np.ndarray, coarse: float) -> float:
    """Maximize the spatially weighted Hann coefficient near one peak."""
    n = observed.shape[1]
    times = np.arange(n)
    windowed = (observed - observed.mean(axis=1, keepdims=True)) * np.hanning(n)

    def negative_power(angular: float) -> float:
        wave = np.exp(-1j * angular * times)
        coefficient = np.einsum("in,n->i", windowed, wave, optimize=False)
        return -float(np.sum(AREA * abs(coefficient)**2))

    width = 2 * np.pi / n
    interval = (max(1e-9, coarse - width), min(np.pi - 1e-9, coarse + width))
    result = minimize_scalar(negative_power, bounds=interval, method="bounded",
                             options={"xatol": 1e-11})
    if not result.success:
        raise RuntimeError(f"peak refinement failed: {result.message}")
    return float(result.x)


def screen_peaks(observed: np.ndarray) -> dict:
    power, nfft, floor = weighted_peak_power(observed)
    minimum = NOISE_FLOOR_MULTIPLIER * floor
    separation = max(2, int(SEPARATION_CYCLES * nfft / observed.shape[1]))
    peaks, _ = find_peaks(power, distance=separation)
    peaks = peaks[(peaks > nfft / observed.shape[1]) & (power[peaks] >= minimum)]
    peaks = peaks[np.argsort(power[peaks])[::-1]][:MAX_VISIBLE]
    coarse = 2 * np.pi * peaks / nfft
    refined = np.asarray([refine_peak(observed, float(value)) for value in coarse])
    return {"coarse": coarse.tolist(), "refined": refined.tolist(),
            "peak_power": power[peaks].tolist(), "noise_floor": floor,
            "power_threshold": minimum, "visible_count": len(peaks)}


def true_index(modes: tuple[tuple[int, ...], ...]) -> int:
    matrix = np.asarray(modes, dtype=int)
    dimension = matrix.shape[1]
    if dimension == 1:
        result = 0
        for number in matrix[:, 0]:
            result = gcd(result, abs(int(number)))
        return result
    result = 0
    for rows in combinations(matrix, dimension):
        result = gcd(result, abs(int(round(np.linalg.det(np.stack(rows))))))
    return result


def positive_frequency(mode: np.ndarray, alpha: np.ndarray) -> tuple[float, np.ndarray]:
    frequency = float(np.angle(np.exp(1j * float(np.dot(mode, alpha)))))
    return (frequency, mode) if frequency >= 0 else (-frequency, -mode)


def truth_candidates(case: Case, alpha: np.ndarray) -> list[dict]:
    candidates = []
    for base, amplitude in zip(case.modes, case.amplitudes):
        for order in range(1, MAX_POISSON_LABEL_ORDER + 1 if case.infinite else 2):
            mode = order * np.asarray(base, dtype=int)
            frequency, label = positive_frequency(mode, alpha[:case.dimension])
            strength = float(amplitude * (2 * RHO**order if case.infinite else 1))
            candidates.append({"frequency": frequency, "label": label.tolist(),
                               "base": list(base), "order": order,
                               "nominal_amplitude": strength})
    return candidates


def match_truth_labels(peaks: np.ndarray, candidates: list[dict], case: Case) -> dict:
    labels = []
    matched = []
    spurious = []
    ambiguous = []
    for index, peak in enumerate(peaks):
        choices = [item for item in candidates
                   if abs(float(item["frequency"]) - float(peak)) <= TRUTH_MATCH_TOLERANCE]
        unique = {tuple(item["label"]) for item in choices}
        if not choices:
            spurious.append({"peak_index": index, "frequency": float(peak)})
            labels.append(None)
        elif len(unique) != 1:
            ambiguous.append({"peak_index": index, "frequency": float(peak),
                              "candidate_labels": [list(value) for value in sorted(unique)]})
            labels.append(None)
        else:
            label = next(iter(unique))
            labels.append(list(label))
            matched.append({"peak_index": index, "frequency": float(peak),
                            "true_label": list(label),
                            "frequency_error": float(min(abs(item["frequency"]-peak)
                                                         for item in choices))})
    def unoriented(label: list[int] | tuple[int, ...]) -> tuple[int, ...]:
        vector = tuple(label)
        opposite = tuple(-value for value in vector)
        return min(vector, opposite)

    present = {unoriented(label) for label in labels if label is not None}
    missing_base = [list(mode) for mode in case.modes
                    if unoriented(mode) not in present]
    return {"labels": labels, "matched_peaks": matched,
            "spurious_peaks": spurious, "ambiguous_peaks": ambiguous,
            "missing_base_modes": missing_base,
            "all_peaks_matched": not spurious and not ambiguous}


def inferred_index(matrix: np.ndarray) -> int | None:
    if matrix.shape[0] != matrix.shape[1]:
        return None
    if matrix.shape[0] == 1:
        return abs(int(matrix[0, 0]))
    return abs(int(round(np.linalg.det(matrix))))


def phase_readouts(observed: np.ndarray, true_phases: np.ndarray,
                   peaks: np.ndarray, lattice: dict, labels: np.ndarray) -> dict:
    products = np.asarray(lattice["products"], dtype=int)
    character_matrix = products @ labels
    expected = phase_products(true_phases[:, :N-L+1], character_matrix)
    rank = min(2 * len(peaks) + 2, 24)
    design = LagDesign(np.arange(L), np.ones(L) / L, float("nan"), L)
    direct = phase_products(peaks[:, None] * np.arange(N-L+1)[None, :], products)
    scores = {"direct_spectrum": {"phase_mae_rad": circular_mae(direct, expected)}}
    for name, spatial in (("spatial_weighted_ssa", AREA),
                          ("classical_mssa", np.ones(Q) / Q)):
        try:
            trajectory = trajectory_matrix(observed, spatial, design)
            dmd = dmd_phase_coordinates(trajectory, peaks, rank=rank)
            phase = phase_products(dmd["phases"], products)
            mismatch = np.abs(np.angle(dmd["eigenvalues"] * np.exp(-1j * peaks)))
            scores[name] = {
                "phase_mae_rad": circular_mae(phase, expected),
                "max_target_eigenfrequency_error": float(np.max(mismatch)),
                "frequency_gate_passed": bool(np.all(mismatch <= FREQUENCY_GATE)),
                "selected_eigenfrequencies": np.angle(dmd["eigenvalues"]).tolist(),
            }
        except (FloatingPointError, np.linalg.LinAlgError, ValueError,
                StopIteration) as exc:
            scores[name] = {"failure": f"{type(exc).__name__}: {exc}"}
    return {"character_matrix_in_physical_coordinates": character_matrix.tolist(),
            "inferred_lattice_index": inferred_index(character_matrix),
            "rank_used": rank, "scores": scores}


def evaluate(case: Case, noise: float, seed: int) -> dict:
    alpha = ALPHA[:case.dimension]
    clean, phases = clean_field(case, ALPHA)
    scale = np.sqrt(np.mean(np.sum(AREA[:, None] * clean**2, axis=0)))
    observed = clean + noise * scale * np.random.default_rng(seed).standard_normal(clean.shape)
    row = {"case": case.name, "noise": noise, "seed": seed,
           "true_phase_rank": case.dimension, "true_observable_lattice_index": true_index(case.modes),
           "relative_noise_rms": noise}
    try:
        screened = screen_peaks(observed)
        row["screen"] = screened
        peaks = np.asarray(screened["refined"], dtype=float)
        if len(peaks) == 0:
            row["status"] = "no_visible_peak"
            return row
        old = infer_frequency_lattice(peaks, tolerance=RELATION_TOLERANCE,
                                      max_order=MAX_ORDER, max_rank=MAX_RANK)
        row["old_lattice"] = old
        try:
            incremental = infer_character_basis(peaks, tolerance=RELATION_TOLERANCE,
                                                max_order=MAX_ORDER, max_rank=MAX_RANK)
        except (RuntimeError, ValueError, np.linalg.LinAlgError, ArithmeticError) as exc:
            incremental = {"rank": None, "exception": f"{type(exc).__name__}: {exc}"}
        row["incremental_lattice"] = incremental
        new = infer_character_basis_global(peaks, tolerance=RELATION_TOLERANCE,
                                           max_order=MAX_ORDER, max_rank=MAX_RANK)
        row["character_lattice"] = new
        matched = match_truth_labels(peaks, truth_candidates(case, alpha), case)
        row["truth_label_audit"] = matched
        if new.get("rank") is None:
            row["status"] = "global_character_inference_failed"
            return row
        if not matched["all_peaks_matched"]:
            row["status"] = "unmatched_or_ambiguous_peak"
            return row
        labels = np.asarray(matched["labels"], dtype=int)
        readouts = phase_readouts(observed, phases, peaks, new, labels)
        row["phase_readouts"] = readouts
        row["observable_module_complete"] = bool(
            new["rank"] == case.dimension
            and readouts["inferred_lattice_index"] == true_index(case.modes))
        row["status"] = "evaluated"
    except Exception as exc:
        row["status"] = "pipeline_exception"
        row["exception"] = f"{type(exc).__name__}: {exc}"
    return row


def alias_negative() -> dict:
    alpha = np.array([2 * np.pi / 10, ALPHA[1]])
    case = next(item for item in CASES if item.name == "r2_primitive_no_coordinate_fundamentals")
    exact = np.asarray([positive_frequency(np.asarray(mode), alpha)[0]
                        for mode in case.modes])
    incremental = infer_character_basis(exact, tolerance=1e-10,
                                        max_order=MAX_ORDER, max_rank=MAX_RANK)
    module = infer_character_basis_global(exact, tolerance=1e-10,
                                          max_order=MAX_ORDER, max_rank=MAX_RANK)
    n = np.arange(100)
    first_phase = .23 + alpha[0] * n
    invisible = np.sin(10 * (first_phase - .23))
    return {"alias_alpha": alpha.tolist(), "sampled_first_phase_states": 10,
            "exact_module_rank": module.get("rank"),
            "exact_module_basis": module.get("basis"),
            "exact_incremental_rank": incremental.get("rank"),
            "invisible_perturbation_max_on_samples": float(np.max(abs(invisible))),
            "invisible_perturbation_off_trajectory": float(np.sin(10 * np.pi / 20)),
            "full_continuous_torus_identifiable": False,
            "interpretation": "module rank does not imply a dense sampled orbit"}


def exact_fixtures() -> list[dict]:
    """Small exact-spectrum controls, independent of peak screening and DMD."""
    rows = []
    for case in CASES[:5]:
        alpha = ALPHA[:case.dimension]
        frequencies_and_labels = [positive_frequency(np.asarray(mode), alpha)
                                  for mode in case.modes]
        peaks = np.asarray([entry[0] for entry in frequencies_and_labels])
        labels = np.stack([entry[1] for entry in frequencies_and_labels])
        incremental = infer_character_basis(peaks, tolerance=1e-10,
                                            max_order=MAX_ORDER, max_rank=MAX_RANK)
        global_result = infer_character_basis_global(peaks, tolerance=1e-10,
                                                     max_order=MAX_ORDER, max_rank=MAX_RANK)
        old = infer_frequency_lattice(peaks, tolerance=1e-10,
                                      max_order=MAX_ORDER, max_rank=MAX_RANK)
        character_matrix = (np.asarray(global_result["products"], dtype=int) @ labels
                            if global_result.get("rank") is not None else None)
        rows.append({"case": case.name, "exact_peaks": peaks.tolist(),
                     "true_peak_labels": labels.tolist(),
                     "true_rank": case.dimension,
                     "true_lattice_index": true_index(case.modes),
                     "old_rank": old.get("rank"),
                     "incremental_rank": incremental.get("rank"),
                     "global_rank": global_result.get("rank"),
                     "global_basis_in_physical_coordinates": (
                         character_matrix.tolist() if character_matrix is not None else None),
                     "global_index": (inferred_index(character_matrix)
                                      if character_matrix is not None else None)})
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    results = []
    for case in CASES:
        for noise in NOISE:
            group = [row for row in rows if row["case"] == case.name and row["noise"] == noise]
            evaluated = [row for row in group if row["status"] == "evaluated"]
            complete = [row for row in evaluated if row["observable_module_complete"]]
            medians = {}
            for method in ("direct_spectrum", "spatial_weighted_ssa", "classical_mssa"):
                values = [row["phase_readouts"]["scores"][method]["phase_mae_rad"]
                          for row in complete if "phase_mae_rad" in
                          row["phase_readouts"]["scores"][method]]
                medians[method] = float(np.median(values)) if values else None
            results.append({
                "case": case.name, "noise": noise, "trials": len(group),
                "evaluated": len(evaluated), "complete_observable_module": len(complete),
                "new_correct_rank": sum(row.get("character_lattice", {}).get("rank") == case.dimension
                                        for row in group),
                "incremental_correct_rank": sum(row.get("incremental_lattice", {}).get("rank") == case.dimension
                                                for row in group),
                "old_correct_rank": sum(row.get("old_lattice", {}).get("rank") == case.dimension
                                        for row in group),
                "status_counts": {status: sum(row["status"] == status for row in group)
                                  for status in sorted({row["status"] for row in group})},
                "median_phase_mae_on_complete_runs": medians,
            })
    return results


def main() -> None:
    rows = []
    for case in CASES:
        for noise in NOISE:
            for seed in SEEDS:
                rows.append(evaluate(case, noise, seed))
            latest = [row for row in rows if row["case"] == case.name and row["noise"] == noise]
            print(case.name, noise, {status: sum(row["status"] == status for row in latest)
                                      for status in sorted({row["status"] for row in latest})},
                  flush=True)
            DESTINATION.write_text(json.dumps({
                "complete": False, "completed_groups": len(rows) // len(SEEDS),
                "expected_groups": len(CASES) * len(NOISE), "rows": rows,
            }, indent=2) + "\n")
    payload = {
        "description": "conditional global character-module SSA on one spatial field; old and incremental stages retained; no truth in inference",
        "samples": N, "lag_window": L, "sensor_count": Q,
        "sensors": SENSORS.tolist(), "voronoi_weights": AREA.tolist(),
        "alpha": ALPHA.tolist(), "noise_levels": NOISE, "seeds": SEEDS,
        "spectral_screen": {"pad_factor": PAD_FACTOR,
                            "separation_cycles": SEPARATION_CYCLES,
                            "max_visible": MAX_VISIBLE,
                            "noise_floor_multiplier": NOISE_FLOOR_MULTIPLIER,
                            "peak_refinement_half_width": 2 * np.pi / N},
        "relation_search": {"max_order": MAX_ORDER, "max_rank": MAX_RANK,
                            "tolerance": RELATION_TOLERANCE},
        "truth_label_matching": {"tolerance": TRUTH_MATCH_TOLERANCE,
                                 "max_poisson_harmonic_order": MAX_POISSON_LABEL_ORDER,
                                 "used_only_for_evaluation": True},
        "dmd_frequency_gate": FREQUENCY_GATE,
        "poisson_rho": RHO,
        "cases": [{"name": case.name, "modes": case.modes,
                   "amplitudes": case.amplitudes, "infinite_harmonics": case.infinite,
                   "true_phase_rank": case.dimension,
                   "true_observable_lattice_index": true_index(case.modes)}
                  for case in CASES],
        "exact_fixtures": exact_fixtures(),
        "alias_negative_exact": alias_negative(),
        "rows": rows, "summary": summarize(rows),
        "interpretation": "rank/index are conditional on visible peaks; physical phase is not claimed when index>1",
    }
    payload["complete"] = True
    DESTINATION.write_text(json.dumps(payload, indent=2) + "\n")
    print(DESTINATION, flush=True)


if __name__ == "__main__":
    main()
