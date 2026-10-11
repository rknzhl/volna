"""Train/test assay of conditional family subtraction followed by ordinary SSA.

The first frequency family is learned from the noisy training field, then
subtracted with frozen coefficients from a separate test field. The test
residual supplies a candidate for SSA/DMD. Truth is used only for scoring and
the explicitly marked ideal-oracle residual control. This is a next-character
assay, not a blind unknown-r phase-module algorithm.
"""

from __future__ import annotations

from project_paths import RESULTS_DIR

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.signal import find_peaks

from method import LagDesign, circular_mae, trajectory_matrix, voronoi_weights_grid


OUT = RESULTS_DIR
N_TRAIN = 2400
N_TEST = 2400
L = 32
Q = 8
SENSOR_SEED = 6812
SENSORS = np.random.default_rng(SENSOR_SEED).uniform(.07, .93, (Q, 2))
AREA = voronoi_weights_grid(SENSORS, resolution=80)
ALPHA_1 = .173*np.sqrt(2)
ALPHA_2 = .217*np.sqrt(3)
NOISE_LEVELS = (.03, .08, .20)
SEEDS = tuple(range(52000, 52004))
K_CANDIDATES = (1, 2, 4, 8, 16, 24, 32)
NOISE_FLOOR_MULTIPLIER = 25.0
PAD_FACTOR = 8
FIRST_FAMILY_GATE = 2*np.pi/N_TEST
MAX_GATE_ORDER = 64
DESIGN = LagDesign(np.arange(L), np.full(L, 1/L), float("nan"), L)
TIME_TRAIN = np.arange(N_TRAIN)
TIME_TEST = np.arange(N_TRAIN, N_TRAIN+N_TEST)


@dataclass(frozen=True)
class Case:
    name: str
    kind: str
    rho: float | None = None
    near_cycles: float | None = None


CASES = (
    Case("a_finite_strong_r2", "finite"),
    Case("b_poisson70_r2", "poisson", .70),
    Case("c_poisson88_r2", "poisson", .88),
    Case("d_poisson70_mixed_r2", "mixed", .70),
    Case("e_poisson88_r1_negative", "negative", .88),
    Case("f_poisson70_2theta_cos3_cover", "cover", .70),
    Case("g_partial_cover_index2", "partial_cover", .70),
    Case("h_nearres_half_cycle", "near", None, .5),
    Case("h_nearres_eight_cycles", "near", None, 8.0),
    Case("h_nearres_eighty_cycles", "near", None, 80.0),
)


def profile(number: int) -> np.ndarray:
    x, y = SENSORS.T
    spatial = (1 + .22*np.cos(2*np.pi*((number % 3+1)*x+.37*y))
               + .15*np.sin(2*np.pi*(x-(number % 2+1)*y)))
    return spatial/np.sqrt(np.sum(AREA*spatial**2))


PROFILES = np.stack([profile(j) for j in range(7)])


def poisson_zero_mean(angle: np.ndarray, rho: float) -> np.ndarray:
    return (1-rho*rho)/(1-2*rho*np.cos(angle)+rho*rho)-1


def analytic_strong_haar_rms(case: Case) -> float:
    if case.kind in ("finite", "near"):
        # Distinct integer harmonics are orthogonal; every spatial profile
        # has unit AREA norm.
        return float(np.sqrt(.5*sum(value*value for value in (1., .8, .6, .4))))
    if case.rho is None:
        raise ValueError("Poisson case requires rho")
    # P_rho(theta)-1 = 2 sum_{m>=1} rho^m cos(m theta).
    return float(np.sqrt(2*case.rho**2/(1-case.rho**2)))


def component_fields(case: Case, times: np.ndarray,
                     normalize_strong: bool = False) -> tuple[np.ndarray, np.ndarray, dict]:
    theta1 = .29+ALPHA_1*times
    theta2_frequency = (2*ALPHA_1+case.near_cycles/N_TRAIN
                        if case.kind == "near" else ALPHA_2)
    theta2 = .87+theta2_frequency*times
    strong = np.zeros((Q, len(times)))
    weak = np.zeros_like(strong)
    targets = []
    if case.kind in ("finite", "near"):
        for harmonic, amplitude in enumerate((1., .8, .6, .4), start=1):
            strong += amplitude*PROFILES[harmonic-1, :, None]*np.cos(harmonic*theta1)[None, :]
    elif case.kind in ("poisson", "mixed", "negative"):
        strong = PROFILES[0, :, None]*poisson_zero_mean(theta1, case.rho)
    elif case.kind in ("cover", "partial_cover"):
        strong = PROFILES[0, :, None]*poisson_zero_mean(2*theta1, case.rho)
    else:
        raise ValueError(case.kind)
    strong_haar_rms = analytic_strong_haar_rms(case)
    if normalize_strong:
        strong = strong/strong_haar_rms
    if case.kind in ("finite", "poisson", "near"):
        weak += .08*PROFILES[5, :, None]*np.cos(theta2)[None, :]
        targets.append({"name": "weak_theta2", "frequency": theta2_frequency,
                        "phase": theta2, "mode": [0, 1]})
    elif case.kind == "mixed":
        phase = theta1+theta2
        weak += .08*PROFILES[5, :, None]*np.cos(phase)[None, :]
        targets.append({"name": "weak_mixed_1_1", "frequency": ALPHA_1+ALPHA_2,
                        "phase": phase, "mode": [1, 1]})
    elif case.kind == "cover":
        phase = 3*theta1
        weak += .08*PROFILES[5, :, None]*np.cos(phase)[None, :]
        targets.append({"name": "weak_cover_3theta", "frequency": 3*ALPHA_1,
                        "phase": phase, "mode": [3]})
    elif case.kind == "partial_cover":
        weak += .20*PROFILES[5, :, None]*np.cos(theta2)[None, :]
        targets.append({"name": "weak_theta2_partial_cover", "frequency": ALPHA_2,
                        "phase": theta2, "mode": [0, 1]})
        phase = 3*theta1
        weak += .08*PROFILES[6, :, None]*np.cos(phase)[None, :]
        targets.append({"name": "unrecovered_cover_3theta", "frequency": 3*ALPHA_1,
                        "phase": phase, "mode": [3, 0]})
    return strong, weak, {"targets": targets,
                          "first_family_frequency": 2*ALPHA_1 if case.kind in ("cover", "partial_cover") else ALPHA_1,
                          "theta1": theta1, "theta2": theta2,
                          "normalized_strong": normalize_strong,
                          "analytic_strong_haar_rms_before_normalization": strong_haar_rms,
                          "strong_applied_multiplier": 1/strong_haar_rms if normalize_strong else 1.0,
                          "strong_lattice_index": 2 if case.kind in ("cover", "partial_cover") else 1,
                          "next_character_completes_lattice": case.kind != "partial_cover"}


def spectrum(field: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, int]:
    n = field.shape[1]
    nfft = 1 << int(np.ceil(np.log2(PAD_FACTOR*n)))
    centered = field-field.mean(axis=1, keepdims=True)
    transform = np.fft.rfft(centered*np.hanning(n), nfft, axis=1)
    power = np.sum(AREA[:, None]*abs(transform)**2, axis=0)
    grid = 2*np.pi*np.arange(len(power))/nfft
    floor = float(np.median(power[(grid > .03)&(grid < np.pi-.03)]))
    return power, grid, floor, nfft


def refine_hann(field: np.ndarray, coarse: float) -> float:
    n = field.shape[1]
    centered = (field-field.mean(axis=1, keepdims=True))*np.hanning(n)
    times = np.arange(n)

    def objective(frequency: float) -> float:
        coefficient = np.einsum("in,n->i", centered,
                                np.exp(-1j*frequency*times), optimize=False)
        return -float(np.sum(AREA*abs(coefficient)**2))

    width = 2*np.pi/n
    optimum = minimize_scalar(objective, bounds=(max(1e-8, coarse-width),
                                                min(np.pi-1e-8, coarse+width)),
                              method="bounded", options={"xatol": 1e-11})
    return float(optimum.x)


def spectral_candidates(field: np.ndarray, family: float | None = None) -> dict:
    power, grid, floor, nfft = spectrum(field)
    n = field.shape[1]
    peaks, _ = find_peaks(power, distance=max(2, int(2*nfft/n)))
    peaks = peaks[(grid[peaks] > 2*np.pi/n)&(grid[peaks] < np.pi-.03)
                  &(power[peaks] >= NOISE_FLOOR_MULTIPLIER*floor)]
    peaks = peaks[np.argsort(power[peaks])[::-1]]
    accepted = []
    excluded = []
    for index in peaks:
        frequency = refine_hann(field, float(grid[index]))
        gap = None
        order = None
        if family is not None:
            harmonics = np.arange(1, MAX_GATE_ORDER+1)
            distance = np.abs(np.angle(np.exp(1j*(frequency-harmonics*family))))
            reflected = np.abs(np.angle(np.exp(1j*(frequency+harmonics*family))))
            nearest = np.minimum(distance, reflected)
            order = int(np.argmin(nearest)+1)
            gap = float(np.min(nearest))
        record = {"frequency": frequency, "coarse_frequency": float(grid[index]),
                  "power": float(power[index]), "nearest_family_order": order,
                  "distance_to_first_family": gap}
        if gap is not None and gap <= FIRST_FAMILY_GATE:
            excluded.append(record)
        else:
            accepted.append(record)
    return {"candidate": accepted[0] if accepted else None,
            "accepted": accepted[:8], "excluded": excluded[:8],
            "significant_peak_count": len(peaks), "noise_floor": floor,
            "threshold": NOISE_FLOOR_MULTIPLIER*floor}


def design_matrix(times: np.ndarray, frequency: float, order: int) -> np.ndarray:
    angles = np.outer(times, frequency*np.arange(1, order+1))
    matrix = np.empty((len(times), 1+2*order))
    matrix[:, 0] = 1
    matrix[:, 1::2] = np.cos(angles)
    matrix[:, 2::2] = np.sin(angles)
    return matrix


def fit_at_frequency(field: np.ndarray, times: np.ndarray, frequency: float,
                     order: int) -> tuple[np.ndarray, float]:
    basis = design_matrix(times, frequency, order)
    coefficients, *_ = np.linalg.lstsq(basis, field.T, rcond=None)
    residual = field.T-np.einsum("ij,jk->ik", basis, coefficients, optimize=False)
    sse = float(np.sum(residual*residual))
    if not np.isfinite(coefficients).all() or not np.isfinite(sse):
        raise FloatingPointError("nonfinite variable-projection fit")
    return coefficients, sse


def variable_project(field: np.ndarray, times: np.ndarray, initial: float,
                     order: int, halfwidth: float) -> tuple[float, np.ndarray, float]:
    lower, upper = max(1e-7, initial-halfwidth), min(np.pi-1e-7, initial+halfwidth)
    grid = np.linspace(lower, upper, 13)
    values = [fit_at_frequency(field, times, float(value), order)[1] for value in grid]
    best = int(np.argmin(values))
    left, right = grid[max(0, best-1)], grid[min(len(grid)-1, best+1)]
    if left == right:
        frequency = float(grid[best])
    else:
        result = minimize_scalar(lambda value: fit_at_frequency(field, times, value, order)[1],
                                 bounds=(left, right), method="bounded",
                                 options={"xatol": 1e-11})
        frequency = float(result.x)
    coefficients, sse = fit_at_frequency(field, times, frequency, order)
    return frequency, coefficients, sse


def fit_first_family(train: np.ndarray) -> dict:
    first_screen = spectral_candidates(train)
    peak = first_screen["candidate"]
    if peak is None:
        return {"status": "no_first_peak", "screen": first_screen}
    initial = peak["frequency"]
    pilot_frequency, _, _ = variable_project(train, TIME_TRAIN, initial, 8,
                                               2*np.pi/(N_TRAIN*8))
    scored = []
    count = train.size
    for order in K_CANDIDATES:
        _, sse = fit_at_frequency(train, TIME_TRAIN, pilot_frequency, order)
        parameters = Q*(2*order+1)+1
        bic = count*np.log(sse/count)+parameters*np.log(count)
        scored.append({"order": order, "bic": float(bic), "train_sse": sse})
    selected = min(scored, key=lambda item: item["bic"])["order"]
    frequency, coefficients, final_sse = variable_project(
        train, TIME_TRAIN, pilot_frequency, selected,
        np.pi/(N_TRAIN*selected))
    single_frequency, single_coefficients, single_sse = variable_project(
        train, TIME_TRAIN, initial, 1, 2*np.pi/N_TRAIN)
    return {"status": "fit", "first_peak": peak, "frequency": frequency,
            "order": selected, "coefficients": coefficients,
            "train_sse": final_sse, "bic_grid": scored,
            "pilot_frequency": pilot_frequency,
            "single_frequency": single_frequency,
            "single_coefficients": single_coefficients,
            "single_train_sse": single_sse}


def project_test(fit: dict, times: np.ndarray, single: bool = False) -> np.ndarray:
    frequency = fit["single_frequency"] if single else fit["frequency"]
    order = 1 if single else fit["order"]
    coefficients = fit["single_coefficients"] if single else fit["coefficients"]
    predicted = np.einsum("ij,jk->ik", design_matrix(times, frequency, order),
                          coefficients, optimize=False).T
    if not np.isfinite(predicted).all():
        raise FloatingPointError("nonfinite held-out family projection")
    return predicted


def ssa_dmd(field: np.ndarray, frequency: float, rank: int,
            weights: np.ndarray) -> dict:
    h = trajectory_matrix(field, weights, DESIGN)
    centered = h-h.mean(axis=1, keepdims=True)
    covariance = np.einsum("ij,kj->ik", centered[:, :-1], centered[:, :-1],
                           optimize=False)
    if not np.isfinite(covariance).all():
        raise FloatingPointError("nonfinite SSA covariance")
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    basis = eigenvectors[:, -rank:][:, ::-1]
    coordinates = np.einsum("ij,ik->jk", basis, centered, optimize=False)
    z0, z1 = coordinates[:, :-1], coordinates[:, 1:]
    cross = np.einsum("ij,kj->ik", z1, z0, optimize=False)
    gram = np.einsum("ij,kj->ik", z0, z0, optimize=False)
    shift = np.einsum("ij,jk->ik", cross, np.linalg.pinv(gram), optimize=False)
    if not np.isfinite(shift).all():
        raise FloatingPointError("nonfinite projected DMD shift")
    values, vectors = np.linalg.eig(shift)
    choice = int(np.argmin(abs(values-np.exp(1j*frequency))))
    phase = np.angle(np.linalg.solve(vectors, coordinates)[choice])
    return {"phase": phase, "eigenfrequency": float(np.angle(values[choice])),
            "frequency_error": float(abs(np.angle(values[choice]*np.exp(-1j*frequency)))),
            "leading_variance_fraction": float(np.sum(eigenvalues[-rank:])/np.sum(eigenvalues))}


def target_audit(candidate: float | None, metadata: dict) -> dict:
    if candidate is None:
        return {"class": "none", "matched_name": None, "distance": None}
    targets = metadata["targets"]
    distances = [(abs(np.angle(np.exp(1j*(candidate-item["frequency"])))), item)
                 for item in targets]
    if not distances:
        return {"class": "false_positive_single_phase", "matched_name": None,
                "distance": None}
    gap, item = min(distances, key=lambda pair: pair[0])
    if gap <= 2*np.pi/N_TEST:
        return {"class": "weak_target", "matched_name": item["name"],
                "distance": float(gap), "mode": item["mode"]}
    return {"class": "other_or_first_family_leakage", "matched_name": None,
            "distance_to_nearest_weak": float(gap)}


def score_readouts(candidate: float, metadata: dict, match: dict,
                   fields: dict[str, np.ndarray]) -> dict:
    target = next(item for item in metadata["targets"]
                  if item["name"] == match["matched_name"])
    truth = target["phase"][:N_TEST-L+1]
    direct_phase = candidate*np.arange(len(truth))
    scores = {"direct_spectrum": {"phase_mae_rad": circular_mae(
        direct_phase[None, :], truth[None, :])}}
    settings = (("raw_classical_mssa_rank4", "raw", 4, np.full(Q, 1/Q)),
                ("raw_spatial_mssa_rank4", "raw", 4, AREA),
                ("single_sinusoid_residual_ssa_rank2", "single", 2, AREA),
                ("family_residual_ssa_rank2", "family", 2, AREA),
                ("oracle_ideal_conditional_residual_ssa_rank2", "oracle", 2, AREA))
    for name, field_name, rank, weights in settings:
        try:
            readout = ssa_dmd(fields[field_name], candidate, rank, weights)
            scores[name] = {"phase_mae_rad": circular_mae(readout["phase"][None, :],
                                                            truth[None, :]),
                            "eigenfrequency": readout["eigenfrequency"],
                            "frequency_error": readout["frequency_error"],
                            "leading_variance_fraction": readout["leading_variance_fraction"]}
            if name == "family_residual_ssa_rank2":
                scores[name]["phase_sample"] = readout["phase"][::8].tolist()
        except (FloatingPointError, np.linalg.LinAlgError, ValueError) as error:
            scores[name] = {"failure": f"{type(error).__name__}: {error}"}
    return scores


def evaluate(case: Case, noise: float, seed: int,
             normalize_strong: bool = False) -> dict:
    times = np.arange(N_TRAIN+N_TEST)
    strong, weak, metadata = component_fields(case, times, normalize_strong)
    clean = strong+weak
    scale = np.sqrt(np.mean(np.sum(AREA[:, None]*clean**2, axis=0)))
    observed = clean+noise*scale*np.random.default_rng(seed).standard_normal(clean.shape)
    train, test = observed[:, :N_TRAIN], observed[:, N_TRAIN:]
    fitted = fit_first_family(train)
    row = {"case": case.name, "noise": noise, "seed": seed,
           "relative_noise_rms": noise, "field_rms": float(scale),
           "normalized_strong": normalize_strong,
           "analytic_strong_haar_rms_before_normalization":
               metadata["analytic_strong_haar_rms_before_normalization"],
           "strong_applied_multiplier": metadata["strong_applied_multiplier"],
           "truth_metadata_for_scoring_only": {
               "first_family_frequency": metadata["first_family_frequency"],
               "target_frequencies": {item["name"]: item["frequency"] for item in metadata["targets"]},
               "strong_lattice_index": metadata["strong_lattice_index"],
               "next_character_completes_lattice": metadata["next_character_completes_lattice"]}}
    first_family = metadata["first_family_frequency"]
    for target in metadata["targets"]:
        harmonics = np.arange(1, MAX_GATE_ORDER+1)*first_family
        distance = np.minimum(
            np.abs(np.angle(np.exp(1j*(target["frequency"]-harmonics)))),
            np.abs(np.angle(np.exp(1j*(target["frequency"]+harmonics)))))
        gap = float(np.min(distance))
        row["truth_metadata_for_scoring_only"].setdefault("target_family_separation", {})[
            target["name"]] = {"radians": gap, "within_one_rayleigh_gate":
                               bool(gap <= FIRST_FAMILY_GATE)}
    if fitted["status"] != "fit":
        row["status"] = "no_first_family"
        return row
    row["first_family_fit"] = {key: value for key, value in fitted.items()
                               if key not in ("coefficients", "single_coefficients")}
    family_residual = test-project_test(fitted, TIME_TEST)
    single_residual = test-project_test(fitted, TIME_TEST, single=True)
    oracle_residual = test-strong[:, N_TRAIN:]
    row["test_rms"] = {
        "raw": float(np.sqrt(np.mean(np.sum(AREA[:, None]*test**2, axis=0)))),
        "family_residual": float(np.sqrt(np.mean(np.sum(AREA[:, None]*family_residual**2, axis=0)))),
        "single_residual": float(np.sqrt(np.mean(np.sum(AREA[:, None]*single_residual**2, axis=0)))),
        "oracle_residual": float(np.sqrt(np.mean(np.sum(AREA[:, None]*oracle_residual**2, axis=0)))),
    }
    row["residual_screen"] = spectral_candidates(family_residual, fitted["frequency"])
    row["oracle_screen_diagnostic"] = spectral_candidates(oracle_residual,
                                                            metadata["first_family_frequency"])
    selected = row["residual_screen"]["candidate"]
    candidate = None if selected is None else selected["frequency"]
    row["candidate_audit_after_inference"] = target_audit(candidate, metadata)
    if candidate is None:
        row["status"] = "no_residual_candidate"
        return row
    if row["candidate_audit_after_inference"]["class"] != "weak_target":
        row["status"] = "wrong_residual_candidate"
        return row
    fields = {"raw": test, "single": single_residual, "family": family_residual,
              "oracle": oracle_residual}
    row["scores_on_common_candidate"] = score_readouts(
        candidate, {"targets": [{**item, "phase": item["phase"][N_TRAIN:]}
                                for item in metadata["targets"]]},
        row["candidate_audit_after_inference"], fields)
    row["status"] = "weak_candidate_scored"
    return row


def summary(rows: list[dict]) -> list[dict]:
    answer = []
    for case in CASES:
        for noise in NOISE_LEVELS:
            group = [item for item in rows if item["case"] == case.name and item["noise"] == noise]
            if not group:
                continue
            scored = [item for item in group if item["status"] == "weak_candidate_scored"]
            medians = {}
            methods = ("direct_spectrum", "raw_classical_mssa_rank4",
                       "raw_spatial_mssa_rank4", "single_sinusoid_residual_ssa_rank2",
                       "family_residual_ssa_rank2",
                       "oracle_ideal_conditional_residual_ssa_rank2")
            for method in methods:
                values = [item["scores_on_common_candidate"][method]["phase_mae_rad"]
                          for item in scored if "phase_mae_rad" in
                          item["scores_on_common_candidate"][method]]
                medians[method] = float(np.median(values)) if values else None
            failures = {}
            all_positive_failures = {}
            all_positive_successes = {}
            for method in methods:
                failures[method] = sum(
                    item["scores_on_common_candidate"].get(method, {}).get("phase_mae_rad", 0) > .3
                    for item in scored)
                if case.kind != "negative":
                    all_positive_successes[method] = sum(
                        item["scores_on_common_candidate"].get(method, {}).get("phase_mae_rad", float("inf")) <= .3
                        for item in scored)
                    all_positive_failures[method] = len(group)-all_positive_successes[method]
            answer.append({"case": case.name, "noise": noise, "trials": len(group),
                           "weak_candidate_scored": len(scored),
                           "no_candidate": sum(item["status"] == "no_residual_candidate" for item in group),
                           "wrong_candidate": sum(item["status"] == "wrong_residual_candidate" for item in group),
                           "median_mae_on_detected_weak_only": medians,
                           "failures_over_0_3rad_on_detected_weak_only": failures,
                           "positive_phase_successes_all_trials": all_positive_successes if case.kind != "negative" else None,
                           "positive_phase_failures_all_trials_including_detection_misses":
                               all_positive_failures if case.kind != "negative" else None,
                           "negative_false_positive_candidates":
                               sum(item.get("residual_screen", {}).get("candidate") is not None
                                   for item in group) if case.kind == "negative" else None})
    return answer


def align_to_truth(estimated: np.ndarray, truth: np.ndarray) -> np.ndarray:
    offset = np.angle(np.mean(np.exp(1j*(estimated-truth))))
    return estimated-offset


def torus_coordinates(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    radius = 2+.65*np.cos(second)
    return np.stack((radius*np.cos(first), radius*np.sin(first), .65*np.sin(second)))


def plot_example(row: dict, case: Case, normalize_strong: bool = False) -> str | None:
    if row["status"] != "weak_candidate_scored":
        return None
    match = row["candidate_audit_after_inference"]
    if match["matched_name"] not in ("weak_theta2", "weak_mixed_1_1"):
        return None
    _, _, meta = component_fields(case, TIME_TEST, normalize_strong)
    true_first = meta["theta1"][:N_TEST-L+1]
    target = next(item for item in meta["targets"] if item["name"] == match["matched_name"])
    true_second = target["phase"][:N_TEST-L+1]
    first = row["first_family_fit"]["frequency"]*TIME_TEST[:len(true_first)]
    first = align_to_truth(first, true_first)
    sample = np.array(row["scores_on_common_candidate"]["family_residual_ssa_rank2"].get("phase_sample", []))
    if not len(sample):
        return None
    indices = np.arange(0, len(true_first), 8)
    indices = indices[:len(sample)]
    second = align_to_truth(sample[:len(indices)], true_second[indices])
    truth_xyz = torus_coordinates(true_first[indices], true_second[indices])
    estimated_xyz = torus_coordinates(first[indices], second)
    first_mae = circular_mae(first[None, :], true_first[None, :])
    second_mae = row["scores_on_common_candidate"]["family_residual_ssa_rank2"]["phase_mae_rad"]
    figure = plt.figure(figsize=(10, 4.5))
    for number, (points, title) in enumerate(((truth_xyz, "True phase coordinates"),
                                               (estimated_xyz, "Estimated phase coordinates")), start=1):
        axis = figure.add_subplot(1, 2, number, projection="3d")
        axis.scatter(*points, c=np.mod(true_second[indices], 2*np.pi), cmap="hsv",
                     s=4, alpha=.7, vmin=0, vmax=2*np.pi)
        axis.set_title(title)
        axis.set_axis_off()
    figure.suptitle(f"{case.name}: fixed torus map of phases; MAE first={first_mae:.3f}, next={second_mae:.3f} rad\n"
                    "Torus shape is imposed for display, not recovered geometry")
    figure.tight_layout()
    suffix = "_normalized" if normalize_strong else ""
    path = OUT/f"conditional_torus_{case.name}{suffix}.png"
    figure.savefig(path, dpi=170)
    plt.close(figure)
    return str(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--normalize-strong", action="store_true")
    args = parser.parse_args()
    selected_cases = [case for case in CASES if case.name[0] in "abef"] if args.pilot else CASES
    noise_levels = (.08,) if args.pilot else NOISE_LEVELS
    seeds = (52000,) if args.pilot else SEEDS
    rows = []
    figures = []
    for case in selected_cases:
        for noise in noise_levels:
            for seed in seeds:
                row = evaluate(case, noise, seed, args.normalize_strong)
                rows.append(row)
                if not args.pilot and noise == .08 and seed == 52000 and case.name[0] in "abd":
                    path = plot_example(row, case, args.normalize_strong)
                    if path:
                        figures.append(path)
            last = rows[-1]
            print(case.name, noise, "status", last["status"],
                  "K", last.get("first_family_fit", {}).get("order"),
                  "candidate", last.get("residual_screen", {}).get("candidate", {}).get("frequency")
                  if last.get("residual_screen", {}).get("candidate") else None,
                  flush=True)
    payload = {
        "description": "Conditional next-character train/test assay on one irregularly sampled 2D spatial field; spectral first-family fit followed by ordinary residual SSA/DMD.",
        "scope_limitation": "First family comes from training spectrum and variable-projection Fourier regression. The subtracted-family SSA is algebraically the same as family-residual SSA; this assay does not infer a full unknown-r lattice or certify a torus.",
        "test_usage": "First-family coefficients and order are frozen from train. Residual candidate selection and DMD use the full held-out test block (transductive). Truth labels are used only after candidate inference for scoring; oracle residual is an ideal upper-control.",
        "samples": {"train": N_TRAIN, "test": N_TEST, "lag_window": L},
        "normalized_strong": args.normalize_strong,
        "normalization_rule": "Divide only the generated strong component by its exact Haar RMS before adding the unchanged weak component; then set Gaussian noise as the declared fraction of total clean-field RMS.",
        "sensors": SENSORS.tolist(), "voronoi_weights": AREA.tolist(),
        "noise_levels": list(noise_levels), "seeds": list(seeds),
        "parameters": {"BIC_harmonic_orders": list(K_CANDIDATES),
                       "first_family_gate": FIRST_FAMILY_GATE,
                       "max_gate_order": MAX_GATE_ORDER,
                       "noise_floor_multiplier": NOISE_FLOOR_MULTIPLIER,
                       "pad_factor": PAD_FACTOR},
        "cases": [case.__dict__ for case in selected_cases],
        "rows": rows, "summary": summary(rows), "plots": figures,
        "pilot": args.pilot,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    if args.pilot:
        print("PILOT", json.dumps(payload["summary"], indent=2), flush=True)
    else:
        OUT.mkdir(exist_ok=True)
        destination = OUT/("conditional_ssa_normalized.json" if args.normalize_strong
                           else "conditional_ssa.json")
        destination.write_text(json.dumps(payload, indent=2)+"\n")
        print("saved", destination, flush=True)


if __name__ == "__main__":
    main()
