"""Deterministic algebra checks for an annihilator contrast on delay fields.

For a Fourier sublattice Gamma, A_Gamma averages translations over its
annihilator. C_Gamma = I - A_Gamma therefore removes every Gamma character.
This checks that algebra, including infinite Poisson Fourier series; it does
not estimate Gamma or its annihilator from observations.
"""

from __future__ import annotations

from project_paths import RESULTS_DIR

import hashlib
import json
from itertools import combinations
from math import gcd, pi
from pathlib import Path

import numpy as np


OUT = RESULTS_DIR / "conditional_projection_checks.json"
SENSORS = np.array([
    [-0.81, -0.72], [-0.63, 0.31], [-0.19, -0.17], [0.04, 0.76],
    [0.27, -0.65], [0.43, 0.19], [0.74, -0.11], [0.88, 0.67],
], dtype=float)
X, Y = SENSORS.T
PROFILE_A = 1.1 + 0.35*X + 0.18*Y
PROFILE_B = 0.75 - 0.26*X + 0.31*Y
PROFILE_C = 0.9 + 0.17*X - 0.29*Y
PROFILE_D = 0.3 + 0.13*X*Y
EPSILON = 0.12
RHO = 0.54
SIGMA = 0.62
ALPHA = np.array([0.173*np.sqrt(2), 0.217*np.sqrt(3)])
LAGS = np.array([0, 1, 3, 8, 13])


def poisson_zero_mean(angle: np.ndarray, rho: float) -> np.ndarray:
    """2 sum_{m>=1} rho^m cos(m angle), evaluated by its closed form."""
    return (1-rho*rho)/(1-2*rho*np.cos(angle)+rho*rho)-1


def strong_2z(theta: np.ndarray) -> np.ndarray:
    return PROFILE_A[:, None]*poisson_zero_mean(2*theta[0], RHO)


def weak_2z(theta: np.ndarray) -> np.ndarray:
    return EPSILON*PROFILE_B[:, None]*np.cos(3*theta[0])


def strong_2z_times_z(theta: np.ndarray) -> np.ndarray:
    first = poisson_zero_mean(2*theta[0], RHO)
    second = poisson_zero_mean(theta[1], SIGMA)
    return (PROFILE_A[:, None]*first + PROFILE_C[:, None]*second
            + PROFILE_D[:, None]*first*second)


def weak_2z_times_z(theta: np.ndarray) -> np.ndarray:
    return EPSILON*PROFILE_B[:, None]*np.cos(3*theta[0])


def strong_rank_one(theta: np.ndarray) -> np.ndarray:
    return PROFILE_A[:, None]*poisson_zero_mean(theta[0], RHO)


def weak_rank_one(theta: np.ndarray) -> np.ndarray:
    return EPSILON*PROFILE_B[:, None]*poisson_zero_mean(theta[1], SIGMA)


def strong_index_three(theta: np.ndarray) -> np.ndarray:
    first = poisson_zero_mean(2*theta[0]+theta[1], RHO)
    second = poisson_zero_mean(theta[0]+2*theta[1], SIGMA)
    return (PROFILE_A[:, None]*first + PROFILE_C[:, None]*second
            + PROFILE_D[:, None]*first*second)


def weak_index_three(theta: np.ndarray) -> np.ndarray:
    return EPSILON*PROFILE_B[:, None]*np.cos(theta[0])


def shifts_2z() -> np.ndarray:
    return np.array([[0.], [pi]])


def shifts_2z_times_z() -> np.ndarray:
    return np.array([[0., 0.], [pi, 0.]])


def shifts_rank_one(order: int) -> np.ndarray:
    return np.stack((np.zeros(order), 2*pi*np.arange(order)/order), axis=1)


def shifts_index_three() -> np.ndarray:
    j = np.arange(3)
    return np.stack((2*pi*j/3, 2*pi*j/3), axis=1)


def average(field, theta: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    return sum(field(theta + shift[:, None]) for shift in shifts)/len(shifts)


def contrast(field, theta: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    return field(theta)-average(field, theta, shifts)


def delay(field, theta: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    return np.concatenate([field(theta + alpha[:, None]*lag) for lag in LAGS], axis=0)


def maximum_error(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.max(np.abs(first-second)))


def lattice_signature(modes: list[tuple[int, ...]], dimension: int) -> dict:
    if dimension == 1:
        index = gcd(*(abs(mode[0]) for mode in modes))
        return {"rank": int(index != 0), "index_if_full_rank": index or None}
    if dimension != 2:
        raise ValueError("this checker only uses phase ranks 1 and 2")
    minors = [abs(a[0]*b[1]-a[1]*b[0]) for a, b in combinations(modes, 2)]
    index = gcd(*minors) if minors else 0
    rank = 2 if index else int(any(any(value for value in mode) for mode in modes))
    return {"rank": rank, "index_if_full_rank": index or None}


def character_action(theta: np.ndarray, shifts: np.ndarray,
                     mode: tuple[int, ...], retained: bool) -> dict:
    mode_array = np.array(mode)
    character = np.exp(1j*(mode_array@theta))
    averaged = sum(np.exp(1j*(mode_array@(theta+shift[:, None])))
                   for shift in shifts)/len(shifts)
    expected = character if retained else np.zeros_like(character)
    return {"mode": list(mode), "retained_by_average": retained,
            "max_error": maximum_error(averaged, expected)}


def case_check(name: str, strong, weak, shifts: np.ndarray,
               phase: np.ndarray, alpha: np.ndarray,
               strong_modes: list[tuple[int, ...]],
               weak_mode: tuple[int, ...],
               retained_modes: list[tuple[int, ...]],
               removed_modes: list[tuple[int, ...]]) -> dict:
    full = lambda argument: strong(argument)+weak(argument)
    strong_value = strong(phase)
    weak_value = weak(phase)
    averaged_strong = average(strong, phase, shifts)
    averaged_weak = average(weak, phase, shifts)
    contrast_full = contrast(full, phase, shifts)
    delayed_contrast = delay(lambda argument: contrast(full, argument, shifts),
                             phase, alpha)
    contrast_delayed = (delay(full, phase, alpha)
                        - average(lambda argument: delay(full, argument, alpha),
                                  phase, shifts))
    action = ([character_action(phase, shifts, mode, True)
               for mode in retained_modes]
              + [character_action(phase, shifts, mode, False)
                 for mode in removed_modes])
    result = {
        "name": name, "annihilator_shifts": shifts.tolist(),
        "strong_generator_modes": [list(mode) for mode in strong_modes],
        "weak_generator_mode": list(weak_mode),
        "strong_module": lattice_signature(strong_modes, phase.shape[0]),
        "after_weak_mode": lattice_signature(strong_modes+[weak_mode], phase.shape[0]),
        "max_A_strong_minus_strong": maximum_error(averaged_strong, strong_value),
        "max_A_weak": float(np.max(np.abs(averaged_weak))),
        "max_C_strong": float(np.max(np.abs(strong_value-averaged_strong))),
        "max_C_full_minus_weak": maximum_error(contrast_full, weak_value),
        "max_delay_commutator": maximum_error(delayed_contrast, contrast_delayed),
        "mode_actions": action,
    }
    return result


def cover_check(strong, weak, base: np.ndarray, shifts: np.ndarray) -> dict:
    full = lambda argument: strong(argument)+weak(argument)
    translated = [base+shift[:, None] for shift in shifts]
    strong_values = [strong(point) for point in translated]
    full_values = [full(point) for point in translated]
    pairwise = list(combinations(range(len(shifts)), 2))
    return {
        "base_phase": base[:, 0].tolist(),
        "max_strong_fiber_difference": max(maximum_error(strong_values[i], strong_values[j])
                                            for i, j in pairwise),
        "min_full_fiber_distance": min(float(np.linalg.norm(full_values[i]-full_values[j]))
                                       for i, j in pairwise),
    }


def rank_one_quadrature(phase: np.ndarray) -> list[dict]:
    """A_Q weak = epsilon*b*(P_{sigma^Q}(Q theta_2)-1) exactly."""
    results = []
    full = lambda argument: strong_rank_one(argument)+weak_rank_one(argument)
    for order in (8, 16, 32):
        shifts = shifts_rank_one(order)
        averaged = average(full, phase, shifts)
        exact_alias = (strong_rank_one(phase)
                       + EPSILON*PROFILE_B[:, None]
                       * poisson_zero_mean(order*phase[1], SIGMA**order))
        residual = contrast(full, phase, shifts)-weak_rank_one(phase)
        # The uniform-in-phase Poisson tail is exact at theta_2 = 0.
        bound = (EPSILON*float(np.max(np.abs(PROFILE_B)))
                 * 2*SIGMA**order/(1-SIGMA**order))
        results.append({
            "order": order,
            "max_discrete_average_minus_closed_alias": maximum_error(averaged, exact_alias),
            "max_contrast_minus_ideal_weak": float(np.max(np.abs(residual))),
            "analytic_uniform_tail_bound": bound,
            "max_delay_commutator": maximum_error(
                delay(lambda argument: contrast(full, argument, shifts), phase, ALPHA),
                delay(full, phase, ALPHA)
                - average(lambda argument: delay(full, argument, ALPHA), phase, shifts)),
        })
    return results


def main() -> None:
    rng = np.random.default_rng(24680)
    phase_one = np.column_stack((np.zeros(1), rng.uniform(0, 2*pi, size=(1, 63))))
    phase_two = np.column_stack((np.zeros((2, 1)), rng.uniform(0, 2*pi, size=(2, 63))))
    cases = [
        case_check("rank1_2Z_infinite_strong_plus_cos3", strong_2z, weak_2z,
                   shifts_2z(), phase_one, ALPHA[:1], [(2,)], (3,),
                   [(2,), (4,), (6,)], [(1,), (3,), (5,)]),
        case_check("rank2_2Z_times_Z_infinite_strong_plus_cos3", strong_2z_times_z,
                   weak_2z_times_z, shifts_2z_times_z(), phase_two, ALPHA,
                   [(2, 0), (0, 1)], (3, 0),
                   [(2, 0), (4, 1), (0, 3)], [(1, 0), (3, 0), (1, 1)]),
        case_check("rankdef_Z_1_0_continuous_annihilator", strong_rank_one,
                   weak_rank_one, shifts_rank_one(32), phase_two, ALPHA,
                   [(1, 0)], (0, 1),
                   [(1, 0), (2, 0)], [(0, 1), (1, 2)]),
        case_check("rank2_noncoordinate_index3_infinite_strong_plus_cos1",
                   strong_index_three, weak_index_three, shifts_index_three(),
                   phase_two, ALPHA, [(2, 1), (1, 2)], (1, 0),
                   [(2, 1), (1, 2), (3, 3)], [(1, 0), (0, 1)]),
    ]
    covers = {
        "rank1_index2_to1": cover_check(strong_2z, weak_2z,
                                         np.array([[0.31]]), shifts_2z()),
        "rank2_index2_to1": cover_check(strong_2z_times_z, weak_2z_times_z,
                                         np.array([[0.31], [0.47]]), shifts_2z_times_z()),
        "rank2_index3_to1": cover_check(strong_index_three, weak_index_three,
                                         np.array([[0.31], [0.47]]), shifts_index_three()),
    }
    quadrature = rank_one_quadrature(phase_two)
    # If Gamma already equals the complete observable lattice, Ann(Gamma)
    # is trivial and the contrast cannot reveal another character.
    full_lattice_zero = {
        "rank1_2_3_module_Z": float(np.max(np.abs(contrast(
            lambda argument: strong_2z(argument)+weak_2z(argument),
            phase_one, np.zeros((1, 1)))))),
        "rank2_2Z_times_Z_plus_3_0_module_Z2": float(np.max(np.abs(contrast(
            lambda argument: strong_2z_times_z(argument)+weak_2z_times_z(argument),
            phase_two, np.zeros((1, 2)))))),
        "rank2_index3_plus_1_0_module_Z2": float(np.max(np.abs(contrast(
            lambda argument: strong_index_three(argument)+weak_index_three(argument),
            phase_two, np.zeros((1, 2)))))),
    }
    finite_cases = [cases[0], cases[1], cases[3]]
    maximum_finite_error = max(
        max(case["max_A_strong_minus_strong"], case["max_A_weak"],
            case["max_C_strong"], case["max_C_full_minus_weak"],
            case["max_delay_commutator"],
            *(entry["max_error"] for entry in case["mode_actions"]))
        for case in finite_cases)
    assert maximum_finite_error < 1e-11
    assert all(row["max_discrete_average_minus_closed_alias"] < 1e-11
               and row["max_contrast_minus_ideal_weak"]
               <= row["analytic_uniform_tail_bound"]+1e-11
               for row in quadrature)
    assert all(value["max_strong_fiber_difference"] < 1e-11
               and value["min_full_fiber_distance"] > 1e-4
               for value in covers.values())
    assert all(value == 0 for value in full_lattice_zero.values())
    payload = {
        "description": "Exact annihilator averaging removes the nominated Fourier sublattice via I-A_Gamma.",
        "limitation": "The Fourier sublattice, annihilator shifts, and weak mode are supplied exactly; this checker does not extract them from observed data or establish a blind SSA method.",
        "field_type": "closed-form Poisson kernels with infinitely many harmonics and nonconstant spatial profiles",
        "sensor_coordinates": SENSORS.tolist(),
        "spatial_profiles": {"a": PROFILE_A.tolist(), "b": PROFILE_B.tolist(),
                             "c": PROFILE_C.tolist(), "d": PROFILE_D.tolist()},
        "parameters": {"epsilon": EPSILON, "rho": RHO, "sigma": SIGMA,
                       "alpha": ALPHA.tolist(), "lags": LAGS.tolist(),
                       "phase_samples_per_case": 64, "rng_seed": 24680},
        "cases": cases,
        "continuous_annihilator_quadrature": quadrature,
        "indexed_cover_refinement": covers,
        "full_observable_lattice_zero": full_lattice_zero,
        "maximum_finite_group_check_error": maximum_finite_error,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2)+"\n")
    print("max finite-group error:", maximum_finite_error)
    for row in quadrature:
        print("Q", row["order"], "quadrature error",
              row["max_contrast_minus_ideal_weak"], "bound",
              row["analytic_uniform_tail_bound"])
    print("saved", OUT)


if __name__ == "__main__":
    main()
