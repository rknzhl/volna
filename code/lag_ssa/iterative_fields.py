"""Fixed spatial-field fixtures and truth-only scoring for iterative phase SSA.

The data generator uses one scalar field sampled at irregular spatial points.
All cases except ``spatial_hidden_weak_negative`` have profiles in the same
known six-dimensional degree-two polynomial space. Infinite Poisson families
are evaluated by closed kernels, never by a truncated Fourier synthesis.

``expected_observable_mode_groups`` and ``observable_lattice`` are ground
truth for evaluation. They must not be passed to an inference algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
from math import gcd, pi

import numpy as np
from sympy import Matrix
from sympy.matrices.normalforms import hermite_normal_form

from method import voronoi_weights_grid


SENSOR_SEED = 1741
SENSORS = np.random.default_rng(SENSOR_SEED).uniform(.06, .94, (12, 2))
AREA = voronoi_weights_grid(SENSORS, resolution=80)
ALPHA = np.array([.151*np.sqrt(2), .259*np.sqrt(3), .227*np.sqrt(5)])
ORIGIN = np.array([.23, .87, 1.19])
DEFAULT_N_TRAIN = 3200
NEAR_CYCLES = .5
SPATIAL_DIMENSION = 6


def _bounded_relation_gap(alpha: np.ndarray) -> float:
    """Audit only: smallest nonzero relation for |k_i|<=4, ||k||_1<=8."""
    values = []
    for coefficients in product(range(-4, 5), repeat=len(alpha)):
        if not any(coefficients) or sum(map(abs, coefficients)) > 8:
            continue
        values.append(abs(float(np.angle(np.exp(1j*np.dot(coefficients, alpha))))))
    return min(values)


BASE_ALPHA_BOUNDED_RELATION_GAP = {
    rank: _bounded_relation_gap(ALPHA[:rank]) for rank in (2, 3)
}


def _monomials(positions: np.ndarray) -> np.ndarray:
    points = np.asarray(positions, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("positions must have shape (sensors, 2)")
    x, y = (2*(points-.5)).T
    return np.stack((np.ones(len(points)), x, y, x*x, x*y, y*y), axis=1)


_RAW_AT_SENSORS = _monomials(SENSORS)
_RAW_GRAM = _RAW_AT_SENSORS.T@(AREA[:, None]*_RAW_AT_SENSORS)
_GRAM_EIGENVALUES, _GRAM_EIGENVECTORS = np.linalg.eigh(_RAW_GRAM)
if _GRAM_EIGENVALUES[0] <= 0:
    raise ArithmeticError("the degree-two spatial basis is not full rank")
_WHITENER = (_GRAM_EIGENVECTORS
             @ np.diag(1/np.sqrt(_GRAM_EIGENVALUES))
             @ _GRAM_EIGENVECTORS.T)
SPATIAL_GRAM_CONDITION = float(_GRAM_EIGENVALUES[-1]/_GRAM_EIGENVALUES[0])


def spatial_basis(positions: np.ndarray) -> np.ndarray:
    """The same known six polynomial functions evaluated at any 2D points."""
    return _monomials(positions)@_WHITENER


SPATIAL_BASIS = spatial_basis(SENSORS)
SPATIAL_BASIS_GRAM = SPATIAL_BASIS.T@(AREA[:, None]*SPATIAL_BASIS)
SPATIAL_BASIS_GRAM_CONDITION = float(np.linalg.cond(SPATIAL_BASIS_GRAM))


def _profile_coefficients(index: int) -> np.ndarray:
    angle = .73*(index+1)
    coefficients = np.array([
        1., .46*np.cos(angle), .44*np.sin(angle),
        .30*np.cos(2*angle), .27*np.sin(2*angle), .19*np.cos(3*angle),
    ])
    return coefficients/np.linalg.norm(coefficients)


PROFILE_COEFFICIENTS = np.stack([_profile_coefficients(j) for j in range(12)])


def hidden_profile(positions: np.ndarray) -> np.ndarray:
    """Smooth profile vanishing at every training sensor; outside known V."""
    points = np.asarray(positions, dtype=float)
    squared = np.sum((points[:, None, :]-SENSORS[None, :, :])**2, axis=2)
    value = np.prod(-np.expm1(-squared/.035), axis=1)
    return value/_HIDDEN_GRID_MAX


_GRID_AXIS = np.linspace(.05, .95, 31)
_GRID_POINTS = np.array(list(product(_GRID_AXIS, repeat=2)))
_GRID_SQUARED = np.sum((_GRID_POINTS[:, None, :]-SENSORS[None, :, :])**2, axis=2)
_HIDDEN_GRID_MAX = float(np.max(np.prod(-np.expm1(-_GRID_SQUARED/.035), axis=1)))
if _HIDDEN_GRID_MAX <= 0:
    raise ArithmeticError("hidden-profile normalization failed")


@dataclass(frozen=True)
class FieldCase:
    name: str
    phase_rank: int
    kind: str
    description: str
    violates_stable_spatial_model: bool = False


CASES = (
    FieldCase("r1_poisson2_cos3", 1, "poisson2_cos3",
              "Unit-Haar-RMS infinite Poisson(2 theta_1) plus weak cos(3 theta_1); no fundamental Fourier line."),
    FieldCase("r1_even_2_4_factor", 1, "even_factor",
              "Only characters 2 and 4: the observable circle is a twofold phase factor."),
    FieldCase("r2_poisson_dominant_weak", 2, "dominant_weak",
              "Unit-Haar-RMS infinite Poisson(theta_1) plus weak cos(theta_2)."),
    FieldCase("r2_partial_cover", 2, "partial_cover",
              "Infinite Poisson(2 theta_1), then theta_2, then weak 3 theta_1; the first two groups retain index two."),
    FieldCase("r2_mixed_no_fundamentals", 2, "mixed_no_fundamentals",
              "Finite modes (2,0), (0,3), (1,1), (1,-1); neither coordinate fundamental is present."),
    FieldCase("r2_infinite_mixed_products", 2, "infinite_mixed",
              "Two infinite Poisson families and their product, containing infinitely many mixed characters."),
    FieldCase("r3_three_phases_weak_third", 3, "three_phases",
              "One infinite family and two further phases, with the third amplitude 0.08."),
    FieldCase("r1_poisson_negative", 1, "negative_one_phase",
              "Single infinite Poisson family; no second phase exists."),
    FieldCase("r2_near_resonant_half_cycle", 2, "near_resonant",
              "Second temporal increment is 2 alpha_1 + 0.5/n_train, below a Rayleigh bin."),
    FieldCase("spatial_hidden_weak_negative", 2, "spatial_hidden",
              "Physical weak second phase has a profile zero at all training sensors; violates the known spatial model.",
              True),
)
CASE_BY_NAME = {case.name: case for case in CASES}


@dataclass(frozen=True)
class _Term:
    name: str
    family: str
    modes: tuple[tuple[int, ...], ...]
    amplitude: float
    profile_index: int | None
    strong: bool
    rho: float | None = None
    sigma: float | None = None
    hidden: bool = False


def _terms(case: FieldCase) -> tuple[_Term, ...]:
    rank = case.phase_rank

    def mode(*coordinates: int) -> tuple[int, ...]:
        if len(coordinates) != rank:
            raise ArithmeticError("case mode rank mismatch")
        return tuple(coordinates)

    kind = case.kind
    if kind == "poisson2_cos3":
        return (_Term("strong_even_poisson", "poisson", (mode(2),), 1., 0, True, .70),
                _Term("weak_cover_cos3", "cos", (mode(3),), .08, 1, False))
    if kind == "even_factor":
        return (_Term("cos2", "cos", (mode(2),), 1., 0, True),
                _Term("cos4", "cos", (mode(4),), .75, 1, True))
    if kind == "dominant_weak":
        return (_Term("strong_poisson_theta1", "poisson", (mode(1, 0),), 1., 0, True, .70),
                _Term("weak_cos_theta2", "cos", (mode(0, 1),), .08, 1, False))
    if kind == "partial_cover":
        return (_Term("strong_even_poisson", "poisson", (mode(2, 0),), 1., 0, True, .70),
                _Term("second_phase_cos", "cos", (mode(0, 1),), .20, 1, False),
                _Term("cover_refinement_cos3", "cos", (mode(3, 0),), .08, 2, False))
    if kind == "mixed_no_fundamentals":
        return (_Term("cos_2_0", "cos", (mode(2, 0),), 1., 0, True),
                _Term("cos_0_3", "cos", (mode(0, 3),), .90, 1, True),
                _Term("cos_1_1", "cos", (mode(1, 1),), .32, 2, False),
                _Term("cos_1_minus1", "cos", (mode(1, -1),), .27, 3, False))
    if kind == "infinite_mixed":
        return (_Term("poisson_theta1", "poisson", (mode(1, 0),), 1., 0, True, .70),
                _Term("poisson_theta2", "poisson", (mode(0, 1),), .55, 1, True, .66),
                _Term("poisson_product", "poisson_product",
                       (mode(1, 0), mode(0, 1)), .35, 2, True, .70, .66))
    if kind == "three_phases":
        return (_Term("poisson_theta1", "poisson", (mode(1, 0, 0),), 1., 0, True, .70),
                _Term("cos_theta2", "cos", (mode(0, 1, 0),), .35, 1, False),
                _Term("weak_cos_theta3", "cos", (mode(0, 0, 1),), .08, 2, False))
    if kind == "negative_one_phase":
        return (_Term("single_poisson_theta1", "poisson", (mode(1),), 1., 0, True, .82),)
    if kind == "near_resonant":
        return tuple(_Term(f"strong_cos_{harmonic}", "cos", (mode(harmonic, 0),),
                           amplitude, harmonic-1, True)
                     for harmonic, amplitude in enumerate((1., .8, .6, .4), start=1)) + (
                         _Term("near_weak_cos_theta2", "cos", (mode(0, 1),), .08, 5, False),)
    if kind == "spatial_hidden":
        return (_Term("visible_poisson_theta1", "poisson", (mode(1, 0),), 1., 0, True, .70),
                _Term("spatially_invisible_cos_theta2", "cos", (mode(0, 1),),
                      .08, None, False, hidden=True))
    raise ValueError(f"unknown fixture kind {kind}")


def _unit_poisson(angle: np.ndarray, rho: float) -> np.ndarray:
    centered = (1-rho*rho)/(1-2*rho*np.cos(angle)+rho*rho)-1
    return centered/np.sqrt(2*rho*rho/(1-rho*rho))


def _temporal(term: _Term, theta: np.ndarray) -> np.ndarray:
    angles = [np.einsum("i,ij->j", np.asarray(mode), theta, optimize=False)
              for mode in term.modes]
    if term.family == "cos":
        return np.cos(angles[0])
    if term.family == "poisson":
        return _unit_poisson(angles[0], term.rho)
    if term.family == "poisson_product":
        return (_unit_poisson(angles[0], term.rho)
                *_unit_poisson(angles[1], term.sigma))
    raise ValueError(term.family)


def _variance(term: _Term) -> float:
    return .5 if term.family == "cos" else 1.


def _profile(term: _Term, positions: np.ndarray) -> np.ndarray:
    if term.hidden:
        return hidden_profile(positions)
    return spatial_basis(positions)@PROFILE_COEFFICIENTS[term.profile_index]


def _group_generators(term: _Term) -> list[tuple[int, ...]]:
    if term.family != "poisson_product":
        return [term.modes[0]]
    first, second = map(np.asarray, term.modes)
    # The product contains these characters among infinitely many others;
    # their two-row lattice is the complete product character lattice.
    return [tuple(first+second), tuple(2*first+second)]


def _lattice_signature(generators: list[tuple[int, ...]],
                       phase_rank: int) -> dict:
    if not generators:
        return {"rank": 0, "basis": [], "index": None, "cover_degree": None}
    integer = Matrix(generators)
    rank = int(integer.rank())
    hnf = hermite_normal_form(integer.T)
    basis = [[int(number) for number in row] for row in hnf.T.tolist()]
    if rank < phase_rank:
        index = None
    else:
        index = 0
        for rows in combinations(generators, phase_rank):
            determinant = abs(int(Matrix(rows).det()))
            index = gcd(index, determinant)
    return {"rank": rank, "basis": basis, "index": index,
            "cover_degree": index}


def resolve_case(case: FieldCase | str) -> FieldCase:
    if isinstance(case, FieldCase):
        if case.name not in CASE_BY_NAME or CASE_BY_NAME[case.name] != case:
            raise ValueError("FieldCase must belong to CASES")
        return case
    return CASE_BY_NAME[case]


def generate_field(case: FieldCase | str, times: np.ndarray,
                   positions: np.ndarray = SENSORS,
                   n_train: int = DEFAULT_N_TRAIN) -> dict:
    """Return a clean scalar field and truth metadata at arbitrary space/time.

    ``times`` are absolute sample indices; no fixed record length is assumed.
    Only the intentional near-resonance uses ``n_train``. Strong terms have
    exact unit Haar RMS after spatial weighting at the original sensors.
    """
    fixture = resolve_case(case)
    sample_times = np.asarray(times, dtype=float).reshape(-1)
    points = np.asarray(positions, dtype=float)
    if not len(sample_times) or points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("times nonempty and positions shaped (sensors,2) required")
    if n_train <= 0:
        raise ValueError("n_train must be positive")
    alpha = ALPHA[:fixture.phase_rank].copy()
    if fixture.kind == "near_resonant":
        alpha[1] = 2*alpha[0]+NEAR_CYCLES/n_train
    theta = ORIGIN[:fixture.phase_rank, None]+alpha[:, None]*sample_times[None, :]
    terms = _terms(fixture)
    strong_haar_rms_before = float(np.sqrt(sum(term.amplitude**2*_variance(term)
                                               for term in terms if term.strong)))
    strong_multiplier = 1/strong_haar_rms_before
    clean = np.zeros((len(points), len(sample_times)))
    covariance = np.zeros((len(points), len(points)))
    groups = []
    cumulative_generators: list[tuple[int, ...]] = []
    cumulative = []
    for term in terms:
        spatial = _profile(term, points)
        visible = bool(np.linalg.norm(spatial) > 1e-12)
        amplitude = term.amplitude*(strong_multiplier if term.strong else 1.)
        clean += amplitude*spatial[:, None]*_temporal(term, theta)[None, :]
        covariance += amplitude**2*_variance(term)*np.outer(spatial, spatial)
        generators = _group_generators(term)
        if visible:
            cumulative_generators.extend(generators)
        signature = _lattice_signature(cumulative_generators, fixture.phase_rank)
        cumulative.append({"after_group": term.name, **signature})
        groups.append({
            "name": term.name, "family": term.family,
            "infinite_harmonics": term.family != "cos",
            "base_modes": [list(mode) for mode in term.modes],
            "lattice_generators_from_support": [list(mode) for mode in generators],
            "amplitude_after_strong_normalization": amplitude,
            "profile_index_in_known_basis": term.profile_index,
            "profile_coefficients_in_known_basis":
                PROFILE_COEFFICIENTS[term.profile_index].tolist()
                if term.profile_index is not None else None,
            "profile_visible_on_these_positions": visible,
            "outside_known_spatial_basis": term.hidden,
            "rho": term.rho, "sigma": term.sigma,
            "strong_term": term.strong,
        })
    lattice = _lattice_signature(cumulative_generators, fixture.phase_rank)
    lattice["cumulative_group_indices"] = cumulative
    lattice["spatial_covariance_haar"] = covariance
    lattice["scope"] = ("Fourier character module and translation cover only; "
                        "lattice rank/index do not certify injectivity of "
                        "the instantaneous field or of a finite delay map")
    return {
        "clean": clean,
        "theta": theta,
        "expected_observable_mode_groups": groups,
        "observable_lattice": lattice,
        "metadata": {
            "case": fixture.name, "description": fixture.description,
            "physical_phase_rank": fixture.phase_rank,
            "alpha": alpha,
            "near_delta": NEAR_CYCLES/n_train if fixture.kind == "near_resonant" else None,
            "n_train_parameter": n_train,
            "base_alpha_bounded_relation_gap_audit":
                BASE_ALPHA_BOUNDED_RELATION_GAP.get(fixture.phase_rank),
            "relation_gap_audit_bound": "|k_i|<=4, l1(k)<=8; not a global Diophantine certificate",
            "strong_haar_rms_before_normalization": strong_haar_rms_before,
            "strong_applied_multiplier": strong_multiplier,
            "known_spatial_basis": not fixture.violates_stable_spatial_model,
            "spatial_model_exception": fixture.violates_stable_spatial_model,
            "infinite_terms_evaluated_by_closed_poisson_kernel":
                any(term.family != "cos" for term in terms),
            "sensor_count": len(points),
        },
    }


def truth_mode_catalog(case: FieldCase | str, max_harmonic: int = 8,
                       n_train: int = DEFAULT_N_TRAIN,
                       positions: np.ndarray = SENSORS) -> list[dict]:
    """Finite scoring catalog; the generated Poisson field is still infinite.

    Mode labels and frequencies are ground truth and must only be consulted
    after frequency inference. A finite catalog cannot certify no higher line.
    """
    fixture = resolve_case(case)
    if max_harmonic < 1:
        raise ValueError("max_harmonic must be positive")
    alpha = ALPHA[:fixture.phase_rank].copy()
    if fixture.kind == "near_resonant":
        alpha[1] = 2*alpha[0]+NEAR_CYCLES/n_train
    records = []
    seen = set()
    for term in _terms(fixture):
        if np.linalg.norm(_profile(term, positions)) <= 1e-12:
            continue
        if term.family == "cos":
            modes = [term.modes[0]]
        elif term.family == "poisson":
            base = np.asarray(term.modes[0])
            modes = [tuple(harmonic*base) for harmonic in range(1, max_harmonic+1)]
        else:
            first, second = map(np.asarray, term.modes)
            modes = [tuple(sign1*m*first+sign2*n*second)
                     for m in range(1, max_harmonic+1)
                     for n in range(1, max_harmonic+1)
                     for sign1, sign2 in ((1, 1), (1, -1))]
        for mode in modes:
            angle = float(np.angle(np.exp(1j*np.dot(mode, alpha))))
            oriented = tuple(int(value) for value in
                             (mode if angle >= 0 else -np.asarray(mode)))
            positive = abs(angle)
            key = (term.name, oriented)
            if key not in seen:
                seen.add(key)
                records.append({"group": term.name, "mode": list(oriented),
                                "positive_frequency": positive,
                                "infinite_family": term.family != "cos"})
    return records


def score_frequency(case: FieldCase | str, inferred_frequency: float,
                    tolerance: float, max_harmonic: int = 8,
                    n_train: int = DEFAULT_N_TRAIN,
                    positions: np.ndarray = SENSORS) -> dict:
    """Truth-only post-inference label with explicit ambiguity and truncation."""
    catalog = truth_mode_catalog(case, max_harmonic, n_train, positions)
    matched = [entry for entry in catalog
               if abs(np.angle(np.exp(1j*(inferred_frequency-entry["positive_frequency"]))))
               <= tolerance]
    return {"matches": matched, "unique_match": matched[0] if len(matched) == 1 else None,
            "ambiguous": len(matched) > 1,
            "unmatched_within_bounded_catalog": len(matched) == 0,
            "catalog_max_harmonic": max_harmonic,
            "tolerance": tolerance}


def validate_library() -> dict:
    """Small deterministic fixture checks; no inference or stochastic test."""
    checks = []
    for fixture in CASES:
        result = generate_field(fixture, np.arange(512))
        observed = result["observable_lattice"]
        clean = result["clean"]
        if not np.isfinite(clean).all():
            raise AssertionError(f"nonfinite fixture: {fixture.name}")
        if fixture.kind == "spatial_hidden":
            if np.max(abs(_profile(_terms(fixture)[-1], SENSORS))) != 0:
                raise AssertionError("hidden weak profile is visible at a training sensor")
            if observed["rank"] != 1:
                raise AssertionError("hidden weak mode changed the observable rank")
        checks.append({"case": fixture.name, "physical_rank": fixture.phase_rank,
                       "observable_rank": observed["rank"],
                       "index": observed["index"],
                       "basis": observed["basis"],
                       "strong_haar_rms_before_normalization":
                           result["metadata"]["strong_haar_rms_before_normalization"],
                       "infinite_closed_kernel":
                           result["metadata"]["infinite_terms_evaluated_by_closed_poisson_kernel"]})
    return {"sensor_count": len(SENSORS),
            "spatial_basis_dimension": SPATIAL_DIMENSION,
            "raw_spatial_gram_condition": SPATIAL_GRAM_CONDITION,
            "whitened_spatial_gram_condition": SPATIAL_BASIS_GRAM_CONDITION,
            "max_whitened_gram_error": float(np.max(abs(SPATIAL_BASIS_GRAM-np.eye(SPATIAL_DIMENSION)))),
            "base_alpha_bounded_relation_gap": BASE_ALPHA_BOUNDED_RELATION_GAP,
            "cases": checks}


if __name__ == "__main__":
    import json
    print(json.dumps(validate_library(), indent=2))
