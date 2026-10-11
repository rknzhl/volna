"""Six predeclared fresh fields: public Y,s,t and separate evaluation archives.

Generation knows the analytic field. The inference runner receives only the
public observation path/hash, run name and common sample counts. No generator
rank, frequencies, origins, equation or character catalogue enters fitting.
"""

from itertools import product
import hashlib
from pathlib import Path

import numpy as np


TRAIN, VALIDATION, FUTURE = 2400, 1200, 1200
NOISE_RATIO = .08
CASES = (
    dict(name="fresh_r2_primitive_cover6", rank=2, spatial_dimension=2, sensor_count=10,
         family="primitive", network="spread", seed_index=1),
    dict(name="fresh_r3_infinite_weak", rank=3, spatial_dimension=3, sensor_count=16,
         family="three_infinite", network="spread", seed_index=2),
    dict(name="fresh_r3_nofund_cover3", rank=3, spatial_dimension=2, sensor_count=12,
         family="mixed_finite", network="anisotropic", seed_index=3),
    dict(name="fresh_r2_irrational_travelling_1d", rank=2, spatial_dimension=1, sensor_count=10,
         family="travelling_1d", network="irregular", seed_index=4),
    dict(name="fresh_weak_phase06", rank=2, spatial_dimension=2, sensor_count=10,
         family="weak", weak_amplitude=.06, network="boundary_cluster", seed_index=5),
    dict(name="fresh_weak_phase008", rank=2, spatial_dimension=2, sensor_count=10,
         family="weak", weak_amplitude=.008, network="boundary_cluster", seed_index=5),
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def positions(specification):
    rng = np.random.default_rng(814100+specification["seed_index"])
    q, d = specification["sensor_count"], specification["spatial_dimension"]
    if specification["network"] == "boundary_cluster":
        return np.column_stack((rng.uniform(.02, .22, q), rng.uniform(.05, .95, q)))
    if d == 1:
        return np.sort(np.r_[.025, .975, rng.uniform(.04, .96, q-2)])[:, None]
    if specification["network"] == "anisotropic":
        corners = np.asarray([[x, y] for x, y in product((.20, .95), (.04, .40))])
        return np.vstack((corners, np.column_stack((rng.uniform(.24, .91, q-4),
                                                    rng.uniform(.07, .37, q-4)))))
    corners = np.asarray(list(product((.05, .95), repeat=d)))
    return np.vstack((corners, rng.uniform(.10, .90, (q-len(corners), d))))


def rotations(specification):
    rng = np.random.default_rng(714100+specification["seed_index"])
    rank = specification["rank"]
    primes = np.asarray([2, 7] if specification["family"] == "travelling_1d" else [2, 3, 5])[:rank]
    denominators = np.asarray([9, 12] if specification["family"] == "travelling_1d" else [9, 7, 6])[:rank]
    rational = rng.uniform(.9, 1.1, rank)
    return rational*np.sqrt(primes)/denominators, rational, primes, denominators


def terms(specification):
    family = specification["family"]
    if family == "primitive":
        return [("cos", row, amplitude, None) for row, amplitude in
                (((3, 0), 1.), ((0, 2), .7), ((1, 1), .22), ((1, -1), .18))]
    if family == "three_infinite":
        return [("poisson", (1, 0, 0), 1., .72), ("cos", (0, 1, 0), .28, None),
                ("cos", (0, 0, 1), .055, None), ("cos", (1, 1, -1), .11, None)]
    if family == "mixed_finite":
        initial = [("cos", row, amplitude, None) for row, amplitude in
                   (((2, 1, 0), 1.), ((0, 1, 1), .7), ((1, 0, 1), .45), ((1, 1, 1), .16))]
        rng = np.random.default_rng(914100+specification["seed_index"])
        retained = {item[1] for item in initial}
        while len(initial) < 12:
            row = rng.integers(-3, 4, 3)
            order = int(np.sum(abs(row)))
            if not 1 < order <= 4:
                continue
            if next(x for x in row if x) < 0:
                row = -row
            mode = tuple(int(x) for x in row)
            if mode in retained:
                continue
            retained.add(mode)
            initial.append(("cos", mode, float(.13*rng.uniform(.7, 1.3)/(1+order)), None))
        return initial
    if family == "travelling_1d":
        return [("cos", row, amplitude, None) for row, amplitude in
                (((1, 0), 1.), ((0, 1), .5), ((1, 1), .18), ((2, -1), .12))]
    if family == "weak":
        return [("poisson", (1, 0), 1., .68),
                ("cos", (0, 1), specification["weak_amplitude"], None)]
    raise ValueError(family)


def profile_and_spatial_phase(specification, sensor_positions, group, rng):
    x = sensor_positions[:, 0]
    y = sensor_positions[:, 1] if sensor_positions.shape[1] > 1 else np.zeros_like(x)
    z = sensor_positions[:, 2] if sensor_positions.shape[1] > 2 else np.zeros_like(x)
    profile = (.82+.18*np.cos(np.pi*(x+.19*group))
               +.12*np.sin(np.pi*(y+.29*group)))
    if sensor_positions.shape[1] > 2:
        profile += .08*np.cos(np.pi*(z+.13*group))
    if specification["family"] == "weak" and group == 1:
        profile = .18+.82*np.sin(np.pi*x)**2
    wavevector = rng.uniform(.6, 1.6, sensor_positions.shape[1])
    phase = (2*np.pi*np.einsum("id,d->i", sensor_positions, wavevector, optimize=False)
             +.12*np.sin(2*np.pi*x)*np.cos(np.pi*y)+rng.uniform(0, 2*np.pi))
    return profile, phase


def generate(root):
    """Generate all6 before inference; return input-only fitting metadata."""
    root = Path(root)
    for name in ("inputs", "evaluation"):
        (root/name).mkdir(parents=True, exist_ok=False)
    times = np.arange(TRAIN+VALIDATION+FUTURE)
    manifest = []
    for specification in CASES:
        seed_index = specification["seed_index"]
        sensor_positions = positions(specification)
        alpha, rational, primes, denominators = rotations(specification)
        origin = np.random.default_rng(614100+seed_index).uniform(0, 2*np.pi, specification["rank"])
        theta = origin[:, None]+alpha[:, None]*times[None, :]
        clean = np.zeros((len(sensor_positions), len(times)))
        profile_rng = np.random.default_rng(914100+seed_index)
        catalogue, coefficient_catalogue, catalogue_group, group_modes = [], [], [], []
        for group, (family, mode, amplitude, rho) in enumerate(terms(specification)):
            mode = np.asarray(mode, dtype=int)
            group_modes.append(mode.tolist())
            profile, spatial_phase = profile_and_spatial_phase(specification, sensor_positions, group, profile_rng)
            angle = np.einsum("i,it->t", mode, theta, optimize=False)[None, :]-spatial_phase[:, None]
            if family == "poisson":
                clean += amplitude*profile[:, None]*(((1-rho*rho)/(1-2*rho*np.cos(angle)+rho*rho)-1)
                                                     /np.sqrt(2*rho*rho/(1-rho*rho)))
                for harmonic in range(1, 65):
                    catalogue.append((harmonic*mode).tolist())
                    coefficient_catalogue.append(amplitude*profile*np.sqrt((1-rho*rho)/2)
                        *rho**(harmonic-1)*np.exp(-1j*harmonic*spatial_phase))
                    catalogue_group.append(group)
            else:
                clean += amplitude*profile[:, None]*np.cos(angle)
                catalogue.append(mode.tolist())
                coefficient_catalogue.append(.5*amplitude*profile*np.exp(-1j*spatial_phase))
                catalogue_group.append(group)
        sigma = NOISE_RATIO*float(np.sqrt(np.mean(clean[:, :TRAIN]**2)))
        standard_noise = np.random.default_rng(514100+seed_index).standard_normal(clean.shape)
        run = specification["name"]
        input_path, evaluation_path = root/"inputs"/(run+".npz"), root/"evaluation"/(run+".npz")
        np.savez_compressed(input_path, Y=clean+sigma*standard_noise, s=sensor_positions, t=times)
        np.savez_compressed(evaluation_path, clean=clean, theta=theta, alpha=alpha, origin=origin,
            sigma=np.asarray(sigma), rank=np.asarray(specification["rank"]),
            group_modes=np.asarray(group_modes), catalog_modes=np.asarray(catalogue),
            catalog_coefficients=np.asarray(coefficient_catalogue), catalog_groups=np.asarray(catalogue_group),
            rational_frequency_multipliers=rational, frequency_squarefree_primes=primes,
            frequency_denominators=denominators, standard_noise_sha256=np.asarray(hashlib.sha256(standard_noise.tobytes()).hexdigest()))
        manifest.append(dict(run=run, case=run, kind="fresh", input=str(input_path),
            input_sha256=digest(input_path), evaluation=str(evaluation_path),
            evaluation_sha256=digest(evaluation_path), n_train=TRAIN, n_validation=VALIDATION,
            n_future=FUTURE))
    return manifest


PROTOCOL = dict(sample_counts=dict(train=TRAIN, validation=VALIDATION, future=FUTURE),
    noise_ratio_to_training_clean_sensor_rms=NOISE_RATIO, cases=CASES,
    seed_rules=dict(frequency="714100+seed_index", origin="614100+seed_index",
        positions="814100+seed_index", profile="914100+seed_index", noise="514100+seed_index"),
    frequency_rule="Uniform(.9,1.1) rational coefficients times sqrt(distinct squarefree primes)/(9,7,6); 1D wave sqrt(2,7)/(9,12)",
    paired_weak_cases_share_network_rotation_origin_profiles_and_standard_noise=True,
    random_extra_mixed_mode_amplitude=".13*Uniform(.7,1.3)/(1+integer_l1_order)",
    intended_exact_sampled_rotation_nonresonance="algebraic rational combinations of distinct squarefree roots cannot equal nonzero2pi winding; distinct roots rationally independent",
    floating_point_values_are_representation_not_observed_proof=True,
    fitting_inputs="Y,s,t only; no generator catalog, phase rank, origin, frequencies or law",
    source_equation_or_pde_given_to_inference=False)
