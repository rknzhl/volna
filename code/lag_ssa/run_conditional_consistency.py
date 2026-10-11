"""Isolated oracle-phase conditional-mean learning assay, not inference.

The known infinite mixed field is sampled along its SAME fixed quasiperiodic
trajectory. Exact theta_1 is supplied to an unweighted kNN factor estimator.
Its target is Q_{Z(1,0)}F, scored at independent Haar points on the full T^2.
Noisy and noiseless controls separate smoothing, finite-fiber leakage and
decoded sensor noise. Finite curves do not prove consistency or completeness.
"""

from __future__ import annotations

from project_paths import RESULTS_DIR

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

from iterative_fields import PROFILE_COEFFICIENTS, generate_field
from spatial_field import SPATIAL_MODEL


OUT = RESULTS_DIR
DESTINATION = OUT / "conditional_consistency.json"
PLOT = OUT / "conditional_consistency.png"
CASE = "r2_infinite_mixed_products"
COUNTS = (800, 1600, 3200, 6400, 12800)
SEEDS = (54000, 54001, 54002)
HAAR_SEED = 54301
SIGMA = .08
RULES = ("fixed_32", "fixed_64", "floor_0.5_sqrt_N", "floor_1.0_sqrt_N")


def _serializable(value):
    if isinstance(value, np.ndarray):
        return _serializable(value.tolist())
    if isinstance(value, np.generic):
        return _serializable(value.item())
    if isinstance(value, dict):
        return {str(key): _serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serializable(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _poisson(angle: np.ndarray, rho: float) -> np.ndarray:
    return (((1-rho*rho)/(1-2*rho*np.cos(angle)+rho*rho)-1)
            / np.sqrt(2*rho*rho/(1-rho*rho)))


def _embedding(theta: np.ndarray) -> np.ndarray:
    return np.column_stack((np.cos(theta), np.sin(theta)))


def _k(rule: str, count: int) -> int:
    if rule == "fixed_32":
        return 32
    if rule == "fixed_64":
        return 64
    return max(1, int((.5 if rule == "floor_0.5_sqrt_N" else 1.)*np.sqrt(count)))


def _mse(values: np.ndarray) -> float:
    """Euclidean mass-coordinate squared norm equals physical field energy."""
    return float(np.mean(np.sum(values*values, axis=0)))


def _save(payload: dict, path: Path) -> None:
    path.parent.mkdir(exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(_serializable(payload), indent=2,
                                    allow_nan=False)+"\n")
    temporary.replace(path)


def _plot(rows: list[dict], destination: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    colors = ("#305e9c", "#8e57a6", "#c87920", "#26816a")
    for rule, color in zip(RULES, colors):
        subset = [row for row in rows if row["rule"] == rule]
        x = [row["n_train"] for row in subset]
        axes[0].loglog(x, [row["noiseless_Q_rmse_physical"] for row in subset],
                      "o-", color=color, label=rule)
        means = [row["noisy_Q_rmse_seed_mean_physical"] for row in subset]
        deviations = [row["noisy_Q_rmse_seed_sd_physical"] for row in subset]
        axes[1].errorbar(x, means, yerr=deviations, fmt="o-", color=color)
        axes[2].loglog(x, [np.sqrt(row["expected_noise_prediction_energy"])
                          for row in subset], "o-", color=color)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    for axis, title in zip(axes, ("Noiseless conditional-mean error",
                                 "Noisy error: mean ± seed SD",
                                 "Exact noise RMSE: sqrt(tr Sigma / k)")):
        axis.set_title(title, fontsize=10)
        axis.set_xlabel("Training samples N")
        axis.set_ylabel("Physical field L2 RMSE")
        axis.grid(True, which="both", alpha=.2)
    axes[0].legend(fontsize=7)
    fig.suptitle("Oracle theta1 factor; one fixed quasiperiodic trajectory; Haar queries", fontsize=11)
    fig.savefig(destination, dpi=170)
    plt.close(fig)


def evaluate(haar_queries: int = 4096) -> dict:
    maximum = max(COUNTS)
    generated = generate_field(CASE, np.arange(maximum), n_train=maximum)
    theta = generated["theta"]
    groups = {group["name"]: group for group in generated["expected_observable_mode_groups"]}
    first, second, mixed = (groups[name] for name in
                            ("poisson_theta1", "poisson_theta2", "poisson_product"))
    amplitudes = np.array([group["amplitude_after_strong_normalization"]
                           for group in (first, second, mixed)])
    profiles = np.column_stack([PROFILE_COEFFICIENTS[group["profile_index_in_known_basis"]]
                                for group in (first, second, mixed)])
    mass_profiles = np.einsum("ij,jk->ik", SPATIAL_MODEL.mass_sqrt, profiles,
                              optimize=False)
    g1 = _poisson(theta[0], first["rho"])
    g2 = _poisson(theta[1], second["rho"])
    temporal_terms = np.stack((g1, g2, g1*g2))
    clean_mass = np.einsum("ij,jt->it", mass_profiles*amplitudes[None, :],
                           temporal_terms, optimize=False)
    decoded_clean_mass = SPATIAL_MODEL.to_mass_coordinates(
        SPATIAL_MODEL.recover(generated["clean"]))
    decomposition_error = float(np.max(abs(clean_mass-decoded_clean_mass)))
    if decomposition_error > 1e-10:
        raise AssertionError("oracle component decomposition disagrees with field generator")
    haar = np.random.default_rng(HAAR_SEED).uniform(0, 2*np.pi, (2, haar_queries))
    query_embedding = _embedding(haar[0])
    test_g1, test_g2 = _poisson(haar[0], first["rho"]), _poisson(haar[1], second["rho"])
    ideal_Q = amplitudes[0]*mass_profiles[:, 0, None]*test_g1[None, :]
    irreducible = (amplitudes[1]*mass_profiles[:, 1, None]*test_g2[None, :]
                   + amplitudes[2]*mass_profiles[:, 2, None]*(test_g1*test_g2)[None, :])
    ideal_irreducible_energy = float(
        amplitudes[1]**2*np.sum(mass_profiles[:, 1]**2)
        + amplitudes[2]**2*np.sum(mass_profiles[:, 2]**2))
    ideal_Q_energy = float(amplitudes[0]**2*np.sum(mass_profiles[:, 0]**2))
    noise_covariance = SPATIAL_MODEL.noise_covmass(SIGMA)
    noise_trace = float(np.trace(noise_covariance))
    noise_mass = {}
    for seed in SEEDS:
        raw_noise = SIGMA*np.random.default_rng(seed).standard_normal(generated["clean"].shape)
        noise_mass[seed] = SPATIAL_MODEL.to_mass_coordinates(SPATIAL_MODEL.recover(raw_noise))
    rows = []
    for count in COUNTS:
        tree = cKDTree(_embedding(theta[0, :count]))
        for rule in RULES:
            k = _k(rule, count)
            distances, neighbours = tree.query(query_embedding, k=k)
            means = temporal_terms[:, neighbours].mean(axis=2)
            components = [
                amplitudes[0]*mass_profiles[:, 0, None]*(means[0]-test_g1)[None, :],
                amplitudes[1]*mass_profiles[:, 1, None]*means[1][None, :],
                amplitudes[2]*mass_profiles[:, 2, None]*means[2][None, :],
            ]
            bias = np.sum(components, axis=0)
            noiseless_prediction = ideal_Q+bias
            # This check verifies that the estimator uses full field observations,
            # not the ideal Q or its separated truth components.
            direct_prediction = decoded_clean_mass[:, neighbours].mean(axis=2)
            identity_error = float(np.max(abs(direct_prediction-noiseless_prediction)))
            if identity_error > 1e-10:
                raise AssertionError("kNN decomposition disagrees with full-field averaging")
            names = ("retained_theta1_smoothing", "off_factor_theta2", "off_factor_product")
            component_energies = {name: _mse(component) for name, component in zip(names, components)}
            cross_terms = {f"{names[i]}__{names[j]}": float(
                2*np.mean(np.sum(components[i]*components[j], axis=0)))
                for i in range(3) for j in range(i+1, 3)}
            bias_energy = _mse(bias)
            decomposition_energy = sum(component_energies.values())+sum(cross_terms.values())
            expected_noise_energy = noise_trace/k
            seeds = []
            for seed in SEEDS:
                noise_prediction = noise_mass[seed][:, neighbours].mean(axis=2)
                noise_energy = _mse(noise_prediction)
                total_energy = _mse(bias+noise_prediction)
                cross = float(2*np.mean(np.sum(bias*noise_prediction, axis=0)))
                seeds.append({"seed": seed, "noisy_Q_mse_physical": total_energy,
                    "noisy_Q_rmse_physical": float(np.sqrt(total_energy)),
                    "noise_prediction_energy": noise_energy,
                    "noise_energy_over_exact_expectation": noise_energy/expected_noise_energy,
                    "bias_noise_cross_energy": cross,
                    "total_energy_decomposition_error": abs(total_energy-bias_energy-noise_energy-cross)})
            rmses = np.array([seed["noisy_Q_rmse_physical"] for seed in seeds])
            angular_radius = 2*np.arcsin(np.clip(distances[:, -1]/2, 0, 1))
            rows.append({"n_train": count, "rule": rule, "k": k, "k_over_N": k/count,
                "neighbour_angular_radius_mean_rad": float(angular_radius.mean()),
                "neighbour_angular_radius_max_rad": float(angular_radius.max()),
                "component_bias_energies_physical": component_energies,
                "component_bias_cross_energies_physical": cross_terms,
                "off_factor_combined_bias_energy_physical": _mse(components[1]+components[2]),
                "noiseless_Q_mse_physical": bias_energy,
                "noiseless_Q_rmse_physical": float(np.sqrt(bias_energy)),
                "noiseless_Q_relative_rmse": float(np.sqrt(bias_energy/ideal_Q_energy)),
                "expected_noise_prediction_energy": expected_noise_energy,
                "expected_noisy_Q_energy": bias_energy+expected_noise_energy,
                "noisy_Q_mse_seed_mean_physical": float(np.mean([seed["noisy_Q_mse_physical"] for seed in seeds])),
                "noisy_Q_rmse_seed_mean_physical": float(rmses.mean()),
                "noisy_Q_rmse_seed_sd_physical": float(rmses.std(ddof=1)),
                "seeds": seeds,
                "scalar_fiber_diagnostics": {
                    "mean_squared_neighbour_average_g_theta2": float(np.mean(means[1]**2)),
                    "mean_squared_neighbour_average_g_product": float(np.mean(means[2]**2))},
                "component_prediction_identity_max_error": identity_error,
                "component_energy_identity_error": abs(bias_energy-decomposition_energy)})
            print(json.dumps({"N": count, "rule": rule, "k": k,
                "noiseless_rmse": float(np.sqrt(bias_energy)),
                "noisy_rmse_mean": float(rmses.mean()),
                "smoothing_energy": component_energies[names[0]],
                "off_factor_energy": _mse(components[1]+components[2]),
                "noise_energy_expected": expected_noise_energy}), flush=True)
    return {"description": "isolated oracle-phase Q_Gamma learning on infinite mixed field",
        "case": CASE, "oracle_control": True,
        "true_phase_supplied_to_neighbour_search": "theta1 only, Gamma=Z(1,0)",
        "phase_or_rank_inference_claim": False, "completeness_claim": False,
        "finite_curves_prove_asymptotic_consistency": False,
        "training_design": "nested prefixes of the SAME fixed alpha, theta0 quasiperiodic trajectory",
        "alpha_truth_only_oracle_control": generated["metadata"]["alpha"],
        "training_counts": COUNTS, "noise_seeds": SEEDS, "raw_noise_sigma": SIGMA,
        "noise_scale": "fixed across N; .08 of the fixture's unit-Haar sensor RMS",
        "noise_prefixes_nested_per_seed": True,
        "query_design": "independent uniform Haar samples on the full T2; no times or trajectory phases reused",
        "haar_query_count": haar_queries, "haar_query_seed": HAAR_SEED,
        "neighbour_weights": "uniform over k distinct neighbours, same across field coordinates",
        "k_rules_predeclared_and_not_selected_from_queries": RULES,
        "ideal_Q": "a1 * profile1 * unit_Poisson_0.70(theta1); theta2 and product Haar means vanish",
        "term_amplitudes_truth_only": amplitudes,
        "ideal_Q_energy_exact_physical": ideal_Q_energy,
        "ideal_Q_energy_query_Monte_Carlo_physical": _mse(ideal_Q),
        "irreducible_conditional_residual_energy_exact_physical": ideal_irreducible_energy,
        "irreducible_conditional_residual_energy_query_Monte_Carlo_physical": _mse(irreducible),
        "mass_coordinate_noise_covariance": noise_covariance,
        "mass_coordinate_noise_trace": noise_trace,
        "exact_noise_prediction_energy_formula": "tr(Sigma_mass)/k; holds for oracle neighbours independent of sensor noise",
        "component_decomposition_max_error": decomposition_error,
        "limits": ["one fixed quasiperiodic increment and origin",
                   "exact oracle phases exclude frequency, character and lattice errors",
                   "three noise seeds are diagnostics, not calibrated coverage",
                   "finite-N off-factor averaging depends on arithmetic of the training trajectory",
                   "noisy response variance is nonvanishing for fixed k"],
        "run_complete": True, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--haar-queries", type=int, default=4096)
    parser.add_argument("--output", type=Path, default=DESTINATION)
    parser.add_argument("--plot", type=Path, default=PLOT)
    args = parser.parse_args()
    if args.haar_queries < 100:
        parser.error("at least 100 independent Haar queries required")
    payload = evaluate(args.haar_queries)
    _plot(payload["rows"], args.plot)
    payload["plot"] = str(args.plot)
    payload["created_utc"] = datetime.now(timezone.utc).isoformat()
    _save(payload, args.output)
    print(str(args.output), flush=True)


if __name__ == "__main__":
    main()
