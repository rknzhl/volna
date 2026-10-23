"""Independent analytic and frozen-query checks of the Koopman implementation."""

import argparse
import hashlib
from itertools import product
import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits

from giannakis_koopman import (
    extend_diffusion_basis,
    fit_diffusion_basis,
    fit_generator,
    fit_observable,
    fit_phase_observable,
    forecast_observable,
    reconstruct_observable,
    reconstruct_phase_observable,
    select_independent_phases,
    transform_generator,
)
from project_paths import RESULTS_DIR


HERE = Path(__file__).resolve().parent
OUTPUT = RESULTS_DIR / "giannakis_koopman_checks_2026_10_23.json"


def circular_error(phase, target):
    return np.abs(np.angle(np.exp(1j * (phase - target))))


def check_diffusion_invariants(basis):
    x = basis["train_x"]
    kernel = np.exp(-cdist(x, x, "sqeuclidean") / (
        basis["epsilon"] * basis["rho"][:, None] * basis["rho"][None, :]))
    transition = kernel / basis["q"][None, :] / basis["d"][:, None]
    weights = basis["weights"]
    phi = basis["train_basis"]
    gram = np.einsum("ni,n,nj->ij", phi, weights, phi)
    row_error = float(np.max(abs(transition.sum(axis=1) - 1.)))
    stationary_error = float(np.max(abs(np.einsum("i,ij->j", weights, transition) - weights)))
    orthogonality_error = float(np.max(abs(gram - np.eye(phi.shape[1]))))
    eigenproblem_error = float(np.max(abs(np.einsum("ij,jk->ik", transition, phi) - phi * basis["kappa"])))
    assert row_error < 1e-12
    assert stationary_error < 1e-12
    assert orthogonality_error < 1e-10
    assert eigenproblem_error < 1e-10
    return dict(row_stochastic_error=row_error, stationary_error=stationary_error,
        weighted_orthogonality_error=orthogonality_error,
        markov_eigenproblem_error=eigenproblem_error)


def check_circle():
    samples, dt = 512, .1
    step = 2. * np.pi * 17 / samples
    theta = step * np.arange(samples)
    observed = np.column_stack((np.cos(theta), np.sin(theta)))
    basis = fit_diffusion_basis(observed, n_basis=13)
    invariants = check_diffusion_invariants(basis)
    model = fit_generator(basis, dt=dt, regularization=.003)
    frequencies = model["eigenvalues"].imag
    constant = int(np.argmin(abs(model["eigenvalues"])))
    assert abs(model["eigenvalues"][constant]) < 1e-12
    assert model["dirichlet_energy"][constant] < 1e-12
    assert np.max(model["eigenvalues"].real) < 1e-12
    dissipative_error = float(np.max(abs(model["eigenvalues"].real
        + model["regularization"] * model["dirichlet_energy"])))
    assert dissipative_error < 1e-10
    selected = select_independent_phases(model)
    assert selected["phase_dimension"] == 1
    selected_index = selected["indices"][0]
    found = float(frequencies[selected_index])
    expected = float(np.sin(step) / dt)
    relative_frequency_error = abs(found - expected) / expected
    assert relative_frequency_error < .02
    train_phase = np.angle(model["train_eigenfunctions"][:, selected_index])
    gauge = float(np.angle(np.mean(np.exp(1j * (train_phase - theta)))))
    direction_coherence = float(abs(np.mean(np.exp(1j * (train_phase - theta)))))
    assert direction_coherence > .99
    replay = extend_diffusion_basis(basis, observed)
    replay_error = float(np.max(abs(replay - basis["train_basis"])))
    assert replay_error < 2e-7
    query_theta = np.linspace(-2.4, 8.9, 137)
    query = np.column_stack((np.cos(query_theta), np.sin(query_theta)))
    transformed = transform_generator(model, query, batch_size=137)
    chunked = transform_generator(model, query, batch_size=7)
    batched_error = float(np.max(abs(transformed["eigenfunctions"] - chunked["eigenfunctions"])))
    assert batched_error < 1e-10
    query_phase_error = float(np.mean(circular_error(
        transformed["phases"][:, selected_index], query_theta + gauge)))
    assert query_phase_error < .05
    signal = np.column_stack((1. + .4 * np.cos(2. * theta) + .3 * np.sin(theta),
                             .6 * np.cos(theta) - .2 * np.sin(3. * theta)))
    query_signal = np.column_stack((1. + .4 * np.cos(2. * query_theta) + .3 * np.sin(query_theta),
                                   .6 * np.cos(query_theta) - .2 * np.sin(3. * query_theta)))
    readout = fit_observable(model, signal)
    reconstructed = reconstruct_observable(readout, transformed)
    reconstruction_rmse = float(np.sqrt(np.mean((reconstructed - query_signal) ** 2)))
    assert reconstruction_rmse < .02
    phase_readout = fit_phase_observable(model, selected, observed, spectral_order=1)
    phase_result = reconstruct_phase_observable(phase_readout, transformed, return_fields=True)
    expected_field = step / dt * np.column_stack((-np.sin(query_theta), np.cos(query_theta)))
    vector_field_rmse = float(np.sqrt(np.mean((phase_result["vector_fields"][:, 0] - expected_field) ** 2)))
    assert vector_field_rmse < .05
    lead_times = np.linspace(0., .6, 25)
    forecast = forecast_observable(model, readout, model["train_eigenfunctions"][0], lead_times)
    forecast_theta = step / dt * lead_times
    expected_forecast = np.column_stack((1. + .4 * np.cos(2. * forecast_theta) + .3 * np.sin(forecast_theta),
        .6 * np.cos(forecast_theta) - .2 * np.sin(3. * forecast_theta)))
    forecast_rmse = float(np.sqrt(np.mean((forecast - expected_forecast) ** 2)))
    assert forecast_rmse < .05
    return dict(control="rotational_circle", sample_count=samples, **invariants,
        constant_eigenvalue_absolute=float(abs(model["eigenvalues"][constant])),
        dissipative_energy_identity_error=dissipative_error,
        known_angular_frequency=step / dt, expected_central_difference_frequency=expected,
        computed_frequency=found, central_frequency_relative_error=relative_frequency_error,
        selected_increment_frequency=float(selected["frequencies"][0]),
        positive_frequency_direction_coherence=direction_coherence,
        nystrom_train_replay_max_error=replay_error, query_batching_max_error=batched_error,
        query_phase_MAE_radians=query_phase_error,
        query_observable_RMSE=reconstruction_rmse,
        query_commuting_field_RMSE=vector_field_rmse,
        short_forecast_RMSE=forecast_rmse,
        phase_dimension=selected["phase_dimension"])


def check_product_torus():
    rng = np.random.default_rng(481)
    samples, dt = 768, .2
    frequency = np.array([.7, np.sqrt(2.) * .7])
    theta = dt * np.arange(samples)[:, None] * frequency[None, :]

    def embedding(phase):
        return np.column_stack((np.cos(phase[:, 0]), np.sin(phase[:, 0]),
            1.2 * np.cos(phase[:, 1]), 1.2 * np.sin(phase[:, 1])))

    observed = embedding(theta)
    basis = fit_diffusion_basis(observed, n_basis=49, bandwidth_scale=4.)
    invariants = check_diffusion_invariants(basis)
    model = fit_generator(basis, dt=dt, regularization=.001)
    selection = select_independent_phases(model)
    assert selection["phase_dimension"] == 2
    labels = np.asarray([label for label in product(range(-2, 3), repeat=2) if label != (0, 0)])
    target_characters = np.exp(1j * np.einsum("ij,kj->ik", theta, labels))
    query_theta = rng.uniform(-np.pi, np.pi, (173, 2))
    query = embedding(query_theta)
    transformed = transform_generator(model, query, batch_size=29)
    separate = np.vstack([transform_generator(model, row[None, :])["eigenfunctions"][0] for row in query])
    batch_error = float(np.max(abs(transformed["eigenfunctions"] - separate)))
    assert batch_error < 1e-10
    matched_labels, errors, coherence = [], [], []
    for index in selection["indices"]:
        train_character = np.exp(1j * np.angle(model["train_eigenfunctions"][:, index]))
        correlation = np.mean(train_character[:, None] * target_characters.conj(), axis=0)
        best = int(np.argmax(abs(correlation)))
        label, gauge = labels[best], float(np.angle(correlation[best]))
        matched_labels.append(label.tolist())
        coherence.append(float(abs(correlation[best])))
        errors.append(float(np.mean(circular_error(transformed["phases"][:, index],
            np.einsum("ij,j->i", query_theta, label) + gauge))))
    winding = np.asarray(matched_labels)
    assert abs(round(np.linalg.det(winding))) == 1
    assert min(coherence) > .95
    assert max(errors) < .25
    readout = fit_observable(model, observed)
    reconstruction = reconstruct_observable(readout, transformed)
    rmse = float(np.sqrt(np.mean((reconstruction - query) ** 2)))
    assert rmse < .1
    return dict(control="product_torus", sample_count=samples, **invariants,
        estimated_manifold_dimension=basis["dimension_estimate"],
        selected_phase_dimension=selection["phase_dimension"],
        selected_frequencies=selection["frequencies"],
        evaluated_integer_characters=matched_labels,
        training_character_coherence=coherence,
        query_phase_MAE_radians=errors,
        query_batching_max_error=batch_error, query_observable_RMSE=rmse,
        evaluation_truth_used_only_after_fit=True)


def check_harmonic_selection():
    samples, dt = 384, .1
    step = 2. * np.pi * 47 / samples
    theta = step * np.arange(samples)
    observed = np.column_stack((np.cos(theta), np.sin(theta)))
    basis = fit_diffusion_basis(observed, n_basis=9)
    model = fit_generator(basis, dt=dt, regularization=.003)
    selection = select_independent_phases(model, max_phases=3)
    assert selection["phase_dimension"] == 1
    assert abs(selection["frequencies"][0] - step / dt) / (step / dt) < .01
    decisions = selection["metadata"]["decisions"]
    relations = [item for item in decisions if item["decision"] == "rejected_bounded_integer_relation"]
    assert len(relations) >= 2
    return dict(control="coarse_time_step_harmonic_rejection",
        sample_count=samples, angular_step_radians=step,
        supplied_phase_cap=3, selected_phase_dimension=selection["phase_dimension"],
        selected_frequencies=selection["frequencies"],
        rejected_harmonic_count=len(relations),
        truth_used_for_selection=False)


def check_group_relations():
    rng = np.random.default_rng(174)
    count, dt = 1280, .2
    phase = rng.uniform(-np.pi, np.pi, (count, 3))
    characters = np.column_stack((np.ones(count), np.exp(1j * phase[:, 0]),
        np.exp(1j * phase[:, 1]), np.exp(1j * (phase[:, 0] + phase[:, 1])),
        np.exp(1j * phase[:, 2])))
    frequency = np.array([0., .53, .83, .53 + .83, .53 + 1e-5])
    model = dict(basis=dict(train_x=phase, weights=np.full(count, 1. / count),
        dimension_estimate=3.), dt=dt,
        train_eigenfunctions=characters,
        eigenvalues=-.001 * np.arange(5) + 1j * np.sin(dt * frequency) / dt,
        train_phase_frequency=frequency,
        phase_step_coherence=np.ones(5), radius_cv=np.zeros(5),
        temporal_residual=np.zeros(5), dirichlet_energy=np.arange(5))
    selection = select_independent_phases(model)
    assert selection["indices"] == [1, 2, 4]
    rejected = [item for item in selection["metadata"]["decisions"]
        if item["decision"] == "rejected_bounded_integer_relation"]
    assert [item["index"] for item in rejected] == [3]
    assert rejected[0]["relation"]["training_phase_product_coherence"] > 1. - 1e-12
    return dict(control="exact_character_group_selection_fixture",
        third_candidate_is_product_of_first_two=True,
        independent_character_frequency_gap=1e-5,
        numerical_frequency_resolution=selection["metadata"]["frequency_tolerance"],
        selected_candidate_indices=selection["indices"],
        selected_phase_dimension=selection["phase_dimension"],
        combination_rejected=True, close_independent_character_retained=True)


def main(output=OUTPUT):
    output = Path(output)
    if output.exists():
        raise FileExistsError("preserve completed numerical checks")
    with threadpool_limits(limits=1):
        controls = [check_circle(), check_product_torus(), check_harmonic_selection(),
                    check_group_relations()]
    result = dict(status="passed", analytic_controls=controls,
        reference="https://arxiv.org/html/1507.02338v4",
        training_and_query_maps_fit_without_true_phases_or_frequencies=True,
        source_hashes={name:hashlib.sha256((HERE / name).read_bytes()).hexdigest()
            for name in ("giannakis_koopman.py", "check_giannakis_koopman.py")})
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    main(parser.parse_args().output)
