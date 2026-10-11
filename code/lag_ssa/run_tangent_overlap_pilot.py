"""Small frozen protocol for the Hankel-overlap tangent-frame candidate.

Generation, inference, and evaluation are separate functions. A fit consumes
only Y; evaluation reads the hidden phases/clean readings after fitting.
The first phase is a feasibility test, not a claim of a new general SSA.
"""

from __future__ import annotations

from project_paths import RESULTS_DIR

import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy import linalg

from tangent_overlap import fit, windows, fold


HERE = Path(__file__).resolve().parent
OUT = RESULTS_DIR / 'tangent_overlap_pilot'


def poisson(angle, rho):
    den = 1 - 2 * rho * np.cos(angle) + rho * rho
    value = (1 - rho * rho) / den - 1
    deriv = -(1 - rho * rho) * 2 * rho * np.sin(angle) / den ** 2
    return value, deriv


def generate(count=4096, channels=12):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "inputs").mkdir(exist_ok=True)
    (OUT / "evaluation").mkdir(exist_ok=True)
    rng = np.random.default_rng(63021)
    s = rng.uniform(0, 1, (channels, 2))
    a = 0.8 + 0.3 * s[:, 0]
    b = 0.7 + 0.3 * s[:, 1]
    c = 0.7 + 0.2 * np.sin(s[:, 0] + s[:, 1])
    beta = 2 * np.pi * (1.1 * s[:, 0] - 0.4 * s[:, 1])
    gamma = 2 * np.pi * (0.3 * s[:, 0] + 0.9 * s[:, 1])
    mixed = 2 * np.pi * (0.7 * s[:, 0] + 0.4 * s[:, 1])
    alpha = np.array([0.115 * np.sqrt(2), 0.107 * np.sqrt(3)])
    theta2 = np.arange(count)[:, None] * alpha + [0.3, 1.1]
    cases = [("one_phase", 1, .02), ("two_phase_clean", 2, .0),
             ("two_phase", 2, .02), ("two_phase_cover_witness", 2, .02)]
    for name, r, noise in cases:
        theta = theta2[:, :r]
        if r == 1:
            p, dp = poisson(theta[:, 0, None] + beta, .5)
            q, dq = poisson(2 * theta[:, 0, None] + gamma, .3)
            clean = a * p + .3 * b * q
            derivative = (a * dp + .6 * b * dq)[:, :, None]
        else:
            k1, k2 = (3, 2) if "cover" in name else (1, 1)
            p, dp = poisson(k1 * theta[:, 0, None] + beta, .5)
            q, dq = poisson(k2 * theta[:, 1, None] + gamma, .5)
            if "cover" in name:
                angle = theta[:, 0, None] + theta[:, 1, None] + mixed
                clean = a * p + .7 * b * q + .15 * c * np.cos(angle)
                derivative = np.stack((k1 * a * dp - .15 * c * np.sin(angle),
                                       k2 * .7 * b * dq - .15 * c * np.sin(angle)), axis=-1)
            else:
                mixed_angle = theta[:, 0, None] - theta[:, 1, None] + mixed
                v, dv = poisson(mixed_angle, .3)
                clean = a * p + .7 * b * q + .2 * c * v
                derivative = np.stack((a * dp + .2 * c * dv,
                                       .7 * b * dq - .2 * c * dv), axis=-1)
        sigma = noise * np.sqrt(np.mean(clean ** 2))
        y = clean + rng.normal(0, sigma, clean.shape)
        np.savez_compressed(OUT / "inputs" / f"{name}.npz", Y=y)
        np.savez_compressed(OUT / "evaluation" / f"{name}.npz", clean=clean,
                            theta=theta, derivative=derivative, sensors=s,
                            alpha=alpha[:r], sigma=sigma)
    protocol = {"count": count, "channels": channels, "train_count": count // 2,
                "lag": 24, "neighbors": 32, "max_local_dimension": 5,
                "lambdas": [.03, .003, .0003], "mssa_ranks": [2, 4, 8, 16, 32],
                "fit_input_keys": ["Y"], "no_true_phases_or_dimension_in_fit": True,
                "future_metric": "encoding of observed future windows, not forecasting",
                "dimension_estimator": "largest median consecutive local singular-value ratio",
                "cases": [a[0] for a in cases]}
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2))


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value))


def relative_error(pred, truth):
    return float(np.linalg.norm(pred - truth) / np.linalg.norm(truth))


def mssa_baselines(train_y, future_y, clean_future, lag, ranks):
    train = windows(train_y, lag)
    center = train.mean(axis=0)
    _, _, vt = linalg.svd(train - center, full_matrices=False)
    h_future = windows(future_y, lag)
    result = []
    for rank in ranks:
        u = vt[:rank].T
        reconstruction = (h_future - center) @ u @ u.T + center
        yhat = fold(reconstruction, train_y.shape[1], lag)
        result.append({"rank": rank, "future_clean_nrmse": relative_error(yhat, clean_future)})
    return result


def evaluate_frame(model, evaluation, train_count):
    rtrue = evaluation["theta"].shape[1]
    derivative = evaluation["derivative"][:train_count]
    # True derivative frame is used only to score, never to construct Q or A.
    d = np.stack([windows(derivative[:, :, j] / model["scale"], model["lag"])
                  for j in range(rtrue)], axis=-1)
    d /= np.sqrt(model["lag"] * model["channels"])
    q = model["geometry"]["Q"]
    coefficients = np.einsum("npa,npb->nab", q, d)
    projected = np.einsum("npa,nab->npb", q, coefficients)
    tangent_error = relative_error(projected, d)
    transported = np.einsum("nab,nbc->nac", model["maps"], coefficients[:-1])
    overlap_error = relative_error(transported, coefficients[1:])
    rows = []
    for fitted in model["fits"]:
        v = fitted["field"]
        # One globally fixed real change of basis; no pointwise truth correction.
        transform = linalg.lstsq(v.reshape(-1, v.shape[-1]), d.reshape(-1, rtrue))[0]
        estimate = np.einsum("npa,ab->npb", v, transform)
        residual = relative_error(estimate, d)
        rows.append({"lambda": fitted["lambda"], "eigenvalues": fitted["eigenvalues"],
                     "transport_energy": fitted["transport_energy"],
                     "roughness": fitted["roughness"],
                     "frame_min_singular_quantiles": fitted["frame_min_singular_quantiles"],
                     "frame_relative_error_after_global_basis": residual})
    return {"true_dimension_evaluation_only": rtrue,
            "local_tangent_relative_error": tangent_error,
            "overlap_transfer_relative_error": overlap_error, "frames": rows}


def run_case(name):
    protocol = json.loads((OUT / "protocol.json").read_text())
    source = np.load(OUT / "inputs" / f"{name}.npz")
    y = source["Y"]
    ntrain = protocol["train_count"]
    start = time.perf_counter()
    model = fit(y[:ntrain], lag=protocol["lag"], neighbors=protocol["neighbors"],
                lambdas=protocol["lambdas"])
    inference_seconds = time.perf_counter() - start
    # Freeze the inferred geometry before opening evaluator-only material.
    np.savez_compressed(OUT / f"{name}_inferred.npz", X=model["X"], Q=model["geometry"]["Q"],
                        neighbors=model["geometry"]["neighbors"], maps=model["maps"],
                        fields=np.stack([f["field"] for f in model["fits"]]),
                        mean=model["mean"], scale=model["scale"])
    hidden = np.load(OUT / "evaluation" / f"{name}.npz")
    result = {"case": name, "estimated_dimension": model["geometry"]["dimension"],
              "median_singular_gaps": model["geometry"]["median_singular_gaps"],
              "median_singular_values": model["geometry"]["median_singular_values"],
              "overlap_min_singular_quantiles": model["overlap_min_singular_quantiles"],
              "inference_seconds": inference_seconds,
              "frame_evaluation": evaluate_frame(model, hidden, ntrain),
              "mssa": mssa_baselines(y[:ntrain], y[ntrain:], hidden["clean"][ntrain:],
                                      protocol["lag"], protocol["mssa_ranks"])}
    (OUT / f"{name}.json").write_text(json.dumps(result, indent=2, default=json_value))
    print(json.dumps(result, default=json_value), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["generate", "fit"])
    parser.add_argument("--case", default="two_phase_clean")
    args = parser.parse_args()
    if args.action == "generate":
        generate()
    else:
        run_case(args.case)
