"""Evaluator and decoder for frozen angle outputs of the overlap pilot.

Hidden phases determine evaluation labels only. Decoder and future encoding
use observed readings and frozen inferred functions. A good-looking torus
does not count as success unless the evaluated winding matrix is unimodular.
"""

from __future__ import annotations

from project_paths import RESULTS_DIR

import itertools
import json
from pathlib import Path

import numpy as np
from scipy import linalg
from sklearn.neighbors import NearestNeighbors

from tangent_overlap import windows, fold


HERE = Path(__file__).resolve().parent
OUT = RESULTS_DIR / 'tangent_overlap_pilot'


def match_phases(angles, truth):
    dim = truth.shape[1]
    labels = np.asarray([m for m in itertools.product(range(-4, 5), repeat=dim)
                         if any(m)], dtype=int)
    characters = np.exp(1j * (truth @ labels.T))
    correlation = np.exp(1j * angles).T @ characters.conj() / len(angles)
    best = np.argmax(abs(correlation), axis=1)
    basis = labels[best]
    origins = np.angle(correlation[np.arange(angles.shape[1]), best])
    target = truth @ basis.T + origins
    error = np.angle(np.exp(1j * (angles - target)))
    determinant = int(round(np.linalg.det(basis))) if basis.shape == (dim, dim) else None
    return {"winding_candidate": basis, "absolute_determinant":
            None if determinant is None else abs(determinant),
            "candidate_is_finite_character_match_not_global_certificate": True,
            "origins": origins, "phase_mae_rad": float(np.mean(abs(error))),
            "phase_max_abs_rad": float(np.max(abs(error))),
            "coherence": abs(correlation[np.arange(angles.shape[1]), best])}


def fourier_design(angles, degree=6):
    r = angles.shape[1]
    labels = np.asarray(list(itertools.product(range(-degree, degree + 1), repeat=r)))
    labels = np.asarray([m for m in labels if any(m) and
                         next(int(a) for a in m if a) > 0])
    argument = angles @ labels.T
    return np.column_stack((np.ones(len(angles)), np.cos(argument), np.sin(argument)))


def decode_windows(train_angles, future_angles, train_windows, degree=6):
    x = fourier_design(train_angles, degree)
    z = fourier_design(future_angles, degree)
    ridge = 1e-5 * len(x)
    coefficient = linalg.solve(x.T @ x + ridge * np.eye(x.shape[1]),
                               x.T @ train_windows, assume_a="pos")
    return z @ coefficient, x.shape[1], linalg.svdvals(x)[-1] / linalg.svdvals(x)[0]


def future_angles(source, characters, y_future, protocol):
    lag, channels = protocol["lag"], y_future.shape[1]
    x_future = windows((y_future - source["mean"]) / source["scale"], lag)
    x_future /= np.sqrt(lag * channels)
    distance, index = NearestNeighbors(n_neighbors=16).fit(source["X"]).kneighbors(x_future)
    width = np.maximum(distance[:, -1], 1e-9)
    weight = np.exp(-(distance / width[:, None]) ** 2)
    weight /= weight.sum(axis=1, keepdims=True)
    value = np.einsum("nk,nkr->nr", weight, characters[index])
    return np.angle(value), abs(value)


def evaluate(name, angles_file, version=""):
    protocol = json.loads((OUT / "protocol.json").read_text())
    inferred = np.load(OUT / f"{name}_inferred.npz")
    learned = np.load(angles_file)
    characters = learned["characters"]
    angles = np.angle(characters)
    ntrain = protocol["train_count"]
    source = np.load(OUT / "inputs" / f"{name}.npz")
    y = source["Y"]
    future, radii = future_angles(inferred, characters, y[ntrain:], protocol)
    prediction_windows, features, condition = decode_windows(angles, future,
        windows(y[:ntrain], protocol["lag"]))
    prediction = fold(prediction_windows, y.shape[1], protocol["lag"])
    hidden = np.load(OUT / "evaluation" / f"{name}.npz")
    train_score = match_phases(angles, hidden["theta"][:len(angles)])
    # The integer label and phase origin are fixed on TRAIN, then scored on FUTURE.
    target = hidden["theta"][ntrain:ntrain + len(future)] @ train_score["winding_candidate"].T
    target += train_score["origins"]
    error = np.angle(np.exp(1j * (future - target)))
    result = {"case": name, "train_phase_evaluation": train_score,
              "future_phase_mae_rad": float(np.mean(abs(error))),
              "future_phase_max_abs_rad": float(np.max(abs(error))),
              "future_interpolated_character_modulus_quantiles": np.quantile(radii, [0, .1, .5]),
              "future_clean_nrmse": float(np.linalg.norm(prediction - hidden["clean"][ntrain:]) /
                                           np.linalg.norm(hidden["clean"][ntrain:])),
              "decoder_fourier_degree": 6, "decoder_features": features,
              "decoder_relative_min_singular": float(condition),
              "future_is_encoding_observed_readings_not_forecast": True}
    result["angle_stage_version"] = version or "v1"
    suffix = f"_{version}" if version else ""
    (OUT / f"{name}{suffix}_angles_evaluation.json").write_text(json.dumps(result, indent=2,
                        default=lambda a: np.asarray(a).tolist()))
    np.savez_compressed(OUT / f"{name}{suffix}_phase_evaluated.npz", train_angles=angles,
                        future_angles=future, prediction=prediction)
    print(json.dumps(result, default=lambda a: np.asarray(a).tolist()), flush=True)
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("case")
    parser.add_argument("angles_file")
    parser.add_argument("--version", default="")
    args = parser.parse_args()
    evaluate(args.case, Path(args.angles_file), version=args.version)
