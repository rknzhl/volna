"""One capacity check of classical horizontal MSSA on two infinite fields.

Fixed L=512; rank chosen from a declared expanded grid by observed validation
MSE. No frequencies, source law, clean field or phase rank enters fitting.
Both chosen forecasts are saved before generator truth is opened for scoring.
FFT convolution evaluates exactly the same diagonal average as the reference
MSSA implementation; it does not change the projection or recurrence.
"""

from __future__ import annotations

from project_paths import RESULTS_DIR

import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.fft import irfft, next_fast_len, rfft
from scipy.linalg import eigh
from scipy.linalg.blas import dgemm

from run_actual_wave_revised import fixture_inputs


HERE = Path(__file__).resolve().parent
OUTPUT = RESULTS_DIR / 'mssa_capacity_check.json'
CASES = ("r2_infinite_mixed_products", "r2_poisson_dominant_weak")
WINDOW = 512
RANKS = (2, 4, 8, 16, 32, 48, 64, 96, 128, 192, 256, 320, 384)
SOURCE_FILES = ("run_mssa_capacity_check.py", "run_actual_wave_revised.py", "iterative_fields.py")


def fingerprint():
    return {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in SOURCE_FILES}


def save(payload):
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    temporary.replace(OUTPUT)


def fit_capacity_mssa(train, validation, future_count, name):
    started = time.perf_counter()
    q, n = train.shape
    mean = train.mean(axis=1, keepdims=True)
    centered = train - mean
    k = n - WINDOW + 1
    blocks = [np.lib.stride_tricks.sliding_window_view(row, WINDOW).T for row in centered]
    hankel = np.asfortranarray(np.hstack(blocks))
    covariance = dgemm(1., hankel, hankel, trans_b=True)
    eigenvalues, eigenvectors = eigh(covariance, driver="evr", check_finite=True)
    order = np.argsort(eigenvalues)[::-1]
    u = np.asfortranarray(eigenvectors[:, order[:max(RANKS)]])
    coefficients = dgemm(1., u, hankel, trans_a=True).reshape(max(RANKS), q, k)
    del hankel, blocks
    counts = np.convolve(np.ones(WINDOW), np.ones(k))
    nfft = next_fast_len(n)
    basis_fft = rfft(u.T, n=nfft, axis=1)
    smoothed = np.empty((len(RANKS), q, n))
    diagonal_average_max_error = None
    for sensor in range(q):
        cumulative = np.cumsum(basis_fft * rfft(coefficients[:, sensor], n=nfft, axis=1), axis=0)
        smoothed[:, sensor] = irfft(cumulative[np.asarray(RANKS)-1], n=nfft, axis=1)[:, :n] / counts
        if sensor == 0:
            # Exact algebra check against the reference implementation at rank2.
            projected = dgemm(1., np.asfortranarray(u[:, :2]),
                              np.asfortranarray(coefficients[:2, sensor]))
            reference = np.asarray([np.mean(np.diagonal(projected[::-1], offset=index-WINDOW+1))
                                    for index in range(n)])
            diagonal_average_max_error = float(np.max(abs(reference-smoothed[0, sensor])))
            if diagonal_average_max_error > 1e-10*max(1., float(np.max(abs(reference)))):
                raise ArithmeticError("FFT diagonal average differs from the reference")
    del coefficients
    grid, candidates = [], []
    for index, rank in enumerate(RANKS):
        denominator = 1. - float(np.sum(u[-1, :rank] ** 2))
        item = {"rank": rank, "recurrence_denominator": denominator}
        if denominator <= 1e-8:
            item["failure"] = "unstable recurrence denominator"
            grid.append(item)
            continue
        recurrence = np.einsum("ij,j->i", u[:-1, :rank], u[-1, :rank], optimize=False) / denominator
        history = np.zeros((q, n+validation.shape[1]+future_count))
        history[:, :n] = smoothed[index]
        with np.errstate(over="ignore", invalid="ignore"):
            for position in range(n, history.shape[1]):
                history[:, position] = np.einsum("i,qi->q", recurrence,
                    history[:, position-len(recurrence):position], optimize=False)
        prediction = history[:, n:] + mean
        if not np.isfinite(prediction).all():
            item["failure"] = "nonfinite recursive forecast"
            grid.append(item)
            print(name + " rank " + str(rank) + " nonfinite", flush=True)
            continue
        error = float(np.mean((prediction[:, :validation.shape[1]]-validation) ** 2))
        if not np.isfinite(error):
            item["failure"] = "nonfinite validation MSE"
            grid.append(item)
            continue
        item.update(validation_sensor_mse=error, finite_forecast=True)
        grid.append(item)
        candidates.append((error, rank, prediction.copy(), smoothed[index].copy()))
        print(name + " rank " + str(rank) + " validation MSE " + format(error, ".8g"), flush=True)
    if not candidates:
        raise RuntimeError("no finite MSSA candidate")
    mse, rank, forecast, reconstruction = min(candidates, key=lambda item: item[:2])
    return {"rank": rank, "window": WINDOW, "rank_grid": list(RANKS),
            "validation_rank_selection_grid": grid, "selected_validation_sensor_mse": mse,
            "prediction": forecast[:, validation.shape[1]:],
            "validation_prediction": forecast[:, :validation.shape[1]],
            "training_reconstruction": reconstruction + mean,
            "descending_covariance_eigenvalues": eigenvalues[order],
            "FFT_diagonal_average_max_absolute_difference_from_reference": diagonal_average_max_error,
            "elapsed_seconds": time.perf_counter()-started}


def score(prediction, clean):
    error = prediction-clean
    return {"sensor_rmse": float(np.sqrt(np.mean(error**2))),
            "sensor_nrmse": float(np.linalg.norm(error)/np.linalg.norm(clean)),
            "all_prediction_values_finite": bool(np.isfinite(prediction).all())}


def main():
    if OUTPUT.exists():
        raise FileExistsError("preserve the existing capacity assay")
    started = time.perf_counter()
    hashes = fingerprint()
    rows = []
    payload = {"status": "running", "rows": rows, "source_fingerprints": hashes,
               "cases": list(CASES), "window": WINDOW, "declared_rank_grid": list(RANKS),
               "sample_split": {"train": 3200, "validation": 1600, "future": 1600},
               "method": "classical horizontal pooled-covariance MSSA and recurrence",
               "rank_selection": "minimum observed validation sensor MSE, rank breaks exact ties",
               "generator_truth_supplied_to_fitting": False,
               "truth_opening_scope": "after BOTH fits and saved frozen forecasts",
               "additional_windows_tested": False}
    save(payload)
    frozen = []
    for name in CASES:
        print("starting capacity " + name, flush=True)
        y, positions, times, _ = fixture_inputs(name)
        fitted = fit_capacity_mssa(y[:, :3200], y[:, 3200:4800], 1600, name)
        arrays = {key: fitted[key] for key in ("prediction", "validation_prediction", "training_reconstruction",
                                              "descending_covariance_eigenvalues")}
        for key, values in arrays.items():
            if not np.isfinite(values).all():
                raise FloatingPointError("nonfinite saved array: " + key)
        path = RESULTS_DIR / ("mssa_capacity_check_" + name + "_frozen.npz")
        np.savez_compressed(path, **arrays)
        row = {"case": name, "run_status": "fitted_and_frozen",
               **{key: value for key, value in fitted.items() if key not in arrays},
               "input_noisy_field_sha256": hashlib.sha256(y.tobytes()).hexdigest(),
               "saved_frozen_arrays": str(path),
               "saved_frozen_arrays_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
               "saved_arrays_all_finite": True}
        rows.append(row)
        frozen.append((name, positions, times, fitted))
        save(payload)

    # Both inferences are now immutable. Open generator clean fields and saved
    # comparison results exclusively for scoring; no fitted parameter changes.
    from iterative_fields import generate_field
    comparison_path = RESULTS_DIR / 'final_phase_factor_assay_fixtures.json'
    comparison = {row["case"]: row for row in json.loads(comparison_path.read_text())["rows"]}
    for row, (name, positions, times, fitted) in zip(rows, frozen):
        clean = generate_field(name, times, positions, n_train=3200)["clean"][:, 4800:]
        previous = comparison[name]
        before = previous["future_clean_sensor_field"]
        row.update(run_status="finished", future_clean_sensor_field=score(fitted["prediction"], clean),
                   comparison_with_saved_frozen_assay={
                       "classical_mssa_L128": before["mssa_window128"],
                       "phase_factor_method": before["revised_phase_family"],
                       "direct_fourier_same_learned_dictionary": before["direct_fourier_same_learned_dictionary"],
                       "MSSA_L128_selected_rank": previous["additional_lawblind_baselines"]["mssa_window128"]["rank"],
                       "direct_learned_dictionary_line_count": previous["additional_lawblind_baselines"]["direct_fourier_same_learned_dictionary"]["dictionary_fit"]["selected_dictionary"]["line_count"],
                       "comparison_does_not_change_inference": True})
        print(json.dumps({"case": name, "selected_rank": row["rank"], "future": row["future_clean_sensor_field"]}), flush=True)
    payload.update(status="finished", invocation_finished=True,
                   source_fingerprints_unchanged=fingerprint()==hashes,
                   comparison_source=str(comparison_path),
                   total_elapsed_seconds=time.perf_counter()-started)
    save(payload)


if __name__ == "__main__":
    main()
