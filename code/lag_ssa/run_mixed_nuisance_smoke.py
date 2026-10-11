"""Targeted mixed-nuisance audit with training-only candidate starts."""

from __future__ import annotations

from project_paths import RESULTS_DIR

from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
from scipy.signal import find_peaks
from scipy.stats import chi2

from iterative_fields import AREA, generate_field, score_frequency
from iterative_ssa import noise_from_spatial_orthogonal_component
from mixed_nuisance_frequency_fit import fit_joint_frequencies_with_sparse_mixed_nuisance
from phase_lattice import infer_character_basis_global
from spatial_field import SPATIAL_MODEL


OUT = RESULTS_DIR
SOURCE = OUT/"iterative_ssa_baseline_full.json"
DESTINATION = OUT/"mixed_nuisance_smoke.json"
CASES = ("r2_infinite_mixed_products", "r2_mixed_no_fundamentals")


def _serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serializable(item) for item in value]
    return value


def main():
    source = json.loads(SOURCE.read_text())
    rows = []
    for name in CASES:
        started = time.monotonic()
        saved = next(row for row in source["rows"] if row["case"] == name)
        settings = saved["settings"]
        n = settings["n_train"]
        times = np.arange(n+settings["n_validation"]+settings["n_future"])
        generated = generate_field(name, times, n_train=n)
        clean = generated["clean"]
        sigma = saved["relative_sensor_noise"]*float(np.sqrt(np.mean(np.sum(
            AREA[:, None]*clean[:, :n]**2, axis=0))))
        observed = clean+sigma*np.random.default_rng(saved["seed"]).standard_normal(clean.shape)
        train = observed[:, :n]
        if name == "r2_infinite_mixed_products":
            mass = SPATIAL_MODEL.to_mass_coordinates(SPATIAL_MODEL.recover(train))
            transform = np.fft.rfft((mass-mass.mean(axis=1, keepdims=True))*np.hanning(n),
                                    n=8*n, axis=1)
            power = np.sum(abs(transform)**2, axis=0)
            frequencies = np.fft.rfftfreq(8*n)*2*np.pi
            peaks = find_peaks(power)[0]
            selected = sorted(peaks, key=lambda index: power[index], reverse=True)[:2]
            starting = np.sort(frequencies[selected])
            start_source = "two largest training physical-mass-coordinate Hann peaks"
        else:
            stages = saved["inference"]["stages"]
            starting = np.array([stage["candidate"]["frequency"] for stage in stages
                if stage["status"] == "accepted_character_conditional_lattice"])
            start_source = "all accepted training SSA/Hann candidates in archived baseline"
        noise_estimate = noise_from_spatial_orthogonal_component(train)
        fitted = fit_joint_frequencies_with_sparse_mixed_nuisance(
            train, starting, noise_sigma=noise_estimate["sigma"])
        radius = float(np.sqrt(chi2.ppf(.99, len(starting))))
        lattice = infer_character_basis_global(fitted["frequencies"], tolerance=1e-12,
            max_order=settings["max_relation_order"], max_rank=settings["max_phase_rank"],
            frequency_covariance=fitted["covariance"], confidence_radius=radius)
        future_start = n+settings["n_validation"]
        future_times = times[future_start:]-fitted["times_center"]
        line_frequencies = np.einsum("ij,j->i", fitted["line_labels"], fitted["frequencies"],
                                     optimize=False)
        angles = future_times[:, None]*line_frequencies[None, :]
        design = np.column_stack((np.ones(len(future_times)), np.cos(angles), np.sin(angles)))
        prediction = np.einsum("nj,ji->in", design, fitted["coefficients"], optimize=False)
        # Physical mode matching and clean future scoring occur after fitting.
        matches = [score_frequency(name, float(value), 2*np.pi/n, max_harmonic=12, n_train=n)
                   for value in fitted["frequencies"]]
        errors = [float(value-match["unique_match"]["positive_frequency"])
                  if match["unique_match"] else None for value, match in zip(fitted["frequencies"], matches)]
        row = {"case": name, "starting_frequencies": starting, "starting_source": start_source,
               "frequencies": fitted["frequencies"], "frequency_errors_truth_only": errors,
               "truth_frequency_matches": matches,
               "frequency_standard_errors_local_iid": fitted["frequency_standard_errors"],
               "optimizer_success": fitted["optimizer_success"],
               "stationary_in_frequency_uncertainty": fitted["stationary_in_frequency_uncertainty"],
               "nuisance_audit": fitted["nuisance_audit"], "conditional_lattice": lattice,
               "future_clean_sensor_rmse": float(np.sqrt(np.mean((prediction-clean[:, future_start:])**2))),
               "elapsed_seconds": time.monotonic()-started}
        rows.append(row)
        payload = {"source": str(SOURCE), "case_names": CASES, "rows": rows,
                   "created_utc": datetime.now(timezone.utc).isoformat(),
                   "complete": len(rows) == len(CASES), "truth_supplied_to_fitting": False,
                   "future_used_to_fit_or_select_dictionary": False,
                   "scope": "two targeted diagnostics with finite pairwise dictionaries"}
        DESTINATION.write_text(json.dumps(_serializable(payload), indent=2, allow_nan=False)+"\n")
        audit = fitted["nuisance_audit"]
        print(name, "mixed", len(audit["actual_selected_mixed_rows"]),
              "K", audit["selected_harmonic_order"],
              "noiseinflation", round(audit["initial_harmonic_audit"]["residual_variance_over_supplied_noise_variance"],3),
              "to", round(audit["residual_variance_over_supplied_noise_variance"],3),
              "errors", errors, "rank", lattice.get("rank"),
              "seconds", round(row["elapsed_seconds"],2), flush=True)


if __name__ == "__main__":
    main()
