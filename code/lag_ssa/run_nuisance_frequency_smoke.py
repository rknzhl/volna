"""Three targeted smoke cases for the harmonic nuisance frequency helper."""

from __future__ import annotations

from project_paths import RESULTS_DIR

from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
from scipy.stats import chi2

from iterative_fields import AREA, generate_field
from iterative_ssa import noise_from_spatial_orthogonal_component
from nuisance_frequency_fit import fit_joint_frequencies_with_harmonic_nuisance
from phase_lattice import infer_character_basis_global


OUT = RESULTS_DIR
SOURCE = OUT/"iterative_ssa_baseline_full.json"
DESTINATION = OUT/"nuisance_frequency_smoke.json"
CASES = ("r2_poisson_dominant_weak", "r2_partial_cover", "r3_three_phases_weak_third")


def serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(item) for item in value]
    return value


def main():
    source = json.loads(SOURCE.read_text())
    rows = []
    for name in CASES:
        start_time = time.monotonic()
        saved = next(row for row in source["rows"] if row["case"] == name)
        settings = saved["settings"]
        n = settings["n_train"]
        accepted = [stage for stage in saved["inference"]["stages"]
                    if stage["status"] == "accepted_character_conditional_lattice"]
        starting = np.array([stage["candidate"]["frequency"] for stage in accepted])
        times = np.arange(n+settings["n_validation"]+settings["n_future"])
        generated = generate_field(name, times, n_train=n)
        clean = generated["clean"]
        sigma = saved["relative_sensor_noise"]*float(np.sqrt(np.mean(np.sum(
            AREA[:, None]*clean[:, :n]**2, axis=0))))
        observed = clean+sigma*np.random.default_rng(saved["seed"]).standard_normal(clean.shape)
        raw_sigma = noise_from_spatial_orthogonal_component(observed[:, :n])["sigma"]
        fit = fit_joint_frequencies_with_harmonic_nuisance(observed[:, :n], starting,
                                                         noise_sigma=raw_sigma)
        radius = float(np.sqrt(chi2.ppf(.99, len(starting))))
        lattice = infer_character_basis_global(fit["frequencies"], tolerance=1e-12,
            max_order=settings["max_relation_order"], max_rank=settings["max_phase_rank"],
            frequency_covariance=fit["covariance"], confidence_radius=radius)
        future_start = n+settings["n_validation"]
        future_times = times[future_start:]-fit["times_center"]
        angles = future_times[:, None]*(fit["line_labels"]@fit["frequencies"])[None, :]
        design = np.column_stack((np.ones(len(future_times)), np.cos(angles), np.sin(angles)))
        prediction = np.einsum("nj,ji->in", design, fit["coefficients"], optimize=False)
        # Begin truth-only scoring after fit and conditional lattice inference.
        truth = generated["metadata"]["alpha"]
        true_peaks = np.array([2*truth[0], truth[1], 3*truth[0]]) if name == "r2_partial_cover" else truth
        row = {"case": name, "starting_frequencies_train_selected": starting,
               "noise_sigma_estimated_from_training_spatial_orthogonal_data": raw_sigma,
               "frequencies": fit["frequencies"], "frequency_errors_truth_only": fit["frequencies"]-true_peaks,
               "standard_errors_local_iid": fit["frequency_standard_errors"],
               "optimizer_success": fit["optimizer_success"],
               "stationary_in_frequency_uncertainty": fit["stationary_in_frequency_uncertainty"],
               "nuisance_audit": fit["nuisance_audit"], "conditional_lattice": lattice,
               "future_clean_sensor_rmse": float(np.sqrt(np.mean((prediction-clean[:, future_start:])**2))),
               "elapsed_seconds": time.monotonic()-start_time}
        rows.append(row)
        payload = {"source": str(SOURCE), "case_names": CASES, "rows": rows,
                   "created_utc": datetime.now(timezone.utc).isoformat(),
                   "complete": len(rows) == len(CASES), "truth_supplied_to_fit": False,
                   "future_used_to_fit_or_select_order": False,
                   "scope": "three targeted smoke cases; no infinite-tail or completeness guarantee"}
        DESTINATION.write_text(json.dumps(serializable(payload), indent=2, allow_nan=False)+"\n")
        print(name, "K", fit["harmonic_order"], "rank", lattice.get("rank"),
              "freq_errors", row["frequency_errors_truth_only"], "seconds", round(row["elapsed_seconds"],2), flush=True)


if __name__ == "__main__":
    main()
