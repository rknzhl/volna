"""Monte Carlo audit of finite-time Gaussian energy intervals, not of phases."""

from project_paths import RESULTS_DIR
from pathlib import Path
import json
import numpy as np
from noise_energy import prediction_energy_interval


def main():
    rng = np.random.default_rng(54401)
    dimension, samples, repeats = 6, 320, 500
    rows = []
    for spectrum in (np.full(dimension, .0064),
                     .0064*np.array([.04, .12, .30, .80, 1.50, 4.])):
        for energy in (0., .005, .05):
            direction = rng.standard_normal((dimension, samples))
            direction *= np.sqrt(energy/max(np.mean(np.sum(direction**2, axis=0)), 1e-30))
            covered = 0
            widths, corrected = [], []
            for _ in range(repeats):
                observed = direction+np.sqrt(spectrum)[:, None]*rng.standard_normal(direction.shape)
                interval = prediction_energy_interval(observed, np.diag(spectrum), alpha=.01)
                covered += interval["lower"] <= energy <= interval["upper"]
                widths.append(interval["upper"]-interval["lower"])
                corrected.append(interval["corrected_validation_energy"])
            rows.append({"noise_eigenvalues": spectrum.tolist(), "true_energy": energy,
                         "covered": covered, "repeats": repeats,
                         "coverage": covered/repeats, "mean_width": float(np.mean(widths)),
                         "mean_corrected_energy": float(np.mean(corrected))})
    output = RESULTS_DIR/"noise_energy_checks.json"
    output.write_text(json.dumps({"rows": rows, "seed": 54401,
        "samples": samples, "alpha": .01, "phase_rank_tested": False}, indent=2)+"\n")
    print([(row["true_energy"], row["coverage"]) for row in rows])


if __name__ == "__main__":
    main()
