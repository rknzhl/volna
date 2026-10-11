"""Unlabeled rank diagnostic on 30 and 60 days of Delaware tide data."""

from __future__ import annotations

from project_paths import RESULTS_DIR

import json
from pathlib import Path

import numpy as np

from method import estimate_peaks, infer_frequency_lattice, voronoi_weights_1d


HERE = Path(__file__).resolve().parents[2]
OUT = RESULTS_DIR


def spectral_median(observed: np.ndarray, area: np.ndarray) -> float:
    y = observed - observed.mean(axis=1, keepdims=True)
    n = y.shape[1]
    nfft = 1 << int(np.ceil(np.log2(8*n)))
    power = np.sum(area[:, None]*abs(np.fft.rfft(
        y*np.hanning(n), n=nfft, axis=1))**2, axis=0)
    power[:max(2, nfft//n)] = 0
    return float(np.median(power[power > 0]))


def main() -> None:
    data = np.load(HERE / "data/real/tides.npz")
    distance = data["distance_km"]
    area = voronoi_weights_1d(distance,
                               (float(distance.min()), float(distance.max())))
    rows = []
    for n in [7200, 14400]:
        observed = data["eta"][:, :n].astype(float)
        peaks, strengths = estimate_peaks(observed, area, count=16,
                                          suppress_harmonics=False)
        floor = spectral_median(observed, area)
        selected = peaks[strengths > 3.5*floor][:8]
        for retained in [4, 6, 8]:
            subset = selected[:retained]
            relation = infer_frequency_lattice(subset, tolerance=2*np.pi/n,
                                               max_rank=3, max_order=6)
            rows.append({"samples": n, "days": n*6/60/24,
                         "max_peaks": retained, "selected_frequencies": subset.tolist(),
                         "selected_power_to_median": (strengths[:len(subset)]/floor).tolist(),
                         "conditional_lattice": relation})
    payload = {"source": "local data/real/tides.npz",
               "constituents_in_metadata": data["constituents"].tolist(),
               "interpretation": "unlabeled diagnostic; inferred rank depends on window length and peak count",
               "rows": rows}
    (OUT / "tides_rank_diagnostic.json").write_text(json.dumps(payload, indent=2) + "\n")
    for row in rows:
        print(row["days"], row["max_peaks"], row["conditional_lattice"]["rank"])


if __name__ == "__main__":
    main()
