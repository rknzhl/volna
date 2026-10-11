"""Finite-sample visibility threshold for an added third torus phase.

The true observable rank changes from two to three at any positive third-mode
amplitude, but no finite noisy procedure can detect arbitrarily small modes.
"""

from __future__ import annotations

from project_paths import RESULTS_DIR

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from method import estimate_peaks, infer_frequency_lattice
from run_controlled import AREA, N, PROFILES
from run_dimension_validation import MAX_VISIBLE, POWER_TO_MEDIAN, spectral_median
from run_spectral_baseline import refine_frequency


OUT = RESULTS_DIR
SEEDS = list(range(20000, 20030))
NOISE = [.10, .20, .40]
AMPLITUDES = [0.0, .01, .02, .05, .10, .20]


def clean_field(amplitude: float) -> np.ndarray:
    frequencies = 2*np.pi*np.array([.061*np.sqrt(2), .073*np.sqrt(3), .083*np.sqrt(5)])
    phases = frequencies[:, None]*np.arange(N)[None, :] + np.arange(3)[:, None]*.37
    a, b, c = PROFILES
    return (a[:, None]*np.cos(phases[0])[None, :]
            + b[:, None]*np.cos(phases[1])[None, :]
            + amplitude*c[:, None]*np.cos(phases[2])[None, :])


def estimate_rank(observed: np.ndarray) -> tuple[int | None, int]:
    peaks, strengths = estimate_peaks(observed, AREA, count=16,
                                      suppress_harmonics=False)
    floor = spectral_median(observed)
    selected = peaks[strengths > POWER_TO_MEDIAN*floor][:MAX_VISIBLE]
    if len(selected) == 0:
        return None, 0
    refined = np.array([refine_frequency(observed, float(x)) for x in selected])
    relation = infer_frequency_lattice(refined, tolerance=2*np.pi/N,
                                       max_rank=3, max_order=6)
    return relation["rank"], len(selected)


def main() -> None:
    rows = []
    for amplitude in AMPLITUDES:
        clean = clean_field(amplitude)
        scale = np.sqrt(np.mean(np.sum(AREA[:, None]*clean**2, axis=0)))
        for noise in NOISE:
            for seed in SEEDS:
                observed = clean + noise*scale*np.random.default_rng(seed).standard_normal(clean.shape)
                rank, count = estimate_rank(observed)
                rows.append({"third_amplitude": amplitude, "noise": noise,
                             "seed": seed, "true_observable_rank": 2 if amplitude == 0 else 3,
                             "estimated_rank": rank, "selected_peak_count": count})
        print("amplitude", amplitude, "complete", flush=True)
    summary = []
    for amplitude in AMPLITUDES:
        for noise in NOISE:
            chosen = [r for r in rows if r["third_amplitude"] == amplitude
                      and r["noise"] == noise]
            truth_rank = 2 if amplitude == 0 else 3
            summary.append({"third_amplitude": amplitude, "noise": noise,
                            "true_observable_rank": truth_rank,
                            "detected_three_rate": float(np.mean([r["estimated_rank"] == 3
                                                                  for r in chosen])),
                            "correct_rate": float(np.mean([r["estimated_rank"] == truth_rank
                                                            for r in chosen])),
                            "median_selected_peaks": float(np.median([
                                r["selected_peak_count"] for r in chosen]))})
    payload = {"method": "spectral threshold plus bounded integer relations",
               "power_to_median": POWER_TO_MEDIAN,
               "seeds": SEEDS, "rows": rows, "summary": summary}
    (OUT / "visibility_sweep.json").write_text(json.dumps(payload, indent=2) + "\n")
    fig, ax = plt.subplots(figsize=(6.5, 4))
    colors = {0.1: "#0072B2", 0.2: "#009E73", 0.4: "#D55E00"}
    for noise in NOISE:
        selected = [r for r in summary if r["noise"] == noise]
        ax.plot([r["third_amplitude"] for r in selected],
                [r["detected_three_rate"] for r in selected],
                marker="o", color=colors[noise], label=f"noise {noise:.0%}")
    ax.set(xlabel="amplitude of third phase mode",
           ylabel="fraction classified as rank 3", ylim=(-.04, 1.04))
    ax.grid(alpha=.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "visibility_sweep.png", dpi=160)
    plt.close(fig)
    for entry in summary:
        print(entry, flush=True)


if __name__ == "__main__":
    main()
