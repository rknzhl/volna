"""Held-out diagnostic for spectral frequency-module rank.

This estimates the observable frequency rank from a finite peak list. The
power threshold is fixed, not a theorem-level confidence certificate.
"""

from __future__ import annotations

from project_paths import RESULTS_DIR

import json
from pathlib import Path

import numpy as np

from method import estimate_peaks, infer_frequency_lattice
from run_controlled import AREA, N, PROFILES, case_data, field
from run_dimension import one_phase, three_phase
from run_spectral_baseline import refine_frequency


OUT = RESULTS_DIR
SEEDS = list(range(20000, 20030))
NOISE = [.08, .20, .40]
POWER_TO_MEDIAN = 3.5
MAX_VISIBLE = 8


def spectral_median(observed: np.ndarray) -> float:
    y = observed - observed.mean(axis=1, keepdims=True)
    n = y.shape[1]
    nfft = 1 << int(np.ceil(np.log2(8*n)))
    power = np.sum(AREA[:, None] * abs(np.fft.rfft(
        y*np.hanning(n), n=nfft, axis=1))**2, axis=0)
    power[:max(2, nfft//n)] = 0
    return float(np.median(power[power > 0]))


def clean_cases() -> list[tuple[str, np.ndarray, int]]:
    cases = []
    for name in ["two_modes", "infinite_weak", "infinite_strong", "weak_primitive"]:
        truth, _, nonlinear, weak = case_data(name)
        cases.append((name, field(truth, nonlinear, weak), 2))
    truth, _, _, _ = case_data("weak_primitive")
    cover = field(truth, nonlinear=0.0, weak=0.0)
    one_harmonic_family = PROFILES[0][:, None] * (
        np.cos(2*truth[0]) + .035*np.cos(truth[0]) + .2*np.cos(3*truth[0]))[None, :]
    cases.extend([("double_cover_observable", cover, 2),
                  ("one_phase_three_harmonics", one_harmonic_family, 1)])
    n = np.arange(N)
    one_freq = 2*np.pi*np.array([.061*np.sqrt(2)])
    one_truth = one_freq[:, None]*n[None, :]
    three_freq = 2*np.pi*np.array([.061*np.sqrt(2), .073*np.sqrt(3), .083*np.sqrt(5)])
    three_truth = three_freq[:, None]*n[None, :] + np.arange(3)[:, None]*.37
    cases.extend([("one_phase_infinite", one_phase(one_truth), 1),
                  ("three_phase", three_phase(three_truth), 3)])
    return cases


def main() -> None:
    rows = []
    for name, clean, observable_rank in clean_cases():
        scale = np.sqrt(np.mean(np.sum(AREA[:, None]*clean**2, axis=0)))
        for noise in NOISE:
            for seed in SEEDS:
                observed = clean + noise*scale*np.random.default_rng(seed).standard_normal(clean.shape)
                peaks, strengths = estimate_peaks(observed, AREA, count=16,
                                                  suppress_harmonics=False)
                floor = spectral_median(observed)
                selected = peaks[strengths > POWER_TO_MEDIAN*floor][:MAX_VISIBLE]
                if len(selected) == 0:
                    rank, relation = None, {"reason": "no peak above threshold"}
                else:
                    refined = np.array([refine_frequency(observed, float(x))
                                        for x in selected])
                    relation = infer_frequency_lattice(refined,
                                                       tolerance=2*np.pi/N,
                                                       max_rank=3, max_order=6)
                    rank = relation["rank"]
                rows.append({"case": name, "true_observable_rank": observable_rank,
                             "noise": noise, "seed": seed,
                             "selected_peaks": selected.tolist(),
                             "selected_count": len(selected),
                             "estimated_rank": rank, "relation": relation})
        print(name, "complete", flush=True)
    summary = []
    for name, _, truth_rank in clean_cases():
        for noise in NOISE:
            chosen = [r for r in rows if r["case"] == name and r["noise"] == noise]
            ranks = [r["estimated_rank"] for r in chosen]
            summary.append({"case": name, "noise": noise,
                            "true_observable_rank": truth_rank,
                            "correct": sum(x == truth_rank for x in ranks),
                            "trials": len(ranks),
                            "rank_counts": {str(k): ranks.count(k) for k in set(ranks)}})
    payload = {"method": "top spectral peaks above 3.5 times median power, bounded integer relations",
               "power_to_median": POWER_TO_MEDIAN, "max_visible": MAX_VISIBLE,
               "relation_tolerance": 2*np.pi/N, "max_integer_order": 6,
               "seeds": SEEDS, "rows": rows, "summary": summary}
    (OUT / "dimension_validation.json").write_text(json.dumps(payload, indent=2) + "\n")
    for entry in summary:
        print(entry, flush=True)


if __name__ == "__main__":
    main()
