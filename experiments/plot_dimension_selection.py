"""Plot automatic phase-count selection from three saved experiment reports.

Run from the repository root:
    .venv/bin/python experiments/plot_dimension_selection.py

No model fitting, SVD, persistence or flow integration is performed.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/torusflow_dimension_mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, ScalarFormatter
import numpy as np

HERE = Path(__file__).resolve().parent
PURPLE, OLIVE, GRAY = "#76528B", "#7E8651", "#B8B8B8"
CASES = (("Sine toy", "toy_sine.json", 0),
         ("Two-wave toy", "toy_torus.json", 0),
         ("Curved torus", "noise00.json", 1))


def read_case(name, filename, attempt_index):
    path = HERE / "data" / filename
    report = json.loads(path.read_text())
    attempt = report["fit"]["attempts"][attempt_index]
    if attempt["lag"] != report["fit"]["selected_lag"]:
        raise ValueError(f"{filename}: not the selected lag")
    diagnostic = attempt["diagnostics"]["dimension_diagnostic"]
    local = diagnostic["geometry_dimension_diagnostic"]
    gaps = np.asarray(local["median_singular_gaps"])
    bars = np.asarray(diagnostic["h1_diagram"])
    r = diagnostic["selected_dimension"]
    t = diagnostic["rips_threshold"]
    threshold = local["gap_threshold"]
    if np.flatnonzero(gaps >= threshold)[0] + 1 != r:
        raise ValueError(f"{filename}: local proposal differs from selected count")
    active = (bars[:, 0] < t) & (t < bars[:, 1])
    if active.sum() != r or diagnostic["active_b1_mod_prime"] != r:
        raise ValueError(f"{filename}: wrong active cycle count")
    if any(diagnostic["h1_right_censored"]):
        raise ValueError(f"{filename}: censored bars require separate plotting")
    left, right = diagnostic["selected_scale_interval"]
    if not left < t < right:
        raise ValueError(f"{filename}: selected threshold is outside its interval")
    # Check every subinterval induced by the entire saved diagram.
    events = np.unique(np.r_[left, right, bars.ravel()])
    events = events[(events >= left) & (events <= right)]
    for midpoint in (events[:-1] + events[1:]) / 2:
        if np.sum((bars[:, 0] < midpoint) & (midpoint < bars[:, 1])) != r:
            raise ValueError(f"{filename}: cycle count is not stable")
    if not (diagnostic["cover_radius"] > diagnostic["train_fill_distance"]
            and diagnostic["train_uncovered_count"] == 0):
        raise ValueError(f"{filename}: saved landmarks do not cover the windows")
    order = np.argsort(-(bars[:, 1] - bars[:, 0]), kind="stable")[:8]
    if np.any(active[np.setdiff1d(np.arange(len(bars)), order)]):
        raise ValueError(f"{filename}: a selected cycle would be omitted")
    return dict(name=name, path=path, attempt=attempt_index, diagnostic=diagnostic,
                gaps=gaps, threshold=threshold, bars=bars, active=active,
                order=order, r=r, t=t, interval=(left, right))


def plain_axes(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#A3A3A3")
    ax.tick_params(length=3, width=.6, color="#A3A3A3")


def main():
    cases = [read_case(*case) for case in CASES]
    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"],
                         "mathtext.fontset": "dejavuserif", "font.size": 11,
                         "axes.labelsize": 11, "xtick.labelsize": 11,
                         "ytick.labelsize": 11, "pdf.fonttype": 42})
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 6.65))
    fig.subplots_adjust(left=.105, right=.975, bottom=.075, top=.89,
                        hspace=1.05, wspace=.42)
    fig.text(.28, .978, "Local SVD", ha="center", va="top", fontsize=13)
    fig.text(.77, .978, "Cycle lifetimes", ha="center", va="top", fontsize=13)
    for row, case in enumerate(cases):
        ax, barcode = axes[row]
        r, t = case["r"], case["t"]
        d = case["diagnostic"]["ambient_dimension"]
        pos = ax.get_position()
        fig.text(.105, pos.y1 + .025, f"{case['name']}   " + rf"$D={d}$",
                 fontsize=12, weight="bold", ha="left", va="bottom")

        gaps = case["gaps"]
        k = np.arange(1, len(gaps) + 1)
        ax.axhline(case["threshold"], color=OLIVE, linestyle="--", linewidth=1)
        ax.plot(k, gaps, color=GRAY, linewidth=1, marker="o", markersize=4)
        ax.plot(r, gaps[r-1], "o", color=PURPLE, markersize=6)
        ax.set_yscale("log")
        ax.set_ylim(.9, max(6., 1.45 * gaps.max()))
        ax.set_xlim(.6, max(1.4, len(k) + .4))
        ticks = [1, 2, 5] + ([10, 50] if gaps.max() > 10 else [])
        ax.yaxis.set_major_locator(FixedLocator(ticks))
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_minor_locator(FixedLocator([]))
        ax.set_xticks(k if len(k) <= 4 else [1, 3, 5, 7, 9, 11])
        ax.set_xlabel(r"Direction index $k$", labelpad=3)
        ax.set_ylabel(r"Ratio $\sigma_k/\sigma_{k+1}$", labelpad=3)
        ax.text(.98, .88, rf"$r={r}$", transform=ax.transAxes,
                ha="right", color=PURPLE, fontsize=12,
                bbox=dict(facecolor="white", edgecolor="none", pad=1))
        plain_axes(ax)

        bars, order = case["bars"], case["order"]
        barcode.axvspan(*case["interval"], color=OLIVE, alpha=.10, linewidth=0)
        barcode.axvline(t, color=OLIVE, linewidth=1.3, zorder=3)
        for y, index in enumerate(order):
            color = PURPLE if case["active"][index] else GRAY
            barcode.plot(bars[index], [y, y], color=color, linewidth=2.1,
                         solid_capstyle="round", zorder=4)
        barcode.set_ylim(len(order)-.45, -.85)
        barcode.set_xlim(0, 1.05 * bars[order, 1].max())
        barcode.set_xticks([0, .5, 1.0] if bars[order, 1].max() < 1.4
                          else [0, .5, 1.0, 1.5])
        barcode.set_yticks([])
        barcode.set_xlabel(r"Connection distance $t$", labelpad=3)
        barcode.spines[["left", "top", "right"]].set_visible(False)
        barcode.spines["bottom"].set_color("#A3A3A3")
        barcode.tick_params(length=3, width=.6, color="#A3A3A3")
        barcode.text(.99, 1.02, f"{r} {'cycle' if r == 1 else 'cycles'} at " + r"$t$",
                     transform=barcode.transAxes, ha="right", va="bottom",
                     color=PURPLE, fontsize=11)

    output = HERE / "figures" / "dimension_selection.pdf"
    output.parent.mkdir(exist_ok=True)
    fig.savefig(output, facecolor="white", metadata={
        "Title": "Automatic phase count in three saved TorusFlow experiments",
        "Creator": Path(__file__).name})
    plt.close(fig)
    provenance = dict(
        computations="Read saved reports, check counts and coverage, plot ratios and bars only.",
        plot=output.relative_to(HERE).as_posix(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        local_gap="Median over neighbourhoods of adjacent local singular-value ratios; first ratio at least 2.",
        barcode="Up to eight longest saved lifetimes. All cycles crossing the selected threshold are shown.",
        checks="Counts agree with r throughout the selected open interval; no censored bars; all training windows covered.",
        cases=[dict(name=c["name"], source=c["path"].relative_to(HERE).as_posix(),
                    source_sha256=hashlib.sha256(c["path"].read_bytes()).hexdigest(),
                    selected_attempt=c["attempt"], dimension=c["r"],
                    ambient_dimension=c["diagnostic"]["ambient_dimension"],
                    landmark_count=c["diagnostic"]["landmarks"],
                    threshold=c["t"], shown_bar_indices=c["order"].tolist(),
                    total_bar_count=len(c["bars"])) for c in cases])
    (HERE / "data" / "dimension_selection_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
