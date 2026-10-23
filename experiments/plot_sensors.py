"""Sensor layouts for the two successful toys; saved data are checked, not fitted.

Run from the repository root:
    .venv/bin/python experiments/plot_sensors.py
"""
from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/takers_sensors_mpl")
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
PURPLE, OLIVE = "#76528B", "#7E8651"


def main():
    # These are the exact seed and wave vectors of fundamental2waves.
    positions = np.random.default_rng(31).uniform(0, 1, (12, 2))
    wavevectors = 2 * np.pi * np.array([[1., .5], [-.4, 1.1]])
    with np.load(HERE / "data/toy_torus_data.npz") as saved:
        theta = saved["reference_theta"]
        offsets = positions @ wavevectors.T
        generated = (np.cos(theta[:, :1] - offsets[:, 0])
                     + .6 * np.cos(theta[:, 1:] - offsets[:, 1]))
        np.testing.assert_array_equal(generated, saved["clean"])
        np.testing.assert_array_equal(generated, saved["Y"])
    with np.load(HERE / "data/toy_sine_data.npz") as saved:
        t, y = saved["t"], saved["Y"][:, 0]
        np.testing.assert_array_equal(y, np.sin(t))

    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "dejavuserif", "font.size": 10,
        "axes.titlesize": 11, "axes.labelsize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "axes.linewidth": .7, "pdf.fonttype": 42,
    })
    fig = plt.figure(figsize=(7.4, 3.45), facecolor="white")
    fig.text(.065, .935, "(a) One sensor", fontsize=12, va="top")
    fig.text(.565, .935, "(b) Twelve sensors", fontsize=12, va="top")

    # The scalar toy specifies a time series, not a physical sensor position.
    schema = fig.add_axes([.07, .68, .35, .15])
    schema.axis("off")
    schema.set(xlim=(0, 1), ylim=(0, 1))
    schema.plot(.12, .5, "o", color=PURPLE, ms=9)
    schema.annotate("", xy=(.44, .5), xytext=(.20, .5),
                    arrowprops=dict(arrowstyle="->", color=".25", lw=1.1))
    schema.text(.50, .50, r"$y(t)=\sin t$", va="center", fontsize=12)
    ax = fig.add_axes([.095, .19, .335, .41])
    show = t <= 4 * np.pi
    ax.plot(t[show], y[show], color=PURPLE, lw=1.5)
    ax.axhline(0, color=".8", lw=.6, zorder=0)
    ax.set(xlim=(0, 4 * np.pi), ylim=(-1.2, 1.2), xlabel="$t$", ylabel="$y$")
    ax.set_xticks([0, 2 * np.pi, 4 * np.pi], ["0", r"$2\pi$", r"$4\pi$"])
    ax.set_yticks([-1, 0, 1])
    ax.spines[["right", "top"]].set_visible(False)
    ax.tick_params(length=3)

    ax = fig.add_axes([.565, .19, .36, .65])
    ax.set_aspect("equal")
    ax.scatter(*positions.T, s=28, color=".23", edgecolors="white", linewidth=.5, zorder=3)
    label_offsets = [(-8, 8), (-9, -13), (-10, 8), (10, 5), (-13, 5),
                     (-10, -13), (9, 7), (-10, 8), (9, 5), (11, 5),
                     (8, 6), (-10, 8)]
    for number, (point, offset) in enumerate(zip(positions, label_offsets), start=1):
        ax.annotate(str(number), point, xytext=offset, textcoords="offset points",
                    ha="center", va="center", fontsize=8.5, color=".18")
    # Separate plane-wave arrows and orthogonal fronts avoid suggesting a source.
    starts = np.array([[.70, .70], [.24, .58]])
    for j, color in enumerate([PURPLE, OLIVE]):
        start = starts[j]
        direction = wavevectors[j] / np.linalg.norm(wavevectors[j])
        normal = np.array([-direction[1], direction[0]])
        for shift in (0, .07):
            centre = start + shift * direction
            front = np.array([centre - .075 * normal, centre + .075 * normal])
            ax.plot(*front.T, color=color, alpha=.45, lw=1.0, zorder=1)
        endpoint = start + .25 * direction
        ax.annotate("", xy=endpoint, xytext=start,
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5,
                                    mutation_scale=12), zorder=4)
    ax.text(.75, .91, "Wave 1", color=PURPLE, fontsize=9, ha="center")
    ax.text(.22, .94, "Wave 2", color=OLIVE, fontsize=9, ha="center")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel=r"$\xi_1$", ylabel=r"$\xi_2$")
    ax.set_xticks([0, .5, 1], ["0", "0.5", "1"])
    ax.set_yticks([0, .5, 1], ["0", "0.5", "1"])
    ax.tick_params(length=3)
    for spine in ax.spines.values():
        spine.set_color(".55")
    destination = HERE / "figures/toy_sensors.pdf"
    destination.parent.mkdir(exist_ok=True)
    fig.savefig(destination, facecolor="white")
    plt.close(fig)
    print(f"Saved {destination}; both toy signals match their saved arrays exactly.")


if __name__ == "__main__":
    main()
