"""Schematic cycle count: the same neighbour graph before/after triangle filling.

Run from the repository root: .venv/bin/python paper/plot_cycle_count.py
This is an explanatory diagram, not an experiment or fitted reconstruction.
"""
from itertools import combinations
from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/torusflow_cycle_mpl")
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.patches import FancyArrowPatch
import numpy as np

PURPLE, OLIVE, GRAY = "#76528B", "#7E8651", "#858585"


def main():
    angles = 2 * np.pi * np.arange(12) / 12
    inner = np.column_stack((np.cos(angles), np.sin(angles)))
    outer = 1.2 * np.column_stack((np.cos(angles + np.pi / 12),
                                   np.sin(angles + np.pi / 12)))
    points = np.vstack((inner, outer))
    distances = np.linalg.norm(points[:, None] - points[None, :], axis=2)
    adjacency = (distances < .7) & (distances > 0)
    edges = np.argwhere(np.triu(adjacency, 1))
    triangles = np.array([face for face in combinations(range(len(points)), 3)
                          if all(adjacency[a, b] for a, b in combinations(face, 2))])
    edge_index = {tuple(edge): i for i, edge in enumerate(edges)}
    boundary = np.zeros((len(edges), len(triangles)))
    for col, (a, b, c) in enumerate(triangles):
        for edge, sign in (((a, b), 1), ((b, c), 1), ((a, c), -1)):
            boundary[edge_index[edge], col] = sign
    rank = np.linalg.matrix_rank(boundary)
    assert (len(points), len(edges), len(triangles), rank) == (24, 48, 24, 24)
    assert len(edges) - len(points) + 1 - rank == 1

    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "dejavuserif", "font.size": 10,
        "axes.titlesize": 11, "pdf.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.65))
    fig.subplots_adjust(left=.025, right=.985, bottom=.14, top=.84, wspace=.20)
    titles = ["(a) Windows", "(b) Add nearby edges", "(c) Fill triangles"]
    footers = ["A sampled cycle", "Local loops appear", "One independent cycle"]
    patch = points[[0, 1, 13, 12, 0]]
    for stage, ax in enumerate(axes):
        ax.set_aspect("equal")
        ax.set(xlim=(-1.37, 1.37), ylim=(-1.37, 1.37))
        ax.axis("off")
        ax.set_title(titles[stage], pad=10, loc="center")
        if stage == 2:
            ax.add_collection(PolyCollection(points[triangles], facecolors=OLIVE,
                                             alpha=.24, edgecolors="none", zorder=0))
        if stage:
            ax.add_collection(LineCollection(points[edges], colors=GRAY,
                                             linewidths=.65, alpha=.8, zorder=1))
            # The outline of two adjacent triangles is also a local boundary.
            ax.plot(*patch.T, color=OLIVE, lw=1.7, zorder=3)
        ax.scatter(*points.T, s=11, color="#555555", zorder=4)
        if stage == 2:
            closed = np.vstack((inner, inner[0]))
            ax.plot(*closed.T, color=PURPLE, lw=1.8, zorder=5)
            ax.annotate("", xy=inner[3], xytext=inner[2],
                        arrowprops=dict(arrowstyle="-|>", color=PURPLE,
                                        lw=1.8, mutation_scale=10), zorder=6)
        ax.text(.5, -.12, footers[stage], ha="center", va="top",
                transform=ax.transAxes, fontsize=10)
    for start, end in ((.326, .351), (.657, .682)):
        fig.add_artist(FancyArrowPatch((start, .49), (end, .49),
                       transform=fig.transFigure, arrowstyle="-|>",
                       mutation_scale=11, lw=.8, color=GRAY))
    target = Path(__file__).resolve().parent / "figures/cycle_count_violet_olive.pdf"
    fig.savefig(target, facecolor="white")
    plt.close(fig)
    print(f"Saved {target}; 25 graph cycles minus 24 independent triangle boundaries = 1.")


if __name__ == "__main__":
    main()
