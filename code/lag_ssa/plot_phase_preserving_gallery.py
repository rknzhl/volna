"""Fixed phase charts of saved observed readouts; no truth or phase clocks.

Reads only the completed selected-dictionary JSON and its selected parameter
and output NPZ archives. Color is sample order. Grey geometry is an explicit
chart scaffold, not an inferred physical metric or a topology certificate.
The third panel intentionally projects a three-coordinate phase model onto
two coordinates; it does not embed the full T^3 into R^3.
"""

from project_paths import RESULTS_DIR

import hashlib
import json
from math import gcd
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np


HERE = Path(__file__).resolve().parent
SOURCE = RESULTS_DIR / 'phase_preserving_dictionary.json'
STEM = RESULTS_DIR / 'phase_preserving_gallery'
CASES = ("r2_primitive_no_fundamentals", "mixed_rotation", "r3_travelling_infinite_weak")
MAJOR_RADIUS, MINOR_RADIUS = 2., .65
SEAM_TOLERANCE = 1e-10
PATH_SAMPLE_LIMIT = 64


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def extended_gcd(first, second):
    r0, r1 = abs(first), abs(second)
    x0, x1, y0, y1 = 1, 0, 0, 1
    while r1:
        quotient = r0//r1
        r0, r1 = r1, r0-quotient*r1
        x0, x1 = x1, x0-quotient*x1
        y0, y1 = y1, y0-quotient*y1
    return r0, x0*(1 if first >= 0 else -1), y0*(1 if second >= 0 else -1)


def fixed_chart(labels, relation):
    rank = labels.shape[1]
    if relation.size:
        if rank != 2 or relation.shape != (2,):
            raise ValueError("the compact gallery panel supports one relation in two coordinates")
        order = gcd(abs(int(relation[0])), abs(int(relation[1])))
        primitive = relation//order
        common, x, y = extended_gcd(int(primitive[0]), int(primitive[1]))
        if common != 1:
            raise ArithmeticError("primitive relation direction required")
        transform = np.asarray([primitive, [-y, x]], dtype=int)
        if int(transform[0, 0])*int(transform[1, 1])-int(transform[0, 1])*int(transform[1, 0]) != 1:
            raise ArithmeticError("phase chart must be unimodular")
        return transform, order
    return np.eye(rank, dtype=int), None


def chart_angles(phase, transform):
    angles = np.einsum("ij,jt->it", transform, phase, optimize=False)
    displayed = np.mod(angles, 2*np.pi)
    # Numerically equivalent representatives at the S^1 seam, not a fit or
    # snapping of the measured phases to a clock or a group component.
    displayed[(displayed < SEAM_TOLERANCE)|(2*np.pi-displayed < SEAM_TOLERANCE)] = 0.
    return displayed


def torus(angles):
    first, second = angles
    radius = MAJOR_RADIUS+MINOR_RADIUS*np.cos(second)
    return np.asarray([radius*np.cos(first), radius*np.sin(first), MINOR_RADIUS*np.sin(second)])


def path_segments(angles, branch=None):
    count = min(PATH_SAMPLE_LIMIT, angles.shape[1])
    indices = []
    for index in range(count-1):
        delta = np.angle(np.exp(1j*(angles[:, index+1]-angles[:, index])))
        if np.max(abs(delta)) > np.pi/3:
            continue
        if branch is not None and branch[index+1] != branch[index]:
            continue
        indices.append(index)
    return indices


def load_panels():
    source = json.loads(SOURCE.read_text())
    if source["status"] != "finished":
        raise ValueError("the selected-dictionary run must already be frozen and complete")
    panels = []
    for name in CASES:
        matches = [row for row in source["rows"] if row["case"] == name]
        if len(matches) != 1:
            raise ValueError("one explicitly specified saved run per gallery case required")
        row = matches[0]
        if row["run_status"] != "finished":
            raise ValueError("incomplete saved gallery run")
        for path_key, hash_key in (("frozen_parameters", "frozen_parameters_sha256"),
                                  ("frozen_output", "frozen_output_sha256")):
            if sha(row[path_key]) != row[hash_key]:
                raise ValueError("saved gallery archive changed")
        with np.load(row["frozen_parameters"], allow_pickle=False) as archive:
            labels = archive["labels"]
            relation = archive["relation"]
        with np.load(row["frozen_output"], allow_pickle=False) as archive:
            phase = archive["phase_at_window_end"]
        if (labels.ndim != 2 or phase.ndim != 2 or phase.shape[0] != labels.shape[1]
                or not np.isfinite(phase).all() or phase.shape[1] != row["future_count"]):
            raise ValueError("matching saved labels and measured endpoint phases required")
        transform, order = fixed_chart(labels, relation)
        angles = chart_angles(phase, transform)[:2]
        if len(angles) != 2:
            raise ValueError("the two-coordinate gallery requires at least two phase rows")
        branch = None if order is None else np.rint(order*angles[0]/(2*np.pi)).astype(int)%order
        panels.append({"case": name, "run": row["run"], "rank": phase.shape[0],
            "labels": labels, "relation": relation, "transform": transform,
            "order": order, "angles": angles, "branch": branch,
            "saved_phase_key": "phase_at_window_end", "samples": phase.shape[1],
            "selected_group_family": row["selected_group_family_frozen"],
            "parameter_file": row["frozen_parameters"], "parameter_sha256": row["frozen_parameters_sha256"],
            "phase_file": row["frozen_output"], "phase_sha256": row["frozen_output_sha256"],
            "source_phase_shape": list(phase.shape), "displayed_coordinate_indices": [0, 1],
            "path_indices": path_segments(angles, branch)})
    if panels[0]["rank"] != 2 or panels[0]["order"] is not None:
        raise ValueError("first panel requires the saved unconstrained two-coordinate model")
    if panels[1]["order"] is None or panels[1]["selected_group_family"] != "compact":
        raise ValueError("middle panel must use the saved selected compact model")
    if panels[2]["rank"] != 3 or panels[2]["order"] is not None:
        raise ValueError("third panel requires the saved three-coordinate model")
    return panels


def draw(panels):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 12, "axes.labelsize": 10, "pdf.fonttype": 42,
        "savefig.facecolor": "white", "figure.facecolor": "white"})
    fig = plt.figure(figsize=(14.7, 8.5))
    grid = fig.add_gridspec(2, 3, left=.065, right=.905, bottom=.14, top=.825,
        height_ratios=(1., 1.15), hspace=.22, wspace=.28)
    norm = colors.Normalize(vmin=0, vmax=max(panel["samples"]-1 for panel in panels))
    cmap = plt.get_cmap("viridis")
    titles = [r"Learned $\mathbb{T}^2$"+"\nprimitive-cover example",
              r"Selected $\mathbb{Z}_{"+str(panels[1]["order"])+r"}\times\mathbb{S}^1$"+"\nseven components; no full torus surface",
              r"Projection of learned $\mathbb{T}^3$"+"\nfirst two coordinates; third omitted"]
    for column, panel in enumerate(panels):
        angles = panel["angles"]
        flat = angles/(2*np.pi)
        indices = np.arange(panel["samples"])
        top = fig.add_subplot(grid[0, column])
        top.set_title(titles[column], pad=13)
        if panel["order"] is not None:
            for component in range(panel["order"]):
                top.axvline(component/panel["order"], color="#b9c1cd", lw=.7, zorder=0)
        top.scatter(flat[0], flat[1], c=indices, cmap=cmap, norm=norm,
            s=10, alpha=.82, linewidths=0, rasterized=True, zorder=2)
        flat_segments = [np.column_stack((flat[:, index], flat[:, index+1])).T
            for index in panel["path_indices"] if np.max(abs(flat[:, index+1]-flat[:, index])) < .5]
        if flat_segments:
            top.add_collection(LineCollection(flat_segments, colors="#23374d", linewidths=.6, alpha=.45, zorder=1))
        top.scatter(flat[0, 0], flat[1, 0], s=38, facecolors="none", edgecolors="black", linewidths=.9, zorder=3)
        top.set_xlim(-.025, 1.025);top.set_ylim(-.025, 1.025)
        top.set_aspect("equal", adjustable="box")
        ticks = [0., .25, .5, .75, 1.]
        top.set_xticks(ticks);top.set_yticks(ticks)
        top.grid(alpha=.16, linewidth=.5)
        top.set_xlabel(r"$z_1/(2\pi)$" if panel["order"] else r"$\theta_1/(2\pi)$")
        top.set_ylabel(r"$z_2/(2\pi)$" if panel["order"] else r"$\theta_2/(2\pi)$")
        if panel["order"]:
            matrix = panel["transform"]
            top.text(.5, -.24, r"$z=U\theta$"+",  U = [["+
                str(matrix[0, 0])+", "+str(matrix[0, 1])+"], ["+str(matrix[1, 0])+", "+str(matrix[1, 1])+
                "]]", transform=top.transAxes, ha="center", fontsize=9)
        bottom = fig.add_subplot(grid[1, column], projection="3d")
        if panel["order"] is None:
            first, second = np.meshgrid(np.linspace(0, 2*np.pi, 30), np.linspace(0, 2*np.pi, 17))
            scaffold = torus(np.asarray([first, second]))
            bottom.plot_wireframe(*scaffold, rstride=2, cstride=2, color="#aeb7c4", linewidth=.35, alpha=.4)
        else:
            second = np.linspace(0, 2*np.pi, 180)
            for component in range(panel["order"]):
                circle = torus(np.asarray([np.full_like(second, 2*np.pi*component/panel["order"]), second]))
                bottom.plot(*circle, color="#98a5b7", linewidth=.9, alpha=.65)
        embedded = torus(angles)
        bottom.scatter(*embedded, c=indices, cmap=cmap, norm=norm,
            s=9, alpha=.87, linewidths=0, depthshade=False, rasterized=True)
        segments = [np.column_stack((embedded[:, index], embedded[:, index+1])).T
                    for index in panel["path_indices"]]
        if segments:
            bottom.add_collection3d(Line3DCollection(segments, colors="#23374d", linewidths=.55, alpha=.5))
        bottom.scatter(*embedded[:, 0], s=34, facecolors="none", edgecolors="black", linewidths=.9, depthshade=False)
        bottom.view_init(elev=24, azim=-57)
        bottom.set_box_aspect((1., 1., .42), zoom=1.22)
        bottom.set_xlim(-2.85, 2.85);bottom.set_ylim(-2.85, 2.85);bottom.set_zlim(-.85, .85)
        bottom.set_axis_off()
        bottom.text2D(.5, -.015, "Fixed torus chart" if panel["rank"] == 2 and panel["order"] is None else
            "Seven circles; time-colored sampled states" if panel["order"] else
            r"$\mathbb{T}^3\to\mathbb{T}^2\to\mathbb{R}^3$; coordinate projection only",
            transform=bottom.transAxes, ha="center", fontsize=10)
    fig.suptitle("Saved observed phase readouts after dictionary selection", fontsize=16, y=.955)
    fig.text(.065, .912, "Flat phase charts above; fixed geometric charts below. Black ring marks the first saved state.", fontsize=10, color="#384554")
    color_axis = fig.add_axes([.933, .29, .014, .43])
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=color_axis)
    bar.set_label("Saved sample order", labelpad=9)
    bar.set_ticks([0, 300, 600, 900, 1199])
    fig.text(.065, .09,
        "Colored points come only from saved endpoint-phase arrays. Grey curves/grids are chart scaffolds; no clock-generated trajectory is shown.",
        fontsize=9, color="#384554")
    fig.text(.065, .062,
        "The compact group is a selected approximate model. These charts establish neither a physical metric nor exact topology or phase rank.",
        fontsize=9, color="#384554")
    fig.text(.065, .035,
        "The right panel omits the third phase and cannot display all T3 state distinctions. No continuous path is interpolated between compact components.",
        fontsize=9, color="#384554")
    fig.savefig(STEM.with_suffix(".png"), dpi=190)
    fig.savefig(STEM.with_suffix(".pdf"))
    plt.close(fig)


def main(rerender=False):
    if rerender:
        existing = json.loads(STEM.with_suffix(".json").read_text())
        if (existing.get("plot_code") != str(Path(__file__).resolve()) or
                existing.get("source_json") != str(SOURCE) or
                existing.get("source_json_sha256") != sha(SOURCE)):
            raise ValueError("only this plot's own unchanged-source render can be revised")
    for suffix in (".png", ".pdf", ".json"):
        if STEM.with_suffix(suffix).exists() and not rerender:
            raise FileExistsError("preserve existing gallery outputs")
    source_sha = sha(SOURCE)
    code_sha = sha(__file__)
    panels = load_panels()
    draw(panels)
    metadata = {"status": "rendered", "source_json": str(SOURCE), "source_json_sha256": source_sha,
        "plot_code": str(Path(__file__).resolve()), "plot_code_sha256": code_sha,
        "phase_source": "saved observed phase_at_window_end only",
        "evaluation_archives_opened": False, "truth_basis_phase_or_metric_used": False,
        "frequency_or_clock_generated_phase_used": False, "new_discovery_or_tests_run": False,
        "selected_models_or_existing_artifacts_changed": False,
        "fixed_torus_chart_radii": [MAJOR_RADIUS, MINOR_RADIUS],
        "seam_representative_cleanup_tolerance_rad": SEAM_TOLERANCE,
        "short_path_segment_sample_limit": PATH_SAMPLE_LIMIT,
        "path_rule": "consecutive saved states only, short angular steps; no compact-component connections; flat seam connections omitted",
        "scaffold_scope": "explicit mathematical chart, not inferred geometry",
        "exact_physical_group_rank_or_metric_certified": False,
        "panels": [{key: (value.tolist() if isinstance(value, np.ndarray) else value)
            for key, value in panel.items() if key not in ("angles", "branch")}
            for panel in panels],
        "artifacts": {suffix[1:]: {"path": str(STEM.with_suffix(suffix)), "sha256": sha(STEM.with_suffix(suffix))}
            for suffix in (".png", ".pdf")}}
    metadata["source_json_unchanged_after_render"] = sha(SOURCE) == source_sha
    metadata["plot_code_unchanged_after_render"] = sha(__file__) == code_sha
    STEM.with_suffix(".json").write_text(json.dumps(metadata, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"status": "rendered", "png": str(STEM.with_suffix(".png")),
        "pdf": str(STEM.with_suffix(".pdf")), "metadata": str(STEM.with_suffix(".json"))}, indent=2))


if __name__ == "__main__":
    main()
