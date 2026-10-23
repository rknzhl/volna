"""Six explanatory panels from one successful frozen two-wave experiment.

Run from the repository root:
    .venv/bin/python experiments/plot_walkthrough.py

Only saved arrays are read. No fitting, persistence, field evaluation or flow
integration is performed here. The canonical torus is a display of recovered
phases, not an asserted toroidal shape of the projected window cloud.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/takers_walkthrough_mpl")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from matplotlib.ticker import MaxNLocator
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FIGURES = HERE / "figures"
PURPLE, OLIVE, GRAY = "#76528B", "#7E8651", "#7C7C7C"
MAJOR_RADIUS, MINOR_RADIUS = 2.0, .65


def wrap(value):
    return np.angle(np.exp(1j * value))


def torus_position(phase):
    """A fixed display map F:T^2->R^3, evaluated on recovered angles only."""
    phase = np.asarray(phase)
    u, v = phase[..., 0], phase[..., 1]
    radius = MAJOR_RADIUS + MINOR_RADIUS * np.cos(v)
    return np.stack((radius * np.cos(u), radius * np.sin(u),
                     MINOR_RADIUS * np.sin(v)), axis=-1)


def torus_derivative(phase):
    phase = np.asarray(phase)
    u, v = phase[..., 0], phase[..., 1]
    radius = MAJOR_RADIUS + MINOR_RADIUS * np.cos(v)
    du = np.stack((-radius * np.sin(u), radius * np.cos(u), np.zeros_like(u)), axis=-1)
    dv = MINOR_RADIUS * np.stack((-np.sin(v) * np.cos(u),
                                  -np.sin(v) * np.sin(u), np.cos(v)), axis=-1)
    return np.stack((du, dv), axis=-1)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def choose_triangles(triangles, phase_chart, count=6):
    """Show a small connected subset of actual 2-simplices incident to vertex 0."""
    incident = triangles[np.any(triangles == 0, axis=1)]
    if not len(incident):
        raise ValueError("The saved finite complex has no triangle incident to vertex 0")
    distance = np.linalg.norm(phase_chart, axis=1)
    order = np.argsort(np.max(distance[incident], axis=1), kind="stable")
    chosen, used = [], {0}
    for index in order:
        triangle = incident[index]
        if set(map(int, triangle)) - used:
            chosen.append(triangle)
            used.update(map(int, triangle))
        if len(chosen) == count:
            break
    return np.asarray(chosen, dtype=int)


def export_plot_data():
    source = DATA / "toy_torus_data.npz"
    report_path = DATA / "toy_torus.json"
    report = json.loads(report_path.read_text())
    if not (report["status"] == "transformed_conditionally"
            and report["hat_r"] == 2 and report["primitive_phase_match"]
            and report["prediction_frozen_before_truth_evaluation"]):
        raise ValueError("The walkthrough requires the successful frozen two-phase toy")
    with np.load(source) as saved:
        data = {key: saved[key] for key in saved.files}
    phase = data["train_phases"]
    x = data["model_X"]
    if phase.shape[1] != 2 or x.shape[1] < 3:
        raise ValueError("Expected two phases and at least three retained window coordinates")
    window_ids = np.unique(np.linspace(0, len(x) - 1, 600, dtype=int))
    torus_ids = np.unique(np.linspace(0, len(phase) - 1, 600, dtype=int))
    landmark_ids = data["complex_landmark_indices"].astype(int)
    landmark_phase = phase[landmark_ids]
    chart = wrap(landmark_phase - landmark_phase[0])
    selected_triangles = choose_triangles(data["complex_triangles"].astype(int), chart)
    edges = {tuple(sorted(map(int, edge))) for edge in data["complex_edges"]}
    for a, b, c in selected_triangles:
        assert (min(a, b), max(a, b)) in edges
        assert (min(a, c), max(a, c)) in edges
        assert (min(b, c), max(b, c)) in edges

    # Actual normalized fields in the display chart: DF(Phi) D Phi E.
    # Their magnitudes are retained; no per-arrow normalization is applied.
    probe_take = np.arange(0, len(data["field_probe_indices"]), 4, dtype=int)
    probe_phase = data["field_probe_phases"][probe_take]
    phase_velocity = np.einsum("nrd,ndj->nrj", data["field_probe_Dphi"][probe_take],
                               data["field_probe_E"][probe_take], optimize=False)
    display_velocity = np.einsum("nkr,nrj->nkj", torus_derivative(probe_phase),
                                 phase_velocity, optimize=False)
    probe_ids = data["field_probe_indices"][probe_take].astype(int)
    np.testing.assert_allclose(wrap(probe_phase - phase[probe_ids]), 0, atol=1e-10)
    if not np.isfinite(data["flow_trace_phases"]).all():
        raise ValueError("The saved flow trace leaves the phase cover")
    query_count = min(48, len(data["raw_time_indices"]))
    query_ids = data["raw_time_indices"][:query_count].astype(int)
    arrays = dict(
        window_indices=window_ids, window_coordinates=x[window_ids, :3],
        consecutive_window_coordinates=x[:8, :3],
        torus_indices=torus_ids, torus_phase=phase[torus_ids],
        torus_positions=torus_position(phase[torus_ids]),
        finite_triangle_vertex_indices=selected_triangles,
        finite_phase_chart=chart, finite_complex_root=np.asarray(0),
        field_source_indices=probe_ids, field_phases=probe_phase,
        field_positions=torus_position(probe_phase),
        field_display_velocity=display_velocity,
        field_phase_velocity=phase_velocity, field_arrow_parameter=np.asarray(.35),
        path_states=data["flow_trace_states"], path_phases=data["flow_trace_phases"],
        path_positions=torus_position(data["flow_trace_phases"]),
        path_parameter=data["flow_trace_parameter"], path_axis=data["flow_trace_axis"],
        query_raw_indices=query_ids, query_relative_indices=np.arange(query_count),
        reference_sensor=data["clean"][query_ids, 0],
        reconstructed_sensor=data["reconstruction"][:query_count, 0],
        sensor_channel=np.asarray(0),
    )
    target = DATA / "toy_walkthrough_plot_data.npz"
    np.savez_compressed(target, **arrays)
    provenance = dict(
        case="clean fundamental2waves", role="successful toy walkthrough",
        source_arrays=dict(path=str(source.relative_to(HERE)), sha256=file_hash(source)),
        source_report=dict(path=str(report_path.relative_to(HERE)), sha256=file_hash(report_path)),
        plotting_script=dict(path=Path(__file__).name, sha256=file_hash(__file__)),
        plotted_arrays=dict(path=str(target.relative_to(HERE)), sha256=file_hash(target)),
        estimated_dimension=report["hat_r"], lag=report["lag"],
        production_source_hashes=report["fit"]["provenance"],
        computations="Array selection, fixed display map and its derivative only; no fitting, persistence, field evaluation or flow integration.",
        panels=dict(
            a="Three retained SVD window coordinates; 600 uniformly indexed TRAIN windows, with the first 8 chronological windows highlighted.",
            b=dict(label="Phase torus", meaning="Recovered phase torus", display="F(u,v)=((2+0.65 cos(v)) cos(u),(2+0.65 cos(v)) sin(u),0.65 sin(v))",
                   note="A canonical display of estimated TRAIN phases, not an assertion that the projected window cloud is a doughnut."),
            c=dict(source="Selected actual 2-simplices of the fitted finite complex", triangle_count=len(selected_triangles),
                   chart="arg exp(i(Phi(landmark)-Phi(landmark0)))", selection="Small connected subset incident to vertex 0; ordered by recovered-chart radius.",
                   note="This is a subset of a clique complex, not an inferred surface triangulation."),
            d=dict(source="Actual fitted phase_field E and actual encoder derivative D Phi at 6 saved TRAIN probes",
                   transformation="DF(Phi) D Phi E", common_arrow_parameter=.35,
                   per_arrow_normalization=False),
            e=dict(source="Saved actual RK4 flow of the fitted normalized field", axis=int(data["flow_trace_axis"]),
                   parameter_first=float(data["flow_trace_parameter"][0]), parameter_last=float(data["flow_trace_parameter"][-1]),
                   state_count=len(data["flow_trace_states"]), display="F(Phi(saved RK4 state))",
                   note="No ideal angular path or generator derivative is substituted."),
            f=dict(source="Actual frozen held-out reconstruction after overlap averaging", sensor=1,
                   shown_samples=query_count, first_raw_index=int(query_ids[0]), last_raw_index=int(query_ids[-1]),
                   reference="Clean generator readings; used only for display after predictions were frozen.")),
        output_files=["figures/toy_walkthrough.pdf"],
    )
    (DATA / "toy_walkthrough_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return arrays


def plain_axes(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#A0A0A0")
    ax.tick_params(color="#A0A0A0", width=.6, length=3)
    ax.grid(color="#E8E8E8", linewidth=.5)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_locator(MaxNLocator(3))
    ax.yaxis.set_major_locator(MaxNLocator(3))


def display_axes(ax, torus=False):
    ax.set_axis_off()
    ax.view_init(elev=24, azim=35)
    ax.set_proj_type("ortho")
    if torus:
        ax.set_xlim(-2.9, 2.9)
        ax.set_ylim(-2.9, 2.9)
        ax.set_zlim(-1.2, 1.2)
        ax.set_box_aspect((1, 1, .5), zoom=1.13)


def draw_torus(ax):
    u, v = np.meshgrid(np.linspace(0, 2 * np.pi, 25), np.linspace(0, 2 * np.pi, 13))
    xyz = torus_position(np.stack((u, v), axis=-1))
    ax.plot_wireframe(xyz[..., 0], xyz[..., 1], xyz[..., 2],
                      color="#CDCDCD", alpha=.55, linewidth=.35, rstride=1, cstride=1)
    display_axes(ax, torus=True)


def save(figure, stem):
    FIGURES.mkdir(exist_ok=True)
    figure.savefig(FIGURES / f"{stem}.pdf", facecolor="white",
                   metadata={"Title": stem, "Creator": "plot_walkthrough.py"})
    plt.close(figure)


def plot_geometry(figure, axes, data):
    windows, torus, triangles = axes
    xyz = data["window_coordinates"]
    windows.scatter(*xyz.T, color="#A8A8A8", s=3, alpha=.5, linewidths=0, rasterized=False)
    track = data["consecutive_window_coordinates"]
    windows.plot(*track.T, color=PURPLE, lw=.9, marker="o", ms=2.4)
    display_axes(windows)
    windows.set_box_aspect((1, 1, .9), zoom=1.13)
    windows.text2D(.04, .01, "SVD projection", transform=windows.transAxes, fontsize=10)
    draw_torus(torus)
    torus.scatter(*data["torus_positions"].T, s=2.5, color=PURPLE, alpha=.6, linewidths=0)
    coordinates = data["finite_phase_chart"]
    selected = data["finite_triangle_vertex_indices"]
    for triangle in selected:
        triangles.add_patch(Polygon(coordinates[triangle], facecolor=OLIVE,
                                    edgecolor=OLIVE, alpha=.24, linewidth=.9))
    used = np.unique(selected)
    triangles.scatter(*coordinates[used].T, color=PURPLE, s=14, zorder=3)
    triangles.scatter(*coordinates[0], s=30, color=PURPLE, zorder=4)
    triangles.autoscale_view()
    triangles.margins(.13)
    triangles.set_aspect("equal", adjustable="box")
    plain_axes(triangles)
    triangles.set_xlabel(r"$\Delta\varphi_1$ (rad)", labelpad=3)
    triangles.set_ylabel(r"$\Delta\varphi_2$ (rad)", labelpad=3)


def plot_dynamics(figure, axes, data):
    fields, path, signal = axes
    draw_torus(fields)
    positions = data["field_positions"]
    scale = float(data["field_arrow_parameter"])
    for j, color in enumerate((PURPLE, OLIVE)):
        velocity = data["field_display_velocity"][:, :, j] * scale
        fields.quiver(*positions.T, *velocity.T, length=1, normalize=False,
                       color=color, linewidth=1.0, arrow_length_ratio=.28)
    fields.scatter(*positions.T, color="#A0A0A0", s=5, linewidths=0)
    fields.text2D(.1, .01, r"$E_1$", color=PURPLE, fontsize=11, transform=fields.transAxes)
    fields.text2D(.34, .01, r"$E_2$", color=OLIVE, fontsize=11, transform=fields.transAxes)
    draw_torus(path)
    xyz = data["path_positions"]
    path.plot(*xyz.T, color=PURPLE, lw=2.1)
    path.scatter(*xyz[0], color=GRAY, s=20, depthshade=False)
    path.scatter(*xyz[-1], color=OLIVE, s=28, marker="^", depthshade=False)
    handles = [Line2D([], [], linestyle="none", marker="o", color=GRAY, ms=4, label="Start"),
               Line2D([], [], linestyle="none", marker="^", color=OLIVE, ms=4, label="End")]
    path.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, -.08),
                ncol=2, frameon=False, fontsize=10, handletextpad=.25, columnspacing=.8)
    signal.plot(data["query_relative_indices"], data["reference_sensor"], color=GRAY,
                 lw=1.35, label="Reference")
    signal.plot(data["query_relative_indices"], data["reconstructed_sensor"], color=PURPLE,
                 lw=1.35, linestyle=(0, (4, 2)), label="Recovered")
    plain_axes(signal)
    signal.set_xlabel("Query index", labelpad=3)
    signal.set_ylabel("Sensor 1", labelpad=3)
    signal.legend(frameon=False, fontsize=8.5, loc="upper left", handlelength=1.5,
                   handletextpad=.4, borderaxespad=.1)
    signal.set_ylim(-2.25, 2.6)


def main():
    plt.rcParams.update({"font.family": "DejaVu Serif", "mathtext.fontset": "dejavuserif",
        "font.size": 9.5, "axes.titlesize": 10.5, "axes.labelsize": 9.5,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.linewidth": .6,
        "pdf.fonttype": 42, "ps.fonttype": 42})
    data = export_plot_data()
    figure = plt.figure(figsize=(6.8, 4.8))
    grid = figure.add_gridspec(2, 3, left=.01, right=.98, bottom=.105, top=.89,
                              hspace=.54, wspace=.35, width_ratios=(1, 1, 1.12))
    axes = [[figure.add_subplot(grid[row, col], projection="3d" if col < 2 else None)
             for col in range(3)] for row in range(2)]
    plot_geometry(figure, axes[0], data)
    plot_dynamics(figure, axes[1], data)
    for row, titles in enumerate((("(a) Windows", "(b) Phase torus", "(c) Triangles"),
                                  ("(d) Fields", "(e) Flow path", "(f) Reconstruction"))):
        title_y = grid[row, 0].get_position(figure).y1 + .035
        for ax, title in zip(axes[row], titles):
            box = ax.get_position()
            figure.text((box.x0 + box.x1) / 2, title_y, title,
                        ha="center", va="bottom", fontsize=10.5)
    save(figure, "toy_walkthrough")
    print("Saved one six-panel vector figure from the successful frozen torus toy.")


if __name__ == "__main__":
    main()
