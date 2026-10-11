"""Scientific coordinate figure from three frozen observed-phase outputs.

No public measurement or evaluation/truth archive is read. Cases are fixed in
advance; the selected model and relation come from frozen parameter archives.
The torus is a fixed coordinate chart, not a learned physical embedding.
"""

from project_paths import RESULTS_DIR

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np


HERE = Path(__file__).resolve().parent
RESULTS = RESULTS_DIR
SOURCE = RESULTS / "compact_group_hankel.json"
PNG = RESULTS / "compact_group_phases.png"
PDF = RESULTS / "compact_group_phases.pdf"
PROVENANCE = RESULTS / "compact_group_phases.json"
CASES = (
    ("wave_1d_traveling", "Модель: 50 окружностей", "Бегущая волна 1D"),
    ("mixed_rotation_seed202610091", "Модель: 7 окружностей", "Контроль смешанных частот"),
    ("r2_travelling_weak08_seed202610081", "Модель: тор", "Две фазы, слабая компонента"),
)
MAJOR_RADIUS, MINOR_RADIUS = 2., .78


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def extended_gcd(first, second):
    old_r, r = abs(first), abs(second)
    old_s, s, old_t, t = 1, 0, 0, 1
    while r:
        quotient = old_r//r
        old_r, r = r, old_r-quotient*r
        old_s, s = s, old_s-quotient*s
        old_t, t = t, old_t-quotient*t
    return old_r, old_s*(1 if first>=0 else -1), old_t*(1 if second>=0 else -1)


def primitive_chart(relation):
    """U=[w; -t,s], with s*w1+t*w2=1 and det(U)=1 exactly."""
    first, second = map(int, relation)
    order, _, _ = extended_gcd(first, second)
    if order<1:
        raise ValueError("one nonzero integer relation is required")
    w1, w2 = first//order, second//order
    divisor, s, t = extended_gcd(w1, w2)
    if divisor!=1:
        raise ArithmeticError("the divided relation must be primitive")
    matrix = np.asarray([[w1, w2], [-t, s]], dtype=int)
    if int(matrix[0, 0]*matrix[1, 1]-matrix[0, 1]*matrix[1, 0])!=1:
        raise ArithmeticError("chart matrix must be unimodular")
    return order, matrix


def chart_points(first, second):
    radius = MAJOR_RADIUS+MINOR_RADIUS*np.cos(second)
    return radius*np.cos(first), radius*np.sin(first), MINOR_RADIUS*np.sin(second)


def load_frozen_case(row):
    phase_path = Path(row["frozen_output"])
    parameter_path = Path(row["frozen_parameters"])
    if sha(phase_path)!=row["frozen_output_sha256"] or sha(parameter_path)!=row["frozen_parameters_sha256"]:
        raise ValueError("frozen source hashes changed")
    with np.load(parameter_path, allow_pickle=False) as parameters:
        selected = str(parameters["selected_model"].item())
        relation = parameters["relation"].copy()
    key = selected+"__phase_at_window_end"
    with np.load(phase_path, allow_pickle=False) as phases:
        observed = phases[key].copy()
    if observed.shape[0]!=2 or not np.isfinite(observed).all():
        raise ValueError("two finite observed endpoint phases are required")
    if selected=="compact":
        if relation.shape!=(2,):
            raise ValueError("one rank-one relation in two phase coordinates is required")
        order, integer_chart = primitive_chart(relation)
    else:
        order, integer_chart = None, np.eye(2, dtype=int)
    coordinates = np.mod(np.einsum("ij,jn->in", integer_chart, observed, optimize=False), 2*np.pi)
    # The two ends of the plotting interval represent the same circle point.
    coordinates[np.isclose(coordinates, 2*np.pi, atol=1e-12, rtol=0)] = 0.
    metadata = {"run":row["run"], "selected_model":selected,
        "source_phase_archive":str(phase_path), "source_phase_archive_sha256":sha(phase_path),
        "source_parameter_archive":str(parameter_path), "source_parameter_archive_sha256":sha(parameter_path),
        "source_phase_key":key, "phase_points":observed.shape[1],
        "relation":relation.tolist(), "integer_chart_U":integer_chart.tolist(),
        "chart_formula":"z=U*theta (mod2*pi)", "compact_component_count":order,
        "model_or_orbit_closure_certified":False,
        "measurement_truth_or_absolute_time_read":False}
    if order is not None:
        component = np.rint(order*coordinates[0]/(2*np.pi)).astype(int) % order
        metadata["observed_component_count"] = len(np.unique(component))
        metadata["maximum_relation_residue_radians"] = float(np.max(abs(np.angle(np.exp(1j*
            np.einsum("i,in->n", relation, observed, optimize=False))))))
        metadata["meridian_origin"] = "zero gauge of the frozen TRAIN-refitted compact model"
    return coordinates, metadata


def main():
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10,
        "axes.labelsize":10, "axes.titlesize":14, "xtick.labelsize":9, "ytick.labelsize":9,
        "pdf.fonttype":42, "ps.fonttype":42, "savefig.facecolor":"white"})
    source_hash = sha(SOURCE)
    source = json.loads(SOURCE.read_text())
    rows = {row["run"]:row for row in source["rows"]}
    frozen = [load_frozen_case(rows[name]) for name, _, _ in CASES]
    expected = (50, 7, None)
    if tuple(metadata["compact_component_count"] for _, metadata in frozen)!=expected:
        raise ValueError("frozen selected models differ from the requested figure cases")
    figure = plt.figure(figsize=(11.8, 7.6), facecolor="white")
    grid = figure.add_gridspec(2, 3, left=.065, right=.96, bottom=.17, top=.855,
                              height_ratios=(1., 1.10), hspace=.29, wspace=.27)
    phi = np.linspace(0, 2*np.pi, 240)
    axes = []
    for column, ((_, title, subtitle), (coordinates, metadata)) in enumerate(zip(CASES, frozen)):
        order = metadata["compact_component_count"]
        color = np.linspace(0, 1, coordinates.shape[1])
        flat = figure.add_subplot(grid[0, column])
        axes.append(flat)
        flat.set_title(title, pad=23, fontweight="medium")
        flat.text(.5, 1.045, subtitle, transform=flat.transAxes, ha="center", va="bottom",
                  fontsize=9, color="#55606d")
        if order is not None:
            for component in range(order):
                flat.axvline(component/order, color="#8d9aab", linewidth=.55, alpha=.20, zorder=0)
        flat.scatter(coordinates[0]/(2*np.pi), coordinates[1]/(2*np.pi), c=color,
                     cmap="viridis", norm=Normalize(0, 1), s=5.5 if order==50 else 6.5,
                     alpha=.85, linewidths=0, rasterized=False, zorder=2)
        flat.set(xlim=(-.015, 1.015), ylim=(-.015, 1.015),
                 xticks=(0, .5, 1), yticks=(0, .5, 1),
                 xlabel="Фаза 1 / 2π", ylabel="Фаза 2 / 2π")
        flat.set_aspect("equal", adjustable="box")
        flat.grid(color="#d7dde5", linewidth=.5, alpha=.6)
        for spine in flat.spines.values():
            spine.set_color("#b8c1cc")
        torus = figure.add_subplot(grid[1, column], projection="3d", computed_zorder=False)
        if order is None:
            first, second = np.meshgrid(np.linspace(0, 2*np.pi, 72), np.linspace(0, 2*np.pi, 36))
            x, y, z = chart_points(first, second)
            torus.plot_surface(x, y, z, color="#c5ced9", alpha=.15, linewidth=0,
                               edgecolor="none", shade=False, zorder=0)
        else:
            for component in range(order):
                x, y, z = chart_points(np.full_like(phi, 2*np.pi*component/order), phi)
                torus.plot(x, y, z, color="#66758a", alpha=.21 if order==50 else .40,
                           linewidth=.55 if order==50 else .8, zorder=1)
        x, y, z = chart_points(coordinates[0], coordinates[1])
        torus.scatter(x, y, z, c=color, cmap="viridis", norm=Normalize(0, 1),
                      s=8 if order==50 else 7, alpha=.9, linewidths=0,
                      depthshade=False, zorder=3, rasterized=False)
        torus.view_init(elev=27, azim=-48)
        torus.set_proj_type("ortho")
        torus.set_box_aspect((1, 1, .52), zoom=1.12)
        extent = MAJOR_RADIUS+MINOR_RADIUS+.08
        torus.set(xlim=(-extent, extent), ylim=(-extent, extent),
                  zlim=(-MINOR_RADIUS-.12, MINOR_RADIUS+.12))
        torus.set_axis_off()
        torus.text2D(.5, -.035, "Координатная диаграмма тора", transform=torus.transAxes,
                     ha="center", va="top", fontsize=9, color="#55606d")
    figure.suptitle("Наблюдаемые фазовые состояния", x=.5, y=.985, fontsize=16, fontweight="medium")
    figure.text(.065, .100,
        "Точки — фазы выбранной модели в конце наблюдаемых окон; цвет показывает порядок отсчётов.",
        fontsize=9.3, color="#3f4a58")
    figure.text(.065, .047,
        "Тор — координатная диаграмма, не физическая геометрия поля. Показана выбранная модель;\n"
        "её ограничения не доказывают рациональность частот и точное замыкание траектории.",
        fontsize=9.3, linespacing=1.45, color="#3f4a58")
    figure.savefig(PNG, dpi=220)
    figure.savefig(PDF)
    plt.close(figure)
    provenance = {"source_script":str(Path(__file__).resolve()), "source_script_sha256":sha(__file__),
        "source_model_selection_audit":str(SOURCE), "source_model_selection_audit_sha256":source_hash,
        "source_model_selection_audit_unchanged":sha(SOURCE)==source_hash,
        "inputs": [metadata for _, metadata in frozen],
        "chart": {"formula":"((R+r*cos(z2))*cos(z1), (R+r*cos(z2))*sin(z1), r*sin(z2))",
            "major_radius":MAJOR_RADIUS, "minor_radius":MINOR_RADIUS,
            "compact_surface_drawn":False, "compact_geometry":"q meridian circles only",
            "continuous_surface":"light fixed chart; no density or exact closure certificate",
            "lines_joining_observed_states_drawn":False,
            "view_elevation":27, "view_azimuth":-48,
            "physical_geometry_or_spatial_interpolation_claimed":False},
        "evaluation_archives_read":False, "new_inference_or_experiment_run":False,
        "outputs": {"png":{"path":str(PNG), "sha256":sha(PNG)},
                    "pdf":{"path":str(PDF), "sha256":sha(PDF)}}}
    provenance["all_frozen_source_hashes_unchanged"] = all(
        sha(row["source_phase_archive"])==row["source_phase_archive_sha256"]
        and sha(row["source_parameter_archive"])==row["source_parameter_archive_sha256"]
        for row in provenance["inputs"])
    PROVENANCE.write_text(json.dumps(provenance, ensure_ascii=False, indent=2, allow_nan=False))
    print(json.dumps({"png":str(PNG), "pdf":str(PDF), "provenance":str(PROVENANCE),
        "components_and_points":[{"run":row["run"], "components":row["compact_component_count"],
            "points":row["phase_points"], "chart_U":row["integer_chart_U"]} for row in provenance["inputs"]],
        "all_sources_unchanged":provenance["all_frozen_source_hashes_unchanged"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
