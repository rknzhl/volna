"""Three original field geometries: actual recovered window-end phases.

The torus drawing is a fixed visualization of the saved phase coordinates,
not evidence of identification of unobserved states. Frequencies are not used
to draw trajectories; every point comes from a frozen raw-window estimate.
"""

from project_paths import RESULTS_DIR

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np


HERE = Path(__file__).resolve().parent
SOURCE = RESULTS_DIR / 'original_all_waves_robust_hankel.json'
CASES = (("wave_1d_traveling", "Поле 1D · 8 датчиков"),
         ("wave_2d_uniform", "Поле 2D · 16 датчиков"),
         ("wave_3d_uniform", "Поле 3D · 27 датчиков"))


def main():
    payload = json.loads(SOURCE.read_text())
    if payload["status"] != "finished" or not payload["source_fingerprints_unchanged"]:
        raise ValueError("a finished source-coherent audit is required")
    rows = {row["case"]: row for row in payload["rows"]}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    figure = plt.figure(figsize=(13, 7.8), constrained_layout=False)
    figure.suptitle("Фазы из исходных волновых полей", fontsize=17, y=.965)
    figure.text(.5, .913, "Оценки из одного сырого окна 128 отсчётов; точка соответствует концу окна", ha="center", fontsize=11)
    phase_surface = np.linspace(0, 2*np.pi, 45)
    first, second = np.meshgrid(phase_surface, phase_surface)
    major, minor = 2., .72
    surface = ((major+minor*np.cos(second))*np.cos(first),
               (major+minor*np.cos(second))*np.sin(first), minor*np.sin(second))
    grid = figure.add_gridspec(2, 3, left=.065, right=.965, top=.85, bottom=.22,
                              height_ratios=(1, 1.2), wspace=.24, hspace=.22)
    last_scatter = None
    for column, (case, title) in enumerate(CASES):
        row = rows[case]
        with np.load(row["frozen_output"]) as archive:
            phase = np.mod(archive["hankel_phase"], 2*np.pi)
            index = archive["time_end_indices"]
        if phase.shape[0] != 2:
            raise ValueError("exactly two inferred phases expected for this visualization")
        progress = np.arange(phase.shape[1])/max(1, phase.shape[1]-1)
        plane = figure.add_subplot(grid[0, column])
        last_scatter = plane.scatter(phase[0], phase[1], c=progress, cmap="viridis", norm=Normalize(0, 1), s=12, lw=0)
        plane.set_title(title+"\n"+f"Ошибка поля {100*row['heldout_field']['hankel']['sensor_nrmse']:.3f}%", fontsize=11)
        plane.set(xlim=(0, 2*np.pi), ylim=(0, 2*np.pi), xlabel=r"$\widehat\theta_1$", ylabel=r"$\widehat\theta_2$")
        plane.set_xticks((0, np.pi, 2*np.pi), labels=("0", r"$\pi$", r"$2\pi$"))
        plane.set_yticks((0, np.pi, 2*np.pi), labels=("0", r"$\pi$", r"$2\pi$"))
        plane.grid(alpha=.18)
        torus = figure.add_subplot(grid[1, column], projection="3d")
        torus.plot_wireframe(*surface, rstride=3, cstride=3, color="#9aa8b3", alpha=.24, linewidth=.45)
        x = (major+minor*np.cos(phase[1]))*np.cos(phase[0])
        y = (major+minor*np.cos(phase[1]))*np.sin(phase[0])
        z = minor*np.sin(phase[1])
        torus.scatter(x, y, z, c=progress, cmap="viridis", norm=Normalize(0, 1), s=7, depthshade=False)
        torus.set_box_aspect((1, 1, .48))
        torus.view_init(elev=26, azim=43)
        torus.set_axis_off()
        torus.set_title(f"{len(index)} восстановленных состояний", fontsize=10, y=.02)
    color_axis = figure.add_axes((.36, .178, .30, .012))
    colorbar = figure.colorbar(last_scatter, cax=color_axis, orientation="horizontal")
    colorbar.set_ticks((0, 1), labels=("начало", "конец"))
    figure.text(.51, .204, "Время внутри проверочного участка", ha="center", fontsize=9)
    figure.text(.065, .117,
        r"Дискретное вращение первой исходной частоты: $3/50$ оборота за отсчёт. Замыкание орбиты: $\mathbb{Z}_{50}\times\mathbb{S}^{1}$.",
        fontsize=10)
    figure.text(.065, .077,
        "Поверхность — стандартное отображение восстановленных фаз; полный непрерывный тор этими данными не идентифицирован.",
        fontsize=10)
    figure.text(.065, .039,
        "Контроль со стохастическим источником сохранён отдельно: он выходит за класс детерминированных полей на торе.",
        fontsize=10, color="#55616a")
    destination = RESULTS_DIR / 'original_all_waves_hankel_phases'
    figure.savefig(destination.with_suffix(".png"), dpi=180, facecolor="white")
    figure.savefig(destination.with_suffix(".pdf"), facecolor="white")
    plt.close(figure)
    print(str(destination.with_suffix(".png")))


if __name__ == "__main__":
    main()
