"""Phase charts from actual frozen window encodings; no true phases in plots."""

from project_paths import RESULTS_DIR

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
PANELS = (
    ("r1_travelling_infinite", None, "Одна фаза, бесконечные обертоны"),
    ("r2_travelling_weak08", (0, 1), "Две фазы, слабая компонента"),
    ("r2_primitive_no_fundamentals", (0, 1), "Две фазы без основных гармоник"),
    ("r3_random_smooth_no_fundamentals", (0, 1), "Три фазы: проекция (1, 2)"),
    ("r3_random_smooth_no_fundamentals", (0, 2), "Три фазы: проекция (1, 3)"),
    ("r3_random_smooth_no_fundamentals", (1, 2), "Три фазы: проекция (2, 3)"),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=RESULTS_DIR / 'robust_strong_raw_hankel_audit.json')
    parser.add_argument("--stem", default="hankel_phase_gallery")
    args = parser.parse_args()
    result = json.loads(args.result.read_text())
    if result["status"] != "finished" or not result["source_fingerprints_unchanged"]:
        raise ValueError("terminal source-consistent result required")
    rows = {row["case"]: row for row in result["rows"] if row["seed"] == 202610081}
    figure = plt.figure(figsize=(13, 8), facecolor="white")
    provenance = []
    for number, (case, pair, title) in enumerate(PANELS, 1):
        row = rows[case]
        path = Path(row["frozen_output"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["frozen_output_sha256"]:
            raise ValueError("frozen phase output changed")
        with np.load(path, allow_pickle=False) as archive:
            phase = archive["hankel_phase"]
        rank, count = phase.shape
        colors = np.arange(count)
        if pair is None:
            if rank != 1:
                raise ValueError("circle panel requires one inferred phase")
            ax = figure.add_subplot(2, 3, number)
            ax.scatter(np.cos(phase[0]), np.sin(phase[0]), s=5, c=colors,
                       cmap="viridis", linewidths=0, alpha=.85)
            ax.set_aspect("equal")
            ax.set_xlim(-1.15, 1.15)
            ax.set_ylim(-1.15, 1.15)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        else:
            ax = figure.add_subplot(2, 3, number, projection="3d")
            def torus(angles):
                first, second = angles[list(pair)]
                radius = 2+.7*np.cos(second)
                return np.asarray((radius*np.cos(first), radius*np.sin(first), .7*np.sin(second)))
            ax.scatter(*torus(phase), s=4, c=colors, cmap="viridis", linewidths=0,
                       alpha=.85, rasterized=True)
            steps = min(100, count-1)
            initial = phase[:, :steps]
            increments = np.angle(np.exp(1j*(phase[:, 1:steps+1]-initial)))
            fine = (initial[:, :, None]+increments[:, :, None]*np.linspace(0, 1, 12)[None, None, :]).reshape(rank, -1)
            ax.plot(*torus(fine), color="#244e70", linewidth=.35, alpha=.6)
            ax.set_box_aspect((1, 1, .46))
            ax.view_init(24, 35)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
        error = row["sensor_nrmse"]["hankel"]
        noun = "угол" if rank == 1 else "угла" if rank in (2, 3, 4) else "углов"
        ax.set_title(title+"\n"+str(rank)+" "+noun+"; ошибка поля "+f"{100*error:.2f}%", fontsize=10, pad=8)
        provenance.append({"case": case, "seed": row["seed"], "inferred_phase_count": rank,
            "pairwise_projection": pair, "frozen_output": str(path),
            "sha256": row["frozen_output_sha256"], "phase_source": "actual nonlinear encoding of held-out raw windows",
            "true_phases_frequencies_and_rank_used_for_plot_coordinates": False,
            "sensor_nrmse": error})
    figure.suptitle("Фазы, восстановленные из новых окон поля", fontsize=14, y=.98)
    figure.text(.5, .028,
        "Цвет — порядок наблюдаемых окон. Положение каждой точки вычислено по данным всех датчиков.\n"
        "Форма тора — координатная карта; для трёх фаз показаны только двумерные проекции.",
        ha="center", fontsize=9)
    figure.subplots_adjust(left=.02, right=.98, bottom=.09, top=.90, wspace=.12, hspace=.3)
    png, svg = RESULTS_DIR/(args.stem+".png"), RESULTS_DIR/(args.stem+".svg")
    figure.savefig(png, dpi=180, facecolor="white")
    figure.savefig(svg, facecolor="white")
    plt.close(figure)
    (RESULTS_DIR/(args.stem+".json")).write_text(json.dumps({"source_result": str(args.result),
        "source_sha256": hashlib.sha256(args.result.read_bytes()).hexdigest(), "rows": provenance,
        "png": str(png), "svg": str(svg)}, ensure_ascii=False, indent=2)+"\n")
    print(png)


if __name__ == "__main__":
    main()
