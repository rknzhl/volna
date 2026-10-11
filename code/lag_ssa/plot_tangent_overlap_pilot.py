"""One scientific figure from frozen evaluated overlap-pilot outputs."""

from project_paths import RESULTS_DIR

import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import linalg


OUT = RESULTS_DIR / 'tangent_overlap_pilot'


def plot():
    name = "two_phase"
    inferred = np.load(OUT / f"{name}_inferred.npz")
    encoded = np.load(OUT / f"{name}_v2_phase_evaluated.npz")
    hidden = np.load(OUT / "evaluation" / f"{name}.npz")
    protocol = json.loads((OUT / "protocol.json").read_text())
    score = json.loads((OUT / f"{name}_v2_angles_evaluation.json").read_text())
    x = inferred["X"]
    _, _, vt = linalg.svd(x - x.mean(axis=0), full_matrices=False)
    cloud = np.einsum("np,ap->na", x - x.mean(axis=0), vt[:3])
    theta = encoded["train_angles"]
    torus = np.column_stack(((2 + np.cos(theta[:, 1])) * np.cos(theta[:, 0]),
                            (2 + np.cos(theta[:, 1])) * np.sin(theta[:, 0]),
                            np.sin(theta[:, 1])))
    color = np.mod(hidden["theta"][:len(x), 0], 2 * np.pi)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(13.0, 4.3), constrained_layout=True)
    ax = fig.add_subplot(131, projection="3d")
    ax.scatter(*cloud.T, c=color, cmap="twilight", s=3, alpha=.7)
    ax.set_title("Исходные окна: проекция PCA")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax = fig.add_subplot(132, projection="3d")
    ax.scatter(*torus.T, c=color, cmap="twilight", s=3, alpha=.7)
    ax.set_title("Восстановленные углы на торе")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_box_aspect([1, 1, .45])
    ax = fig.add_subplot(133)
    first = protocol["train_count"]
    points = np.arange(240)
    ax.plot(points, hidden["clean"][first:first + len(points), 0],
            color="#202a35", lw=1.6, label="Чистое поле")
    ax.plot(points, encoded["prediction"][:len(points), 0],
            color="#2463eb", lw=1.1, label="Реконструкция")
    ax.set_title("Новые наблюдения: датчик 1")
    ax.set_xlabel("Отсчёт проверочного блока")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=.15)
    mae = score["future_phase_mae_rad"]
    nr = score["future_clean_nrmse"]
    fig.suptitle(f"Две фазы · 12 датчиков · шум 2% RMS\n"
                 f"Ошибка фаз {mae:.3f} рад · ошибка реконструкции {100 * nr:.1f}%", fontsize=13)
    fig.savefig(OUT / "two_phase_example.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    plot()
