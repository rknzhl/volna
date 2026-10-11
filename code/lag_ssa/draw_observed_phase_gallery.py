"""Phase charts from saved independent SSA readouts and inferred integer bases.

Only fitted phase arrays and fitted basis products determine chart points.
The standard circle/torus is a coordinate chart, not a fitted physical metric.
No generator phases, frequencies, or spatial equation enter the plotting.
"""

from project_paths import RESULTS_DIR

from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = RESULTS_DIR
WAVES = OUT / "final_phase_factor_assay.json"
FIXTURES = OUT / "final_phase_factor_assay_fixtures.json"
CASES = (
    ("wave_1d_traveling", "Бегущая волна, 1D"),
    ("wave_2d_lens", "Линза, 2D"),
    ("wave_3d_lens", "Линза, 3D"),
    ("r2_poisson_dominant_weak", "Бесконечные обертоны + слабая фаза"),
    ("r2_partial_cover", "Уточнение фазового накрытия"),
    ("r1_poisson2_cos3", "Две гармоники одной фазы"),
)


def main():
    inputs = [WAVES, FIXTURES]
    rows, sources = {}, {}
    for path in inputs:
        payload = json.loads(path.read_text())
        if payload["status"] != "finished":
            raise ValueError("plot only terminal complete assays")
        for row in payload["rows"]:
            rows[row["case"]] = (row, path)
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    figure = plt.figure(figsize=(13.0, 8.0), facecolor="white")
    records = []
    for panel, (case, title) in enumerate(CASES, 1):
        row, path = rows[case]
        lattice = row["inference"]["final_lattice"]
        if lattice is None:
            raise ValueError(f"{case} has no supported inferred phase basis")
        coordinates = Path(row.get("coordinates_npz", OUT /
            (path.stem+"_"+case+"_coordinates.npz")))
        with np.load(coordinates, allow_pickle=False) as archive:
            acquired_train = archive["train_phase"]
            acquired_future = archive["future_phase"]
        products = np.asarray(lattice["products"], dtype=int)
        train = np.angle(np.exp(1j*np.einsum("ij,jt->it",products,acquired_train,optimize=False)))
        future = np.angle(np.exp(1j*np.einsum("ij,jt->it",products,acquired_future,optimize=False)))
        # Fix a displayed reference point using the first training readout.
        origin = train[:, :1].copy()
        train = np.angle(np.exp(1j*(train-origin)))
        future = np.angle(np.exp(1j*(future-origin)))
        phases = np.concatenate((train,future),axis=1)
        stride = max(1,phases.shape[1]//1800)
        phases = phases[:,::stride]
        rank = products.shape[0]
        if rank == 2:
            ax = figure.add_subplot(2,3,panel,projection="3d")
            first, second = phases
            radius = 2.0+.7*np.cos(second)
            xyz = np.column_stack((radius*np.cos(first),radius*np.sin(first),.7*np.sin(second)))
            ax.scatter(*xyz.T,c=np.mod(second,2*np.pi),cmap="twilight",s=2.0,
                       alpha=.75,linewidths=0,rasterized=True)
            ax.set_box_aspect((1,1,.46))
            ax.view_init(24,35)
            ax.set_xticks([]);ax.set_yticks([]);ax.set_zticks([])
        elif rank == 1:
            ax = figure.add_subplot(2,3,panel)
            ax.scatter(np.cos(phases[0]),np.sin(phases[0]),
                c=np.mod(phases[0],2*np.pi),cmap="twilight",s=5,linewidths=0,
                alpha=.7,rasterized=True)
            ax.set_aspect("equal");ax.set_xlim(-1.2,1.2);ax.set_ylim(-1.2,1.2)
            ax.set_xticks([]);ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        else:
            raise ValueError("this gallery displays complete one/two-phase charts only")
        field = row["future_clean_sensor_field"]
        metric_key = "iterative_hierarchy" if "iterative_hierarchy" in field else "revised_phase_family"
        nrmse = field[metric_key]["sensor_nrmse"]
        ax.set_title(f"{title}\nУгловых координат: {rank}; ошибка поля {100*nrmse:.2f}%",
                     fontsize=10,pad=8)
        records.append({"case":case,"assay":str(path),
            "coordinate_file":str(coordinates),
            "coordinate_sha256":hashlib.sha256(coordinates.read_bytes()).hexdigest(),
            "rank_candidate":rank,"basis_products":products.tolist(),
            "display_origin_first_training_point":origin[:,0].tolist(),
            "future_clean_sensor_nrmse":nrmse,
            "points_use_generator_truth":False,
            "physical_torus_geometry_inferred":False})
    figure.suptitle("Фазовые координаты одного поля, восстановленные по датчикам",
                   fontsize=14,y=.98)
    figure.text(.5,.026,
        "Точки — замороженные SSA-readout. Окружность и тор задают координатную форму.\n"
        "У исходных волн первая фаза дискретно циклическая; заполнение непрерывной поверхности не утверждается.",
        ha="center",fontsize=9)
    figure.subplots_adjust(left=.02,right=.98,bottom=.09,top=.9,wspace=.12,hspace=.30)
    png = OUT / "observed_phase_gallery.png"
    svg = OUT / "observed_phase_gallery.svg"
    figure.savefig(png,dpi=180,facecolor="white")
    figure.savefig(svg,facecolor="white")
    plt.close(figure)
    (OUT / "observed_phase_gallery.json").write_text(json.dumps(
        {"sources_sha256":sources,"rows":records,"png":str(png),"svg":str(svg)},
        ensure_ascii=False,indent=2)+"\n")
    print(png)


if __name__ == "__main__":
    main()
