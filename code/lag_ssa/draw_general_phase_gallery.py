"""Circle/torus charts from independently frozen phase readouts.

For a three-phase field only pairwise projections are shown. No generator
phase, true rank or true frequency determines plotted point coordinates.
"""

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
OUT = RESULTS_DIR
SEED = 202610081
PANELS = (
    ("r1_travelling_infinite",None,"Одна фаза, бесконечные обертоны"),
    ("r2_travelling_weak08",(0,1),"Две фазы, слабая компонента"),
    ("r2_primitive_no_fundamentals",(0,1),"Две фазы без основных гармоник"),
    ("r3_random_smooth_no_fundamentals",(0,1),"Три фазы: проекция (1, 2)"),
    ("r3_random_smooth_no_fundamentals",(0,2),"Три фазы: проекция (1, 3)"),
    ("r3_random_smooth_no_fundamentals",(1,2),"Три фазы: проекция (2, 3)"),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result",type=Path,default=OUT/"general_robustness_cover.json")
    parser.add_argument("--stem",default="general_phase_gallery")
    args = parser.parse_args()
    result = json.loads(args.result.read_text())
    if result["status"]!="finished" or not result["source_fingerprints_unchanged"]:
        raise ValueError("terminal source-consistent assay required")
    rows = {row["case"]:row for row in result["rows"] if row["seed"]==SEED}
    figure = plt.figure(figsize=(13,8),facecolor="white")
    provenance = []
    for number,(case,pair,title) in enumerate(PANELS,1):
        row = rows[case]
        path = Path(row["frozen_arrays"])
        with np.load(path,allow_pickle=False) as archive:
            acquired_train,acquired_future = archive["train_phase"],archive["future_phase"]
        products = np.asarray(row["inference"]["final_lattice"]["products"],dtype=int)
        train = np.angle(np.exp(1j*np.einsum("ij,jt->it",products,acquired_train,optimize=False)))
        future = np.angle(np.exp(1j*np.einsum("ij,jt->it",products,acquired_future,optimize=False)))
        origin = train[:,:1].copy()
        train = np.angle(np.exp(1j*(train-origin)))
        future = np.angle(np.exp(1j*(future-origin)))
        rank = train.shape[0]
        if pair is None:
            if rank!=1:
                raise ValueError("circle panel requires one inferred generator")
            ax = figure.add_subplot(2,3,number)
            ax.scatter(np.cos(train[0,::2]),np.sin(train[0,::2]),s=3,c="#aeb5bd",alpha=.45,linewidths=0)
            ax.scatter(np.cos(future[0]),np.sin(future[0]),s=4,c=np.arange(future.shape[1]),
                cmap="viridis",alpha=.8,linewidths=0)
            ax.set_aspect("equal")
            ax.set_xlim(-1.15,1.15);ax.set_ylim(-1.15,1.15)
            ax.set_xticks([]);ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        else:
            ax = figure.add_subplot(2,3,number,projection="3d")
            def torus(angles):
                first,second = angles[list(pair)]
                radius = 2+.7*np.cos(second)
                return np.array((radius*np.cos(first),radius*np.sin(first),.7*np.sin(second)))
            ax.scatter(*torus(train[:,::3]),s=2,c="#aeb5bd",alpha=.25,linewidths=0,rasterized=True)
            points = torus(future)
            ax.scatter(*points,s=3,c=np.arange(future.shape[1]),cmap="viridis",alpha=.8,
                linewidths=0,rasterized=True)
            # Interpolate angles, so the displayed connecting arc stays on
            # the chosen torus chart instead of cutting across its interior.
            count = min(100,future.shape[1]-1)
            first = future[:,:count]
            increments = np.angle(np.exp(1j*(future[:,1:count+1]-first)))
            fine = (first[:,:,None]+increments[:,:,None]*np.linspace(0,1,12)[None,None,:]).reshape(rank,-1)
            ax.plot(*torus(fine),color="#1c4b70",linewidth=.35,alpha=.5)
            ax.set_box_aspect((1,1,.46));ax.view_init(24,35)
            ax.set_xticks([]);ax.set_yticks([]);ax.set_zticks([])
        metric = row["future_field"]["future_sensor_nrmse"]
        ax.set_title(title+"\nУгловых координат: "+str(rank)+"; прогноз поля "+f"{100*metric:.2f}%",
            fontsize=10,pad=8)
        provenance.append(dict(case=case,seed=SEED,rank=rank,pairwise_projection=pair,
            inferred_basis_products=products.tolist(),origin_first_training_point=origin[:,0].tolist(),
            frozen_arrays=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            plotted_coordinates_use_generator_truth=False))
    figure.suptitle("Фазовые траектории после проверки конечных накрытий",fontsize=14,y=.98)
    figure.text(.5,.028,
        "Серый — обучение; цвет — независимые будущие наблюдения. Форма тора — выбранная координатная карта.\n"
        "Для трёх фаз показаны только двумерные проекции; полное представление имеет три угла.",
        ha="center",fontsize=9)
    figure.subplots_adjust(left=.02,right=.98,bottom=.09,top=.90,wspace=.12,hspace=.3)
    png,svg = OUT/(args.stem+".png"),OUT/(args.stem+".svg")
    figure.savefig(png,dpi=180,facecolor="white")
    figure.savefig(svg,facecolor="white")
    plt.close(figure)
    (OUT/(args.stem+".json")).write_text(json.dumps(dict(source_result=str(args.result),
        source_sha256=hashlib.sha256(args.result.read_bytes()).hexdigest(),rows=provenance,
        png=str(png),svg=str(svg)),ensure_ascii=False,indent=2)+"\n")
    print(png)


if __name__=="__main__":
    main()
