"""Reproducible canonical phase tori from the frozen SSA readout.

The ordinary runner fits on train/validation, reads a new future record with
frozen filters, and uses truth only for evaluation. This script captures its
already assessed phase arrays and saves a standalone gallery and NPZs.
The chart is the standard embedding of S1×S1, not a learned spatial metric.
"""

from project_paths import RESULTS_DIR
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import run_iterative_ssa as runner
from iterative_ssa import Settings
from method import torus_xyz


CASES = ("r2_poisson_dominant_weak", "r2_partial_cover",
         "r2_mixed_no_fundamentals", "r2_infinite_mixed_products")
LABELS = ("Strong infinite family + weak phase", "Cover refinement: 2θ₁, θ₂, 3θ₁",
          "Mixed modes; no fundamentals", "Infinite mixed products")
OUT = RESULTS_DIR


def main():
    settings = Settings(n_train=3200, n_validation=1600, n_future=1600,
        lag_window=128, frontend_mode="dynamic", joint_frequency_mode="harmonic_nuisance",
        phase_metric_mode="noise", temporal_gate_mode="iid_reference")
    runner.RUN_VARIANT, runner.SEED, runner.NOISE = "residual_v4_gallery", 54000, .20
    runner.PLOTS_ENABLED = True
    original_plot = runner._torus_plot
    captured = {}

    def capture(case, truth, iterative, clock):
        captured[case] = (truth.copy(), iterative.copy(), clock.copy())
        return original_plot(case, truth, iterative, clock)

    runner._torus_plot = capture
    rows = []
    for case in CASES:
        row = runner.evaluate(case, settings)
        if case in captured:
            truth, iterative, clock = captured[case]
            np.savez_compressed(OUT/f"phase_gallery_{case}.npz", truth=truth,
                                iterative=iterative, clock=clock)
            reference = torus_xyz(truth)
            row["canonical_torus_geometry_error"] = {
                name: {"rmse": float(np.sqrt(np.mean(np.sum((torus_xyz(phase)-reference)**2, axis=1)))),
                       "max_distance": float(np.max(np.linalg.norm(torus_xyz(phase)-reference, axis=1)))}
                for name, phase in (("frozen_ssa", iterative), ("estimated_clock", clock))}
        rows.append(row)
        print(case, row.get("phase_comparison_status"), flush=True)
    fig = plt.figure(figsize=(11.5, 13.5))
    for i, (case, label) in enumerate(zip(CASES, LABELS)):
        for j, method in enumerate(("Reference", "Frozen SSA", "Estimated clock")):
            ax = fig.add_subplot(4, 3, i*3+j+1, projection="3d")
            if case not in captured:
                ax.text2D(.1, .5, "Phase module unresolved", transform=ax.transAxes)
                ax.set_axis_off()
                continue
            phases = captured[case][j]
            xyz = torus_xyz(phases)
            ax.scatter(*xyz.T, c=np.arange(len(xyz)), cmap="viridis", s=2.5, alpha=.7)
            ax.set(xlim=(-2.8,2.8), ylim=(-2.8,2.8), zlim=(-.8,.8))
            ax.set_box_aspect((1, 1, .35))
            ax.view_init(27, 40)
            ax.set_xticks((-2,0,2)); ax.set_yticks((-2,0,2)); ax.set_zticks((-.7,0,.7))
            ax.tick_params(labelsize=7, pad=0)
            if i == 0:
                ax.set_title(method, fontsize=11, pad=3)
            if j == 0:
                ax.text2D(.02, .98, label, transform=ax.transAxes, fontsize=8)
            elif j > 0:
                scores = rows[i]["phase_comparison"]
                score = scores["iterative_frozen_ssa" if j == 1 else "direct_inferred_clock"]
                ax.text2D(.03, .97, f"Phase MAE = {score['future_train_origin_aligned_mae_rad']:.3f} rad",
                          transform=ax.transAxes, fontsize=8)
    fig.suptitle("Canonical phase tori of one spatial field\n"
                 "Noise 20%; future observations; standard chart R=2, a=0.7", fontsize=12, y=.985)
    fig.subplots_adjust(left=.03, right=.98, bottom=.02, top=.94, hspace=.08, wspace=.02)
    png, pdf = OUT/"iterative_ssa_phase_gallery.png", OUT/"iterative_ssa_phase_gallery.pdf"
    fig.savefig(png, dpi=180); fig.savefig(pdf); plt.close(fig)
    payload = {"settings": settings.__dict__, "seed": 54000, "noise": .20,
        "rows": rows, "gallery": str(png), "export_pdf": str(pdf),
        "chart": "standard S1×S1 embedding with R=2, a=.7",
        "field_delay_embedding_metric_learned": False, "truth_supplied_to_fitting": False,
        "origins_calibrated_from_training_only_for_assessment": True}
    (OUT/"iterative_ssa_phase_gallery.json").write_text(
        json.dumps(runner._serializable(payload), indent=2, allow_nan=False)+"\n")
    print(png, flush=True)


if __name__ == "__main__":
    main()
