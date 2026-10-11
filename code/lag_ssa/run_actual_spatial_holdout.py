"""Spatial holdout of law-blind phase inference and geometry-only decoding.

One interior sensor is absent from fitting, frequency selection, and the
P1 mesh. Evaluation files are opened after all fits; the wave equation and
source parameters never enter the model. This tests sparse spatial decoding,
not a phase-completeness theorem.
"""

from project_paths import RESULTS_DIR

from pathlib import Path
import hashlib
import json

import numpy as np

from nodal_spatial import NodalSpatialModel
from run_actual_wave_mssa_baseline import fit_joint_baseline

HERE = Path(__file__).resolve().parent
DATA = RESULTS_DIR / 'wave_actual_data'
CASES = (("wave_1d_traveling", 3), ("wave_2d_lens", 5),
         ("wave_3d_uniform", 13))


def main():
    fitted = []
    for slug, held in CASES:
        path = DATA / "inputs" / (slug + ".npz")
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"Y", "s", "t"}:
                raise ValueError("inference inputs must be exactly Y,s,t")
            y, s, t = archive["Y"], archive["s"], archive["t"]
        keep = np.delete(np.arange(len(s)), held)
        model = NodalSpatialModel.from_positions(s[keep])
        fit = fit_joint_baseline(y[keep], t)
        prediction = model.evaluate(fit["future_prediction"], s[[held]])
        fitted.append((slug, held, fit, prediction, keep, s,
                       hashlib.sha256(path.read_bytes()).hexdigest()))
        print(slug, "fit finished; withheld sensor absent from inference", flush=True)
    rows = []
    for slug, held, fit, prediction, keep, s, digest in fitted:
        with np.load(DATA / "evaluation" / (slug + ".npz"), allow_pickle=False) as archive:
            clean = archive["clean_Y"]
        start = fit["future_start"]
        held_target = clean[[held], start:]
        trained_target = clean[keep, start:]
        rows.append({"slug": slug, "input_sha256": digest,
            "spatial_dimension": s.shape[1], "held_sensor_index": held,
            "held_sensor_position": s[held].tolist(),
            "training_sensor_indices": keep.tolist(),
            "learned_frequencies_hz": fit["frequencies_hz"].tolist(),
            "selected_ssa_rank": fit["selected_ssa_rank"],
            "future_start": start,
            "held_spatial_sensor_future_clean_nrmse": float(
                np.linalg.norm(prediction-held_target)/np.linalg.norm(held_target)),
            "training_spatial_sensors_future_clean_nrmse": float(
                np.linalg.norm(fit["future_prediction"]-trained_target)/np.linalg.norm(trained_target)),
            "decoder": "coordinate-only P1 interpolation on retained sensor convex hull",
            "held_sensor_used_for_fit_or_selection": False})
    payload = {"inference_inputs": ["Y of retained sensors", "s of retained sensors", "t"],
        "evaluation_archives_opened_after_all_fits": True,
        "meaning": "phase inference at measured points can succeed while spatial interpolation fails",
        "rows": rows}
    destination = RESULTS_DIR / 'actual_wave_spatial_holdout.json'
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(rows, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
