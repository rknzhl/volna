"""Law-blind spatial profile decoding with outer sensor and future holdouts.

Only retained Y,s,t enter temporal inference and spatial CV. All predictions
are frozen on disk before any evaluation archive is opened. Phase lifts and
smoothness are tested hypotheses; no source, PDE or true frequency is used.
"""

from __future__ import annotations

from project_paths import RESULTS_DIR

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull

from frequency_fit import sinusoid_design
from run_actual_wave_mssa_baseline import fit_joint_baseline, predict_joint_decoder
from spatial_phase_decoder import coefficient_rows, complex_profiles, fit_spatial_phase_decoder


HERE = Path(__file__).resolve().parent
DATA = RESULTS_DIR / 'wave_actual_data'
DESTINATION = RESULTS_DIR / 'spatial_phase_holdout.json'
CASES = (("wave_1d_traveling", 3), ("wave_2d_lens", 5), ("wave_3d_uniform", 13))


def _serializable(value):
    if isinstance(value, np.ndarray):
        return _serializable(value.tolist())
    if isinstance(value, np.generic):
        return _serializable(value.item())
    if isinstance(value, dict):
        return {str(key): _serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serializable(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _save(payload: dict, destination: Path):
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(_serializable(payload), indent=2, allow_nan=False)+"\n")
    temporary.replace(destination)


def _fingerprints():
    return {name: hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in
            ("run_spatial_phase_holdout.py", "spatial_phase_decoder.py", "nodal_spatial.py",
             "run_actual_wave_mssa_baseline.py", "frequency_fit.py")}


def _nrmse(prediction, reference):
    return float(np.linalg.norm(prediction-reference)/max(np.linalg.norm(reference), 1e-30))


def _fit_and_freeze(slug: str, held: int, data: Path, destination: Path):
    source = data/"inputs"/(slug+".npz")
    with np.load(source, allow_pickle=False) as archive:
        if set(archive.files) != {"Y", "s", "t"}:
            raise ValueError("input archive must contain exactly Y,s,t")
        positions, times = archive["s"], archive["t"]
        keep = np.delete(np.arange(len(positions)), held)
        retained = archive["Y"][keep]
    fit = fit_joint_baseline(retained, times)
    decoder = fit_spatial_phase_decoder(positions[keep], fit["decoder"]["coefficients"],
                                        fit["frequencies_hz"])
    query = positions[[held]]
    selected_rows = coefficient_rows(decoder["prepared"], query, decoder["selected_spec"])
    p1_rows = coefficient_rows(decoder["prepared"], query, {"method": "complex_p1"})
    future_times = times[fit["future_start"]:]
    selected_prediction = predict_joint_decoder(fit, future_times, coefficients=selected_rows)
    p1_prediction = predict_joint_decoder(fit, future_times, coefficients=p1_rows)
    frozen_path = destination.with_name(destination.stem+"_"+slug+"_held_"+str(held)+"_frozen.npz")
    np.savez_compressed(frozen_path, held_index=held, retained_indices=keep,
        query_positions=query, frequencies_hz=fit["frequencies_hz"],
        frequencies_rad_per_sample=fit["frequencies_rad_per_sample"], future_times=future_times,
        selected_coefficients=selected_rows, p1_coefficients=p1_rows,
        selected_prediction=selected_prediction, p1_prediction=p1_prediction)
    return {"slug": slug, "held": held, "keep": keep, "fit": fit,
        "selected_rows": selected_rows, "p1_rows": p1_rows,
        "selected_prediction": selected_prediction, "p1_prediction": p1_prediction,
        "public": {"slug": slug, "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "held_sensor_index": held, "held_sensor_position": query[0].tolist(),
            "retained_sensor_indices": keep.tolist(), "spatial_dimension": positions.shape[1],
            "learned_frequencies_hz": fit["frequencies_hz"].tolist(),
            "selected_ssa_rank": fit["selected_ssa_rank"], "future_start": fit["future_start"],
            "frozen_prediction_file": str(frozen_path), "spatial_selection": decoder["audit"],
            "held_signal_used_for_fit_or_selection": False,
            "P1_and_selected_share_identical_temporal_model": True}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, default=DESTINATION)
    parser.add_argument("--all-interior", action="store_true",
                        help="all strictly interior nodes of the same three predeclared input fields")
    args = parser.parse_args()
    args.output.parent.mkdir(exist_ok=True)
    fingerprints = _fingerprints()
    cases = list(CASES)
    if args.all_interior:
        cases = []
        for slug, _ in CASES:
            with np.load(args.data/"inputs"/(slug+".npz"), allow_pickle=False) as archive:
                positions = archive["s"]
            if positions.shape[1] == 1:
                interior = ((positions[:, 0] > positions[:, 0].min())
                            & (positions[:, 0] < positions[:, 0].max()))
            else:
                hull = ConvexHull(positions)
                offsets = np.einsum("ij,kj->ik", positions, hull.equations[:, :-1], optimize=False)
                offsets += hull.equations[None, :, -1]
                interior = np.all(offsets < -1e-10, axis=1)
            cases.extend((slug, int(held)) for held in np.flatnonzero(interior))
    payload = {"created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "fitting", "source_sha256": fingerprints,
        "inference_inputs": ["Y of retained sensors", "s of retained sensors", "t"],
        "evaluation_archives_opened_after_all_predictions_frozen": False,
        "known_PDE_source_spatial_law_true_frequencies_supplied": False,
        "spatial_model_selection": "predeclared candidate grid, retained-sensor profile CV only",
        "outer_holdout_scope": "all strictly interior nodes of the three fixed fields" if args.all_interior else
                               "three predeclared interior nodes",
        "outer_holdout_count": len(cases),
        "sources": ["https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RBFInterpolator.html",
                    "https://arxiv.org/abs/1211.0374"],
        "novelty_claim": False, "torus_recovery_claim": False, "rows": []}
    _save(payload, args.output)
    frozen = []
    for slug, held in cases:
        result = _fit_and_freeze(slug, held, args.data, args.output)
        frozen.append(result)
        payload["rows"].append(result["public"])
        _save(payload, args.output)
        print(json.dumps({"case": slug, "predictions": "frozen before evaluation",
            "selected": result["public"]["spatial_selection"]["selected_spec"],
            "cv_profile_mse": result["public"]["spatial_selection"]["selected_cv_relative_profile_mse"]}), flush=True)
    payload["status"] = "predictions_frozen"
    _save(payload, args.output)
    for result in frozen:
        public, fit = result["public"], result["fit"]
        with np.load(args.data/"evaluation"/(result["slug"]+".npz"), allow_pickle=False) as archive:
            clean = archive["clean_Y"]
        target = clean[[result["held"]], fit["future_start"]:]
        retained_target = clean[result["keep"], fit["future_start"]:]
        future_indices = np.arange(fit["future_start"], clean.shape[1])
        design = sinusoid_design(np.asarray(fit["frequencies_rad_per_sample"]),
                                 future_indices-fit["decoder"]["times_center_index"])
        reference_rows = np.linalg.lstsq(design, target.T, rcond=None)[0]
        reference_profile = complex_profiles(reference_rows, len(fit["frequencies_hz"]))[0]
        metrics = {}
        for name, rows, prediction in (("complex_P1", result["p1_rows"], result["p1_prediction"]),
                                        ("CV_selected", result["selected_rows"], result["selected_prediction"])):
            profile = complex_profiles(rows, len(fit["frequencies_hz"]))[0]
            metrics[name] = {"held_sensor_future_clean_nrmse": _nrmse(prediction, target),
                "phase_error_vs_clean_future_fixed_frequency_fit_rad":
                    np.angle(profile*np.conj(reference_profile)).tolist(),
                "amplitude_ratio_vs_clean_future_fixed_frequency_fit":
                    (abs(profile)/np.maximum(abs(reference_profile), 1e-30)).tolist()}
        public["evaluation"] = metrics | {
            "retained_sensor_future_clean_nrmse": _nrmse(fit["future_prediction"], retained_target),
            "reference_profile_fit_is_truth_only_after_freeze": True}
        print(json.dumps({"case": result["slug"], "P1_nrmse": metrics["complex_P1"]["held_sensor_future_clean_nrmse"],
            "selected_nrmse": metrics["CV_selected"]["held_sensor_future_clean_nrmse"]}), flush=True)
    if _fingerprints() != fingerprints:
        raise RuntimeError("source changed during frozen assay")
    payload["status"] = "complete"
    payload["evaluation_archives_opened_after_all_predictions_frozen"] = True
    payload["limits"] = ["only three predeclared input fields; interior holdouts from each field are correlated",
        "phase smoothness and winding lifts remain hypotheses selected by retained-node CV",
        "sparse spatial sampling permits alternative profiles with identical retained values",
        "improvement does not certify a unique field, propagation law, or complete torus"]
    _save(payload, args.output)
    print(str(args.output), flush=True)


if __name__ == "__main__":
    main()
