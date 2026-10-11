"""Analytic source controls of the geometric coordinate/lift implementation."""

from project_paths import RESULTS_DIR

import hashlib
import argparse
from itertools import product
import json
from pathlib import Path
import time

import numpy as np

from geometric_phase_frontend import fit_hankel_topology, transform_hankel_topology, SETTINGS, library_fingerprints
from geometric_phase_encoding import encode_hankel_blocks


HERE = Path(__file__).resolve().parent
OUTPUT = RESULTS_DIR / 'geometric_phase_frontend_controls.json'


def angle_error(values, target, train_gauge):
    difference = values - target
    shift = np.angle(np.mean(np.exp(1j * difference[:, train_gauge]), axis=1))
    return np.abs((difference - shift[:, None] + np.pi) % (2 * np.pi) - np.pi)


def main(output=OUTPUT):
    output = Path(output)
    if output.exists():
        raise FileExistsError("preserve completed controls")
    rows = []
    for dimension in (1, 2):
        alpha = np.array([2*np.pi*np.sqrt(2)/10, 2*np.pi*np.sqrt(3)/11])[:dimension]
        theta = alpha[:, None] * np.arange(768)[None, :]
        H = np.vstack((np.cos(theta), np.sin(theta)))
        blocks = {name:dict(H=H[:,a:b], window_start_index=np.arange(a,b),
                            window_end_index=np.arange(a,b))
                  for name,a,b in (("train",0,512),("validation",512,640),("future",640,768))}
        started = time.perf_counter()
        model = fit_hankel_topology(blocks["train"]["H"], SETTINGS)
        assert model["diagnostics"]["selected_rank"] == dimension
        assert model["certificate"]["full_primitive_free_cohomology_basis_certified"]
        encoded = encode_hankel_blocks(model, blocks, SETTINGS)
        assert encoded["diagnostics"]["status"] == "conditional_source_graph_phase_encoding"
        assert encoded["diagnostics"]["future_encoded_count"] == 128
        # Analytic loops are evaluation controls only, never fitting inputs.
        winding = np.zeros((dimension, dimension), dtype=int)
        for axis in range(dimension):
            loop = np.zeros((dimension, 513))
            loop[axis] = np.linspace(0, 2*np.pi, 513)
            mapped = transform_hankel_topology(model, np.vstack((np.cos(loop), np.sin(loop))))
            assert mapped["covered"].all()
            lifted = np.unwrap(mapped["phi"], axis=1)
            winding[:,axis] = np.rint((lifted[:,-1]-lifted[:,0])/(2*np.pi)).astype(int)
        determinant = int(round(np.linalg.det(winding)))
        assert abs(determinant) == 1
        raw = np.concatenate((encoded["arrays"]["initial_train_coordinates"],
            encoded["arrays"]["initial_validation_coordinates"],
            encoded["arrays"]["initial_future_coordinates"]), axis=1)
        averaged = np.concatenate((encoded["arrays"]["train_coordinates"],
            encoded["arrays"]["validation_coordinates"],
            encoded["arrays"]["future_coordinates"]), axis=1)
        target = winding @ theta
        raw_error = angle_error(raw, target, slice(0,512))
        avg_error = angle_error(averaged, target, slice(SETTINGS.averaging_window-1,512))
        rows.append(dict(control="circle" if dimension==1 else "product_torus",
            algorithm_selected_rank=dimension, evaluated_loop_winding=winding.tolist(),
            evaluated_winding_determinant=determinant,
            finite_complex_primitive_certificate=True,
            integer_closure_defect=model["diagnostics"]["integer_closure"]["maximum_integer_closure_defect"],
            frozen_training_replay_error=model["diagnostics"]["frozen_training_replay_error"],
            all_validation_future_states_encoded=True,
            future_raw_phase_MAE_radians=float(np.mean(raw_error[:,640:])),
            future_averaged_phase_MAE_radians=float(np.mean(avg_error[:,640:])),
            elapsed_seconds=time.perf_counter()-started))
    sources = ("geometric_phase_frontend.py","geometric_phase_encoding.py",
               "phase_lift_average.py","cohomology_primitivity.py","check_geometric_frontend.py")
    result = dict(status="passed", analytic_controls=rows,
        no_field_archives_or_evaluators_read=True,
        true_dimension_not_given_to_fitting=True,
        synthetic_control_truth_used_only_for_assertions_after_fit=True,
        source_hashes={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in sources},
        library=library_fingerprints())
    output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(dict(status=result["status"],analytic_controls=rows),indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    main(parser.parse_args().output)
