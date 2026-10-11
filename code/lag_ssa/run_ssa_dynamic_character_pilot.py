"""Focused first-character check for the multi-mode SSA readout.

No physical phase labels/frequencies enter the readout. This finite assay
checks mixture handling and a Gaussian negative; it does not calibrate gates.
"""

from project_paths import RESULTS_DIR
from pathlib import Path
import hashlib
import json
import numpy as np

from iterative_fields import AREA, generate_field
from spatial_field import SPATIAL_MODEL
from ssa_character import apply_ssa_character
from ssa_dynamic_character import extract_ssa_dynamic_character


def main():
    rows = []
    for case in ("r2_infinite_mixed_products", "r2_mixed_no_fundamentals",
                 "r1_even_2_4_factor", "r2_poisson_dominant_weak", "gaussian"):
        rng = np.random.default_rng(54000)
        if case == "gaussian":
            mass = rng.standard_normal((6, 3200))
        else:
            clean = generate_field(case, np.arange(3200))["clean"]
            sigma = .08*np.sqrt(np.mean(np.sum(AREA[:, None]*clean**2, axis=0)))
            observed = clean+sigma*rng.standard_normal(clean.shape)
            mass = SPATIAL_MODEL.to_mass_coordinates(SPATIAL_MODEL.recover(observed))
        fitted = extract_ssa_dynamic_character(mass, np.ones(6), 128)
        row = {key: value for key, value in fitted.items() if key not in
               ("phase", "complex_coordinate", "normalized_character", "lag_filter", "signal_mean")}
        replay = None
        if "phase" in fitted:
            replay = float(np.max(abs(apply_ssa_character(fitted, mass)["normalized_character"]
                                      -fitted["normalized_character"])))
        row.update(case=case, frozen_filter_replay_error=replay)
        rows.append(row)
        print(case, fitted["status"], fitted.get("candidate_frequency"), flush=True)
    source = Path(__file__).with_name("ssa_dynamic_character.py")
    output = RESULTS_DIR/"ssa_dynamic_character_pilot.json"
    payload = {"rows": rows, "gates_uncalibrated": True, "seed": 54000,
               "relative_sensor_noise": .08, "n_train": 3200, "window": 128,
               "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    output.write_text(json.dumps(payload, indent=2)+"\n")


if __name__ == "__main__":
    main()
