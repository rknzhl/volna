"""MSSA comparator with the same TRAIN sensor scaling as the overlap pilot."""

from project_paths import RESULTS_DIR

import json
from pathlib import Path
import numpy as np
from scipy import linalg

from tangent_overlap import windows, fold


OUT = RESULTS_DIR / 'tangent_overlap_pilot'


def run():
    protocol = json.loads((OUT / "protocol.json").read_text())
    results = []
    for name in protocol["cases"]:
        y = np.load(OUT / "inputs" / f"{name}.npz")["Y"]
        inferred = np.load(OUT / f"{name}_inferred.npz")
        standardized = (y - inferred["mean"]) / inferred["scale"]
        ntrain, lag = protocol["train_count"], protocol["lag"]
        h = windows(standardized[:ntrain], lag)
        center = np.mean(h, axis=0)
        _, _, vt = linalg.svd(h - center, full_matrices=False)
        future = windows(standardized[ntrain:], lag)
        predictions = []
        for rank in protocol["mssa_ranks"]:
            u = vt[:rank].T
            coded = np.einsum("ni,ia->na", future - center, u)
            recovered = np.einsum("na,ia->ni", coded, u) + center
            prediction = fold(recovered, y.shape[1], lag)
            prediction = prediction * inferred["scale"] + inferred["mean"]
            predictions.append((rank, prediction))
        # The fitted projector and all predictions are fixed before truth is opened.
        clean = np.load(OUT / "evaluation" / f"{name}.npz")["clean"][ntrain:]
        rows = [{"rank": rank, "future_clean_nrmse": float(np.linalg.norm(p - clean) /
                                                            np.linalg.norm(clean))}
                for rank, p in predictions]
        results.append({"case": name, "train_sensor_scaling_matches_new_method": True,
                        "future_is_encoding_observed_readings_not_forecast": True,
                        "mssa": rows})
    destination = OUT / "matched_scaling_mssa.json"
    destination.write_text(json.dumps(results, indent=2))
    print(json.dumps(results), flush=True)


if __name__ == "__main__":
    run()
