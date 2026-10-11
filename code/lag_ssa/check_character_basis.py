"""Exact character-lattice checks, including limits of the order budget."""

from __future__ import annotations

from project_paths import RESULTS_DIR

import itertools
import json
from math import gcd
from pathlib import Path

import numpy as np

from method import infer_frequency_lattice
from phase_lattice import infer_character_basis, infer_character_basis_global


OUT = RESULTS_DIR
ALPHA = np.array([.173*np.sqrt(2), .217*np.sqrt(3)])


def lattice_index(labels: np.ndarray) -> int:
    r = labels.shape[1]
    answer = 0
    for selected in itertools.combinations(range(len(labels)), r):
        determinant = round(float(np.linalg.det(labels[list(selected)])))
        answer = gcd(answer, abs(determinant))
    return answer


def main() -> None:
    cases = {
        "periodic_2_3": np.array([[2], [3]]),
        "periodic_even_cover": np.array([[2], [4]]),
        "torus_coprime_overtones": np.array([[2, 0], [3, 0], [0, 2], [0, 3]]),
        "torus_mixed_primitive": np.array([[2, 0], [0, 3], [1, 1], [1, -1]]),
        "torus_index3_cover": np.array([[2, 1], [1, 2], [1, -1]]),
        "order_budget_limit_4_6_9": np.array([[4], [6], [9]]),
        "changing_basis_order_limit_2_3_4_8": np.array([[2], [3], [4], [8]]),
    }
    rows = []
    for name, labels in cases.items():
        alpha = ALPHA[:labels.shape[1]]
        frequencies = np.angle(np.exp(1j*(labels @ alpha)))
        expected_rank = labels.shape[1]
        expected_index = lattice_index(labels)
        for order in [6, 12]:
            correct = 0
            global_correct = 0
            samples = []
            for permutation in itertools.permutations(range(len(labels))):
                index = np.asarray(permutation)
                result = infer_character_basis(frequencies[index], 1e-10,
                                               max_order=order, max_rank=3)
                global_result = infer_character_basis_global(frequencies[index], 1e-10,
                                                             max_order=order, max_rank=3)
                if result["rank"] == expected_rank:
                    products = np.asarray(result["products"], dtype=int)
                    recovered_labels = products @ labels[index]
                    recovered_index = lattice_index(recovered_labels)
                    success = recovered_index == expected_index
                    assert result["max_peak_residual"] < 1e-8
                else:
                    recovered_index, success = None, False
                correct += success
                global_index = None
                if global_result["rank"] == expected_rank:
                    global_labels = np.asarray(global_result["products"]) @ labels[index]
                    global_index = lattice_index(global_labels)
                    global_correct += global_index == expected_index
                samples.append({"permutation": list(permutation), "rank": result["rank"],
                                "lattice_index": recovered_index,
                                "products": result.get("products"),
                                "global_rank": global_result["rank"],
                                "global_index": global_index,
                                "global_products": global_result.get("products")})
            old = infer_frequency_lattice(frequencies, 1e-10, max_rank=3, max_order=order)
            row = {"case": name, "expected_rank": expected_rank,
                   "expected_index": expected_index, "max_order": order,
                   "old_rank_on_original_order": old["rank"],
                   "correct_permutations": correct, "permutations": len(samples),
                   "global_correct_permutations": global_correct,
                   "samples": samples}
            rows.append(row)
            print(name, "order", order, "correct", correct, "/", len(samples),
                  "global", global_correct, "old rank", old["rank"], flush=True)
            if order == 12:
                assert global_correct == len(samples), (name, row)
    payload = {"description": "exact module rank and finite-cover index, all peak permutations",
               "frequencies_rad": ALPHA.tolist(), "rows": rows}
    (OUT / "character_basis_checks.json").write_text(json.dumps(payload, indent=2)+"\n")


if __name__ == "__main__":
    main()
