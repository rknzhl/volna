"""Exact finite-budget conditioning example for four signed torus modes.

Five widely spaced delays are compared with five consecutive delays and a
full consecutive horizon. This checks a geometric claim, not phase recovery.
"""

from __future__ import annotations

from project_paths import RESULTS_DIR

import json
from pathlib import Path

import numpy as np


OUT = RESULTS_DIR


def smallest_frame_eigenvalue(lags: np.ndarray, frequencies: np.ndarray) -> float:
    steering = np.exp(1j * np.outer(lags, frequencies)) / np.sqrt(len(lags))
    gram = np.einsum("lm,ln->mn", steering.conj(), steering, optimize=False)
    return float(np.linalg.eigvalsh(gram)[0])


def main() -> None:
    rows = []
    for d in [20, 40, 80, 160]:
        first = 2*np.pi/(5*d) + np.sqrt(2)/d**3
        second = 4*np.pi/(5*d) + np.sqrt(3)/d**3
        frequencies = np.array([first, -first, second, -second])
        designs = {"five_consecutive": np.arange(5),
                   "five_spaced": d*np.arange(5),
                   "full_horizon": np.arange(4*d+1)}
        rows.append({"D": d, "frequencies": [first, second],
                     "smallest_frame_eigenvalue": {
                         name: smallest_frame_eigenvalue(lags, frequencies)
                         for name, lags in designs.items()}})
    (OUT / "budget_conditioning.json").write_text(json.dumps({
        "field": "2*g(s)*(cos(theta_1)+cos(theta_2)), norm_s(g)=1",
        "signed_modes": ["+e1", "-e1", "+e2", "-e2"],
        "rows": rows,
    }, indent=2) + "\n")
    for row in rows:
        print(row["D"], row["smallest_frame_eigenvalue"])


if __name__ == "__main__":
    main()
