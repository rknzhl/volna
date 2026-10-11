"""Same frozen assay with actual SSA phase-product relation gates enabled."""

from __future__ import annotations

from project_paths import RESULTS_DIR

import hashlib
from pathlib import Path
import sys

import run_actual_wave_revised as revised


HERE = Path(__file__).resolve().parent
BASE_FINGERPRINTS = revised.fingerprints


def fingerprints():
    result = BASE_FINGERPRINTS()
    result[Path(__file__).name] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return result


def main():
    revised.fingerprints = fingerprints
    revised.main(["--phase-relations", "observed", "--output",
                  str(RESULTS_DIR / 'actual_wave_observed_relations.json'),
                  "--fixture-output",
                  str(RESULTS_DIR / 'actual_wave_observed_relations_fixtures.json')]
                 + sys.argv[1:])


if __name__ == "__main__":
    main()
