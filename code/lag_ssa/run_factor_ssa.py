"""Same law-blind assay with one global factor-field fit instead of a hierarchy."""

from project_paths import RESULTS_DIR

import hashlib
from pathlib import Path
import sys

import final_phase_factor_assay as assay
import run_actual_wave_iterative as wave_driver
import run_actual_wave_revised as fixture_driver
from factor_ssa import fit_factor_ssa


HERE = Path(__file__).resolve().parent
BASE_FINGERPRINTS = assay.fingerprints


def fingerprints():
    result = BASE_FINGERPRINTS()
    for name in ("factor_ssa.py", "run_factor_ssa.py"):
        result[name] = hashlib.sha256((HERE/name).read_bytes()).hexdigest()
    return result


def main():
    wave_driver.fit_iterative_ssa = fit_factor_ssa
    fixture_driver.fit_iterative_ssa = fit_factor_ssa
    assay.fingerprints = fingerprints
    sys.argv = [sys.argv[0], "--output", str(RESULTS_DIR / 'factor_ssa_waves.json'),
                "--fixture-output", str(RESULTS_DIR / 'factor_ssa_fixtures.json')] + sys.argv[1:]
    assay.main()


if __name__ == "__main__":
    main()
