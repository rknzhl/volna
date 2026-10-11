"""Original8 trailing-window audit with two matched-context initializers."""

from project_paths import RESULTS_DIR

import hashlib

import run_original_hankel_torus as original
from robust_hankel_torus_projection import RobustHankelTorusProjector


BASE_HASHES = original.hashes


def hashes():
    result = BASE_HASHES()
    for name in ("robust_hankel_torus_projection.py", "run_original_robust_hankel_torus.py"):
        result[name] = hashlib.sha256((original.HERE/name).read_bytes()).hexdigest()
    return result


def main():
    original.ROOT = RESULTS_DIR / 'original_robust_hankel_torus'
    original.OUTPUT = original.ROOT.with_suffix(".json")
    original.HankelTorusProjector = RobustHankelTorusProjector
    original.hashes = hashes
    original.main()


if __name__ == "__main__":
    main()
