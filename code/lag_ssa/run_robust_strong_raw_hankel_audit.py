"""Frozen general18 forward-window audit with two L128 initializers."""

from project_paths import RESULTS_DIR

import hashlib

import run_strong_raw_hankel_audit as audit
from robust_hankel_torus_projection import RobustHankelTorusProjector


BASE_HASHES = audit.hashes


def hashes():
    result = BASE_HASHES()
    for name in ("robust_hankel_torus_projection.py", "run_robust_strong_raw_hankel_audit.py"):
        result[name] = hashlib.sha256((audit.HERE/name).read_bytes()).hexdigest()
    return result


def main():
    audit.ROOT = RESULTS_DIR / 'robust_strong_raw_hankel_audit'
    audit.OUTPUT = audit.ROOT.with_suffix(".json")
    audit.HankelTorusProjector = RobustHankelTorusProjector
    audit.hashes = hashes
    audit.main()


if __name__ == "__main__":
    main()
