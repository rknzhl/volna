"""Final matched FFT reuse with both observation-derived L128 initializers.

The six independent FFT field parameter archives remain unchanged. Reuse the
matched raw-Hann initialization and unconditional free-span initialization;
compare to the final robust SSA audit on all 1073 identical future starts.
The common runner freezes all six outputs before opening evaluation archives.
"""

from project_paths import RESULTS_DIR

import hashlib
from pathlib import Path

import run_fft_strong_raw_hankel_audit as audit
from robust_hankel_torus_projection import RobustHankelTorusProjector


BASE_HASHES = audit.hashes


def hashes():
    result = BASE_HASHES()
    for name in ("robust_hankel_torus_projection.py", Path(__file__).name):
        result[name] = hashlib.sha256((audit.HERE/name).read_bytes()).hexdigest()
    return result


def main():
    audit.ROOT = RESULTS_DIR / 'fft_robust_strong_raw_hankel_audit'
    audit.OUTPUT = audit.ROOT.with_suffix(".json")
    audit.BASELINE = RESULTS_DIR / 'robust_strong_raw_hankel_audit.json'
    audit.matched.HankelTorusProjector = RobustHankelTorusProjector
    audit.hashes = hashes
    audit.main()


if __name__ == "__main__":
    main()
