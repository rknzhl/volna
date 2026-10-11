"""All11 original wave observations: frozen robust trailing-L128 observer.

No fixtures or generator are opened during fitting. The eleven cached public
Y,s,t inputs and the original60/20/20 split are reused unchanged. The source
noise scenario is retained as a control outside the pure torus signal class.
"""

from project_paths import RESULTS_DIR

import hashlib

import run_actual_wave_iterative as wave
import run_original_hankel_torus as original
from robust_hankel_torus_projection import RobustHankelTorusProjector


BASE_HASHES = original.hashes
BASE_SCORE = original.score


def hashes():
    result = BASE_HASHES()
    for name in ("robust_hankel_torus_projection.py", "run_all_waves_robust_hankel.py"):
        result[name] = hashlib.sha256((original.HERE/name).read_bytes()).hexdigest()
    return result


def public_manifest():
    return [{"case": name, "kind": "wave",
        "input": str(wave.DATA/"inputs"/(name+".npz")),
        "input_sha256": hashlib.sha256((wave.DATA/"inputs"/(name+".npz")).read_bytes()).hexdigest()}
        for name in wave.CASES]


def score(row, entry):
    BASE_SCORE(row, entry)
    row["pure_torus_class_control"] = {
        "stochastic_source_noise_case": entry["case"] == "wave_1d_source_noise",
        "source_noise_within_pure_quasiperiodic_field_class": False if entry["case"] == "wave_1d_source_noise" else None,
        "interpretation": "source-noise control retains stochastic field dynamics beyond a deterministic torus template"
            if entry["case"] == "wave_1d_source_noise" else "tonal forcing scenarios; finite-window transients may remain"}


def main():
    original.ROOT = RESULTS_DIR / 'original_all_waves_robust_hankel'
    original.OUTPUT = original.ROOT.with_suffix(".json")
    original.HankelTorusProjector = RobustHankelTorusProjector
    original.public_manifest = public_manifest
    original.hashes = hashes
    original.score = score
    original.main()


if __name__ == "__main__":
    main()
