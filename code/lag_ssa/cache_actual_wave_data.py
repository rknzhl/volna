"""Cache the original notebook wave data without changing its generator.

Each inputs NPZ contains only Y (noisy recordings), s (sensor positions), and t.
Clean fields and generator truth live in separate evaluation NPZ files.  The
notebook's one sensor-noise RNG is advanced in scenario order, including when
resuming an existing cache.  This module never fits an inference model.
"""

from __future__ import annotations

from project_paths import RESULTS_DIR

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/waves/synthetic"))
import scenarios  # noqa: E402; loads its sibling wave solver


SLUGS = (
    "wave_1d_traveling",
    "wave_1d_standing",
    "wave_1d_counterpropagating",
    "wave_1d_layered",
    "wave_1d_source_noise",
    "wave_2d_uniform",
    "wave_2d_layered",
    "wave_2d_lens",
    "wave_2d_waveguide",
    "wave_3d_uniform",
    "wave_3d_lens",
)
OUT = RESULTS_DIR / "wave_actual_data"


def fingerprints() -> dict:
    files = {
        "cache_script": Path(__file__).resolve(),
        "scenario_generator": ROOT / "code/waves/synthetic/scenarios.py",
        "wave_solver": ROOT / "code/waves/synthetic/wave.py",
        "notebook": ROOT / "notebooks/01_synthetic.ipynb",
    }
    return {key: {"path": str(path.relative_to(ROOT)),
                  "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for key, path in files.items()}


def save_npz(path: Path, **arrays) -> None:
    """Publish a complete archive atomically, including after interruption."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def save_manifest(directory: Path, metadata: dict, rows: list) -> None:
    save_npz(directory / "manifest.npz",
             metadata_json=np.asarray(json.dumps({**metadata, "rows": rows},
                                                 ensure_ascii=False)),
             slugs=np.asarray([row["slug"] for row in rows]),
             input_files=np.asarray([row["input_file"] for row in rows]),
             evaluation_files=np.asarray([row["evaluation_file"] for row in rows]))


def configuration(source_hashes: dict) -> dict:
    return {
        "provenance": "notebooks/01_synthetic.ipynb cells 1, 5, 10",
        "scenario_order": list(scenarios.SCENARIOS),
        "simulation_seed": 0,
        "sensor_noise_rng_seed": 0,
        "sensor_noise_rng_scope": "one RNG reused in notebook scenario order",
        "sensor_noise_scale": "0.02 * max(abs(clean_Y)) per scenario",
        "source_frequencies_hz": list(scenarios.NU),
        "source_amplitudes": list(scenarios.AMP),
        "grid": scenarios.GRID,
        "duration_seconds": scenarios.DURATION,
        "discard_start_seconds": scenarios.START,
        "ramp_seconds": scenarios.RAMP,
        "output_step_seconds": 0.01,
        "numpy_version": np.__version__,
        "source_fingerprints": source_hashes,
        "input_keys": ["Y", "s", "t"],
        "evaluation_files_must_not_be_used_for_fit": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    source_hashes = fingerprints()
    config = configuration(source_hashes)
    # JSON canonicalizes integer dictionary keys before resume comparison.
    config = json.loads(json.dumps(config))
    manifest_path = directory / "manifest.npz"
    previous_rows = {}
    if manifest_path.exists():
        if not args.resume:
            raise SystemExit("Cache exists; use --resume to preserve completed fields.")
        with np.load(manifest_path, allow_pickle=False) as archive:
            previous = json.loads(str(archive["metadata_json"]))
        if previous["configuration"] != config:
            raise SystemExit("Resume rejected: generator, notebook, or configuration changed.")
        previous_rows = {row["slug"]: row for row in previous.get("rows", [])}
    if len(SLUGS) != len(scenarios.SCENARIOS):
        raise SystemExit("Scenario/slugs count mismatch.")
    rng = np.random.default_rng(0)
    rows = []
    started = time.perf_counter()
    metadata = {"configuration": config, "status": "running",
                "invocation_finished": False}
    save_manifest(directory, metadata, rows)
    for index, ((name, sc), slug) in enumerate(zip(scenarios.SCENARIOS.items(), SLUGS)):
        input_path = directory / "inputs" / (slug + ".npz")
        evaluation_path = directory / "evaluation" / (slug + ".npz")
        previous_row = previous_rows.get(slug)
        if previous_row and input_path.exists() and evaluation_path.exists():
            with np.load(input_path, allow_pickle=False) as archive:
                if set(archive.files) != {"Y", "s", "t"}:
                    raise RuntimeError("Input archive contains unexpected keys.")
                shape = archive["Y"].shape
            # Notebook generates a complete Gaussian matrix for every scenario.
            rng.standard_normal(shape)
            rows.append(previous_row)
            print(f"{index + 1}/11 {slug}: reused {shape}", flush=True)
            save_manifest(directory, metadata, rows)
            continue
        print(f"{index + 1}/11 {slug}: simulating original {name}", flush=True)
        case_started = time.perf_counter()
        run = scenarios.simulate(sc, seed=0, snap_times=())
        clean = run["Y"]
        sigma = 0.02 * np.abs(clean).max()
        noisy = clean + sigma * rng.standard_normal(clean.shape)
        save_npz(input_path, Y=noisy, s=run["sensors"], t=run["t"])
        h = 1 / (scenarios.GRID[sc["d"]] - 1)
        save_npz(evaluation_path,
                 clean_Y=clean,
                 source_frequencies_hz=np.asarray(scenarios.NU),
                 source_amplitudes=np.asarray(scenarios.AMP),
                 source_positions=run["sources"],
                 medium_n=run["n"],
                 physical_t=run["t"] + scenarios.START,
                 sensor_recording_nodes=np.rint(run["sensors"] / h).astype(int),
                 sensor_noise_standard_deviation=np.asarray(sigma),
                 scenario_name=np.asarray(name),
                 scenario_json=np.asarray(json.dumps(sc, ensure_ascii=False)),
                 generator_metadata_json=np.asarray(json.dumps(config, ensure_ascii=False)))
        elapsed = time.perf_counter() - case_started
        row = {"index": index, "slug": slug, "scenario": name,
               "shape": list(clean.shape), "spatial_dimension": sc["d"],
               "input_file": str(input_path.relative_to(directory)),
               "evaluation_file": str(evaluation_path.relative_to(directory)),
               "runtime_seconds": elapsed,
               "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
               "evaluation_sha256": hashlib.sha256(evaluation_path.read_bytes()).hexdigest()}
        rows.append(row)
        save_manifest(directory, metadata, rows)
        print(f"{index + 1}/11 {slug}: saved {clean.shape}, {elapsed:.2f} s", flush=True)
    unchanged = fingerprints() == source_hashes
    metadata.update(status="complete" if unchanged else "source_changed_during_run",
                    invocation_finished=True,
                    source_fingerprints_unchanged=unchanged,
                    invocation_runtime_seconds=time.perf_counter() - started,
                    total_simulation_runtime_seconds=sum(row["runtime_seconds"] for row in rows))
    save_manifest(directory, metadata, rows)
    print(json.dumps({"status": metadata["status"], "rows": len(rows),
                      "runtime_seconds": metadata["invocation_runtime_seconds"],
                      "source_fingerprints_unchanged": unchanged,
                      "directory": str(directory)}, ensure_ascii=False), flush=True)
    if not unchanged:
        raise SystemExit("Source fingerprint changed during data generation.")


if __name__ == "__main__":
    main()
