"""Measured-window polynomial diagnostics on six frozen free spectral banks.

This is a bounded viability assay, not a relation/group selection method.
The TRAIN/VAL-discovered bank is applied to the old FUTURE observation block.
That benchmark was previously examined; this is not a fresh external test.
"""

from project_paths import RESULTS_DIR

from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.linalg import svd

from wick_character_moments import character_moment, overlapping_mean_variance_bound


HERE = Path(__file__).resolve().parent
SOURCE = RESULTS_DIR / 'ledger_group_selection.json'
OUTPUT = RESULTS_DIR / 'wick_character_assay.json'
PROTOCOL = RESULTS_DIR / 'wick_character_assay_protocol.json'
ARCHIVES = RESULTS_DIR / 'wick_character_assay'
SOURCES = ("wick_character_moments.py", "run_wick_character_assay.py")
SETTINGS = dict(window=128, maximum_degree=8, full_bank_rank_rcond=1e-10,
                samples="all contiguous windows wholly within original FUTURE block",
                relation_subset="all original eligible proposals with l1 degree <= 8")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, payload):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def declaration():
    source = json.loads(SOURCE.read_text())
    if source["status"] != "finished_inference_only":
        raise ValueError("completed frozen free banks required")
    public = {item["run"]: item for item in source["manifest"]}
    manifest = []
    for row in source["rows"]:
        original = public[row["run"]]
        model = row["model_files"][0]
        if model["model_kind"] != "frozen_free_ledger_baseline":
            raise ValueError("unconstrained, unchanged bank required")
        selected = []
        for index, proposal in enumerate(row["inference"]["proposals"]["all_proposals"]):
            degree = sum(abs(value) for value in proposal["row"])
            if proposal["eligible_for_nested_refit"] and 0 < degree <= SETTINGS["maximum_degree"]:
                selected.append(dict(proposal_index=index, row=proposal["row"], degree=degree))
        entry = {key: original[key] for key in ("run", "input", "input_sha256", "n_train", "n_validation", "n_future")}
        entry.update(parameters=model["parameters"], parameters_sha256=model["parameters_sha256"],
                     sigma_estimate=row["noise_estimate_train_only"]["sigma"], relations=selected)
        for path, expected in ((entry["input"], entry["input_sha256"]),
                               (entry["parameters"], entry["parameters_sha256"])):
            if digest(path) != expected:
                raise ValueError("frozen input changed")
        manifest.append(entry)
    return dict(status="prepared_not_run", created_at=datetime.now(timezone.utc).isoformat(),
                source=str(SOURCE), source_sha256=digest(SOURCE), settings=SETTINGS,
                source_fingerprints={name: digest(HERE / name) for name in SOURCES}, manifest=manifest,
                relation_count=sum(len(item["relations"]) for item in manifest),
                prior_benchmark_inspection=True, fresh_external_test=False,
                future_observations_used_for_readout=True, observations_given_to_fit=False,
                evaluation_files_opened=False, physical_labels_given_to_assay=False,
                bank_selection="old TRAIN/VAL; FUTURE raw readout only; no refit or relation selection",
                normalization="complex beta left unnormalized; no clipping of corrected mean",
                uncertainty="conditional unit-amplitude fixed-bank variance upper bound; estimated sigma; no model-error bound or unconditional CI",
                overlap="all contiguous windows; independence only for start separation >= L, Cauchy bound for closer starts",
                hypothesis="noise correction may improve measured relation diagnostics; high degree and weak profiles may make variance unusable",
                group_or_rank_inferred=False, novelty_claimed=False)


def validate(prepared):
    current = declaration()
    for key in ("source_sha256", "settings", "source_fingerprints", "manifest", "relation_count"):
        if prepared[key] != current[key]:
            raise ValueError("changed prepared assay: " + key)


def full_bank_readout(parameters, raw, starts, sigma):
    nu = parameters["frequencies"]
    profile = parameters["complex_profiles_sensor_origin"]
    root = parameters["spatial_mass_sqrt"]
    mean = parameters["coefficients_sensor_center"][0]
    length, sensors, count = SETTINGS["window"], len(mean), len(nu)
    weighted = np.einsum("js,ks->jk", profile, root, optimize=False)
    positive = (2 / np.sqrt(length) * np.exp(1j * np.arange(length)[:, None, None] * nu[None, None, :])
                * weighted.T[None, :, :]).reshape(length * sensors, count)
    bank = np.column_stack((positive.real, -positive.imag))
    u, singular, vt = svd(bank, full_matrices=False, lapack_driver="gesvd")
    if np.min(singular) <= SETTINGS["full_bank_rank_rcond"] * np.max(singular):
        raise np.linalg.LinAlgError("full bank unresolved; no truncation allowed")
    inverse = np.einsum("ia,ba->ib", vt.T / singular[None, :], u, optimize=False)
    readout = np.einsum("als,si->ali", inverse.reshape(2 * count, length, sensors), root, optimize=False) / np.sqrt(length)
    centered = raw.T - mean[None, :]
    windows = np.lib.stride_tricks.sliding_window_view(centered, length, axis=0).transpose(0, 2, 1)
    beta = np.einsum("ali,nli->na", readout, windows, optimize=False)
    covariance = sigma**2 * np.einsum("ali,bli->ab", readout, readout, optimize=False)
    covariance = (covariance + covariance.T) / 2
    expected = np.column_stack((np.cos(starts[:, None] * nu), np.sin(starts[:, None] * nu)))
    template = np.einsum("na,pa->np", expected, bank, optimize=False)
    recomposed = np.einsum("ap,np->na", inverse, template, optimize=False)
    bank_identity = np.einsum("ap,pb->ab", inverse, bank, optimize=False)
    return beta, covariance, readout, dict(
        full_bank_columns=2 * count, bank_condition=float(singular[0] / singular[-1]),
        inverse_identity_max_error=float(np.max(abs(bank_identity - np.eye(2 * count)))),
        known_template_coefficient_max_error=float(np.max(abs(recomposed - expected))),
        real_coefficient_covariance_eigenvalues=np.linalg.eigvalsh(covariance).tolist(),
        complex_noise_variances=(np.diag(covariance)[:count] + np.diag(covariance)[count:]).tolist(),
        unnormalized_coordinate_modulus_median=np.median(abs(beta[:, :count] + 1j * beta[:, count:]), axis=0).tolist(),
        sensor_count=sensors, window_count=len(beta))


def run_entry(entry):
    with np.load(entry["parameters"], allow_pickle=False) as archive:
        parameters = {key: archive[key] for key in archive.files}
    first = entry["n_train"] + entry["n_validation"]
    last = first + entry["n_future"]
    with np.load(entry["input"], allow_pickle=False) as archive:
        if set(archive.files) != {"Y", "s", "t"}:
            raise ValueError("public Y,s,t archive required")
        times = archive["t"][first:last]
        positions = archive["s"]
        raw = archive["Y"][:, first:last]
    if not np.array_equal(times, np.arange(first, last)) or not np.array_equal(positions, parameters["sensor_positions"]):
        raise ValueError("changed sample grid or sensors")
    starts = times[:len(times) - SETTINGS["window"] + 1]
    beta, covariance, readout, audit = full_bank_readout(parameters, raw, starts, entry["sigma_estimate"])
    results = []
    raw_series, corrected_series = [], []
    for proposal in entry["relations"]:
        statistic = character_moment(beta, covariance, proposal["row"])
        naive, corrected = statistic["raw"], statistic["corrected"]
        upper = statistic["pointwise_variance_upper_bound"]
        mean_upper = overlapping_mean_variance_bound(upper, len(starts), SETTINGS["window"])
        raw_mean, corrected_mean = complex(np.mean(naive)), complex(np.mean(corrected))
        clock = complex(np.mean(np.exp(1j * starts * np.dot(proposal["row"], parameters["frequencies"]))))
        half = len(starts) // 2
        results.append(dict(proposal, raw_mean=[raw_mean.real, raw_mean.imag],
                            corrected_mean=[corrected_mean.real, corrected_mean.imag],
                            raw_mean_modulus=abs(raw_mean), corrected_mean_modulus=abs(corrected_mean),
                            correction_mean_magnitude=abs(corrected_mean - raw_mean),
                            pointwise_variance_upper_bound=upper,
                            overlapping_mean_variance_upper_bound=mean_upper,
                            conditional_mean_std_upper_bound=float(np.sqrt(mean_upper)),
                            observed_mean_difference_between_halves=float(abs(np.mean(corrected[:half]) - np.mean(corrected[half:]))),
                            free_clock_mean_modulus=abs(clock),
                            corrected_mean_distance_to_free_clock=abs(corrected_mean - clock),
                            finite_outputs=bool(np.isfinite(naive).all() and np.isfinite(corrected).all()),
                            relation_accepted=False, relation_rejected=False))
        raw_series.append(naive)
        corrected_series.append(corrected)
    path = ARCHIVES / (entry["run"] + ".npz")
    np.savez_compressed(path, beta=beta, covariance=covariance, readout_raw=readout,
                        starts=starts, raw_products=np.asarray(raw_series), corrected_products=np.asarray(corrected_series),
                        relation_rows=np.asarray([item["row"] for item in entry["relations"]], dtype=int))
    return dict(run=entry["run"], status="diagnosed", archive=str(path), archive_sha256=digest(path),
                bank=audit, relations=results, observed_sample_interval=[first, last],
                original_future_block_used_for_fitting=False,
                summary=dict(relations=len(results),
                             conditional_std_at_least_one=sum(item["conditional_mean_std_upper_bound"] >= 1 for item in results),
                             corrected_modulus_above_one=sum(item["corrected_mean_modulus"] > 1 for item in results)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    arguments = parser.parse_args()
    if not arguments.run:
        if PROTOCOL.exists() or OUTPUT.exists():
            raise FileExistsError("preserve frozen assay")
        prepared = declaration()
        write_json(PROTOCOL, prepared)
        print(json.dumps(dict(status=prepared["status"], relation_count=prepared["relation_count"])), flush=True)
        return
    prepared = json.loads(PROTOCOL.read_text())
    validate(prepared)
    if OUTPUT.exists() or ARCHIVES.exists():
        raise FileExistsError("preserve frozen assay")
    ARCHIVES.mkdir()
    output = dict(prepared, status="running", protocol_sha256=digest(PROTOCOL), rows=[])
    write_json(OUTPUT, output)
    tick = time.perf_counter()
    for entry in prepared["manifest"]:
        row = run_entry(entry)
        output["rows"].append(row)
        write_json(OUTPUT, output)
        print(json.dumps(dict(run=row["run"], summary=row["summary"], bank_condition=row["bank"]["bank_condition"])), flush=True)
    validate(prepared)
    output.update(status="finished_measured_window_assay", elapsed_seconds=time.perf_counter() - tick,
                  all_six_outputs_frozen_before_new_evaluation=True, source_input_hashes_unchanged=True)
    write_json(OUTPUT, output)
    print(json.dumps(dict(status=output["status"], elapsed_seconds=output["elapsed_seconds"])), flush=True)


if __name__ == "__main__":
    main()
