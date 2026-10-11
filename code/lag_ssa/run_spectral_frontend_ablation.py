"""Independent frequency-discovery ablation: field FFT instead of SSA/DMD.

The factor-selection, observed relation checks and regression remain identical.
This comparator discovers its own candidate frequencies from residual field
periodograms; it does not receive frequencies or dictionaries from SSA.
Two observer windows are declared before any scoring. No frozen source edits.
"""

from __future__ import annotations

from project_paths import RESULTS_DIR

import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np
from scipy.optimize import minimize_scalar

import iterative_ssa
from final_phase_factor_assay import fixed_settings, fingerprints
from phase_readout_refinement import fit_field_character_observer, observer_quality
import run_actual_wave_revised as fixtures
from run_actual_wave_iterative import save


HERE = Path(__file__).resolve().parent
OUT = RESULTS_DIR / 'spectral_frontend_ablation.json'
CASES = tuple(fixtures.FIXTURES)
WINDOWS = (128, 512)
ORIGINAL_FRONTEND = iterative_ssa.character_frontend


def spectral_frontend(window):
    def fit(residual, noise_sigma, settings, sampling_model=None):
        if sampling_model is None or settings.phase_metric_mode != "field":
            raise ValueError("this comparator needs the explicit field mass model")
        y = np.asarray(residual, dtype=float)
        n = y.shape[1]
        if n <= window:
            raise ValueError("record shorter than the declared observation window")
        t = np.arange(n)
        tapered = (y-y.mean(axis=1,keepdims=True))*np.hanning(n)
        transform = np.fft.rfft(tapered,axis=1)
        power = np.sum(abs(transform)**2,axis=0)
        omega = 2*np.pi*np.fft.rfftfreq(n)
        eligible = np.flatnonzero((omega>2*np.pi/n)&(omega<np.pi-2*np.pi/n))
        peak = int(eligible[np.argmax(power[eligible])])
        lower = max(1e-8,float(omega[peak]-2*np.pi/n))
        upper = min(np.pi-1e-8,float(omega[peak]+2*np.pi/n))

        def negative_power(frequency):
            c = np.einsum("qt,t->q",tapered,np.exp(-1j*frequency*t),optimize=False)
            return -float(np.sum(abs(c)**2))

        grid = np.linspace(lower,upper,11)
        best = int(np.argmin([negative_power(value) for value in grid]))
        interval = (grid[max(0,best-1)],grid[min(len(grid)-1,best+1)])
        frequency = float(minimize_scalar(negative_power,bounds=interval,
            method="bounded",options={"xatol":1e-11}).x)
        observer = fit_field_character_observer(y,frequency,window)
        readout = observer.apply(y)
        z = readout["normalized_character"]
        phase = np.unwrap(np.angle(z))
        starts = np.arange(len(phase))
        centered = starts-starts.mean()
        slope = float(np.dot(centered,phase-phase.mean())/np.dot(centered,centered))
        residual_rms = float(np.std(phase-slope*starts))
        quality = observer_quality(readout,frequency,
            sampling_model.noise_covmass(noise_sigma),observer)
        gates = {
            "phase_coherent":quality["phase_step_coherence"]>=.90,
            "radius_stable":quality["radius_cv"]<=.60,
            "linear_phase":residual_rms<=.50,
            "slope_agreement":abs(np.angle(np.exp(1j*(slope-frequency))))<=.01,
            "coordinate_over_noise_reference":quality[
                "median_coordinate_radius_over_noise_rms_diagnostic"]>=3,
        }
        public = dict(status="resolved" if all(gates.values()) else "unresolved",
            reason=None if all(gates.values()) else "spectral observer purity gate failed",
            window=window,hankel_columns=len(z),metrics=dict(quality,
                frequency_temporal_dmd=frequency,frequency_phase_slope=slope,
                phase_linear_fit_rms=residual_rms,peak_periodogram_bin=peak,
                source="independently selected residual field periodogram maximum"),
            gates={key:bool(value) for key,value in gates.items()},
            gate_status="heuristic_uncalibrated",phase_metric_mode="field",
            readout_source="residual",phase_readout_method="field FFT/Hann convolution; no SSA proposal")
        if not all(gates.values()):
            return public
        public.update(candidate_frequency=frequency,
            lag_filter=observer.weights[:,None]*np.exp(-1j*frequency*np.arange(window))[:,None]*
                np.conj(observer.spatial_profile)[None,:],
            signal_mean=observer.signal_mean.copy(),
            complex_coordinate=readout["coordinate"],
            normalized_character=z,phase=np.angle(z))
        return public
    return fit


def main():
    if OUT.exists():
        raise FileExistsError("preserve the existing independent ablation")
    hashes = fingerprints()
    hashes[Path(__file__).name] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    rows = []
    started = time.perf_counter()
    payload = dict(status="running",rows=rows,source_fingerprints=hashes,
        declared_cases=list(CASES),declared_observer_windows=list(WINDOWS),
        inference_inputs=["Y","s","t"],SSA_frequencies_or_dictionary_supplied=False,
        ablation_scope="replace residual SSA/DMD proposal and phase observer by independently selected FFT/convolution; identical factor selection and regression",
        selection_uses_future=False,between_case_tuning=False)
    save(payload,OUT)
    try:
        for window in WINDOWS:
            iterative_ssa.character_frontend = spectral_frontend(window)
            destination = OUT.with_name(OUT.stem+"_window"+str(window)+".json")
            for case in CASES:
                print("starting independent spectral "+str(window)+" "+case,flush=True)
                try:
                    row = fixtures.evaluate_fixture(case,destination,"observed",
                        settings_override=fixed_settings)
                    row["independent_spectral_window"] = window
                except Exception as error:
                    row = dict(case=case,run_status="exception",independent_spectral_window=window,
                        exception=f"{type(error).__name__}: {error}",traceback=traceback.format_exc())
                rows.append(row)
                save(payload,OUT)
                print(json.dumps(dict(case=case,window=window,run_status=row["run_status"],
                    r=row.get("inference",{}).get("r_candidate_conditional"),
                    status=row.get("inference",{}).get("status"),
                    field=row.get("future_clean_sensor_field"),exception=row.get("exception"))),flush=True)
    finally:
        iterative_ssa.character_frontend = ORIGINAL_FRONTEND
    current = fingerprints()
    current[Path(__file__).name] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    payload.update(status="finished",source_fingerprints_unchanged=current==hashes,
        invocation_runtime_seconds=time.perf_counter()-started)
    save(payload,OUT)


if __name__ == "__main__":
    main()
