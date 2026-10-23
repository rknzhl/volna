"""One bounded, observation-only run on sin(t) + 0.1 sin(0.1 t)."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/takers_toy_signal_20261023/mpl')

import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DATA = HERE/'data'
FIGURES = HERE/'figures'
RESULT = DATA/'toy_signal.json'
ARRAYS = DATA/'toy_signal_data.npz'
DT, N_TRAIN, N_QUERY = 1.6, 1600, 400


def write_json(path, value):
    def convert(v):
        return v.tolist() if hasattr(v, 'tolist') else str(v)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=convert, allow_nan=False)+'\n')
    temporary.replace(path)


def write_arrays(value):
    temporary = ARRAYS.with_suffix('.npz.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **value)
    temporary.replace(ARRAYS)


def worker():
    sys.path.insert(0, str(REPO/'code/lag_ssa'))
    from threadpoolctl import threadpool_limits
    from fields_first_sensor_pipeline import SensorPipeline
    from evaluate_tangent_overlap_angles import match_phases
    from run_fields_first_sensor import jsonable
    report = json.loads(RESULT.read_text())
    with np.load(ARRAYS) as saved:
        arrays = {key: saved[key] for key in saved.files}
    y = arrays['Y']; started = time.monotonic()
    with threadpool_limits(limits=1):
        pipeline = SensorPipeline(max_candidates=2).fit(y[:N_TRAIN],
            progress_callback=lambda summary: write_json(DATA/'toy_signal.progress.json', summary))
        prediction = pipeline.transform(y[N_TRAIN:], start_index=N_TRAIN)
    report.update(status=prediction['status'], fit_status=pipeline.status,
        transform_status=prediction['status'], fit=jsonable(pipeline.summary()),
        elapsed_seconds=time.monotonic()-started,
        transform=jsonable({key: value for key, value in prediction.items()
                           if key not in ('reconstruction', 'residual_reconstruction', 'slow',
                                          'conditioned_observations', 'raw_time_indices',
                                          'phases', 'window_raw_time_indices')}))
    if pipeline.model is not None:
        report['hat_r'] = int(pipeline.model['r'])
        arrays['train_phases'] = pipeline.model['phases']
        arrays['train_window_raw_time_indices'] = pipeline.train_window_raw_indices
    if prediction['status'] == 'transformed_conditionally':
        for key in ('reconstruction', 'residual_reconstruction', 'slow',
                    'conditioned_observations', 'raw_time_indices',
                    'phases', 'window_raw_time_indices'):
            arrays[key] = prediction[key]
        # Predictions are saved before any comparison with the reference phase.
        write_arrays(arrays)
        report['prediction_frozen_before_truth_evaluation'] = True
        reference = arrays['reference_theta']
        match = match_phases(arrays['train_phases'],
            reference[arrays['train_window_raw_time_indices']])
        target = reference[prediction['window_raw_time_indices']] @ match['winding_candidate'].T + match['origins']
        phase_error = np.angle(np.exp(1j*(prediction['phases']-target)))
        sigma = arrays['clean'][:N_TRAIN].std(axis=0)
        error = (prediction['reconstruction']-arrays['clean'][prediction['raw_time_indices']])/sigma
        report.update(phase_mae_rad=float(np.mean(abs(phase_error))),
            sensor_nrmse=float(np.sqrt(np.mean(error**2))),
            fundamental_phase_match=bool(report['hat_r'] == 1 and match['absolute_determinant'] == 1),
            N_eval_sensors=len(prediction['raw_time_indices']),
            N_eval_phases=len(prediction['window_raw_time_indices']),
            train_phase_evaluation=jsonable(match))
        arrays['aligned_reference_phases'] = target
        write_arrays(arrays)
    else:
        write_arrays(arrays)
    write_json(RESULT, report)


def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    report = json.loads(RESULT.read_text())
    with np.load(ARRAYS) as saved:
        arrays = {key: saved[key] for key in saved.files}
    recovered = 'reconstruction' in arrays
    phase_panel = recovered and report.get('fundamental_phase_match')
    plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 10,
        'axes.spines.top': False, 'axes.spines.right': False,
        'pdf.fonttype': 42, 'ps.fonttype': 42})
    figure, axes = plt.subplots(2 if phase_panel else 1, 1,
        figsize=(7.1, 4.4 if phase_panel else 2.5), squeeze=False,
        constrained_layout=True)
    ax = axes[0, 0]
    ids = arrays['raw_time_indices'][:96] if recovered else np.arange(41)
    if recovered:
        ax.plot(arrays['t'][ids], arrays['Y'][ids, 0], color='#303030', lw=1.2, label='Signal')
    else:
        dense = np.linspace(arrays['t'][ids[0]], arrays['t'][ids[-1]], 1601)
        ax.plot(dense, np.sin(dense)+.1*np.sin(.1*dense), color='#303030', lw=1.1, label='Signal')
        ax.plot(arrays['t'][ids], arrays['Y'][ids, 0], color='#78833D',
                marker='o', ms=2.8, lw=0, label='Samples')
        ax.legend(frameon=False, loc='upper right', ncol=2)
    if recovered:
        ax.plot(arrays['t'][ids], arrays['reconstruction'][:len(ids), 0],
                color='#78833D', lw=1.2, ls='--', label='Reconstruction')
        ax.legend(frameon=False, loc='upper right', ncol=2)
    ax.set(xlabel=r'$t$', ylabel=r'$y(t)$')
    if phase_panel:
        ax = axes[1, 0]
        ids = arrays['window_raw_time_indices'][:96]
        target = arrays['aligned_reference_phases'][:len(ids), 0]
        inferred = np.unwrap(arrays['phases'][:len(ids), 0])
        inferred += 2*np.pi*np.rint((target[0]-inferred[0])/(2*np.pi))
        origin = target[0]
        ax.plot(arrays['t'][ids], target-origin, color='#303030', lw=1.1, label='Reference')
        ax.plot(arrays['t'][ids], inferred-origin, color='#6A4C93', lw=1.2,
                ls='--', label='Recovered phase')
        ax.set(xlabel=r'$t$', ylabel='Phase change (rad)')
        ax.legend(frameon=False, loc='upper left')
    FIGURES.mkdir(exist_ok=True)
    figure.savefig(FIGURES/'experiments_toy_signal.pdf')
    plt.close(figure)


def main():
    DATA.mkdir(exist_ok=True); FIGURES.mkdir(exist_ok=True)
    t = DT*np.arange(N_TRAIN+N_QUERY)
    signal = (np.sin(t)+.1*np.sin(.1*t))[:, None]
    write_arrays(dict(t=t, Y=signal, clean=signal.copy(),
        t_train=t[:N_TRAIN], t_query=t[N_TRAIN:],
        Y_train=signal[:N_TRAIN], Y_query=signal[N_TRAIN:],
        reference_theta=(.1*t)[:, None], train_indices=np.arange(N_TRAIN),
        query_indices=np.arange(N_TRAIN, len(t))))
    report = dict(name='toy_signal', equation='sin(t) + 0.1 sin(0.1 t)',
        status='running', dt=DT, N_train=N_TRAIN, N_query=N_QUERY,
        m=1, reference_r=1, hat_r=None, noise_fraction=0.,
        phase_mae_rad=None, sensor_nrmse=None,
        fundamental_phase_match=None, N_eval_sensors=0, N_eval_phases=0,
        reference_phase='theta(t)=0.1t; sin(t) is its tenth harmonic',
        sensor_nrmse_normalization='clean oscillatory TRAIN standard deviation (ddof=0)',
        maximum_candidates=2, maximum_numerical_threads=1, walltime_budget_seconds=85,
        no_truth_used_in_fit=True, prediction_frozen_before_truth_evaluation=False)
    write_json(RESULT, report)
    started = time.monotonic()
    try:
        with (DATA/'toy_signal.stdout').open('w') as stream:
            process = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker'],
                cwd=REPO, stdout=stream, stderr=subprocess.STDOUT, timeout=85)
        if process.returncode:
            report = json.loads(RESULT.read_text())
            report.update(status='unresolved_worker_error', worker_returncode=process.returncode)
            write_json(RESULT, report)
    except subprocess.TimeoutExpired:
        report = json.loads(RESULT.read_text())
        report.update(status='unresolved_walltime_budget', elapsed_seconds=time.monotonic()-started)
        write_json(RESULT, report)
    plot()
    report = json.loads(RESULT.read_text())
    print(json.dumps({key: report.get(key) for key in
        ('status', 'hat_r', 'phase_mae_rad', 'sensor_nrmse', 'fundamental_phase_match', 'elapsed_seconds')}), flush=True)


if __name__ == '__main__':
    worker() if '--worker' in sys.argv else main()
