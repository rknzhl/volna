"""Run the Giannakis spectral baseline on the eight frozen article cases."""
import os
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'

import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy import linalg
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = HERE / 'data'
OUT = DATA / 'giannakis'
CODE = ROOT / 'code/lag_ssa'
sys.path.insert(0, str(CODE))

from fields_first_preprocess import fit_preprocess, highpass_decompose
from fields_first_sensor_cases import generate_variant
from tangent_overlap import windows, fold
from evaluate_tangent_overlap_angles import match_phases

CASES = ('toy_sine', 'toy_torus', 'scalar_noise10', 'two_waves',
         'noise00', 'noise01', 'noise05', 'noise10')
SETTINGS = dict(basis_sizes=[32, 64, 128], regularizations=[.0001, .001, .01],
                training_fraction=.75, neighbors=8, bandwidth_scale=1.,
                decoder_rcond=1e-10, maximum_numerical_threads=1)


def digest(value, dtype=None):
    if isinstance(value, Path):
        return hashlib.sha256(value.read_bytes()).hexdigest()
    return hashlib.sha256(np.asarray(value, dtype=dtype).tobytes()).hexdigest()


def jsonable(value):
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, complex):
        return [value.real, value.imag]
    if isinstance(value, Path):
        return str(value.relative_to(ROOT)) if value.is_relative_to(ROOT) else str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(jsonable(value), indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def inputs(name):
    """Keep reference material separate from estimator inputs."""
    path = DATA / (name + '.json')
    source = json.loads(path.read_text())
    if name.startswith('toy_'):
        with np.load(DATA / (name + '_data.npz')) as archive:
            y = archive['Y'].copy()
            ntrain = source['N_train']
            indices = archive['raw_time_indices'].copy()
            windices = archive['window_raw_time_indices'].copy()
            evaluator = dict(theta=archive['reference_theta'].copy(),
                clean_total=archive['clean'].copy(),
                signal_train_std=archive['clean'][:ntrain].std(axis=0))
        dt, lag = source['dt'], source['lag']
    else:
        config = {key: source[key] for key in
                  ('case', 'seed', 'noise', 'trend', 'phase_mode', 'rate_profile')}
        config['trend_strength'] = source.get('generator_metadata', {}).get('trend_strength', .3)
        ntrain = source['ntrain']
        y, evaluator = generate_variant(n=ntrain + source['ntest'],
                                       train_count=ntrain, **config)
        report = source['transform']
        indices = np.arange(report['raw_time_first'], report['raw_time_last'] + 1)
        windices = np.arange(report['raw_time_first'],
                            report['raw_time_first'] + report['retained_window_count'])
        dt, lag = 1., source['fit']['selected_lag']
    return dict(name=name, source=source, source_path=path, train=y[:ntrain],
                query=y[ntrain:], ntrain=ntrain, dt=dt, lag=lag,
                snapshot_indices=indices, window_indices=windices), evaluator


def prepare(data):
    """Use exactly the shared frozen MSSA/highpass preparation of CC and TC."""
    model = fit_preprocess(data['train'], lag=32, maxrank=24, trend_degrees=(0,))
    expected = data['source']['fit']['preprocessing']
    if model['rank'] != expected['rank']:
        raise ValueError('Regenerated preprocessing rank differs from saved experiment')
    np.testing.assert_allclose(model['cv']['selected']['cv_mse_raw'],
        expected['cv_selected']['cv_mse_raw'], rtol=1e-10, atol=1e-14)
    train = highpass_decompose(model, data['train'])
    query = highpass_decompose(model, data['query'], start_index=data['ntrain'])
    take, qtake = np.flatnonzero(train['valid_mask']), np.flatnonzero(query['valid_mask'])
    a, b = train['residual'][take], query['residual'][qtake]
    mean, rawscale = a.mean(axis=0), a.std(axis=0)
    scale = np.where(rawscale > rawscale.max() * 1e-12, rawscale, 1.)
    normal = np.sqrt(data['lag'] * a.shape[1])
    x, z = windows((a - mean) / scale, data['lag']) / normal, windows((b - mean) / scale, data['lag']) / normal
    train_ids = train['time_index'][take][:len(x)].astype(int)
    query_ids = query['time_index'][qtake][:len(z)].astype(int)
    snapshot_ids = query['time_index'][qtake].astype(int)
    np.testing.assert_array_equal(query_ids, data['window_indices'])
    np.testing.assert_array_equal(snapshot_ids, data['snapshot_indices'])
    return dict(x=x, z=z, mean=mean, scale=scale, normal=normal,
                slow=query['slow'][qtake], train_indices=train_ids,
                query_indices=query_ids, snapshot_indices=snapshot_ids,
                preprocessing=dict(rank=model['rank'], cv_selected=model['cv']['selected'],
                    lag=32, highpass_window=129, highpass_polyorder=2,
                    boundary_trim=95, query_refit=False))


def readout(eigenfunctions, target, weights):
    w = np.sqrt(weights / np.sum(weights))[:, None]
    return linalg.lstsq(w * eigenfunctions, w * target,
                        cond=SETTINGS['decoder_rcond'], lapack_driver='gelsd')[0]


def multiply(a, b):
    # Some macOS BLAS builds set stale floating-point flags on finite products.
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        result = a @ b
    if not np.isfinite(result).all():
        raise FloatingPointError('nonfinite spectral matrix product')
    return result


def select_model(x, dt, lag, progress):
    from giannakis_koopman import fit_diffusion_basis, extend_diffusion_basis, fit_generator
    stop = int(len(x) * SETTINGS['training_fraction'])
    fit_x, validate_x = x[:stop], x[stop + lag - 1:]
    basis = fit_diffusion_basis(fit_x, n_basis=max(SETTINGS['basis_sizes']),
        neighbors=SETTINGS['neighbors'], bandwidth_scale=SETTINGS['bandwidth_scale'])
    validation_basis = extend_diffusion_basis(basis, validate_x)
    candidates = []
    for size in SETTINGS['basis_sizes']:
        for epsilon in SETTINGS['regularizations']:
            model = fit_generator(basis, dt=dt, regularization=epsilon, n_basis=size)
            coefficients = readout(model['train_eigenfunctions'], fit_x, basis['weights'])
            values = multiply(validation_basis[:, :model['n_basis']], model['coefficients'])
            current = np.real(multiply(values, coefficients))
            # The real eigenvalue decay is artificial diffusion, not physical damping.
            phase_step = np.exp(1j * model['eigenvalues'].imag * dt)
            shifted = np.real(multiply(values[:-1] * phase_step, coefficients))
            reconstruction = float(np.mean((current - validate_x) ** 2))
            evolution = float(np.mean((shifted - validate_x[1:]) ** 2))
            row = dict(n_basis=size, regularization=epsilon,
                       validation_reconstruction_mse=reconstruction,
                       validation_one_step_mse=evolution,
                       selection_loss=.5 * (reconstruction + evolution))
            candidates.append(row)
            progress(candidates)
    selected = min(candidates, key=lambda row: (row['selection_loss'], row['n_basis'], row['regularization']))
    return selected, dict(candidates=candidates, fit_window_count=len(fit_x),
        validation_window_count=len(validate_x), gap_windows=lag - 1,
        selection='minimum equally weighted validation reconstruction and one-step harmonic prediction MSE',
        validation_target='observed standardized highpass windows within TRAIN',
        preprocessing_scope='shared preprocessing fitted once to original TRAIN only')


def run(name, resume=False):
    from giannakis_koopman import (fit_diffusion_basis, fit_generator,
                                  transform_generator, select_independent_phases)
    OUT.mkdir(exist_ok=True)
    data, evaluator = inputs(name)
    sources = [Path(__file__), CODE / 'giannakis_koopman.py',
               CODE / 'fields_first_sensor_cases.py', CODE / 'fields_first_preprocess.py',
               CODE / 'tangent_overlap.py', CODE / 'evaluate_tangent_overlap_angles.py']
    version = dict(settings=SETTINGS, source_sha256=digest(data['source_path']),
        toy_input_sha256=digest(DATA / (name + '_data.npz')) if name.startswith('toy_') else None,
        code_sha256={str(p.relative_to(ROOT)): digest(p) for p in sources},
        packages={package: importlib.metadata.version(package) for package in ('numpy', 'scipy', 'scikit-learn')})
    key = hashlib.sha256(json.dumps(version, sort_keys=True).encode()).hexdigest()
    destination = OUT / (name + '.json')
    if resume and destination.exists():
        old = json.loads(destination.read_text())
        if old.get('cache_key') == key:
            print('CACHED', name, flush=True)
            return old
    started = time.monotonic()
    print('FIT', name, flush=True)
    p = prepare(data)
    def progress(rows):
        write_json(OUT / (name + '.progress.json'), dict(name=name, stage='training_validation',
            elapsed_seconds=time.monotonic() - started, candidates=rows))
        print('VALIDATE', name, json.dumps(rows[-1]), flush=True)
    selected, validation = select_model(p['x'], data['dt'], data['lag'], progress)
    basis = fit_diffusion_basis(p['x'], n_basis=selected['n_basis'],
        neighbors=SETTINGS['neighbors'], bandwidth_scale=SETTINGS['bandwidth_scale'])
    model = fit_generator(basis, dt=data['dt'], regularization=selected['regularization'])
    selection = select_independent_phases(model)
    encoded = transform_generator(model, p['z'])
    selected_ids = np.asarray(selection['indices'], dtype=int)
    train_phases = np.angle(model['train_eigenfunctions'][:, selected_ids])
    query_phases = np.angle(encoded['eigenfunctions'][:, selected_ids])
    coefficients = readout(model['train_eigenfunctions'], p['x'], basis['weights'])
    predicted_windows = np.real(multiply(encoded['eigenfunctions'], coefficients))
    raw_windows = predicted_windows * p['normal'] * np.tile(p['scale'], data['lag']) + np.tile(p['mean'], data['lag'])
    reconstruction = fold(raw_windows, data['train'].shape[1], data['lag']) + p['slow']
    # Persist every prediction before reading any reference labels for scoring.
    np.savez_compressed(OUT / (name + '.npz'), train_phases=train_phases,
        query_phases=query_phases, reconstruction=reconstruction,
        train_window_indices=p['train_indices'], query_window_indices=p['query_indices'],
        snapshot_indices=p['snapshot_indices'], selected_eigenfunction_indices=selected_ids,
        eigenvalues=model['eigenvalues'], dirichlet_energy=model['dirichlet_energy'])
    frozen_hash = digest(OUT / (name + '.npz'))
    phase_count = len(selected_ids)
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        matched = match_phases(train_phases, evaluator['theta'][p['train_indices']]) if phase_count else None
    primitive = bool(matched is not None and phase_count == evaluator['theta'].shape[1]
                     and matched['absolute_determinant'] == 1)
    diagnostic_error = None
    signed_permutation = False
    if matched is not None:
        target = multiply(evaluator['theta'][p['query_indices']], matched['winding_candidate'].T) + matched['origins']
        diagnostic_error = float(np.mean(np.abs(np.angle(np.exp(1j * (query_phases - target))))))
        matrix = np.asarray(matched['winding_candidate'])
        signed_permutation = bool(primitive and np.isin(matrix, [-1, 0, 1]).all()
            and (np.abs(matrix).sum(axis=0) == 1).all() and (np.abs(matrix).sum(axis=1) == 1).all())
    residual = (reconstruction - evaluator['clean_total'][p['snapshot_indices']]) / evaluator['signal_train_std']
    result = dict(name=name, method='giannakis2019', cache_key=key, version=version,
        reference=dict(title='Data-driven spectral decomposition and forecasting of ergodic dynamical systems',
            author='Dimitrios Giannakis', year=2019, url='https://arxiv.org/html/1507.02338v4',
            algorithms=[1, 2], equations=[21, 23, 24, 26, 29, 32, 33]),
        selected=selected, validation=validation,
        phase_dimension=phase_count, phase_status='evaluated' if primitive else 'phase_basis_unresolved',
        phase_mae_rad=diagnostic_error if primitive else None,
        diagnostic_phase_mae_rad=diagnostic_error, independent_phase_basis=primitive,
        gauge_is_signed_permutation=signed_permutation,
        train_phase_evaluation=matched, phase_selection=selection,
        sensor_nrmse=float(np.sqrt(np.mean(residual ** 2))),
        diffusion_metadata=basis['metadata'], generator_metadata=model['metadata'],
        preprocessing=p['preprocessing'], elapsed_seconds=time.monotonic() - started,
        evaluation=dict(ntrain=len(data['train']), nquery=len(data['query']), lag=data['lag'], dt=data['dt'],
            train_observations_sha256=digest(data['train'], '<f8'),
            query_observations_sha256=digest(data['query'], '<f8'),
            snapshot_indices_sha256=digest(p['snapshot_indices'], '<i8'),
            window_indices_sha256=digest(p['query_indices'], '<i8'),
            snapshot_count=len(p['snapshot_indices']), window_count=len(p['query_indices']),
            frozen_predictions_sha256=frozen_hash, predictions_frozen_before_truth_evaluation=True,
            query_refit=False, truth_used_in_fit=False, truth_used_in_selection=False,
            supplied_true_phase_count=False, shared_training_selected_window_lag=True,
            reconstruction_is_observed_block_not_forecast=True,
            signal_readout='weighted Gram least squares in full regularized Koopman eigenfunction basis',
            phase_metric='wrapped MAE in TRAIN-aligned recovered integer gauge; null unless count matches and determinant is one',
            finite_primitive_match_is_not_global_certificate=True))
    write_json(destination, result)
    print('DONE', name, json.dumps({key: result[key] for key in
        ('phase_dimension', 'phase_status', 'phase_mae_rad', 'sensor_nrmse', 'elapsed_seconds')}), flush=True)
    return jsonable(result)


def summarize():
    results = [json.loads((OUT / (name + '.json')).read_text())
               for name in CASES if (OUT / (name + '.json')).exists()]
    write_json(OUT / 'summary.json', dict(method='giannakis2019', settings=SETTINGS, records=results))
    with (HERE / 'giannakis_results.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=('name', 'phase_dimension', 'phase_status',
            'phase_mae_rad', 'sensor_nrmse', 'gauge_is_signed_permutation'))
        writer.writeheader()
        for result in results:
            writer.writerow({key: result[key] for key in writer.fieldnames})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', nargs='+', choices=CASES, default=list(CASES))
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        for name in args.cases:
            run(name, resume=args.resume)
            summarize()


if __name__ == '__main__':
    main()
