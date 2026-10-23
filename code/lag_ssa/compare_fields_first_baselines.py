"""Compare successful saved runs on the same regenerated observations and indices.

SSA/MSSA are frozen training-subspace denoisers. Circular/toroidal coordinates
are phase baselines, with no decoder and consequently no reconstruction score.
All computations are sequential and restricted to one numerical thread.
"""
import os
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_name] = '1'

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits

from evaluate_tangent_overlap_angles import match_phases
from fields_first_preprocess import (fit_preprocess, transform_preprocess,
                                     highpass_decompose)
from fields_first_sensor_cases import generate_variant
from tangent_overlap import windows


METHODS = ('ssa', 'mssa', 'circular', 'toroidal')


def errors(prediction, target, train_std):
    difference = np.asarray(prediction) - np.asarray(target)
    scaled = difference / np.maximum(train_std, 1e-12)
    return dict(train_channel_std_normalized_rmse=float(np.sqrt(np.mean(scaled ** 2))),
        per_channel_train_std_normalized_rmse=np.sqrt(np.mean(scaled ** 2, axis=0)),
        raw_relative_error=float(np.linalg.norm(difference) / max(np.linalg.norm(target), 1e-15)))


def _indices(source):
    """Saved evaluator uses contiguous snapshot and window-start indices."""
    report = source['transform']
    first, last = int(report['raw_time_first']), int(report['raw_time_last'])
    snapshots = np.arange(first, last + 1)
    if len(snapshots) != report['retained_snapshot_count']:
        raise ValueError('source does not describe a contiguous evaluation block')
    lag = int(source['fit']['selected_lag'])
    starts = snapshots[:len(snapshots) - lag + 1]
    if len(starts) != report['retained_window_count']:
        raise ValueError('saved window count disagrees with selected lag')
    return snapshots, starts, lag


def _selected_dimension(source):
    for attempt in reversed(source['fit']['attempts']):
        if attempt.get('gates', {}).get('passed'):
            return attempt.get('diagnostics', {}).get('dimension_selected')
    return None


def _linear_baseline(train, query, *, separate, start):
    groups = [slice(c, c + 1) for c in range(train.shape[1])] if separate else [slice(None)]
    predictions, models = [], []
    fit_seconds, encode_seconds = 0., 0.
    for group in groups:
        began = time.monotonic()
        model = fit_preprocess(train[:, group], lag=32, maxrank=24,
                               trend_degrees=(0,))
        fit_seconds += time.monotonic() - began
        began = time.monotonic()
        predictions.append(transform_preprocess(model, query[:, group],
                                                start_index=start)['denoised'])
        encode_seconds += time.monotonic() - began
        models.append(model)
    return np.column_stack(predictions), models, dict(fit_seconds=fit_seconds,
        encode_seconds=encode_seconds, fit_seconds_scope='training CV and full-training subspace fit',
        encode_seconds_scope='frozen subspace projection and overlap averaging on the whole query block')


class FrozenCoordinates:
    """Native dreimac fit, followed by evaluation of its frozen classifying map.

    dreimac 0.3.2 does not return fitted harmonic coefficients. A temporary
    wrapper records the outputs of its own LS helper during the TRAIN call;
    it changes neither the solve nor its result. Query evaluation calls only
    the package's integration routine with fixed coefficients and cover scale.
    """
    def __init__(self, topology, classes):
        import dreimac.toroidalcoords as implementation
        self.topology = topology
        self.representatives = []
        native = implementation._integrate_harmonic_representative

        def record(*args, **kwargs):
            result = native(*args, **kwargs)
            self.representatives.append(tuple(np.array(value, copy=True) for value in result))
            return result

        with patch.object(implementation, '_integrate_harmonic_representative', record):
            self.train_phases = np.asarray(topology.get_coordinates(
                perc=.5, cocycle_idxs=list(classes), standard_range=True,
                check_cocycle_condition=True)).T
        self.radius = float(topology._r_cover)
        self.threshold = float(topology._rips_threshold)
        self.change_basis = (np.array(topology._change_basis, copy=True)
                             if len(classes) > 1 else np.eye(1))
        if len(self.representatives) != len(classes):
            raise RuntimeError('native coordinate fit did not expose every harmonic representative')

    def transform(self, distances):
        from dreimac.toroidalcoords import _sparse_integrate
        from dreimac.utils import CircleMapUtils
        distances = np.asarray(distances)
        membership = distances < self.radius
        weights = np.maximum(self.radius - distances, 0.)
        totals = weights.sum(axis=0)
        covered = totals > 0
        weights /= np.where(covered, totals, 1.)[None]
        roots = np.argmax(membership, axis=0)
        theta = np.asarray([_sparse_integrate(harmonic, integral, weights, roots,
            self.topology._dist_land_land, self.threshold,
            self.topology._cns_lookup_table) for harmonic, integral in self.representatives])
        theta = CircleMapUtils.linear_combination(theta, self.change_basis).T
        return dict(phases=theta, covered=covered)


def _phase_baselines(train_split, query_split, lag, *, names, nlandmarks):
    from dreimac import ToroidalCoords
    from fields_first_periods import SETTINGS, _select_prefix
    began = time.monotonic()
    take = np.flatnonzero(train_split['valid_mask'])
    qtake = np.flatnonzero(query_split['valid_mask'])
    train = train_split['residual'][take]
    query = query_split['residual'][qtake]
    mean, rawscale = train.mean(axis=0), train.std(axis=0)
    scale = np.where(rawscale > rawscale.max() * 1e-12, rawscale, 1.)
    normal = np.sqrt(lag * train.shape[1])
    x = windows((train - mean) / scale, lag) / normal
    z = windows((query - mean) / scale, lag) / normal
    geometry_seconds = time.monotonic() - began
    began = time.monotonic()
    metric = cdist(x, x)
    resolutions = [min(nlandmarks, len(x))]
    if nlandmarks == 192 and len(x) > 192:
        resolutions.append(min(384, len(x)))
    attempts = []
    for resolution in resolutions:
        topology = ToroidalCoords(metric, n_landmarks=resolution,
                                  distance_matrix=True, prime=47, maxdim=1, verbose=False)
        r, selection = _select_prefix(topology._dgms[1], SETTINGS)
        attempts.append(dict(n_landmarks=resolution, selected_dimension=r, selection=selection))
        if r is not None:
            break
    persistence_seconds = time.monotonic() - began
    info = dict(phase_dimension=r, selection=selection, n_landmarks=topology._n_landmarks,
        prime=47, cover_percent=.5, selected_window_lag=lag,
        dimension_rule='training persistent H1 count under the torus hypothesis',
        preprocessing='same joint MSSA and fixed highpass as fields-first',
        shared_window_geometry_seconds=geometry_seconds,
        shared_training_persistence_seconds=persistence_seconds,
        topology_resolution_attempts=attempts,
        topology_resolution_rule='retry 192 landmarks once at 384 only if no dimension is selected; unchanged persistence gates',
        query_refit=False, reconstruction_decoder=None)
    if r is None or r >= x.shape[1]:
        return {name: dict(status='unresolved_training_dimension', metadata=info) for name in names}
    began = time.monotonic()
    distances = cdist(x[topology._idx_land], z)
    info['shared_query_landmark_distances_seconds'] = time.monotonic() - began
    results = {}
    for name in names:
        try:
            began = time.monotonic()
            if name == 'toroidal':
                fitted = [FrozenCoordinates(topology, range(r))]
            else:
                fitted = [FrozenCoordinates(topology, [j]) for j in range(r)]
            fit_seconds = time.monotonic() - began
            began = time.monotonic()
            predictions = [model.transform(distances) for model in fitted]
            encode_seconds = time.monotonic() - began
            began = time.monotonic()
            # Check the adapter against the exact native output before querying.
            adapter_error = max(float(np.max(np.abs(np.angle(np.exp(1j * (
                model.transform(topology._dist_land_data)['phases'] - model.train_phases))))))
                for model in fitted)
            if adapter_error > 1e-10:
                raise ValueError('frozen extension disagrees with native train coordinates')
            covered = np.logical_and.reduce([value['covered'] for value in predictions])
            results[name] = dict(status='frozen_prediction' if covered.all() else 'unresolved_query_coverage',
                metadata=dict(info, cover_radii=[model.radius for model in fitted],
                    fit_seconds=fit_seconds, encode_seconds=encode_seconds,
                    fit_seconds_scope='native harmonic coordinate fits; shared preprocessing and persistence reported separately',
                    encode_seconds_scope='fixed classifying map on query windows; shared query preparation reported separately',
                    adapter_verification_seconds=time.monotonic() - began,
                    native_train_adapter_max_error_rad=adapter_error,
                    query_coverage=float(covered.mean()),
                    lattice_reduction=name == 'toroidal',
                    frozen_harmonic_coefficients=True),
                train_phases=np.column_stack([model.train_phases for model in fitted]),
                phases=np.column_stack([value['phases'] for value in predictions]),
                train_indices=train_split['time_index'][take][:len(x)].astype(int),
                query_indices=query_split['time_index'][qtake][:len(z)].astype(int))
        except Exception as exc:
            results[name] = dict(status='unresolved_coordinate_fit', metadata=info,
                                 error_type=type(exc).__name__, error=str(exc))
    return results


def _score_phases(frozen, truth, expected_indices):
    result = dict(method=frozen['method'], status=frozen['status'],
        metadata=frozen['metadata'], phase_dimension=frozen['metadata']['phase_dimension'],
        reconstruction_rmse=None, phase_mae_rad=None, absolute_determinant=None)
    result.update(fit_seconds=frozen['metadata'].get('fit_seconds'),
                  encode_seconds=frozen['metadata'].get('encode_seconds'))
    if frozen['status'] != 'frozen_prediction':
        result.update({key: frozen[key] for key in ('error', 'error_type') if key in frozen})
        return result
    if not np.array_equal(frozen['query_indices'], expected_indices):
        raise ValueError('topological evaluation indices differ from saved field-run indices')
    increments = np.angle(np.exp(1j * np.diff(frozen['train_phases'], axis=0)))
    mean_step = np.angle(np.mean(np.exp(1j * increments), axis=0))
    step_errors = np.angle(np.exp(1j * (increments - mean_step)))
    result['train_phase_step_rmse_rad'] = float(np.sqrt(np.mean(step_errors ** 2)))
    # Generator phases first enter here, after every baseline prediction is frozen.
    matched = match_phases(frozen['train_phases'], truth['theta'][frozen['train_indices']])
    target = truth['theta'][expected_indices] @ matched['winding_candidate'].T + matched['origins']
    delta = np.angle(np.exp(1j * (frozen['phases'] - target)))
    same_dimension = frozen['phases'].shape[1] == truth['theta'].shape[1]
    primitive = same_dimension and matched['absolute_determinant'] == 1
    result.update(status='evaluated' if primitive else 'phase_basis_not_unimodular',
        phase_mae_rad=float(np.mean(np.abs(delta))),
        phase_rmse_rad=float(np.sqrt(np.mean(delta ** 2))),
        absolute_determinant=matched['absolute_determinant'],
        phase_dimension_matches_truth=same_dimension,
        independent_phase_basis=primitive, train_phase_evaluation=matched)
    return result


def compare(source_path, methods=METHODS, nlandmarks=192, generator_evidence=None):
    source_path = Path(source_path)
    source = json.loads(source_path.read_text())
    if source.get('transform_status') != 'transformed_conditionally':
        raise ValueError('comparison requires a successful saved end-to-end source run')
    idx, widx, lag = _indices(source)
    config = {key: source[key] for key in ('case', 'ntrain', 'ntest', 'seed', 'noise', 'trend')}
    config.update(phase_mode=source.get('phase_mode', 'rigid'),
                  rate_profile=source.get('rate_profile', 'default'),
                  trend_strength=source.get('generator_metadata', {}).get('trend_strength', .3))
    ntrain, ntest = config.pop('ntrain'), config.pop('ntest')
    y, evaluator = generate_variant(n=ntrain + ntest, train_count=ntrain, **config)
    train, query = y[:ntrain], y[ntrain:]
    predictions, metadata, phase_predictions = {}, {}, {}
    mssa_model = None
    if 'ssa' in methods:
        predictions['ssa'], models, timings = _linear_baseline(train, query, separate=True, start=ntrain)
        metadata['ssa'] = dict(selected_ranks=[model['rank'] for model in models],
            selected_cv_mse=[model['cv']['selected']['cv_mse_raw'] for model in models],
            **timings)
    if any(name in methods for name in ('mssa', 'circular', 'toroidal')):
        predictions['mssa'], models, timings = _linear_baseline(train, query, separate=False, start=ntrain)
        mssa_model = models[0]
        metadata['mssa'] = dict(selected_rank=mssa_model['rank'],
            selected_cv_mse=mssa_model['cv']['selected']['cv_mse_raw'],
            source_preprocessing_rank=source['fit']['preprocessing']['rank'],
            **timings)
    topology_names = [name for name in methods if name in ('circular', 'toroidal')]
    if topology_names:
        began = time.monotonic()
        train_split = highpass_decompose(mssa_model, train)
        train_split_seconds = time.monotonic() - began
        began = time.monotonic()
        query_split = highpass_decompose(mssa_model, query, start_index=ntrain)
        query_split_seconds = time.monotonic() - began
        try:
            phase_predictions = _phase_baselines(train_split, query_split, lag,
                names=topology_names, nlandmarks=nlandmarks)
        except Exception as exc:
            phase_predictions = {name: dict(status='unresolved_topology',
                metadata=dict(phase_dimension=None), error_type=type(exc).__name__,
                error=str(exc)) for name in topology_names}
        for frozen in phase_predictions.values():
            frozen['metadata'].update(shared_preprocessing_fit_seconds=metadata['mssa']['fit_seconds'],
                shared_training_filter_seconds=train_split_seconds,
                shared_query_filter_seconds=query_split_seconds)
    # No model fit below this point. Truth is used only for scoring.
    rows = [dict(method='fields_first', status=source['transform_status'],
        reconstruction_rmse=source['reconstruction_against_clean_total']['train_channel_std_normalized_rmse'],
        phase_mae_rad=source.get('test_phase_evaluation', {}).get('mean_absolute_rad'),
        absolute_determinant=source.get('train_phase_evaluation', {}).get('absolute_determinant'),
        phase_dimension=_selected_dimension(source), metrics_from_saved_source=True,
        fit_seconds=source['fit'].get('elapsed_fit_seconds'), encode_seconds=None,
        timing_scope='saved full fit; query elapsed time was not recorded'),
        dict(method='observed', status='evaluated', reconstruction_rmse=errors(
            query[idx - ntrain], evaluator['clean_total'][idx], evaluator['signal_train_std'])[
                'train_channel_std_normalized_rmse'], phase_mae_rad=None, phase_dimension=None)]
    for attempt in source['fit']['attempts']:
        if attempt.get('gates', {}).get('passed'):
            rows[0]['train_phase_step_rmse_rad'] = attempt.get('diagnostics', {}).get('train_phase_step_rmse')
    for name in ('ssa', 'mssa'):
        if name not in methods:
            continue
        metrics = errors(predictions[name][idx - ntrain], evaluator['clean_total'][idx],
                         evaluator['signal_train_std'])
        if name == 'mssa' and 'preprocessed_against_clean_total' in source:
            difference = abs(metrics['train_channel_std_normalized_rmse'] - source[
                'preprocessed_against_clean_total']['train_channel_std_normalized_rmse'])
            metadata[name]['saved_conditioner_rmse_absolute_difference'] = difference
            metadata[name]['reproduces_saved_conditioner_rmse'] = difference < 1e-10
        rows.append(dict(method=name, status='evaluated', reconstruction_rmse=metrics[
            'train_channel_std_normalized_rmse'], reconstruction_metrics=metrics,
            phase_mae_rad=None, phase_dimension=None,
            fit_seconds=metadata[name]['fit_seconds'], encode_seconds=metadata[name]['encode_seconds'],
            metadata=dict(metadata[name], lag=32, maxrank=24, trend_degree=0,
                cv_folds=2, holdout_fraction=.1, completion_iterations=10,
                selection='minimum training masked raw-entry MSE; ranks 0..24',
                query_subspace_refit=False, physical_trend_identified=False)))
    for name, frozen in phase_predictions.items():
        frozen['method'] = name
        rows.append(_score_phases(frozen, evaluator, widx))
    manifest_path = source_path.parent / 'source_sha256.json'
    generator = Path(__file__).with_name('fields_first_sensor_cases.py')
    current_hash = hashlib.sha256(generator.read_bytes()).hexdigest()
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    saved_hash = manifest.get('code/lag_ssa/fields_first_sensor_cases.py')
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    train_hash = hashlib.sha256(train.astype('<f8').tobytes()).hexdigest()
    query_hash = hashlib.sha256(query.astype('<f8').tobytes()).hexdigest()
    equivalence_verified = None
    if generator_evidence is not None:
        evidence = json.loads(Path(generator_evidence).read_text())
        candidates = [row for row in evidence.get('runs', []) if row['source_sha256'] == source_hash]
        if candidates:
            row = candidates[0]
            equivalence_verified = bool(evidence['current_generator_sha256'] == current_hash
                and evidence['recovered_packaged_generator_sha256'] == saved_hash
                and row['observations_bitwise_equal']
                and all(row['evaluator_arrays_bitwise_equal'].values())
                and row['train_observations_sha256'] == train_hash
                and row['query_observations_sha256'] == query_hash)
            if not equivalence_verified:
                raise ValueError('saved generator-equivalence evidence does not match this comparison')
    return dict(source=str(source_path.resolve()), source_sha256=source_hash,
        configuration=dict(config, ntrain=ntrain, ntest=ntest),
        evaluation=dict(raw_time_first=int(idx[0]), raw_time_last=int(idx[-1]),
            snapshot_count=len(idx), window_count=len(widx), window_lag=lag,
            snapshot_indices_sha256=hashlib.sha256(idx.astype('<i8').tobytes()).hexdigest(),
            window_indices_sha256=hashlib.sha256(widx.astype('<i8').tobytes()).hexdigest(),
            train_observations_sha256=train_hash, query_observations_sha256=query_hash,
            generator_sha256=current_hash,
            source_generator_hash_matches_current=None if saved_hash is None else saved_hash == current_hash,
            source_generator_observations_bitwise_equivalent=equivalence_verified,
            generator_equivalence_evidence=None if equivalence_verified is None else str(Path(generator_evidence).resolve()),
            predictions_frozen_before_truth_evaluation=True,
            same_saved_query_indices=True, reconstruction_is_observed_block_not_forecast=True,
            topological_decoder=None, metrics_not_provided_by_a_method_are_null=True),
        maximum_numerical_threads=1, dreimac_version=importlib.metadata.version('dreimac'), rows=rows)


def _markdown(comparisons):
    lines = ['| Experiment (train/query) | r: ours / CC / TC | Reading NRMSE, %: ours / SSA / MSSA | Phase MAE, rad: ours / CC / TC |',
             '|---|---|---|---|']

    def number(row, key, scale=1., precision=5):
        value = row.get(key)
        return '-' if value is None else f'{scale * value:.{precision}f}'

    def phase(row):
        if not row:
            return '-'
        if row.get('phase_mae_rad') is None:
            return 'failed' if row.get('status', '').startswith('unresolved') else '-'
        if row.get('absolute_determinant') != 1 or row.get('status') == 'phase_basis_not_unimodular':
            return 'failed'
        return number(row, 'phase_mae_rad')

    for value in comparisons:
        rows = {row['method']: row for row in value['rows']}
        config = value['configuration']
        name = f"{Path(value['source']).stem} ({config['ntrain']}/{config['ntest']})"
        dimensions = ' / '.join(str(rows.get(method, {}).get('phase_dimension') or '-')
                                for method in ('fields_first', 'circular', 'toroidal'))
        readings = ' / '.join(number(rows.get(method, {}), 'reconstruction_rmse', 100., 3)
                              for method in ('fields_first', 'ssa', 'mssa'))
        phases = ' / '.join(phase(rows.get(method, {}))
                            for method in ('fields_first', 'circular', 'toroidal'))
        lines.append(f'| {name} | {dimensions} | {readings} | {phases} |')
    lines.extend(['',
        'All methods receive the same raw measurements, train/query split and evaluation indices; tuning uses training data only. '
        'SSA/MSSA use lag 32; circular (CC) and toroidal (TC) coordinates share our training-selected lag in this controlled comparison. '
        'Phase MAE measures uniform dynamical-phase recovery, not topological correctness; this benchmark is conditional on accepted fields-first runs. '
        'NRMSE uses each clean oscillatory channel\'s training standard deviation; - means unavailable and failed means no valid complete phase basis.'])
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sources', nargs='+', type=Path, help='saved result JSONs or directories')
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--only', nargs='+', help='source filename stems to include')
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    parser.add_argument('--n-landmarks', type=int, default=192)
    parser.add_argument('--generator-evidence', type=Path,
                        help='bitwise equivalence evidence for an older generator snapshot')
    parser.add_argument('--resume', action='store_true',
                        help='reuse complete results with identical source, methods, settings and code hashes')
    args = parser.parse_args()
    from run_fields_first_sensor import write_json
    if not 4 <= args.n_landmarks <= 384:
        parser.error('--n-landmarks must lie between 4 and 384')
    paths = sorted({path.resolve() for item in args.sources for path in
                    (item.glob('*.json') if item.is_dir() else [item])})
    successful = []
    for path in paths:
        if args.only and path.stem not in args.only:
            continue
        value = json.loads(path.read_text())
        if isinstance(value, dict) and value.get('transform_status') == 'transformed_conditionally':
            successful.append(path)
    if not successful:
        parser.error('no successful source runs selected')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    code = Path(__file__).resolve().parent
    version = dict(methods=list(args.methods), n_landmarks=args.n_landmarks,
        generator_evidence_sha256=None if args.generator_evidence is None else
            hashlib.sha256(args.generator_evidence.read_bytes()).hexdigest(),
        code_sha256={name: hashlib.sha256((code / name).read_bytes()).hexdigest() for name in
            ('compare_fields_first_baselines.py', 'fields_first_preprocess.py',
             'fields_first_sensor_cases.py', 'fields_first_periods.py',
             'evaluate_tangent_overlap_angles.py', 'tangent_overlap.py')},
        package_versions={name: importlib.metadata.version(name) for name in
                          ('numpy', 'scipy', 'scikit-learn', 'dreimac', 'ripser')})
    comparisons = []
    with threadpool_limits(limits=1):
        for path in successful:
            print('BASELINE', path.name, flush=True)
            source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            key = hashlib.sha256(json.dumps(dict(source_sha256=source_hash, **version),
                                            sort_keys=True).encode()).hexdigest()
            filename = path.stem + '_' + source_hash[:8] + '.baselines.json'
            destination = args.output_dir / filename
            cached = json.loads(destination.read_text()) if args.resume and destination.exists() else {}
            if cached.get('cache_key') == key:
                result = cached
                print('BASELINE_CACHED', path.name, flush=True)
            else:
                result = compare(path, args.methods, args.n_landmarks, args.generator_evidence)
                result.update(cache_key=key, comparison_version=version)
                write_json(destination, result)
            comparisons.append(result)
            write_json(args.output_dir / 'summary.json', comparisons)
            (args.output_dir / 'COMPARISON.md').write_text(_markdown(comparisons))
            print('BASELINE_DONE', path.name, flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
