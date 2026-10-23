"""Verify frozen Giannakis predictions independently; never fit an estimator.

Regenerate the original observations, check exact evaluation indices and
provenance, and recompute all reported scores from the frozen NPZ predictions.
Unresolved/nonprimitive phase outputs remain unresolved in this report.
"""
import os
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
              'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'

import argparse
import ast
import hashlib
import importlib.metadata
import itertools
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'experiments'
DATA = BASE / 'data'
OUT = DATA / 'giannakis'
CASES = ('toy_sine', 'toy_torus', 'scalar_noise10', 'two_waves',
         'noise00', 'noise01', 'noise05', 'noise10')
sys.path.insert(0, str(ROOT / 'code/lag_ssa'))
from fields_first_sensor_cases import generate_variant


def file_digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_digest(value, dtype):
    return hashlib.sha256(np.asarray(value, dtype=dtype).tobytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def signed_permutation(matrix):
    value = np.asarray(matrix)
    return bool(value.ndim == 2 and value.shape[0] == value.shape[1]
                and np.isin(value, (-1, 0, 1)).all()
                and (np.abs(value).sum(axis=0) == 1).all()
                and (np.abs(value).sum(axis=1) == 1).all())


def finite_product(left, right):
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        value = left @ right
    assert np.isfinite(value).all()
    return value


def phase_match(angles, reference):
    """Independent implementation of the article's finite character metric."""
    labels = np.asarray([row for row in itertools.product(
        range(-4, 5), repeat=reference.shape[1]) if any(row)], dtype=int)
    characters = np.exp(1j * finite_product(reference, labels.T))
    # Some macOS BLAS versions leave floating-point flags on finite products;
    # validate the actual result explicitly, as the preprocessing code does.
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        correlations = np.exp(1j * angles).T @ characters.conj() / len(angles)
    assert np.isfinite(correlations).all()
    best = np.argmax(np.abs(correlations), axis=1)
    winding = labels[best]
    origins = np.angle(correlations[np.arange(angles.shape[1]), best])
    determinant = (abs(int(round(np.linalg.det(winding))))
                   if winding.shape == (reference.shape[1], reference.shape[1]) else None)
    error = np.angle(np.exp(1j * (angles - finite_product(reference, winding.T) - origins)))
    return dict(winding_candidate=winding, origins=origins,
                absolute_determinant=determinant,
                phase_mae_rad=float(np.mean(np.abs(error))),
                phase_max_abs_rad=float(np.max(np.abs(error))),
                coherence=np.abs(correlations[np.arange(angles.shape[1]), best]))


def source_inputs(name):
    """Load source arrays without importing the runner or any estimator."""
    source_path = DATA / (name + '.json')
    source = read(source_path)
    if name.startswith('toy_'):
        array_path = DATA / (name + '_data.npz')
        with np.load(array_path, allow_pickle=False) as saved:
            y, clean, theta = (saved[key].copy() for key in
                               ('Y', 'clean', 'reference_theta'))
            snapshots = saved['raw_time_indices'].copy()
            windows = saved['window_raw_time_indices'].copy()
            train_windows = saved['train_window_raw_time_indices'].copy()
        ntrain, nquery, lag, dt = (source[key] for key in
                                  ('N_train', 'N_query', 'lag', 'dt'))
        std = clean[:ntrain].std(axis=0)
        input_sources = {str(array_path.relative_to(ROOT)): file_digest(array_path)}
    else:
        ntrain, nquery = source['ntrain'], source['ntest']
        config = {key: source[key] for key in
                  ('case', 'seed', 'noise', 'trend', 'phase_mode', 'rate_profile')}
        config['trend_strength'] = source.get('generator_metadata', {}).get('trend_strength', .3)
        y, truth = generate_variant(n=ntrain + nquery, train_count=ntrain, **config)
        clean, theta, std = (truth[key] for key in
                             ('clean_total', 'theta', 'signal_train_std'))
        report = source['transform']
        snapshots = np.arange(report['raw_time_first'], report['raw_time_last'] + 1)
        windows = np.arange(report['raw_time_first'],
                            report['raw_time_first'] + report['retained_window_count'])
        lag, dt = source['fit']['selected_lag'], 1.
        train_windows = np.arange(95, ntrain - 95 - lag + 1)
        assert len(snapshots) == report['retained_snapshot_count']
        input_sources = {}
    # The common preparation trims 31 MSSA samples plus 64 filter samples.
    np.testing.assert_array_equal(snapshots, np.arange(ntrain + 95, ntrain + nquery - 95))
    np.testing.assert_array_equal(windows, snapshots[:len(snapshots) - lag + 1])
    np.testing.assert_array_equal(train_windows, np.arange(95, ntrain - 95 - lag + 1))
    assert len(y) == ntrain + nquery and len(theta) == len(y) and clean.shape == y.shape
    return dict(source=source, source_path=source_path, y=y, clean=clean, theta=theta,
                std=std, ntrain=ntrain, nquery=nquery, lag=lag, dt=dt,
                snapshots=snapshots, windows=windows, train_windows=train_windows,
                input_sources=input_sources)


def estimator_interface_audit():
    """Check the estimator API has no oracle inputs and scoring follows freezing.

    This is a narrow source audit, not a general information-flow proof. Input
    hashes and metric recomputation provide the separate numerical checks.
    """
    estimator_path = ROOT / 'code/lag_ssa/giannakis_koopman.py'
    runner_path = BASE / 'run_giannakis_comparison.py'
    estimator_tree, runner_tree = (ast.parse(path.read_text()) for path in
                                   (estimator_path, runner_path))
    forbidden = {'truth', 'theta', 'reference_theta', 'reference_r', 'true_phase_count'}
    functions = {node.name: node for node in estimator_tree.body
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    api = ('fit_diffusion_basis', 'fit_generator', 'select_independent_phases')
    arguments = {name: [arg.arg for arg in functions[name].args.args] for name in api}
    assert not any(forbidden.intersection(names) for names in arguments.values())
    run = next(node for node in runner_tree.body
               if isinstance(node, ast.FunctionDef) and node.name == 'run')
    save_calls = [node for node in ast.walk(run) if isinstance(node, ast.Call)
                  and isinstance(node.func, ast.Attribute)
                  and node.func.attr == 'savez_compressed']
    assert len(save_calls) == 1
    evaluator_reads = [node for node in ast.walk(run)
                       if isinstance(node, ast.Name) and node.id == 'evaluator'
                       and isinstance(node.ctx, ast.Load)]
    assert evaluator_reads and min(node.lineno for node in evaluator_reads) > save_calls[0].lineno
    select = next(node for node in runner_tree.body
                  if isinstance(node, ast.FunctionDef) and node.name == 'select_model')
    assert not any(isinstance(node, ast.Name) and node.id in forbidden | {'evaluator'}
                   for node in ast.walk(select))
    return dict(scope='estimator signatures, TRAIN selection, and prediction-freeze/scoring source order',
                estimator_arguments=arguments, oracle_arguments_absent=True,
                query_labels_absent_from_model_selection=True,
                evaluator_reads_follow_prediction_save=True,
                static_audit_is_not_general_information_flow_proof=True)


def verify(name):
    data = source_inputs(name)
    result_path, array_path = OUT / (name + '.json'), OUT / (name + '.npz')
    result = read(result_path)
    with np.load(array_path, allow_pickle=False) as saved:
        arrays = {key: saved[key].copy() for key in saved.files}
    checks = {}

    def check(key, condition):
        checks[key] = bool(condition)

    version, evaluation = result['version'], result['evaluation']
    check('source_json_hash', version['source_sha256'] == file_digest(data['source_path']))
    check('toy_input_source_hash', version.get('toy_input_sha256') == (
        file_digest(DATA / (name + '_data.npz')) if name.startswith('toy_') else None))
    expected_key = hashlib.sha256(json.dumps(version, sort_keys=True).encode()).hexdigest()
    check('cache_key', result['cache_key'] == expected_key)
    source_checks = {path: file_digest(ROOT / path) == expected
                     for path, expected in version['code_sha256'].items()}
    check('all_recorded_code_hashes', all(source_checks.values()))
    package_checks = {name: importlib.metadata.version(name) == expected
                      for name, expected in version['packages'].items()}
    check('package_versions', all(package_checks.values()))
    check('frozen_prediction_hash', evaluation['frozen_predictions_sha256'] == file_digest(array_path))
    ntrain = data['ntrain']
    hashes = dict(train_observations_sha256=array_digest(data['y'][:ntrain], '<f8'),
                  query_observations_sha256=array_digest(data['y'][ntrain:], '<f8'),
                  snapshot_indices_sha256=array_digest(data['snapshots'], '<i8'),
                  window_indices_sha256=array_digest(data['windows'], '<i8'))
    for key, expected in hashes.items():
        check(key, evaluation[key] == expected)
    for key, expected in (('train_window_indices', data['train_windows']),
                          ('query_window_indices', data['windows']),
                          ('snapshot_indices', data['snapshots'])):
        check(key, np.array_equal(arrays[key], expected))
    check('split_lag_dt_and_counts', all(evaluation[key] == value for key, value in
        dict(ntrain=ntrain, nquery=data['nquery'], lag=data['lag'], dt=data['dt'],
             snapshot_count=len(data['snapshots']), window_count=len(data['windows'])).items()))
    for key, expected in dict(predictions_frozen_before_truth_evaluation=True,
        query_refit=False, truth_used_in_fit=False, truth_used_in_selection=False,
        supplied_true_phase_count=False, shared_training_selected_window_lag=True,
        reconstruction_is_observed_block_not_forecast=True).items():
        check(key, evaluation[key] is expected)
    check('frozen_diffusion_and_generator', result['diffusion_metadata']['query_refit'] is False
          and result['generator_metadata']['query_refit'] is False)
    check('phase_selection_has_no_oracle_dimension',
          result['phase_selection']['metadata']['truth_used'] is False
          and result['phase_selection']['metadata']['phase_dimension_prescribed'] is False)
    candidates = result['validation']['candidates']
    selected = min(candidates, key=lambda row: (row['selection_loss'], row['n_basis'], row['regularization']))
    check('selected_train_validation_minimum', selected == result['selected'])
    check('selection_grid', {(row['n_basis'], row['regularization']) for row in candidates}
          == set(itertools.product(version['settings']['basis_sizes'], version['settings']['regularizations'])))
    check('selected_generator_parameters', all(result['generator_metadata'][key] == selected[key]
                                              for key in ('n_basis', 'regularization')))
    preprocessing = result['preprocessing']
    check('shared_preprocessing', preprocessing['rank'] == data['source']['fit']['preprocessing']['rank']
          and preprocessing['lag'] == 32 and preprocessing['highpass_window'] == 129
          and preprocessing['highpass_polyorder'] == 2 and preprocessing['boundary_trim'] == 95
          and preprocessing['query_refit'] is False)
    check('shared_preprocessing_cv', preprocessing['cv_selected']
          == data['source']['fit']['preprocessing']['cv_selected'])

    reconstruction = arrays['reconstruction']
    phase_count = result['phase_dimension']
    train_phase, query_phase = arrays['train_phases'], arrays['query_phases']
    check('prediction_shapes', reconstruction.shape == (len(data['snapshots']), data['y'].shape[1])
          and train_phase.shape == (len(data['train_windows']), phase_count)
          and query_phase.shape == (len(data['windows']), phase_count))
    check('finite_predictions', all(np.isfinite(arrays[key]).all() for key in
          ('reconstruction', 'train_phases', 'query_phases', 'eigenvalues', 'dirichlet_energy')))
    ids = arrays['selected_eigenfunction_indices']
    check('selected_phase_ids', np.array_equal(ids, result['phase_selection']['indices'])
          and len(ids) == phase_count and len(np.unique(ids)) == len(ids)
          and ((ids >= 0) & (ids < len(arrays['eigenvalues']))).all())
    normalized = (reconstruction - data['clean'][data['snapshots']]) / data['std']
    sensor_nrmse = float(np.sqrt(np.mean(normalized ** 2)))
    check('recomputed_sensor_nrmse', np.isclose(sensor_nrmse, result['sensor_nrmse'], rtol=1e-12, atol=1e-14))
    matched, diagnostic_error, primitive, permutation = None, None, False, False
    if phase_count:
        matched = phase_match(train_phase, data['theta'][data['train_windows']])
        recorded = result['train_phase_evaluation']
        for key in ('winding_candidate', 'origins', 'coherence', 'phase_mae_rad', 'phase_max_abs_rad'):
            check('train_phase_match_' + key, np.allclose(matched[key], recorded[key], rtol=1e-10, atol=1e-12))
        check('train_phase_match_determinant', matched['absolute_determinant'] == recorded['absolute_determinant'])
        target = finite_product(data['theta'][data['windows']], matched['winding_candidate'].T) + matched['origins']
        diagnostic_error = float(np.mean(np.abs(np.angle(np.exp(1j * (query_phase - target))))))
        primitive = bool(phase_count == data['theta'].shape[1] and matched['absolute_determinant'] == 1)
        permutation = primitive and signed_permutation(matched['winding_candidate'])
        check('recomputed_diagnostic_phase_mae', np.isclose(diagnostic_error,
              result['diagnostic_phase_mae_rad'], rtol=1e-12, atol=1e-14))
    else:
        check('empty_phase_result_has_no_match_or_error', result['train_phase_evaluation'] is None
              and result['diagnostic_phase_mae_rad'] is None)
    check('primitive_phase_flag', result['independent_phase_basis'] is primitive)
    check('signed_permutation_flag', result['gauge_is_signed_permutation'] is bool(permutation))
    check('phase_status_preserves_failure', result['phase_status']
          == ('evaluated' if primitive else 'phase_basis_unresolved'))
    check('phase_score_preserves_failure',
          bool(np.isclose(result['phase_mae_rad'], diagnostic_error, rtol=1e-12, atol=1e-14))
          if primitive else result['phase_mae_rad'] is None)
    frozen_baseline_checks = None
    if name in ('scalar_noise10', 'two_waves'):
        baseline = next(row for row in read(DATA / 'baseline_compatibility.json')['records']
                        if row['name'] == name)
        frozen_baseline_checks = {key: expected == baseline['hashes'][key] for key, expected in hashes.items()}
        check('existing_baseline_inputs_and_indices', all(frozen_baseline_checks.values()))
        check('existing_baseline_lag', data['lag'] == baseline['selected_lag'])
    assert all(checks.values()), (name, {key: value for key, value in checks.items() if not value})
    return dict(name=name, checks=checks, source_code_checks=source_checks,
                package_checks=package_checks, original_input_array_sources=data['input_sources'],
                hashes=hashes, existing_baseline_checks=frozen_baseline_checks,
                phase_dimension=phase_count, reference_phase_dimension=data['theta'].shape[1],
                phase_status=result['phase_status'], phase_mae_rad=result['phase_mae_rad'],
                diagnostic_phase_mae_rad=diagnostic_error, sensor_nrmse=sensor_nrmse,
                winding_candidate=None if matched is None else matched['winding_candidate'].tolist(),
                absolute_determinant=None if matched is None else matched['absolute_determinant'],
                gauge_is_signed_permutation=bool(permutation),
                phase_error_directly_comparable_in_original_coordinate_gauge=bool(permutation),
                phase_metric_caveat=('Recovered mixed unimodular gauge: coordinate-averaged MAE is basis-dependent.'
                    if primitive and not permutation else 'Phase basis unresolved; headline phase score remains null.'
                    if not primitive else 'Signed permutation and TRAIN origin alignment preserve coordinate-averaged MAE.'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', nargs='+', choices=CASES, default=list(CASES))
    parser.add_argument('--output', type=Path, default=BASE / 'giannakis_verification.json')
    args = parser.parse_args()
    records = [verify(name) for name in args.cases]
    output = dict(scope='Independent verification of frozen predictions; no estimator or preprocessing fitting.',
                  estimator_fits_performed=False, preprocessing_fits_performed=False,
                  code_audit=estimator_interface_audit(),
                  complete_eight_case_comparison=set(args.cases) == set(CASES),
                  metrics_units='Sensor NRMSE is clean oscillatory TRAIN-channel-standard-deviation normalized; phase MAE is radians.',
                  phase_metric_policy='Finite TRAIN character alignment in [-4,4]^r; count and absolute determinant one required. Mixed unimodular gauges remain basis-dependent; unresolved scores stay null.',
                  records=records)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')
    print(args.output)
    for row in records:
        print(row['name'], row['phase_status'], row['phase_mae_rad'], row['sensor_nrmse'])


if __name__ == '__main__':
    main()
