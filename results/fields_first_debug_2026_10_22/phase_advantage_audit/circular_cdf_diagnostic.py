"""Two scalar diagnostics: native circular coordinates plus a frozen TRAIN CDF.

This empirical circular-CDF correction is a strengthened diagnostic baseline,
not an implementation or equivalence claim for the full Kralemann method.
"""
import os
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[name] = '1'

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits


def sha_array(array):
    return hashlib.sha256(np.asarray(array, dtype='<f8').tobytes()).hexdigest()


def fit_circular_cdf(train_phases):
    """Average ranks at ties, midpoint empirical CDF, periodic linear extension."""
    phase = np.asarray(train_phases)
    if phase.ndim != 2 or phase.shape[1] != 1 or not np.isfinite(phase).all():
        raise ValueError('CDF diagnostic requires one finite inferred circular coordinate')
    knots, counts = np.unique(np.mod(phase[:, 0], 2*np.pi), return_counts=True)
    if len(knots) < 3:
        raise ValueError('insufficient distinct circular coordinate samples')
    values = 2*np.pi*(np.cumsum(counts) - counts/2)/len(phase)
    return dict(knots=np.r_[knots[-1]-2*np.pi, knots, knots[0]+2*np.pi],
                values=np.r_[values[-1]-2*np.pi, values, values[0]+2*np.pi],
                train_count=len(phase), unique_count=len(knots))


def apply_circular_cdf(model, phase):
    value = np.asarray(phase)
    mapped = np.interp(np.mod(value[:, 0], 2*np.pi), model['knots'], model['values'])
    return np.mod(mapped, 2*np.pi)[:, None]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve(); out = args.output_dir.resolve()
    sys.path.insert(0, str(root/'code/lag_ssa'))
    from compare_fields_first_baselines import _indices, _phase_baselines, _score_phases
    from fields_first_preprocess import fit_preprocess, highpass_decompose
    from fields_first_sensor_cases import generate_variant
    from run_fields_first_sensor import jsonable
    previous = json.loads((root/'results/fields_first_debug_2026_10_22/baselines/summary.json').read_text())
    frozen_cases = []
    with threadpool_limits(limits=1):
        for name in ('scalar_clean', 'scalar_noise10'):
            print('CDF_FIT', name, flush=True)
            path = root/'results/fields_first_debug_2026_10_22/validation'/(name+'.json')
            source = json.loads(path.read_text())
            reference = next(item for item in previous if item['source'] == str(path))
            idx, widx, lag = _indices(source)
            config = {key: source[key] for key in ('case','seed','noise','trend')}
            config.update(phase_mode=source.get('phase_mode','rigid'),
                          rate_profile=source.get('rate_profile','default'),
                          trend_strength=source.get('generator_metadata',{}).get('trend_strength',.3))
            ntrain, ntest = source['ntrain'], source['ntest']
            observations, evaluator = generate_variant(n=ntrain+ntest,train_count=ntrain,**config)
            train_hash, query_hash = sha_array(observations[:ntrain]), sha_array(observations[ntrain:])
            assert train_hash == reference['evaluation']['train_observations_sha256']
            assert query_hash == reference['evaluation']['query_observations_sha256']
            started = time.monotonic()
            preprocess = fit_preprocess(observations[:ntrain],lag=32,maxrank=24,trend_degrees=(0,))
            train_split = highpass_decompose(preprocess, observations[:ntrain])
            query_split = highpass_decompose(preprocess, observations[ntrain:],start_index=ntrain)
            native = _phase_baselines(train_split,query_split,lag,names=['circular'],nlandmarks=192)['circular']
            native['method'] = 'native_circular'
            corrected = None; cdf = None
            if native['status'] == 'frozen_prediction':
                cdf = fit_circular_cdf(native['train_phases'])
                corrected = copy.deepcopy(native)
                corrected['method'] = 'native_circular_plus_train_empirical_cdf'
                corrected['train_phases'] = apply_circular_cdf(cdf,native['train_phases'])
                corrected['phases'] = apply_circular_cdf(cdf,native['phases'])
                corrected['metadata'].update(uniformization='midpoint empirical circular CDF with periodic linear interpolation',
                    uniformization_fit_input='TRAIN native circular-coordinate samples only',
                    hidden_phases_or_frequencies_used_for_uniformization=False,
                    cdf_train_count=cdf['train_count'],cdf_unique_count=cdf['unique_count'],
                    cdf_frozen_before_query_evaluation=True)
                assert np.isfinite(corrected['train_phases']).all() and np.isfinite(corrected['phases']).all()
            frozen_cases.append(dict(name=name,source=source,path=path,reference=reference,
                idx=idx,widx=widx,lag=lag,native=native,corrected=corrected,cdf=cdf,evaluator=evaluator,
                configuration=dict(config,ntrain=ntrain,ntest=ntest),fit_seconds=time.monotonic()-started,
                train_hash=train_hash,query_hash=query_hash))
        # All native maps, CDFs, TRAIN predictions and query predictions are now frozen.
        results = []
        out.mkdir(parents=True,exist_ok=True)
        for case in frozen_cases:
            native = _score_phases(case['native'],case['evaluator'],case['widx'])
            corrected = None if case['corrected'] is None else _score_phases(
                case['corrected'],case['evaluator'],case['widx'])
            old_native = next(row for row in case['reference']['rows'] if row['method']=='circular')
            if native['phase_mae_rad'] is not None:
                assert abs(native['phase_mae_rad']-old_native['phase_mae_rad']) < 1e-12
            result = dict(name=case['name'],configuration=case['configuration'],lag=case['lag'],
                source=str(case['path']),source_sha256=hashlib.sha256(case['path'].read_bytes()).hexdigest(),
                train_observations_sha256=case['train_hash'],query_observations_sha256=case['query_hash'],
                raw_arrays_match_main_comparison_hashes=True,
                snapshot_indices_sha256=hashlib.sha256(case['idx'].astype('<i8').tobytes()).hexdigest(),
                window_indices_sha256=hashlib.sha256(case['widx'].astype('<i8').tobytes()).hexdigest(),
                snapshot_count=len(case['idx']),query_window_count=len(case['widx']),
                all_predictions_frozen_before_any_truth_scoring=True,
                fit_seconds=case['fit_seconds'],ours_from_saved_source=dict(
                    phase_mae_rad=case['source']['test_phase_evaluation']['mean_absolute_rad'],
                    absolute_determinant=case['source']['train_phase_evaluation']['absolute_determinant']),
                native_circular=native,circular_plus_train_cdf=corrected)
            results.append(result)
            if case['cdf'] is not None:
                np.savez_compressed(out/(case['name']+'_frozen_phases.npz'),
                    train_native=case['native']['train_phases'],query_native=case['native']['phases'],
                    train_cdf=case['corrected']['train_phases'],query_cdf=case['corrected']['phases'],
                    cdf_knots=case['cdf']['knots'],cdf_values=case['cdf']['values'],
                    train_indices=case['native']['train_indices'],query_indices=case['native']['query_indices'])
            print('CDF_RESULT',case['name'],'ours',result['ours_from_saved_source']['phase_mae_rad'],
                  'native',native['phase_mae_rad'],'CDF',None if corrected is None else corrected['phase_mae_rad'],flush=True)
    report = dict(scope='two-scalar diagnostic strengthened baseline; not the full Kralemann method',
        maximum_numerical_threads=1,ours_fits_performed=0,query_truth_used_to_fit_or_select=False,
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        dependency_sha256={name:hashlib.sha256((root/'code/lag_ssa'/name).read_bytes()).hexdigest()
            for name in ('compare_fields_first_baselines.py','fields_first_preprocess.py',
                         'fields_first_sensor_cases.py','evaluate_tangent_overlap_angles.py')},results=results)
    (out/'circular_cdf_diagnostic.json').write_text(json.dumps(jsonable(report),indent=2,allow_nan=False)+'\n')
    target = out/'circular_cdf_diagnostic.py'
    if Path(__file__).resolve() != target.resolve():
        shutil.copy2(Path(__file__),target)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
