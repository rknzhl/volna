"""Four fixed multivariate diagnostics: native CC/TC plus frozen marginal CDFs.

Coordinatewise empirical CDF correction is not a complete multidimensional
dynamical uniformization method. No fields-first model is fitted here.
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


def array_sha(array, dtype='<f8'):
    return hashlib.sha256(np.asarray(array, dtype=dtype).tobytes()).hexdigest()


def fit_marginal_cdfs(train):
    """Same midpoint ranks and periodic interpolation used in scalar diagnostic."""
    if not np.isfinite(train).all():
        raise ValueError('finite TRAIN coordinates required')
    models = []
    for coordinate in np.asarray(train).T:
        knots, counts = np.unique(np.mod(coordinate, 2*np.pi), return_counts=True)
        if len(knots) < 3:
            raise ValueError('insufficient distinct TRAIN coordinate samples')
        values = 2*np.pi*(np.cumsum(counts)-counts/2)/len(coordinate)
        models.append(dict(knots=np.r_[knots[-1]-2*np.pi,knots,knots[0]+2*np.pi],
            values=np.r_[values[-1]-2*np.pi,values,values[0]+2*np.pi],
            train_count=len(coordinate),unique_count=len(knots)))
    return models


def apply_marginal_cdfs(models, phase):
    return np.column_stack([np.mod(np.interp(np.mod(coordinate,2*np.pi),
        model['knots'],model['values']),2*np.pi) for model,coordinate in zip(models,np.asarray(phase).T)])


def integer_inverse(matrix):
    a = np.asarray(matrix,dtype=int)
    inverse = np.rint(np.linalg.inv(a)).astype(int)
    if not np.array_equal(a@inverse,np.eye(len(a),dtype=int)):
        raise ValueError('evaluation basis is not unimodular')
    return inverse


def query_step(train, query):
    increments = np.angle(np.exp(1j*np.diff(train,axis=0)))
    step = np.angle(np.mean(np.exp(1j*increments),axis=0))
    error = np.angle(np.exp(1j*(np.diff(query,axis=0)-step)))
    return dict(frozen_train_step_rad=step,
        query_phase_step_rmse_rad=float(np.sqrt(np.mean(error**2))),
        per_coordinate_query_phase_step_rmse_rad=np.sqrt(np.mean(error**2,axis=0)))


def common_basis(score, frozen, evaluator):
    if score['status'] != 'evaluated' or score['absolute_determinant'] != 1:
        return dict(status='unavailable_nonprimitive_or_unresolved_basis'),{}
    match = score['train_phase_evaluation']
    inverse = integer_inverse(match['winding_candidate'])
    origin = np.asarray(match['origins'])
    train = np.mod((frozen['train_phases']-origin)@inverse.T,2*np.pi)
    query = np.mod((frozen['phases']-origin)@inverse.T,2*np.pi)
    train_truth = evaluator['theta'][frozen['train_indices']]
    query_truth = evaluator['theta'][frozen['query_indices']]
    error = np.angle(np.exp(1j*(query-query_truth)))
    increment_error = np.angle(np.exp(1j*(np.diff(query,axis=0)-np.diff(query_truth,axis=0))))
    return dict(status='evaluated',inverse_winding_matrix=inverse,
        query_phase_mae_rad=float(np.mean(np.abs(error))),
        per_coordinate_query_phase_mae_rad=np.mean(np.abs(error),axis=0),
        query_phase_rmse_rad=float(np.sqrt(np.mean(error**2))),
        query_increment_rmse_rad_against_truth=float(np.sqrt(np.mean(increment_error**2))),
        **query_step(train,query)),dict(generator_basis_train_phases=train,
            generator_basis_query_phases=query,query_generator_basis_phase_error=error,
            theta_train_evaluator=train_truth,theta_query_evaluator=query_truth)


def ours_common_basis_bound(source):
    inverse = integer_inverse(source['train_phase_evaluation']['winding_candidate'])
    errors = np.asarray(source['test_phase_evaluation']['per_coordinate_mean_absolute_rad'])
    bound = float(np.mean(np.abs(inverse)@errors))
    signed_permutation = bool(np.all(np.sum(np.abs(inverse),axis=0)==1)
        and np.all(np.sum(np.abs(inverse),axis=1)==1))
    return dict(inverse_winding_matrix=inverse,
        query_phase_mae_rad_upper_bound=bound,
        bound_is_exact_for_signed_permutation=signed_permutation,
        exact_query_phase_mae_rad=bound if signed_permutation else None,
        source_learned_basis_phase_mae_rad=source['test_phase_evaluation']['mean_absolute_rad'],
        source_query_phase_step_rmse_rad=source['transform']['diagnostics'].get(
            'phase_step_rmse_against_frozen_training_step'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path.cwd())
    parser.add_argument('--output-dir',type=Path,required=True)
    args = parser.parse_args(); root=args.root.resolve(); out=args.output_dir.resolve()
    sys.path.insert(0,str(root/'code/lag_ssa'))
    from compare_fields_first_baselines import _indices,_phase_baselines,_score_phases
    from fields_first_preprocess import fit_preprocess,highpass_decompose
    from fields_first_sensor_cases import generate_variant
    from run_fields_first_sensor import jsonable
    references=json.loads((root/'results/fields_first_debug_2026_10_22/baselines/summary.json').read_text())
    cases=[]
    with threadpool_limits(limits=1):
        for name in ('nonlinear_sensors','optical','two_waves','three_phases_linear'):
            print('MARGINAL_CDF_FIT',name,flush=True)
            path=root/'results/fields_first_debug_2026_10_22/validation'/(name+'.json')
            source=json.loads(path.read_text())
            reference=next(value for value in references if value['source']==str(path))
            idx,widx,lag=_indices(source)
            config={key:source[key] for key in ('case','seed','noise','trend')}
            config.update(phase_mode=source.get('phase_mode','rigid'),
                rate_profile=source.get('rate_profile','default'),
                trend_strength=source.get('generator_metadata',{}).get('trend_strength',.3))
            ntrain,ntest=source['ntrain'],source['ntest']
            y,evaluator=generate_variant(n=ntrain+ntest,train_count=ntrain,**config)
            train_hash,query_hash=array_sha(y[:ntrain]),array_sha(y[ntrain:])
            assert train_hash==reference['evaluation']['train_observations_sha256']
            assert query_hash==reference['evaluation']['query_observations_sha256']
            assert array_sha(idx,'<i8')==reference['evaluation']['snapshot_indices_sha256']
            assert array_sha(widx,'<i8')==reference['evaluation']['window_indices_sha256']
            started=time.monotonic()
            preprocessing=fit_preprocess(y[:ntrain],lag=32,maxrank=24,trend_degrees=(0,))
            train_split=highpass_decompose(preprocessing,y[:ntrain])
            query_split=highpass_decompose(preprocessing,y[ntrain:],start_index=ntrain)
            native=_phase_baselines(train_split,query_split,lag,
                names=['circular','toroidal'],nlandmarks=192)
            variants={}; cdfs={}
            for method, prediction in native.items():
                prediction['method']='native_'+method
                variants[method]=prediction
                if prediction['status']!='frozen_prediction':
                    continue
                models=fit_marginal_cdfs(prediction['train_phases'])
                corrected=copy.deepcopy(prediction)
                corrected['method']='native_'+method+'_plus_train_marginal_cdf'
                corrected['train_phases']=apply_marginal_cdfs(models,prediction['train_phases'])
                corrected['phases']=apply_marginal_cdfs(models,prediction['phases'])
                corrected['metadata'].update(uniformization='coordinatewise TRAIN empirical midpoint CDF; periodic linear interpolation',
                    complete_multidimensional_dynamical_uniformization=False,
                    hidden_phases_or_frequencies_used_to_fit=False,
                    cdf_train_counts=[model['train_count'] for model in models],
                    cdf_unique_counts=[model['unique_count'] for model in models])
                variants[method+'_marginal_cdf']=corrected
                cdfs[method]=models
            cases.append(dict(name=name,path=path,source=source,reference=reference,
                idx=idx,widx=widx,lag=lag,configuration=dict(config,ntrain=ntrain,ntest=ntest),
                train_hash=train_hash,query_hash=query_hash,evaluator=evaluator,variants=variants,
                cdfs=cdfs,fit_seconds=time.monotonic()-started))
        # Every native map, marginal CDF and prediction for all four cases is frozen.
        results=[]; out.mkdir(parents=True,exist_ok=True)
        for case in cases:
            scores={}; arrays={}
            for method,frozen in case['variants'].items():
                score=_score_phases(frozen,case['evaluator'],case['widx'])
                if score['status']=='evaluated':
                    score['query_step']=query_step(frozen['train_phases'],frozen['phases'])
                    common,common_arrays=common_basis(score,frozen,case['evaluator'])
                    score['generator_basis']=common
                    arrays[method+'_train_phases']=frozen['train_phases']
                    arrays[method+'_query_phases']=frozen['phases']
                    arrays[method+'_train_indices']=frozen['train_indices']
                    arrays[method+'_query_indices']=frozen['query_indices']
                    arrays.update({method+'_'+key:value for key,value in common_arrays.items()})
                if method in ('circular','toroidal') and score['phase_mae_rad'] is not None:
                    previous=next(row for row in case['reference']['rows'] if row['method']==method)
                    assert abs(score['phase_mae_rad']-previous['phase_mae_rad'])<1e-12
                scores[method]=score
            for method,models in case['cdfs'].items():
                for coordinate,model in enumerate(models):
                    arrays[f'{method}_cdf_knots_{coordinate}']=model['knots']
                    arrays[f'{method}_cdf_values_{coordinate}']=model['values']
            np.savez_compressed(out/(case['name']+'_multifactor_frozen_phases.npz'),**arrays)
            result=dict(name=case['name'],configuration=case['configuration'],lag=case['lag'],
                source=str(case['path']),source_sha256=hashlib.sha256(case['path'].read_bytes()).hexdigest(),
                train_observations_sha256=case['train_hash'],query_observations_sha256=case['query_hash'],
                snapshot_indices_sha256=array_sha(case['idx'],'<i8'),window_indices_sha256=array_sha(case['widx'],'<i8'),
                same_raw_arrays_split_and_evaluation_indices_as_main_comparison=True,
                all_four_case_predictions_frozen_before_any_truth_scoring=True,
                query_window_count=len(case['widx']),fit_seconds=case['fit_seconds'],
                ours_from_saved_source=ours_common_basis_bound(case['source']),methods=scores)
            results.append(result)
            print('MARGINAL_CDF_RESULT',case['name'],'ours',result['ours_from_saved_source']['source_learned_basis_phase_mae_rad'],
                {key:dict(learned_basis_mae=value['phase_mae_rad'],
                    generator_basis_mae=value.get('generator_basis',{}).get('query_phase_mae_rad'),
                    status=value['status']) for key,value in scores.items()},flush=True)
    report=dict(scope='four fixed multivariate diagnostics; coordinatewise marginal CDF correction, not complete dynamical uniformization',
        maximum_numerical_threads=1,ours_fits_performed=0,query_truth_used_for_fit_or_selection=False,
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        dependency_sha256={name:hashlib.sha256((root/'code/lag_ssa'/name).read_bytes()).hexdigest()
            for name in ('compare_fields_first_baselines.py','fields_first_preprocess.py',
                         'fields_first_sensor_cases.py','evaluate_tangent_overlap_angles.py')},results=results)
    (out/'multifactor_marginal_cdf_diagnostic.json').write_text(json.dumps(jsonable(report),indent=2,allow_nan=False)+'\n')
    target=out/'multifactor_marginal_cdf_diagnostic.py'
    if Path(__file__).resolve()!=target.resolve():
        shutil.copy2(Path(__file__),target)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
