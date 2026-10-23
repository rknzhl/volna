"""Evaluate the observation-only pipeline; hidden truth is opened after freezing predictions."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy import linalg
from fields_first_sensor_cases import generate_variant
from fields_first_sensor_pipeline import SensorPipeline
from fields_first_preprocess import highpass_decompose
from evaluate_tangent_overlap_angles import match_phases
from run_fields_first_sensor import write_json
from threadpoolctl import threadpool_limits, threadpool_info


def errors(prediction, target, train_std):
    diff = np.asarray(prediction)-np.asarray(target)
    scaled = diff/np.maximum(train_std, 1e-12)
    return dict(train_channel_std_normalized_rmse=float(np.sqrt(np.mean(scaled**2))),
                per_channel_train_std_normalized_rmse=np.sqrt(np.mean(scaled**2, axis=0)),
                raw_relative_error=float(linalg.norm(diff)/max(linalg.norm(target), 1e-15)))


def benchmark(case='scalarcircle', trend='linear', ntrain=1800, ntest=600,
              seed=31, noise=.05, phase_mode='rigid', rate_profile='default',
              max_candidates=None, progress_path=None):
    Y, truth = generate_variant(case, ntrain+ntest, seed=seed, noise=noise,
        trend=trend, train_count=ntrain, phase_mode=phase_mode, rate_profile=rate_profile)
    def checkpoint(summary):
        if progress_path is not None:
            write_json(progress_path,dict(case=case,noise=noise,
                trend=trend,seed=seed,ntrain=ntrain,ntest=ntest,fit=summary))
    pipeline = SensorPipeline(max_candidates=max_candidates).fit(Y[:ntrain],progress_callback=checkpoint)
    prediction = pipeline.transform(Y[ntrain:], start_index=ntrain)
    # Selection uses training diagnostics only; the following is evaluation.
    result = dict(case=case, trend=trend, noise=noise, seed=seed, ntrain=ntrain,
        ntest=ntest, phase_mode=phase_mode, rate_profile=rate_profile,
        fit=pipeline.summary(), transform_status=prediction['status'],
        prediction_frozen_before_truth_evaluation=True,
        generator_metadata=truth['metadata'], nominal_snr_db=truth['nominal_snr_db'])
    if prediction['status'] != 'transformed_conditionally':
        result['transform'] = {k:v for k,v in prediction.items() if k!='status'}
        return result
    idx = prediction['raw_time_indices']
    widx = prediction['window_raw_time_indices']
    result['transform'] = dict(raw_time_first=int(idx[0]), raw_time_last=int(idx[-1]),
        retained_snapshot_count=len(idx), retained_window_count=len(widx),
        diagnostics=prediction['diagnostics'])
    std = truth['signal_train_std']
    result['reconstruction_against_clean_total'] = errors(prediction['reconstruction'], truth['clean_total'][idx], std)
    result['reconstruction_against_noisy_observations'] = errors(prediction['reconstruction'], Y[idx], std)
    result['preprocessed_against_clean_total'] = errors(prediction['conditioned_observations'], truth['clean_total'][idx], std)
    clean_split=highpass_decompose(pipeline.preprocess,truth['clean_total'][ntrain:],start_index=ntrain)
    result['field_residual_against_filtered_clean']=errors(
        prediction['residual_reconstruction'],clean_split['residual'][idx-ntrain],std)
    # The observed slow component is NOT scored as an identified physical trend.
    match = match_phases(pipeline.model['phases'], truth['theta'][pipeline.train_window_raw_indices])
    target = truth['theta'][widx]@match['winding_candidate'].T+match['origins']
    error = np.angle(np.exp(1j*(prediction['phases']-target)))
    result['train_phase_evaluation'] = match
    result['phase_dimension_matches_truth']=pipeline.model['r']==truth['theta'].shape[1]
    result['test_phase_evaluation'] = dict(mean_absolute_rad=float(np.mean(np.abs(error))),
        per_coordinate_mean_absolute_rad=np.mean(np.abs(error), axis=0),
        per_coordinate_rmse_rad=np.sqrt(np.mean(error**2, axis=0)),
        maximum_absolute_rad=float(np.max(np.abs(error))))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', default='scalarcircle')
    parser.add_argument('--trend', choices=['none','linear','quadratic','smooth_slow'], default='linear')
    parser.add_argument('--ntrain', type=int, default=1800)
    parser.add_argument('--ntest', type=int, default=600)
    parser.add_argument('--seed', type=int, default=31)
    parser.add_argument('--noise', type=float, default=.05)
    parser.add_argument('--phase-mode', choices=['rigid','chirp','coupled'], default='rigid')
    parser.add_argument('--rate-profile', choices=['default','wide','slow','near_resonant'], default='default')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-candidates',type=int,help='stop with an unresolved budget status after this many candidates')
    parser.add_argument('--threads',type=int,default=1,help='maximum numerical-library threads (default: 1)')
    args = parser.parse_args()
    if args.threads<1:parser.error('--threads must be positive')
    with threadpool_limits(limits=args.threads):
        result = benchmark(args.case, None if args.trend=='none' else args.trend,
            args.ntrain, args.ntest, args.seed, args.noise, args.phase_mode, args.rate_profile,
            args.max_candidates,args.output.with_suffix('.progress.json'))
        result['execution']=dict(maximum_numerical_threads=args.threads,threadpools=threadpool_info())
    write_json(args.output,result)
    print(json.dumps({'case':args.case, 'noise':args.noise,
                      'status':result['transform_status'], 'output':str(args.output.resolve())}))


if __name__ == '__main__':
    main()
