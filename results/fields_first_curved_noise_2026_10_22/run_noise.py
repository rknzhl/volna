"""Paired noise tests of the unchanged public sensor pipeline, one at a time."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/takers_curved_noise_20261022/mpl')

import argparse
import copy
import hashlib
from functools import partial
import json
import sys
import time
from pathlib import Path
from threadpoolctl import threadpool_limits,threadpool_info

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code/lag_ssa'))
from benchmark_fields_first_sensors import benchmark
import benchmark_fields_first_sensors as benchmark_module
from validate_fields_first import result_row
from run_fields_first_sensor import write_json
import fields_first_periods as periods
import fields_first_sensor_pipeline as pipeline

LEVELS=(.01,.05,.10)
OUTPUT=Path(__file__).parent


def summarize(level,result):
    row=result_row(f'curved_noise_{level:g}',result)
    attempts=result['fit']['attempts']
    numerical=[a for a in attempts if 'neighbors' in a]
    selected=next((a for a in numerical if a.get('gates',{}).get('passed')),None)
    diagnostic=(selected or (numerical[-1] if numerical else {})).get('diagnostics',{})
    row.update(noise_sigma_over_signal_std=level,
        nominal_snr_db=result['nominal_snr_db'],
        fit_status=result['fit']['status'],
        selected_lag=result['fit']['selected_lag'],
        candidates_tried=len(numerical),
        resource_skipped_lags=[a['lag'] for a in attempts if a.get('status')=='unresolved_resource_budget'],
        candidate_budget_exhausted=result['fit']['status']=='unresolved_candidate_budget_exhausted',
        conditioning_rank=result['fit']['preprocessing']['rank'],
        phase_dimension=diagnostic.get('dimension_selected'),
        geometry=diagnostic.get('geometry_method'),
        topology_landmarks=diagnostic.get('dimension_diagnostic',{}).get('landmarks'),
        query_duality_rms=result.get('transform',{}).get('diagnostics',{}).get('encoder_field_duality_rms'),
        conditioner_rmse_std_normalized=result.get('preprocessed_against_clean_total',{}).get('train_channel_std_normalized_rmse'),
        elapsed_seconds=result['execution']['elapsed_seconds'])
    row['attempts']=[dict(lag=a['lag'],neighbors=a.get('neighbors'),
        status=a.get('status',a.get('gates',{}).get('core_status')),
        phase_dimension=a.get('diagnostics',{}).get('dimension_selected'),
        geometry=a.get('diagnostics',{}).get('geometry_method'),
        failed_training_checks=[k for k,v in a.get('gates',{}).get('checks',{}).items() if not v],
        exception=a.get('exception')) for a in attempts]
    return row


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--noise',type=float,choices=LEVELS,required=True)
    parser.add_argument('--baseline-only',action='store_true',
        help='use the successful clean configuration: 24 neighbours, lags 4 and 8')
    args=parser.parse_args()
    tag=f'noise{int(round(100*args.noise)):02d}'
    started=time.monotonic()
    original=periods.infer_torus_dimension
    original_lags=pipeline.LAGS
    original_factory=benchmark_module.SensorPipeline
    pipeline.LAGS=(4,8)
    if args.baseline_only:
        benchmark_module.SensorPipeline=partial(original_factory,neighbors=(24,))
    cache={};cache_counts=dict(evaluations=0,hits=0)
    def cached_dimension(x,nland=192,**kwargs):
        # Repeated neighbour choices use exactly the same source coordinates.
        key=(x.shape,x.dtype.str,hashlib.sha256(x.tobytes()).digest(),
             nland,tuple(sorted(kwargs.items())))
        if key not in cache:
            print('DIMENSION_EVALUATE',x.shape,flush=True)
            cache[key]=copy.deepcopy(original(x,nland=nland,**kwargs))
            cache_counts['evaluations']+=1
        else:
            print('DIMENSION_CACHE_HIT',x.shape,flush=True)
            cache_counts['hits']+=1
        return copy.deepcopy(cache[key])
    periods.infer_torus_dimension=cached_dimension
    with threadpool_limits(limits=1):
        try:
            result=benchmark(case='threephase4sensor',trend=None,ntrain=2400,ntest=320,
                seed=31,noise=args.noise,max_candidates=12,
                progress_path=OUTPUT/f'{tag}.progress.json')
        finally:
            periods.infer_torus_dimension=original
            pipeline.LAGS=original_lags
            benchmark_module.SensorPipeline=original_factory
        result['execution']=dict(maximum_numerical_threads=1,
            threadpools=threadpool_info(),elapsed_seconds=time.monotonic()-started,
            core_equations_unchanged=True,search_limited_for_memory=True,
            experiment_lags=[4,8],experiment_maximum_full_window_dimension=32,
            experiment_neighbors=[24] if args.baseline_only else [24,48,96],
            excluded_lags=[16,32,48],
            dimension_cache=cache_counts,
            cache_scope='exact source array and parameter keys; deep-copied dimension diagnostics only',
            candidate_budget_reason='all neighbourhoods at L=4,8; larger noisy spans can require several GB')
    write_json(OUTPUT/f'{tag}.json',result)
    summary=[]
    for level in LEVELS:
        path=OUTPUT/f'noise{int(round(100*level)):02d}.json'
        if path.exists():summary.append(summarize(level,json.loads(path.read_text())))
    write_json(OUTPUT/'summary.json',summary)
    print('NOISE_RESULT',json.dumps(summarize(args.noise,result)),flush=True)


if __name__=='__main__':main()
