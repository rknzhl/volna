"""Sequential synthetic validation of one observation-only sensor pipeline."""
import os
for _name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS',
              'NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_name]='1'

import argparse
import json
import time
from pathlib import Path
from threadpoolctl import threadpool_limits
from benchmark_fields_first_sensors import benchmark
from run_fields_first_sensor import write_json


# Fixed examples; their phases are opened only by benchmark's final evaluation.
CASES = dict(
    scalar_clean=dict(case='scalarcircle',noise=0.,trend=None),
    scalar_noise10=dict(case='scalarcircle',noise=.1,trend='linear'),
    two_waves=dict(case='fundamental2waves',noise=.05,trend='quadratic',ntrain=2400),
    nonlinear_sensors=dict(case='nonlinear3sensor',noise=.05,trend='quadratic',rate_profile='wide',ntrain=2400,seed=47),
    optical=dict(case='opticalsteady1d_layered',noise=.05,trend='linear',ntrain=2400),
    three_phases=dict(case='threephase4sensor',noise=0.,trend=None,ntrain=2400),
    three_phases_linear=dict(case='threephase6sensor',noise=0.,trend=None,ntrain=2400),
)


def result_row(name, result):
    accepted=result['transform_status']=='transformed_conditionally'
    rmse=result.get('reconstruction_against_clean_total',{}).get('train_channel_std_normalized_rmse')
    phase=result.get('test_phase_evaluation',{}).get('mean_absolute_rad')
    primitive=result.get('train_phase_evaluation',{}).get('absolute_determinant')==1
    return dict(name=name,status=result['transform_status'],
        phase_dimension=result['fit']['attempts'][-1].get('diagnostics',{}).get('dimension_selected'),
        rmse_std_normalized=rmse,phase_mae_rad=phase,primitive_phase_match=primitive,
        passed=bool(accepted and result.get('phase_dimension_matches_truth')
            and primitive and rmse is not None and phase is not None
            and rmse<.05 and phase<.05))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--only',nargs='+',choices=list(CASES))
    parser.add_argument('--max-candidates',type=int,default=9)
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    summary=[]
    with threadpool_limits(limits=1):
        for name in args.only or CASES:
            config=dict(ntrain=1800,ntest=320,seed=31)
            config.update(CASES[name])
            destination=args.output_dir/(name+'.json')
            print('VALIDATE',name,flush=True)
            result=benchmark(**config,max_candidates=args.max_candidates,
                progress_path=destination.with_suffix('.progress.json'))
            write_json(destination,result)
            row=result_row(name,result)
            summary.append(row)
            saved=[result_row(case_name,json.loads(path.read_text()))
                for case_name in CASES
                if (path:=args.output_dir/(case_name+'.json')).exists()]
            write_json(args.output_dir/'summary.json',saved)
            print('VALIDATION_RESULT',row,flush=True)
            time.sleep(1)
    return 0 if all(row['passed'] for row in summary) else 1


if __name__=='__main__':
    raise SystemExit(main())
