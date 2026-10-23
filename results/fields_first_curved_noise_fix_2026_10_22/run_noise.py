"""Paired noisy curved-torus runs of the public observation-only pipeline."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/takers_curved_noise_fix_20261022/mpl')
import argparse,json,pickle,sys,time
from functools import partial
from pathlib import Path
from threadpoolctl import threadpool_limits,threadpool_info

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code/lag_ssa'))
import benchmark_fields_first_sensors as benchmark_module
import fields_first_sensor_pipeline as pipeline
import fields_first_periods as periods
from run_fields_first_sensor import write_json
from validate_fields_first import result_row

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--noise',type=float,choices=(0.,.01,.05,.10),required=True)
    args=parser.parse_args()
    output=Path(__file__).parent
    tag=f'noise{int(round(100*args.noise)):02d}'
    scratch=Path('/private/tmp/takers_curved_noise_fix_20261022')
    scratch.mkdir(exist_ok=True)
    original_fit=pipeline.core.fit;original_gates=pipeline._training_gates
    original_dimension_probe=periods._infer_torus_dimension_once
    def timed_dimension_probe(x,nland,**kwargs):
        began=time.monotonic()
        print('TOPOLOGY_RESOLUTION_START',nland,'cap',kwargs.get('filtration_cap'),flush=True)
        dimension,diagnostic=original_dimension_probe(x,nland,**kwargs)
        print('TOPOLOGY_RESOLUTION_RESULT',nland,dimension,
            diagnostic.get('failure_reason'),time.monotonic()-began,flush=True)
        return dimension,diagnostic
    periods._infer_torus_dimension_once=timed_dimension_probe
    def save_fit(*a,**kw):
        model=original_fit(*a,**kw)
        pickle.dump(model,(scratch/f'{tag}_candidate.pkl').open('wb'))
        return model
    def save_gates(model):
        print('TRAINING_GATES',model['status'],flush=True)
        result=original_gates(model)
        pickle.dump(model,(scratch/f'{tag}_candidate.pkl').open('wb'))
        print('TRAINING_GATES_RESULT',result,flush=True)
        return result
    pipeline.core.fit=save_fit;pipeline._training_gates=save_gates
    pipeline.LAGS=(4,8)
    benchmark_module.SensorPipeline=partial(pipeline.SensorPipeline,neighbors=(24,))
    started=time.monotonic()
    with threadpool_limits(limits=1):
        result=benchmark_module.benchmark(case='threephase4sensor',trend=None,
            ntrain=2400,ntest=320,seed=31,noise=args.noise,max_candidates=2,
            progress_path=output/f'{tag}.progress.json')
    result['execution']=dict(maximum_numerical_threads=1,threadpools=threadpool_info(),
        elapsed_seconds=time.monotonic()-started,
        experiment_lags=[4,8],experiment_neighbors=[24],
        parameter_selection='TRAIN-only singular gap and quadratic-relation validation',
        final_accuracy_gates_unchanged=True)
    write_json(output/f'{tag}.json',result)
    rows=[]
    for level in (0.,.01,.05,.10):
        path=output/f'noise{int(round(100*level)):02d}.json'
        if path.exists():
            saved=json.loads(path.read_text());row=result_row(path.stem,saved)
            row.update(noise=level,elapsed_seconds=saved['execution']['elapsed_seconds'])
            rows.append(row)
    write_json(output/'summary.json',rows)
    current=result_row(tag,result)
    current.update(noise=args.noise,elapsed_seconds=result['execution']['elapsed_seconds'])
    print('NOISE_RESULT',json.dumps(current),flush=True)

if __name__=='__main__':main()
