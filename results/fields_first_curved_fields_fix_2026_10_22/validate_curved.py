"""Run the public measurement-only pipeline; save candidates for diagnosis."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS',
            'MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/takers_curved_fields_20261022/mpl')
import json,pickle,sys,time
from pathlib import Path
from threadpoolctl import threadpool_limits,threadpool_info
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code/lag_ssa'))
from benchmark_fields_first_sensors import benchmark
import fields_first_sensor_pipeline as pipeline_module
from run_fields_first_sensor import write_json
from validate_fields_first import result_row

def main():
    # These wrappers only save state; they do not change fitting or selection.
    fit=pipeline_module.core.fit;gates=pipeline_module._training_gates
    scratch=Path('/private/tmp/takers_curved_fields_20261022')
    scratch.mkdir(parents=True,exist_ok=True)
    def save_fit(*args,**kwargs):
        model=fit(*args,**kwargs)
        pickle.dump(model,(scratch/'current_candidate.pkl').open('wb'))
        return model
    def save_gates(model):
        print('TRAINING_GATES',model['status'],flush=True)
        result=gates(model)
        pickle.dump(model,(scratch/'current_candidate.pkl').open('wb'))
        print('TRAINING_GATES_RESULT',result,flush=True)
        return result
    pipeline_module.core.fit=save_fit
    pipeline_module._training_gates=save_gates
    out=Path(__file__).parent/'validation';out.mkdir(exist_ok=True)
    started=time.monotonic()
    result=benchmark(case='threephase4sensor',trend=None,ntrain=2400,ntest=320,
        seed=31,noise=0.,max_candidates=2,progress_path=out/'three_phases.progress.json')
    result['execution']=dict(maximum_numerical_threads=1,threadpools=threadpool_info(),
                             elapsed_seconds=time.monotonic()-started)
    write_json(out/'three_phases.json',result)
    row=result_row('three_phases',result)
    write_json(out/'summary.json',[row])
    print('VALIDATION_RESULT',json.dumps(row),flush=True)
    return 0 if row['passed'] else 1

if __name__=='__main__':
    with threadpool_limits(limits=1):raise SystemExit(main())
