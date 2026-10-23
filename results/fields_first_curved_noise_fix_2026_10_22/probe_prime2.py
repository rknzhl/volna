"""Time-bounded persistence parameter probe on cached TRAIN windows only."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/takers_curved_noise_fix_20261022/mpl')
import json,multiprocessing as mp,pickle,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code/lag_ssa'))

def worker():
    from threadpoolctl import threadpool_limits
    from fields_first_periods import _local_dimension_hint,_infer_torus_dimension_once
    from run_fields_first_sensor import write_json
    data=pickle.load(open('/private/tmp/takers_curved_noise_fix_20261022/noise10_inputs.pkl','rb'))
    with threadpool_limits(limits=1):
        x=data['fullx']@data['vh'][:12].T
        hint,hint_report=_local_dimension_hint(x)
        started=time.monotonic()
        r,diagnostic=_infer_torus_dimension_once(x,nland=2048,prime=2,preferred_count=hint)
        row=dict(status='completed',selected_dimension=r,local_hint=hint,
            known_phases_used=False,numerical_threads=1,
            elapsed_seconds=time.monotonic()-started,diagnostic=diagnostic)
        write_json(Path(__file__).with_name('prime2_probe.json'),row)
        print('PRIME2_RESULT',r,row['elapsed_seconds'],diagnostic.get('failure_reason'),flush=True)

if __name__=='__main__':
    process=mp.get_context('spawn').Process(target=worker)
    process.start();process.join(60)
    if process.is_alive():
        process.terminate();process.join(5)
        if process.is_alive():process.kill();process.join()
        Path(__file__).with_name('prime2_probe.json').write_text(json.dumps(
            dict(status='interrupted_at_60_second_budget',numerical_threads=1),indent=2)+'\n')
        raise SystemExit(2)
    raise SystemExit(process.exitcode)
