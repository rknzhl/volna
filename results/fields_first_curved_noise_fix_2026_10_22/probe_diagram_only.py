"""Bounded TRAIN persistence probe without unused cocycle representatives."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/takers_curved_noise_fix_20261022/mpl')
import argparse,json,multiprocessing as mp,pickle,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code/lag_ssa'))

def worker(threshold):
    from ripser import ripser
    from threadpoolctl import threadpool_limits
    from fields_first_periods import _distances,_select_prefix,_select_scale,SETTINGS
    from dataclasses import replace
    from run_fields_first_sensor import write_json
    data=pickle.load(open('/private/tmp/takers_curved_noise_fix_20261022/noise10_inputs.pkl','rb'))
    with threadpool_limits(limits=1):
        x=data['fullx']@data['vh'][:12].T
        metric=_distances(x.T);started=time.monotonic()
        print('DIAGRAM_ONLY_START',flush=True)
        ph=ripser(metric,distance_matrix=True,coeff=47,maxdim=1,n_perm=2048,
            do_cocycles=False,**({} if threshold is None else dict(thresh=threshold)))
        print('DIAGRAM_ONLY_RIPS_DONE',time.monotonic()-started,flush=True)
        diagram=ph['dgms'][1].copy()
        if threshold is not None:diagram[~np.isfinite(diagram[:,1]),1]=threshold
        diagram=diagram[(diagram[:,1]-diagram[:,0]).argsort()[::-1]]
        hint=__import__('fields_first_periods')._local_dimension_hint(x)[0]
        count,selection=_select_prefix(diagram,replace(SETTINGS,n_landmarks=2048),preferred_count=hint)
        threshold,diagnostic=_select_scale(diagram,count,selection,ph['dperm2all'],metric)
        row=dict(status='completed',prime=47,filtration_threshold=threshold,known_phases_used=False,numerical_threads=1,
            elapsed_seconds=time.monotonic()-started,selected_dimension=count,
            threshold=threshold,selection=selection,diagnostic=diagnostic)
        pickle.dump(dict(ph=ph,metric=metric,x=x),open('/private/tmp/takers_curved_noise_fix_20261022/diagram_only.pkl','wb'))
        name='diagram_only_probe.json' if threshold is None else 'diagram_only_bounded_probe.json'
        write_json(Path(__file__).with_name(name),row)
        print('DIAGRAM_ONLY_RESULT',count,threshold,row['elapsed_seconds'],flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--threshold',type=float)
    args=parser.parse_args()
    process=mp.get_context('spawn').Process(target=worker,args=(args.threshold,))
    process.start();process.join(60)
    if process.is_alive():
        process.terminate();process.join(5)
        if process.is_alive():process.kill();process.join()
        name='diagram_only_probe.json' if args.threshold is None else 'diagram_only_bounded_probe.json'
        Path(__file__).with_name(name).write_text(json.dumps(
            dict(status='interrupted_at_60_second_budget',numerical_threads=1),indent=2)+'\n')
        raise SystemExit(2)
    raise SystemExit(process.exitcode)
