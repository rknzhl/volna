"""Frozen-model flow checks at eight TRAIN starts and two integration steps."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/takers_curved_noise_fix_20261022/mpl')
import json,pickle,sys,time
from pathlib import Path
import numpy as np
from scipy import linalg
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code/lag_ssa'))
import fields_first_phase as core
from fields_first_sensor_pipeline import GATES,_source_rms
from run_fields_first_sensor import write_json

def main():
    rows=[]
    for percent in (0,1,5,10):
        path=Path(f'/private/tmp/takers_curved_noise_fix_20261022/noise{percent:02d}_candidate.pkl')
        if not path.exists():continue
        result_path=Path(__file__).with_name(f'noise{percent:02d}.json')
        if not result_path.exists() or json.loads(result_path.read_text())['transform_status']!='transformed_conditionally':
            continue
        start=time.monotonic();model=pickle.load(path.open('rb'))
        if model['status']!='conditional_fields_first_model':continue
        ids=np.unique(np.linspace(0,len(model['X'])-1,8,dtype=int))
        x=model['X'][ids];scale=_source_rms(model)
        closure=[];refinement=[]
        for j in range(model['r']):
            end=core.flow(model,x,2*np.pi,j,step=.12)
            fine=core.flow(model,x,2*np.pi,j,step=.06)
            closure.append(linalg.norm(end-x,axis=1)/scale)
            refinement.append(linalg.norm(fine-end,axis=1)/scale)
        original=core.inverse(model,model['phases'][ids],step=.12)
        reverse=core.inverse(model,model['phases'][ids],step=.12,reverse=True)
        fine=core.inverse(model,model['phases'][ids],step=.06)
        error_squared=np.sum((original-x)**2)+np.sum(model['source_projection_residual_squared'][ids])
        row=dict(noise=percent/100,source_count=len(x),source_indices=ids,
            model_frozen=True,known_phases_or_derivatives_used=False,numerical_threads=1,
            full_turn_relative=closure,full_turn_max=float(np.max(closure)),
            full_turn_step_refinement_relative=refinement,
            full_turn_step_refinement_max=float(np.max(refinement)),
            inverse_rms_relative=float(np.sqrt(error_squared/(len(x)*scale**2))),
            order_rms_relative=float(linalg.norm(reverse-original)/(np.sqrt(len(x))*scale)),
            inverse_step_refinement_relative=float(linalg.norm(fine-original)/(np.sqrt(len(x))*scale)),
            elapsed_seconds=time.monotonic()-start)
        row['passed']=bool(row['full_turn_max']<GATES['full_turn_relative']
            and row['full_turn_step_refinement_max']<GATES['flow_refinement_relative']
            and row['inverse_rms_relative']<GATES['inverse_window_relative']
            and row['order_rms_relative']<GATES['flow_order_relative'])
        rows.append(row);print('FLOW_RESULT',{k:v for k,v in row.items() if not isinstance(v,(np.ndarray,list))},flush=True)
    write_json(Path(__file__).with_name('flow_verification.json'),rows)
    return 0 if rows and all(row['passed'] for row in rows) else 1

if __name__=='__main__':
    with threadpool_limits(limits=1):raise SystemExit(main())
