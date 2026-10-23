"""Additional frozen-model checks, using observations and fields only."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import pickle,sys,time
from pathlib import Path
import numpy as np
from scipy import linalg
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code/lag_ssa'))
import fields_first_phase as core
from fields_first_sensor_pipeline import GATES
from run_fields_first_sensor import write_json

def main():
    start=time.monotonic()
    model=pickle.load(open('/private/tmp/takers_curved_fields_20261022/current_candidate.pkl','rb'))
    ids=np.unique(np.linspace(0,len(model['X'])-1,8,dtype=int));x=model['X'][ids]
    scale=float(np.sqrt(np.mean(np.sum(model['X']**2,axis=1))))
    closure=[];refinement=[]
    for j in range(model['r']):
        end=core.flow(model,x,2*np.pi,j,step=.12)
        fine=core.flow(model,x,2*np.pi,j,step=.06)
        closure.append(linalg.norm(end-x,axis=1)/scale)
        refinement.append(linalg.norm(fine-end,axis=1)/scale)
    original=core.inverse(model,model['phases'][ids],step=.12)
    reversed_order=core.inverse(model,model['phases'][ids],step=.12,reverse=True)
    finer=core.inverse(model,model['phases'][ids],step=.06)
    report=dict(source_count=len(x),source_indices=ids,model_frozen=True,
        known_phases_or_derivatives_used=False,numerical_threads=1,
        full_turn_relative=closure,full_turn_max=float(np.max(closure)),
        full_turn_step_refinement_relative=refinement,
        full_turn_step_refinement_max=float(np.max(refinement)),
        inverse_rms_relative=float(linalg.norm(original-x)/(np.sqrt(len(x))*scale)),
        order_rms_relative=float(linalg.norm(reversed_order-original)/(np.sqrt(len(x))*scale)),
        inverse_step_refinement_relative=float(linalg.norm(finer-original)/(np.sqrt(len(x))*scale)),
        elapsed_seconds=time.monotonic()-start)
    report['passed']=bool(report['full_turn_max']<GATES['full_turn_relative']
        and report['full_turn_step_refinement_max']<GATES['flow_refinement_relative']
        and report['inverse_rms_relative']<GATES['inverse_window_relative']
        and report['order_rms_relative']<GATES['flow_order_relative'])
    write_json(Path(__file__).parent/'flow_verification.json',report)
    print({k:v for k,v in report.items() if not isinstance(v,(np.ndarray,list))},flush=True)
    return 0 if report['passed'] else 1

if __name__=='__main__':
    with threadpool_limits(limits=1):raise SystemExit(main())
