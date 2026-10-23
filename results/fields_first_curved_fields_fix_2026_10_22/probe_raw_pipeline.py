"""Keep the original geometry/topology; test fields fitted by raw overlap."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/takers_curved_fields_20261022/mpl')
import sys,pickle,json,time
from pathlib import Path
import numpy as np
from scipy import linalg
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/lag_ssa'),str(Path(__file__).parent)]
import fields_first_phase as core
from probe_linear_frame import affine_W
from fields_first_periods import fit_periods,reduce_period_basis
from fields_first_sensor_pipeline import _training_gates_once
from run_fields_first_sensor import jsonable
S=Path('/private/tmp/takers_curved_fields_20261022')


def raw_field(model,query):
    x=np.atleast_2d(query);w=affine_W(model['frame'],x)
    if model.get('strict_flows',False):
        g=core.evaluate_tangent_frame(model['atlas'],x);sv=np.linalg.svd(w,compute_uv=False)
        if g['gap'].min()<.1:raise ValueError('flow_tangent_projector_gap_lost')
        if g['nearest_distance'].max()>3*model['atlas']['bandwidth']:raise ValueError('flow_outside_geometric_support')
        if sv[:,-1].min()<.01*model['diagnostics']['frame_min_singular'] or (sv[:,-1]/sv[:,0]).min()<1e-3:raise ValueError('flow_invariant_frame_degenerate')
    return w[0] if np.asarray(query).ndim==1 else w


def raw_beta_action(model,query,direction):
    w=affine_W(model['frame'],np.atleast_2d(query));wt=np.swapaxes(w,1,2)
    return np.linalg.solve(wt@w,(wt@np.asarray(direction)[...,None]))[...,0]


def main():
    started=time.monotonic();m=pickle.loads((S/'augmented_full_model.pkl').read_bytes())
    tag='_low_ridge' if '--low-ridge' in sys.argv else ''
    fitted=pickle.loads((S/('raw_overlap_no_spatial'+tag+'_frame.pkl')).read_bytes())
    m['frame']=fitted;core.field=raw_field;core.evaluate_W=affine_W;core.beta_action=raw_beta_action
    m.pop('strict_flows',None);m.pop('flow_geometry',None);m.pop('origin',None)
    m['cardinal_phase_weights']=True
    x=m['X'];w=raw_field(m,x);d=m['diagnostics'];d['frame_min_singular']=float(np.linalg.svd(w,compute_uv=False)[:,-1].min())
    print('INTEGRALS',flush=True)
    raw,pathdiag=core.fine_integrals(m,m['complex'])
    pm=fit_periods(raw,m['complex']);pm,cx=reduce_period_basis(pm,m['complex'],np.einsum('npa,npb->ab',w,w)/len(w))
    m.update(period_model=pm,complex=cx)
    m['center_normalized_beta']=np.linalg.solve(pm['Pi'],core.beta(m,cx['landmarks']))
    m['origin']=core.encode(m,x[[0]])['phases'][0]
    enc=core.encode(m,x,derivative=True);m['phases']=enc['phases'];ph=m['phases'];r=m['r']
    d['periods']=pm['diagnostics'];d['integration']=pathdiag
    d['encoder_field_duality_relative']=float(linalg.norm(enc['jacobian']@core.phase_field(m,x)-np.eye(r)[None])/np.sqrt(len(x)*r))
    d['phase_step']=np.angle(np.mean(np.exp(1j*np.diff(ph,axis=0)),axis=0))
    d['train_phase_step_rmse']=float(np.sqrt(np.mean(np.angle(np.exp(1j*(np.diff(ph,axis=0)-d['phase_step'])))**2)))
    print('PRE_FLOW',d['periods']['projection_relative'],d['encoder_field_duality_relative'],d['train_phase_step_rmse'],flush=True)
    pickle.dump(m,(S/('raw_full_model'+tag+'.pkl')).open('wb'))
    gate=_training_gates_once(m)
    print('GATES',json.dumps(jsonable(gate)),flush=True)
    pickle.dump(m,(S/('raw_full_model'+tag+'.pkl')).open('wb'))
    Path(__file__).with_name('raw_pipeline'+tag+'_probe.json').write_text(json.dumps(jsonable(dict(gates=gate,fit=fitted['diagnostic'],elapsed_seconds=time.monotonic()-started)),indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
