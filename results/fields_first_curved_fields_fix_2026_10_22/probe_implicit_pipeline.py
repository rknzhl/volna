"""Diagnostic: tangent geometry from measured quadratic equations."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/takers_curved_fields_20261022/mpl')
import sys,pickle,json,time
from pathlib import Path
import numpy as np
from scipy import linalg
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code/lag_ssa'))
import fields_first_phase as core
from fields_first_frame import fit_projected_W
from fields_first_periods import fit_periods,reduce_period_basis
from fields_first_sensor_pipeline import _training_gates_once
from run_fields_first_sensor import jsonable
S=Path('/private/tmp/takers_curved_fields_20261022')


def geometry(atlas,query,directions=None,**kwargs):
    model=atlas['implicit_geometry'];x=np.atleast_2d(query);r=model['r']
    grad=2*np.einsum('kij,nj->nki',model['C'],x)+model['A'][None]
    G=np.einsum('nki,nkj->nij',grad,grad);values,vectors=np.linalg.eigh(G)
    Q=vectors[:,:,:r];normal=vectors[:,:,r:]
    nearest=cKDTree(model['X']).query(x)[0]
    out=dict(Q=Q,P=Q@np.swapaxes(Q,1,2),gap=1-np.maximum(values[:,r-1],0)/values[:,r],nearest_distance=nearest)
    if directions is not None:
        v=np.broadcast_to(np.asarray(directions),x.shape)
        dg=2*np.einsum('kij,nj->nki',model['C'],v)
        half=np.einsum('nki,nkj->nij',grad,dg);dG=half+np.swapaxes(half,1,2)
        cross=np.swapaxes(normal,1,2)@dG@Q
        cross/=values[:,:r,None].transpose(0,2,1)-values[:,r:,None]
        half=normal@cross@np.swapaxes(Q,1,2);out['dP']=half+np.swapaxes(half,1,2)
    return out


def main():
    started=time.monotonic();m=pickle.loads((S/'augmented_full_model.pkl').read_bytes())
    m.pop('origin',None);m.pop('strict_flows',None);m['cardinal_phase_weights']=True
    m['atlas']['implicit_geometry']=pickle.loads((S/'implicit_geometry.pkl').read_bytes())
    core.evaluate_tangent_frame=geometry;core.evaluate_projector=geometry
    x=m['X'];q=geometry(m['atlas'],x)['Q'];a,overlap=core.overlap_maps(q,4,m['basis'])
    m['frame']=fit_projected_W(x,q,a)
    w=core.field(m,x);sv=np.linalg.svd(w,compute_uv=False);d=m['diagnostics']
    d.update(frame_min_singular=float(sv[:,-1].min()),frame_min_relative_singular=float((sv[:,-1]/sv[:,0]).min()),atlas_min_gap=1.,overlap=overlap)
    print('INTEGRALS',flush=True)
    raw,pathdiag=core.fine_integrals(m,m['complex'],curved=False)
    pm=fit_periods(raw,m['complex']);pm,cx=reduce_period_basis(pm,m['complex'],np.einsum('npa,npb->ab',w,w)/len(w));m.update(period_model=pm,complex=cx)
    m['center_normalized_beta']=np.linalg.solve(pm['Pi'],core.beta(m,cx['landmarks']))
    m['origin']=core.encode(m,x[[0]])['phases'][0]
    enc=core.encode(m,x,derivative=True);m['phases']=enc['phases'];ph=m['phases'];r=m['r']
    d['periods']=pm['diagnostics'];d['integration']=pathdiag
    d['encoder_field_duality_relative']=float(linalg.norm(enc['jacobian']@core.phase_field(m,x)-np.eye(r)[None])/np.sqrt(len(x)*r))
    d['phase_step']=np.angle(np.mean(np.exp(1j*np.diff(ph,axis=0)),axis=0))
    d['train_phase_step_rmse']=float(np.sqrt(np.mean(np.angle(np.exp(1j*(np.diff(ph,axis=0)-d['phase_step'])))**2)))
    d['commutators']=core.commutator_diagnostics(m)
    print('PRE_FLOW',d['periods']['projection_relative'],d['encoder_field_duality_relative'],d['train_phase_step_rmse'],flush=True)
    pickle.dump(m,(S/'implicit_full_model.pkl').open('wb'))
    gate=_training_gates_once(m)
    print('GATES',json.dumps(jsonable(gate)),flush=True)
    pickle.dump(m,(S/'implicit_full_model.pkl').open('wb'))
    Path(__file__).with_name('implicit_pipeline_probe.json').write_text(json.dumps(jsonable(dict(gates=gate,diagnostics=d,elapsed_seconds=time.monotonic()-started)),indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
