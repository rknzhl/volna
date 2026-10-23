"""Test projection onto one frozen local graph without moving along it."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/takers_curved_fields_20261022/mpl')
import sys,pickle,json,time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code/lag_ssa'))
import fields_first_phase as core
from fields_first_sensor_pipeline import _training_gates_once
from fields_first_tangent_atlas import _poly
from run_fields_first_sensor import jsonable


def local_retract(atlas,query,**_kwargs):
    x=np.atleast_2d(np.asarray(query,dtype=float))
    tree=atlas.get('_local_graph_tree')
    if tree is None:tree=cKDTree(atlas['X']);atlas['_local_graph_tree']=tree
    distance,ids=tree.query(x)
    if np.max(distance)>3*atlas['bandwidth']:
        raise ValueError('flow_outside_geometric_support')
    q=atlas['Q'][ids];radii=atlas['radius'][ids]
    u=np.einsum('np,npa->na',x-atlas['X'][ids],q)/radii[:,None]
    normal=np.einsum('nt,ntp->np',_poly(u,atlas['powers'])[0],atlas['coefficients'][ids])
    target=atlas['X'][ids]+radii[:,None]*(np.einsum('npa,na->np',q,u)+normal)
    scale=np.sqrt(np.mean(np.sum(atlas['X']**2,axis=1)))
    return target,dict(iterations=1,maximum_normal_residual_relative=0.,
        maximum_displacement_relative=float(np.max(np.linalg.norm(target-x,axis=1))/scale),
        method='frozen nearest polynomial graph; tangential coordinates retained')


def main():
    start=time.monotonic();scratch=Path('/private/tmp/takers_curved_fields_20261022')
    m=pickle.loads((scratch/'current_candidate.pkl').read_bytes())
    core.retract_to_atlas=local_retract
    m.pop('strict_flows',None);m.pop('flow_domain_diagnostics',None)
    core.enable_geometry_constrained_flows(m)
    dual,n=core.constrained_flow_duality(m,m['X'])
    print('DUALITY',dual,'reference',m['diagnostics']['flow_reference_displacement_relative'],flush=True)
    gate=_training_gates_once(m) if dual<.15 else dict(passed=False,duality_rms=dual)
    print('GATES',json.dumps(jsonable(gate)),flush=True)
    pickle.dump(m,(scratch/'local_retracted_model.pkl').open('wb'))
    report=dict(gates=gate,elapsed_seconds=time.monotonic()-start,numerical_threads=1)
    Path(__file__).with_name('local_retraction_probe.json').write_text(json.dumps(jsonable(report),indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
