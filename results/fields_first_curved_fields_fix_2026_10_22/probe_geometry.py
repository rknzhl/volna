"""Small geometry/reference controls; no period or field fits."""
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
sys.path.insert(0,str(ROOT/'code/lag_ssa'))
import fields_first_phase as core
from fields_first_tangent_atlas import fit_atlas,evaluate_tangent_frame
from evaluate_tangent_overlap_angles import match_phases
from run_fields_first_sensor import jsonable
S=Path('/private/tmp/takers_curved_fields_20261022')

def main():
    model=pickle.loads((S/'augmented_full_model.pkl').read_bytes())
    reference=pickle.loads((S/'diagnostic_reference.pkl').read_bytes())
    jac=reference['jacobian'];x=model['X'];started=time.monotonic();rows=[]
    for name,atlas in [('cubic',model['atlas']),
                       ('cubic_half_bandwidth',dict(model['atlas'],bandwidth=model['atlas']['bandwidth']/2)),
                       ('quartic',fit_atlas(x,3,k=48,degree=4))]:
        geo=evaluate_tangent_frame(atlas,x);q=geo['Q']
        projection=np.einsum('npa,nar->npr',q,np.einsum('npa,npr->nar',q,jac))
        row=dict(name=name,tangent_projection_relative=float(linalg.norm(projection-jac)/linalg.norm(jac)),
                 minimum_atlas_gap=float(geo['gap'].min()))
        rows.append(row);print('GEOMETRY',row,flush=True)
        if name!='cubic':pickle.dump(atlas,(S/(name+'_atlas.pkl')).open('wb'))
    # Ideal phase values and derivatives isolate the linear chart extension.
    ids=np.linspace(0,len(x)-1,128,dtype=int)
    match=match_phases(model['phases'],reference['theta']);g=match['winding_candidate']
    cx=model['complex'];land=cx['landmark_indices'];q=model['period_model']['normalized_potential'].T
    exact=reference['theta'][land]@g.T+match['origins']+model['origin']
    lifted=q+np.angle(np.exp(1j*(exact-2*np.pi*q)))/(2*np.pi)
    ideal=dict(model,period_model=dict(model['period_model'],normalized_potential=lifted.T))
    ideal['center_normalized_beta']=g@np.linalg.pinv(jac[land])/(2*np.pi)
    encoded=core.encode(ideal,x[ids],derivative=True)
    E=jac[ids]@np.linalg.inv(g)
    product=encoded['jacobian']@E
    original_basis_error=float(linalg.norm(product-np.eye(3)[None])/np.sqrt(len(ids)*3))
    product=np.linalg.inv(g)@product@g
    report=dict(geometry=rows,oracle_control_only=True,
        ideal_linear_encoder_original_basis_duality=original_basis_error,
        ideal_linear_encoder_source_basis_duality=float(linalg.norm(product-np.eye(3)[None])/np.sqrt(len(ids)*3)),
        elapsed_seconds=time.monotonic()-started,numerical_threads=1)
    Path(__file__).with_name('geometry_diagnosis.json').write_text(json.dumps(jsonable(report),indent=2)+'\n')
    print('ORACLE',report,flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
