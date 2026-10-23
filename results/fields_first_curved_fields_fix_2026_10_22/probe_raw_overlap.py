"""Diagnostic fit: exact window overlap, approximate tangent regularization."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/takers_curved_fields_20261022/mpl')
import sys,pickle,json,time
from pathlib import Path
import numpy as np
from scipy import linalg
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code/lag_ssa'))
from fields_first_frame import dictionary
from run_fields_first_sensor import jsonable
S=Path('/private/tmp/takers_curved_fields_20261022')


def main():
    start=time.monotonic();d=pickle.loads((S/'linear_no_spatial.pkl').read_bytes())
    x,q,basis=d['x'],d['q'],d['basis'];n,p=x.shape;r=q.shape[-1]
    feature=np.column_stack((np.ones(n),x))
    F=np.einsum('pa,nb->npab',np.eye(p),feature).reshape(n,p,-1)
    k=F.shape[-1];Z=np.einsum('npa,npk->nak',q,F)
    prefix=basis[:-4];suffix=basis[4:]
    R=(np.einsum('ap,npk->nak',prefix,F[1:])-np.einsum('ap,npk->nak',suffix,F[:-1])).reshape(-1,k)/np.sqrt((n-1)*len(prefix))
    normal=(F-np.einsum('npa,nak->npk',q,Z)).reshape(-1,k)/np.sqrt(n*p)
    near=NearestNeighbors(n_neighbors=2).fit(x).kneighbors(x,return_distance=False)[:,1]
    spatial=(F-F[near]).reshape(-1,k)/np.sqrt(n*p)
    anchor=Z[0];c0=anchor.T@linalg.solve(anchor@anchor.T,np.eye(r),assume_a='pos');null=linalg.null_space(anchor)
    ridge=1e-10 if '--low-ridge' in sys.argv else 1e-6
    spatial_weight=0. if '--no-spatial' in sys.argv else 1e-3
    system=np.vstack((linalg.qr(R,mode='economic')[1],
        np.sqrt(1e-3)*linalg.qr(normal,mode='economic')[1],
        np.sqrt(spatial_weight)*linalg.qr(spatial,mode='economic')[1],np.sqrt(ridge/k)*np.eye(k)))
    coeff=c0+null@linalg.lstsq(system@null,-system@c0,cond=1e-10,lapack_driver='gelsy')[0]
    W=F@coeff
    reference=pickle.loads((S/'diagnostic_reference.pkl').read_bytes());J=reference['jacobian'];B=linalg.lstsq(q[0].T@J[0],np.eye(r))[0];true=J@B
    out=dict(ridge=ridge,spatial_weight=spatial_weight,full_overlap_relative=float(linalg.norm(R@coeff)/linalg.norm(W)*np.sqrt(n*p)),
        normal_relative=float(linalg.norm((normal@coeff)*np.sqrt(n*p))/linalg.norm(W)),
        reference_relative=float(linalg.norm(W-true)/linalg.norm(true)),
        minimum_singular=float(np.linalg.svd(W,compute_uv=False)[:,-1].min()),seconds=time.monotonic()-start)
    print(json.dumps(out),flush=True)
    tag='_no_spatial' if spatial_weight==0 else ''
    if ridge<1e-6:tag+='_low_ridge'
    Path(__file__).with_name('raw_overlap'+tag+'_probe.json').write_text(json.dumps(jsonable(out),indent=2)+'\n')
    pickle.dump(dict(affine_C=coeff.reshape(p,p+1,r),diagnostic=out),
        (S/('raw_overlap'+tag+'_frame.pkl')).open('wb'))


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
