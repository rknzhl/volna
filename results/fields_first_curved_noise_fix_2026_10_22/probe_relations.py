"""Approximate quadratic relations on TRAIN only; no angles or flow fitting."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[key]='1'
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/takers_curved_noise_fix_20261022/mpl')
import sys,pickle
from pathlib import Path
import numpy as np
from scipy import linalg
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code/lag_ssa'))
from run_fields_first_sensor import write_json

def main():
    report=[]
    for percent in (1,5,10):
        d=pickle.load(open(f'/private/tmp/takers_curved_noise_fix_20261022/noise{percent:02d}_inputs.pkl','rb'))
        for p in (12,13):
            x=d['fullx']@d['vh'][:p].T;rows,cols=np.triu_indices(p)
            f=np.column_stack((np.ones(len(x)),x,x[:,rows]*x[:,cols]));held=np.arange(len(x))%5==0
            _,s,vh=linalg.svd(f[~held],full_matrices=False)
            for tol in (1e-4,3e-4,1e-3,3e-3,6e-3,1e-2):
                v=vh[s<=s[0]*tol];row=dict(noise_percent=percent,p=p,tol=tol,relations=len(v))
                if len(v):
                    residual=np.max(abs(f@v.T),axis=1)/np.linalg.norm(f,axis=1)
                    C=np.zeros((len(v),p,p));C[:,rows,cols]=v[:,1+p:]/2;C[:,cols,rows]+=v[:,1+p:]/2
                    z=x[np.unique(np.linspace(0,len(x)-1,128,dtype=int))]
                    g=2*np.einsum('spq,nq->nsp',C,z)+v[:,1:1+p][None]
                    val=np.linalg.eigvalsh(np.einsum('nsp,nsq->npq',g,g))
                    row.update(validation_max=float(residual[held].max()),training_max=float(residual[~held].max()),
                       gap_min=float(np.min(1-np.maximum(val[:,2],0)/np.maximum(val[:,3],1e-30))),
                       normal_ratio_min=float(np.min(val[:,3]/np.maximum(val[:,-1],1e-30))))
                report.append(row)
                if len(v):print(row,flush=True)
    write_json(Path(__file__).parent/'near_relations.json',report)

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
