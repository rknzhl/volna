"""TRAIN-only spectral diagnostics; no persistence, fields or inverse flows."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/takers_curved_noise_fix_20261022/mpl')
import sys,pickle,time
from pathlib import Path
import numpy as np
from scipy import linalg
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code/lag_ssa'))
from fields_first_sensor_cases import generate_variant
from fields_first_preprocess import fit_preprocess,highpass_decompose
from fields_first_periods import _local_dimension_hint
from tangent_overlap import windows
from run_fields_first_sensor import write_json

def main():
    report=[];scratch=Path('/private/tmp/takers_curved_noise_fix_20261022')
    for noise in (.01,.05,.10):
        began=time.monotonic();Y,_=generate_variant('threephase4sensor',2720,seed=31,noise=noise,trend=None,train_count=2400)
        prep=fit_preprocess(Y[:2400],lag=32,maxrank=24,trend_degrees=(0,))
        split=highpass_decompose(prep,Y[:2400],window=129,polyorder=2)
        y=split['residual'][split['valid_mask']];mean=y.mean(0);scale=y.std(0)
        full=windows((y-mean)/scale,8)/np.sqrt(32)
        _,s,vh=linalg.svd(full,full_matrices=False)
        row=dict(noise=noise,conditioning_rank=prep['rank'],source_singular_relative=s/s[0],span_cutoffs=[])
        for tol in (1e-10,1e-4,1e-3,1e-2):
            ids=s>s[0]*tol;x=full@vh[ids].T
            hint,diag=_local_dimension_hint(x)
            feats=np.column_stack((np.ones(len(x)),x,x[:,np.triu_indices(x.shape[1])[0]]*x[:,np.triu_indices(x.shape[1])[1]]))
            held=np.arange(len(x))%5==0
            fs=linalg.svdvals(feats[~held])
            gaps=fs[:-1]/np.maximum(fs[1:],1e-30)
            largest=np.argsort(gaps)[-6:][::-1]
            row['span_cutoffs'].append(dict(tolerance=tol,dimension=x.shape[1],geometry_hint=hint,
                projection_relative=float(linalg.norm(full-x@vh[ids])/linalg.norm(full)),
                relation_singular_relative=fs/fs[0],largest_relation_gaps=[dict(index=int(i+1),gap=float(gaps[i]),tail=float(fs[i+1]/fs[0])) for i in largest]))
        row['seconds']=time.monotonic()-began;report.append(row)
        pickle.dump(dict(Y=Y,preprocess=prep,residual=y,fullx=full,s=s,vh=vh),open(scratch/f'noise{int(noise*100):02d}_inputs.pkl','wb'))
        print('SPECTRA',noise,'D',[(z['tolerance'],z['dimension'],z['geometry_hint'],z['projection_relative']) for z in row['span_cutoffs']],'seconds',row['seconds'],flush=True)
    write_json(Path(__file__).parent/'spectra.json',report)

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
