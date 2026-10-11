"""Check conditional phase-count inference for one and three phase fields."""

from __future__ import annotations

from project_paths import RESULTS_DIR

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from method import (
    LagDesign, circular_mae, design_min_variance_lags, design_positive_lags,
    dmd_phase_coordinates,
    estimate_peaks, infer_frequency_lattice, torus_xyz, trajectory_matrix,
)
from run_controlled import AREA, N, PROFILES, poisson


OUT = RESULTS_DIR
OUT.mkdir(exist_ok=True)


def one_phase(phases: np.ndarray) -> np.ndarray:
    a, b, _ = PROFILES
    theta = phases[0]
    return (a[:,None] * (poisson(theta, .32)-1)[None,:]
            + .12*b[:,None]*np.cos(3*theta)[None,:])


def three_phase(phases: np.ndarray) -> np.ndarray:
    a,b,c = PROFILES
    theta,phi,psi = phases
    return (a[:,None]*np.cos(theta)[None,:]
            +b[:,None]*np.cos(phi)[None,:]
            +c[:,None]*np.cos(psi)[None,:]
            +.08*a[:,None]*np.cos(theta+phi+psi)[None,:])


def run_one(name: str, frequency: np.ndarray, generator) -> dict:
    n=np.arange(N)
    phases=frequency[:,None]*n[None,:]+np.arange(len(frequency))[:,None]*.37
    clean=generator(phases)
    scale=np.sqrt(np.mean(np.sum(AREA[:,None]*clean**2,axis=0)))
    y=clean+.05*scale*np.random.default_rng(12).standard_normal(clean.shape)
    peaks,power=estimate_peaks(y,AREA,count=8,suppress_harmonics=False)
    visible=peaks[power/power[0]>.0005]
    lattice=infer_frequency_lattice(visible,tolerance=2*np.pi/N,max_rank=3,max_order=6)
    positive=design_positive_lags(np.r_[frequency,-frequency],max_lag=256)
    length=max(max(positive.lags)+1,32)
    dense=design_min_variance_lags(np.r_[frequency,-frequency],length)
    horizon=LagDesign(np.arange(length),np.ones(length)/length,float('nan'),length)
    variants=[]
    saved={}
    for label,design,spatial in [
        ('positive_sparse_oracle',positive,AREA),
        ('positive_dense_oracle',dense,AREA),
        ('mssa_equal_horizon',horizon,np.ones(len(AREA))/len(AREA)),
    ]:
        h=trajectory_matrix(y,spatial,design)
        result=dmd_phase_coordinates(h,frequency,rank=max(8,2*len(frequency)+2))
        error=circular_mae(result['phases'],phases[:,:h.shape[1]])
        variants.append({'design':label,'lag_count':len(design.lags),
                         'max_lag':int(max(design.lags)),
                         'phase_mae_rad':error,
                         'dmd_frequency':np.angle(result['eigenvalues']).tolist()})
        saved[label]=result['phases']
    if len(frequency)==1:
        fig,ax=plt.subplots(figsize=(5,5))
        for label,phase in saved.items():
            q=phase[0,::3]
            ax.scatter(np.cos(q),np.sin(q),s=2,label=label,alpha=.55)
        ax.set_aspect('equal'); ax.legend(); ax.set_title('One-phase circle (recovered)')
    else:
        fig=plt.figure(figsize=(10,4))
        for i,(left,right) in enumerate([(0,1),(0,2)]):
            ax=fig.add_subplot(1,2,i+1,projection='3d')
            chart=torus_xyz(saved['positive_dense_oracle'][[left,right],::3])
            ax.scatter(*chart.T,c=np.mod(phases[left,::3][:len(chart)],2*np.pi),
                       cmap='twilight',s=2)
            ax.set_title(f'Three-torus factor ({left+1},{right+1})')
            ax.set_xticks([]);ax.set_yticks([]);ax.set_zticks([])
    fig.tight_layout();fig.savefig(OUT/f'{name}_phase_chart.png',dpi=150);plt.close(fig)
    return {'case':name,'true_rank':len(frequency),'frequencies':frequency.tolist(),
            'visible_peaks':visible.tolist(),'inferred_lattice':lattice,
            'variants':variants}


def main() -> None:
    one=run_one('one_phase_infinite',2*np.pi*np.array([.061*np.sqrt(2)]),one_phase)
    three=run_one('three_phase',2*np.pi*np.array([.061*np.sqrt(2),.073*np.sqrt(3),
                                                .083*np.sqrt(5)]),three_phase)
    payload=[one,three]
    (OUT/'dimension_metrics.json').write_text(json.dumps(payload,indent=2)+'\n')
    for result in payload:
        print(result['case'],'r=',result['inferred_lattice']['rank'],
              [(v['design'],round(v['phase_mae_rad'],3)) for v in result['variants']])


if __name__=='__main__':
    main()
