"""Recover F(x,theta) at held-out sensor locations under a smooth-space prior."""

from __future__ import annotations

from project_paths import RESULTS_DIR

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline

from method import (
    LagDesign, design_positive_lags, dmd_phase_coordinates,
    trajectory_matrix, voronoi_weights_1d,
)
from run_controlled import N, SENSORS, case_data, field


OUT = RESULTS_DIR
OUT.mkdir(exist_ok=True)
TRAIN = np.array([0,1,3,4,6,7])
TEST = np.array([2,5])


def fourier_features(phases: np.ndarray, degree: int = 3) -> np.ndarray:
    columns = [np.ones(phases.shape[1])]
    for a in range(-degree,degree+1):
        for b in range(-degree,degree+1):
            if (a,b) <= (0,0):
                continue
            angle = a*phases[0]+b*phases[1]
            columns.extend([np.cos(angle),np.sin(angle)])
    return np.stack(columns,axis=1)


def reconstruct_at_test_sensors(phases: np.ndarray, observed_train: np.ndarray,
                                clean_test: np.ndarray) -> tuple[float,np.ndarray]:
    x = fourier_features(phases)
    split = int(.7*len(x))
    coefficients = np.linalg.lstsq(x[:split],observed_train[:,:split].T,rcond=1e-8)[0].T
    predicted_coefficients = CubicSpline(SENSORS[TRAIN],coefficients,axis=0)(SENSORS[TEST])
    prediction = np.einsum('ij,kj->ki',x[split:],predicted_coefficients,optimize=False)
    target = clean_test[:,split:]
    denominator = np.sum((target-target.mean(axis=1,keepdims=True))**2)
    error = np.sqrt(np.sum((prediction-target)**2)/denominator)
    return float(error),prediction


def evaluate(name: str) -> dict:
    truth,frequency,nonlinear,weak=case_data(name)
    clean=field(truth,nonlinear,weak)
    rng=np.random.default_rng(25)
    noisy=clean+.03*np.sqrt(np.mean(clean**2))*rng.standard_normal(clean.shape)
    spatial=voronoi_weights_1d(SENSORS[TRAIN],(0,1))
    sparse=design_positive_lags(np.r_[frequency,-frequency])
    long=LagDesign(np.arange(max(sparse.lags)+1),
                   np.ones(max(sparse.lags)+1)/(max(sparse.lags)+1),
                   float('nan'),max(sparse.lags)+1)
    methods=[('positive_oracle',sparse,spatial),
             ('mssa_equal_horizon',long,np.ones(len(TRAIN))/len(TRAIN))]
    result=[]
    curves=[]
    for label,design,weights in methods:
        h=trajectory_matrix(noisy[TRAIN],weights,design)
        phase=dmd_phase_coordinates(h,frequency,rank=8)['phases']
        k=h.shape[1]
        score,prediction=reconstruct_at_test_sensors(phase,noisy[TRAIN,:k],clean[TEST,:k])
        result.append({'method':label,'space_time_test_nrmse':score,'lag_count':len(design.lags)})
        curves.append(prediction)
    k=N-max(sparse.lags)
    oracle_score,oracle_prediction=reconstruct_at_test_sensors(truth[:,:k],noisy[TRAIN,:k],clean[TEST,:k])
    result.append({'method':'oracle_phases','space_time_test_nrmse':oracle_score})
    curves.append(oracle_prediction)
    fig,ax=plt.subplots(figsize=(11,3.5))
    split=int(.7*k)
    times=np.arange(split,min(split+200,k))
    ax.plot(times,clean[TEST[0],times],color='black',lw=1.5,label='held-out truth')
    for row,curve in zip(result,curves):
        ax.plot(times,curve[0,:len(times)],lw=1,label=f"{row['method']}: {row['space_time_test_nrmse']:.3f}")
    ax.set_title(f'{name}: spatially held-out sensor at x={SENSORS[TEST[0]]:.2f}')
    ax.set_xlabel('sample index');ax.set_ylabel('field value');ax.legend(ncol=2,fontsize=8)
    fig.tight_layout();fig.savefig(OUT/f'{name}_spatial_prediction.png',dpi=150);plt.close(fig)
    return {'case':name,'train_sensor_indices':TRAIN.tolist(),
            'test_sensor_indices':TEST.tolist(),'spatial_prior':'cubic interpolation of Fourier coefficient profiles',
            'results':result}


def main() -> None:
    results=[evaluate(name) for name in ['two_modes','infinite_weak','infinite_strong']]
    (OUT/'spatial_decoder_metrics.json').write_text(json.dumps(results,indent=2)+'\n')
    for case in results:
        print(case['case'],[(x['method'],round(x['space_time_test_nrmse'],3)) for x in case['results']])


if __name__=='__main__':
    main()
