"""Two fixed, bounded toy runs of the unchanged sensor-only pipeline.

Run serially: .venv/bin/python .../run_successful_toys.py sine (then torus).
Reference phases enter only the evaluation after predictions have been saved.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'

import argparse
import json
from pathlib import Path
import pickle
import subprocess
import sys
import time
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DATA = HERE/'data'
sys.path.insert(0, str(REPO/'code/lag_ssa'))


def write_json(path, value):
    def convert(v):
        return v.tolist() if hasattr(v, 'tolist') else str(v)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, default=convert, allow_nan=False)+'\n')
    temp.replace(path)


def write_arrays(path, value):
    temp = path.with_suffix('.npz.tmp')
    with temp.open('wb') as stream:
        np.savez_compressed(stream, **value)
    temp.replace(path)


def worker(case):
    from threadpoolctl import threadpool_limits
    from fields_first_sensor_pipeline import SensorPipeline
    from evaluate_tangent_overlap_angles import match_phases
    from run_fields_first_sensor import jsonable
    result = DATA/f'toy_{case}.json'
    array_path = DATA/f'toy_{case}_data.npz'
    report = json.loads(result.read_text())
    with np.load(array_path) as saved:
        arrays = {key: saved[key] for key in saved.files}
    ntrain = report['N_train']; started = time.monotonic()
    with threadpool_limits(limits=1):
        pipeline = SensorPipeline(max_candidates=2).fit(arrays['Y_train'],
            progress_callback=lambda s: write_json(DATA/f'toy_{case}.progress.json', s))
        fit_seconds = time.monotonic()-started
        prediction = pipeline.transform(arrays['Y_query'], start_index=ntrain)
    report.update(status=prediction['status'], fit_status=pipeline.status,
        transform_status=prediction['status'], fit=jsonable(pipeline.summary()),
        elapsed_seconds=time.monotonic()-started, fit_seconds=fit_seconds,
        transform_seconds=time.monotonic()-started-fit_seconds,
        transform=jsonable({k: v for k, v in prediction.items() if k not in (
            'reconstruction','residual_reconstruction','slow','conditioned_observations',
            'raw_time_indices','phases','window_raw_time_indices')}))
    if pipeline.model is not None:
        model = pipeline.model
        report['hat_r'] = int(model['r'])
        report['lag'] = report['L'] = int(model['lag'])
        arrays['train_phases'] = model['phases']
        arrays['train_window_raw_time_indices'] = pipeline.train_window_raw_indices
        arrays['model_X'] = model['X']
    if prediction['status'] != 'transformed_conditionally':
        write_arrays(array_path, arrays); write_json(result, report)
        return
    for key in ('reconstruction','residual_reconstruction','slow','conditioned_observations',
                'raw_time_indices','phases','window_raw_time_indices'):
        arrays[key] = prediction[key]
    write_arrays(array_path, arrays)
    report['prediction_frozen_before_truth_evaluation'] = True
    match = match_phases(arrays['train_phases'],
        arrays['reference_theta'][arrays['train_window_raw_time_indices']])
    target = arrays['reference_theta'][arrays['window_raw_time_indices']] @ match['winding_candidate'].T + match['origins']
    phase_error = np.angle(np.exp(1j*(arrays['phases']-target)))
    sensor_error = (arrays['reconstruction']-arrays['clean'][arrays['raw_time_indices']])/arrays['clean'][:ntrain].std(axis=0)
    report.update(phase_mae_rad=float(np.mean(abs(phase_error))),
        sensor_nrmse=float(np.sqrt(np.mean(sensor_error**2))),
        primitive_phase_match=bool(report['hat_r']==report['reference_r'] and match['absolute_determinant']==1),
        N_eval_sensors=len(arrays['raw_time_indices']), N_eval_phases=len(arrays['window_raw_time_indices']),
        train_phase_evaluation=jsonable(match))
    arrays['aligned_reference_phases'] = target
    # The frozen model permits later plotting without a new fit.
    payload = pickle.dumps(pipeline, protocol=pickle.HIGHEST_PROTOCOL)
    report['model_pickle_bytes'] = len(payload)
    if len(payload) < 20*1024**2:
        path = DATA/f'toy_{case}_model.pkl'; path.write_bytes(payload)
        report['model_path'] = str(path.relative_to(HERE))
    # Geometry and all arrows below are evaluations of this fitted model.
    try:
        import fields_first_phase as core
        from fields_first_frame import fps
        cx = model['complex']
        for key in ('landmark_indices','landmarks','edges','integer_cocycles'):
            arrays['complex_'+key] = cx[key]
        delta = cx['delta1'].tocsr()
        if np.all(np.diff(delta.indptr)==3):
            tri_edges = delta.indices.reshape(-1,3)
            arrays['complex_triangles'] = np.sort(cx['edges'][tri_edges].reshape(-1,6),axis=1)[:,::2]
        ids = np.asarray(fps(model['X'],min(24,len(model['X']))), dtype=int)
        points = model['X'][ids]
        arrays.update(field_probe_indices=ids,field_probe_points=points,
            field_probe_phases=core.encode(model,points)['phases'],
            field_probe_E=core.phase_field(model,points),field_probe_W=core.field(model,points),
            field_probe_Dphi=core.encode(model,points,derivative=True)['jacobian'],
            field_probe_Q=core.evaluate_tangent_frame(model['atlas'],points)['Q'],
            period_matrix=model['period_model']['Pi'],
            reference_state=model.get('inverse_reference',model['X'][0]))
        states=[arrays['reference_state'].copy()]
        with threadpool_limits(limits=1):
            for _ in range(50):
                states.append(core.flow(model,states[-1],np.pi/50,axis=0)[0])
            arrays['flow_trace_states']=np.asarray(states)
            arrays['flow_trace_phases']=core.encode(model,arrays['flow_trace_states'])['phases']
        arrays['flow_trace_axis']=np.array(0)
        arrays['flow_trace_parameter']=np.linspace(0,np.pi,51)
    except Exception as error:
        report['optional_geometry_export_error']=repr(error)
    write_arrays(array_path, arrays)
    report['elapsed_seconds_with_export']=time.monotonic()-started
    report['array_keys']=sorted(arrays)
    write_json(result, report)


def main(case):
    DATA.mkdir(exist_ok=True)
    if case=='sine':
        dt,ntrain,nquery=.4,1200,400
        t=dt*np.arange(ntrain+nquery); y=np.sin(t)[:,None]
        theta=t[:,None]; equation='sin(t)'
    else:
        from fields_first_sensor_cases import generate_variant
        dt,ntrain,nquery=1.,2400,320
        t=np.arange(ntrain+nquery,dtype=float)
        y,truth=generate_variant('fundamental2waves',len(t),seed=31,noise=0.,trend=None,train_count=ntrain)
        theta=truth['theta']; equation='cos(theta_1-offset_1) + 0.6 cos(theta_2-offset_2)'
    array_path=DATA/f'toy_{case}_data.npz';result=DATA/f'toy_{case}.json'
    write_arrays(array_path,dict(t=t,Y=y,clean=y.copy(),reference_theta=theta,
        t_train=t[:ntrain],t_query=t[ntrain:],Y_train=y[:ntrain],Y_query=y[ntrain:],
        train_indices=np.arange(ntrain),query_indices=np.arange(ntrain,len(t))))
    write_json(result,dict(name=f'toy_{case}',equation=equation,status='running',dt=dt,
        N_train=ntrain,N_query=nquery,m=y.shape[1],reference_r=theta.shape[1],hat_r=None,
        noise_fraction=0.,trend=None,phase_mae_rad=None,sensor_nrmse=None,
        N_eval_sensors=0,N_eval_phases=0,maximum_candidates=2,maximum_numerical_threads=1,
        walltime_budget_seconds=90,no_truth_used_in_fit=True,
        prediction_frozen_before_truth_evaluation=False,
        sensor_nrmse_normalization='clean oscillatory TRAIN standard deviation (ddof=0)'))
    started=time.monotonic()
    try:
        with (DATA/f'toy_{case}.stdout').open('w') as log:
            proc=subprocess.run([sys.executable,str(Path(__file__).resolve()),case,'--worker'],
                cwd=REPO,stdout=log,stderr=subprocess.STDOUT,timeout=90)
        if proc.returncode:
            report=json.loads(result.read_text())
            report.update(status='unresolved_worker_error',worker_returncode=proc.returncode,
                elapsed_seconds=time.monotonic()-started)
            write_json(result,report)
    except subprocess.TimeoutExpired:
        report=json.loads(result.read_text())
        report.update(status='unresolved_walltime_budget',elapsed_seconds=time.monotonic()-started)
        write_json(result,report)
    report=json.loads(result.read_text())
    print(json.dumps({k:report.get(k) for k in ('name','status','hat_r','lag','phase_mae_rad',
        'sensor_nrmse','primitive_phase_match','elapsed_seconds','optional_geometry_export_error')}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('case',choices=('sine','torus'))
    parser.add_argument('--worker',action='store_true');args=parser.parse_args()
    worker(args.case) if args.worker else main(args.case)
