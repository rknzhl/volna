"""Check the new dimension rule at six previously accepted training geometries."""
import os
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[name] = '1'
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/takers_loop_closure_20261022/mpl')

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy import linalg
from threadpoolctl import threadpool_limits

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root/'code/lag_ssa'))
from fields_first_sensor_cases import generate_variant
from fields_first_preprocess import fit_preprocess, highpass_decompose
from fields_first_periods import infer_torus_dimension
from tangent_overlap import windows
from run_fields_first_sensor import jsonable


def main():
    comparisons = json.loads((root/'results/fields_first_debug_2026_10_22/baselines/summary.json').read_text())
    results = []
    with threadpool_limits(limits=1):
        for name in ('scalar_clean', 'scalar_noise10', 'two_waves',
                     'nonlinear_sensors', 'optical', 'three_phases_linear'):
            path = root/'results/fields_first_debug_2026_10_22/validation'/(name+'.json')
            source = json.loads(path.read_text())
            previous = next(row for row in comparisons if row['source'] == str(path))
            ntrain = source['ntrain']
            y, _ = generate_variant(source['case'], ntrain, seed=source['seed'],
                noise=source['noise'], trend=source['trend'], phase_mode=source['phase_mode'],
                rate_profile=source['rate_profile'], train_count=ntrain)
            input_hash = hashlib.sha256(y.astype('<f8').tobytes()).hexdigest()
            assert input_hash == previous['evaluation']['train_observations_sha256']
            preprocessing = fit_preprocess(y, lag=32, maxrank=24, trend_degrees=(0,))
            split = highpass_decompose(preprocessing, y)
            residual = split['residual'][split['valid_mask']]
            lag = source['fit']['selected_lag']
            joint = windows((residual-residual.mean(0))/residual.std(0), lag)/np.sqrt(lag*y.shape[1])
            _, singular, vh = linalg.svd(joint, full_matrices=False)
            states = joint @ vh[singular > singular[0]*1e-10].T
            dimension, diagnostic = infer_torus_dimension(states)
            old = next(attempt['diagnostics'] for attempt in source['fit']['attempts']
                if attempt.get('gates', {}).get('passed'))
            row = dict(name=name, lag=lag, input_sha256=input_hash,
                old_dimension=old['dimension_selected'], new_dimension=dimension,
                old_landmarks=old['dimension_diagnostic']['landmarks'],
                new_landmarks=diagnostic['landmarks'],
                geometry_hint=diagnostic['geometry_dimension_hint'],
                diagnostic=diagnostic)
            row['unchanged'] = (row['old_dimension'], row['old_landmarks']) == (
                row['new_dimension'], row['new_landmarks'])
            results.append(row)
            print(name, dimension, diagnostic['landmarks'], row['unchanged'], flush=True)
    report = dict(scope='dimension/resolution regression at previously accepted training lags',
        maximum_numerical_threads=1, field_fits=0, true_phases_used=False,
        all_unchanged=all(row['unchanged'] for row in results), results=results)
    Path(__file__).with_name('dimension_regression.json').write_text(
        json.dumps(jsonable(report), indent=2, allow_nan=False)+'\n')
    return 0 if report['all_unchanged'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
