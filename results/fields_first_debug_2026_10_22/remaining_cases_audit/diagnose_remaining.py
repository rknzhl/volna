"""Small discriminating probes for two unresolved cases; no field/flow fitting."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
            'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/takers_remaining_audit_mpl')

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits


def wrapped(value):
    return np.angle(np.exp(1j * value))


def prefixes(diagram):
    lives = diagram[:, 1] - diagram[:, 0]
    rows = []
    for count in range(1, min(6, len(diagram)) + 1):
        birth = float(diagram[:count, 0].max())
        death = float(diagram[:count, 1].min())
        gap = float(lives[count-1] / lives[count]) if count < len(lives) else None
        rows.append(dict(count=count, common_birth=birth, common_death=death,
            common_standard_range=2*birth < death, lifetime_gap=gap,
            lifetime_gap_at_least_two=gap is None or gap >= 2))
    return rows


def chirp_probe(root):
    from fields_first_sensor_cases import generate_variant
    from fields_first_sensor_pipeline import GATES
    source = json.loads((root/'results/fields_first_e2e_2026_10_20/scalarcircle_chirp_noise0p01.json').read_text())
    ntrain, ntest, lag, trim = source['ntrain'], source['ntest'], 4, 95
    _, evaluator = generate_variant('scalarcircle', ntrain+ntest, seed=31,
        noise=.01, trend=None, phase_mode='chirp', train_count=ntrain)
    theta = evaluator['theta'][:, 0]
    train_ids = np.arange(trim, ntrain-trim-lag+1)
    query_ids = np.arange(ntrain+trim, ntrain+ntest-trim-lag+1)
    train_step = wrapped(np.diff(theta[train_ids]))
    query_step = wrapped(np.diff(theta[query_ids]))
    frozen_step = float(np.angle(np.mean(np.exp(1j*train_step))))
    train_error = float(np.sqrt(np.mean(wrapped(train_step-frozen_step)**2)))
    query_error = float(np.sqrt(np.mean(wrapped(query_step-frozen_step)**2)))
    alpha = .41*np.sqrt(2)
    kappa = .4*alpha/ntrain
    ell = np.arange(lag)
    current_step = theta[train_ids+1]-theta[train_ids]
    phase = theta[train_ids, None]+ell*current_step[:, None]+kappa*ell*(ell-1)/2
    actual = theta[train_ids[:, None]+ell]
    derivative = (-.4*np.sin(phase)*np.exp(.4*np.cos(phase))
        +.25*np.cos(phase)-.3*np.sin(2*phase))
    jacobian = np.stack((derivative, derivative*ell), axis=2)
    singular = np.linalg.svd(jacobian, compute_uv=False)
    return dict(case='scalarcircle', phase_mode='chirp', ntrain=ntrain,
        ntest=ntest, lag=lag, boundary_trim=trim,
        oracle_quantities_for_diagnosis_only=True, fields_fitted=False,
        exact_increment_formula='alpha * (1 + 0.2*(2*n+1)/ntrain)',
        train_increment_range=[float(train_step.min()), float(train_step.max())],
        train_constant_increment=frozen_step,
        exact_phase_train_step_rmse=train_error,
        exact_phase_query_step_rmse_against_frozen_train_step=query_error,
        configured_phase_step_threshold=GATES['phase_step_rmse'],
        exact_phases_pass_train_gate=train_error < GATES['phase_step_rmse'],
        exact_phases_pass_query_gate=query_error < GATES['phase_step_rmse'],
        phase_frequency_window_map_formula='theta + ell*omega + kappa*ell*(ell-1)/2',
        window_phase_formula_max_error=float(np.max(np.abs(phase-actual))),
        phase_frequency_window_jacobian_min_second_singular=float(singular[:, 1].min()),
        phase_frequency_window_jacobian_second_to_first_ratio_quantiles=np.quantile(
            singular[:, 1]/singular[:, 0], [0, .5, 1]).tolist(),
        classification='nonstationary phase/frequency state violates the rigid phase-only torus model')


def curved_probe():
    from fields_first_sensor_cases import generate_variant
    from fields_first_preprocess import fit_preprocess, highpass_decompose
    from fields_first_periods import _infer_torus_dimension_once
    from tangent_overlap import windows
    y, evaluator = generate_variant('threephase4sensor', 2720, seed=31,
        noise=0., trend=None, train_count=2400)
    preprocessing = fit_preprocess(y[:2400], lag=32, maxrank=24, trend_degrees=(0,))
    split = highpass_decompose(preprocessing, y[:2400])
    valid = split['valid_mask']
    raw, filtered = y[:2400][valid], split['residual'][valid]
    recovered = np.column_stack((np.arctan2(y[:, 1], y[:, 0]),
        np.arctan2(y[:, 3], y[:, 2]), np.arctan2(
            np.hypot(y[:, 2], y[:, 3])-2, np.hypot(y[:, 0], y[:, 1])-2)))
    probes = []
    for label, values in [('raw', raw), ('preprocessed', filtered)]:
        x = windows((values-values.mean(0))/values.std(0), 8)/np.sqrt(8*values.shape[1])
        began = time.monotonic()
        r, diagnostic = _infer_torus_dimension_once(x, nland=384)
        diagram = np.asarray(diagnostic['h1_diagram'])
        probes.append(dict(input=label, states=len(x), lag=8, landmarks=384,
            inferred_dimension=r, selection=diagnostic['selection'],
            top_bars=diagram[:10].tolist(), prefix_diagnostics=prefixes(diagram),
            seconds=time.monotonic()-began))
    return dict(case='threephase4sensor', ntrain=2400, seed=31,
        raw_observations_sha256=hashlib.sha256(y.astype('<f8').tobytes()).hexdigest(),
        oracle_inverse_used_for_diagnosis_only=True,
        explicit_inverse_max_wrapped_phase_error=float(np.max(np.abs(wrapped(
            recovered-evaluator['theta'])))),
        raw_embedding_jacobian_min_singular=1.,
        selected_conditioning_rank=preprocessing['rank'],
        conditioning_relative_error=float(np.linalg.norm(split['denoised'][valid]-raw)/np.linalg.norm(raw)),
        fields_fitted=False, probes=probes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('diagnostics.json'))
    args = parser.parse_args()
    root = args.root.resolve()
    sys.path.insert(0, str(root/'code/lag_ssa'))
    began = time.monotonic()
    with threadpool_limits(limits=1):
        chirp = chirp_probe(root)
        print('CHIRP', json.dumps(chirp), flush=True)
        curved = curved_probe()
        print('CURVED', json.dumps(curved), flush=True)
    report = dict(maximum_numerical_threads=1, field_fits=0, inverse_flows=0,
        elapsed_seconds=time.monotonic()-began, chirp=chirp, curved=curved,
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
