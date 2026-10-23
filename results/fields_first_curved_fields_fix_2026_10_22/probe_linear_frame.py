"""Diagnostic: same atlas/transport/period/phase algorithms, ambient affine W.

No generator phases or true phase count enter this probe. The saved atlas has
already been fitted to measurements; dimension inference is repeated here.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/takers_curved_fields_20261022/mpl')
from pathlib import Path
import argparse
import json
import pickle
import sys
import time
import numpy as np
from scipy import linalg
from threadpoolctl import threadpool_limits
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'code/lag_ssa'))
import fields_first_phase as core
from fields_first_periods import infer_torus_dimension, prepare_period_complex, fit_periods
from fields_first_sensor_pipeline import _training_gates
from run_fields_first_sensor import jsonable
SCRATCH = Path('/private/tmp/takers_curved_fields_20261022')


def affine_W(frame, query, derivative=False):
    x = np.asarray(query); single = x.ndim == 1
    if single: x = x[None]
    c = frame['affine_C']
    w = np.einsum('nb,pbr->npr', np.column_stack((np.ones(len(x)), x)), c)
    if not derivative: return w[0] if single else w
    jac = np.broadcast_to(c[:, 1:, :], (len(x),)+c[:, 1:, :].shape).copy()
    return (w[0], jac[0]) if single else (w, jac)


def main(augmented=False):
    started = time.monotonic()
    data = pickle.loads((SCRATCH/'linear_no_spatial.pkl').read_bytes())
    x, q, a = data['x'], data['q'], data['a']
    r, dimdiag = infer_torus_dimension(x)
    assert r == data['atlas']['r']
    frame = (pickle.loads((SCRATCH/'augmented_frame.pkl').read_bytes()) if augmented else
             dict(affine_C=data['coeff'].reshape(x.shape[1], x.shape[1]+1, r)))
    model = dict(X=x, r=r, atlas=data['atlas'], frame=frame, basis=data['basis'],
        lag=4, channels=4, mean=data['mean'], scale=data['scale'], normal=4.,
        diagnostics=dict(dimension_selected=r, dimension_diagnostic=dimdiag))
    if not augmented: core.evaluate_W = affine_W
    w = core.field(model, x); sv = np.linalg.svd(w, compute_uv=False)
    c = np.einsum('npa,npr->nar', q, w)
    d = model['diagnostics']
    d.update(frame_anchor_error=float(linalg.norm(w[0]-q[0])),
        frame_transfer_relative=float(linalg.norm(c[1:]-a@c[:-1])/linalg.norm(c[1:])),
        frame_min_singular=float(sv[:, -1].min()),
        frame_min_relative_singular=float((sv[:, -1]/sv[:, 0]).min()),
        atlas_min_gap=float(core.evaluate_tangent_frame(model['atlas'], x)['gap'].min()))
    print('COMPLEX', flush=True)
    cx = prepare_period_complex(x, r, nland=dimdiag['landmarks'])
    model['complex'] = cx
    assert cx['status'].startswith('prepared_'), cx['status']
    print('INTEGRALS', flush=True)
    geometry_cache = []
    if augmented:
        def cached_beta_action(fitted, query, direction):
            points = np.atleast_2d(query)
            tangent = core.evaluate_tangent_frame(fitted['atlas'], points)['Q']
            geometry_cache.append((points.copy(), np.asarray(direction).copy(), tangent))
            raw_w = core.evaluate_W(fitted['frame'], points)
            b = np.einsum('npa,npr->nar', tangent, raw_w)
            rhs = np.einsum('npa,np->na', tangent, direction)
            return np.linalg.solve(b, rhs[..., None])[..., 0]
        core.beta_action = cached_beta_action
    raw, pathdiag = core.fine_integrals(model, cx)
    if augmented:
        pickle.dump(geometry_cache, (SCRATCH/'integration_geometry.pkl').open('wb'))
    model['period_model'] = fit_periods(raw, cx)
    pm = model['period_model']
    assert pm['status'].startswith('fitted_'), pm['status']
    d['periods'] = pm['diagnostics']; d['integration'] = pathdiag
    model['center_normalized_beta'] = np.einsum('ab,lbp->lap', np.linalg.inv(pm['Pi']), core.beta(model, cx['landmarks']))
    model['origin'] = core.encode(model, x[[0]])['phases'][0]
    encoded = core.encode(model, x, derivative=True); ph = encoded['phases']
    product = encoded['jacobian']@core.phase_field(model, x)
    d['encoder_field_duality_relative'] = float(linalg.norm(product-np.eye(r)[None])/np.sqrt(len(x)*r))
    d['train_encoder_coverage'] = float(encoded['covered'].mean())
    d['phase_step'] = np.angle(np.mean(np.exp(1j*np.diff(ph, axis=0)), axis=0))
    d['train_phase_step_rmse'] = float(np.sqrt(np.mean(np.angle(np.exp(1j*(np.diff(ph, axis=0)-d['phase_step'])))**2)))
    d['commutators'] = core.commutator_diagnostics(model)
    model.update(phases=ph, status='conditional_fields_first_model')
    name = 'augmented' if augmented else 'linear'
    pickle.dump(model, (SCRATCH/(name+'_full_model.pkl')).open('wb'))
    print('PRE_FLOW', {key:d[key] for key in ('encoder_field_duality_relative', 'train_phase_step_rmse')}, flush=True)
    gate = _training_gates(model)
    pickle.dump(model, (SCRATCH/(name+'_full_model.pkl')).open('wb'))
    report = dict(diagnostics=d, gates=gate, elapsed_seconds=time.monotonic()-started,
        topology_diagnostic=cx['diagnostics'], numerical_threads=1,
        source='measurement-only affine overlap fit; same frozen atlas and original gates')
    Path(__file__).with_name(name+'_probe.json').write_text(json.dumps(jsonable(report), indent=2)+'\n')
    print('GATES', json.dumps(jsonable(gate)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--augmented', action='store_true')
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        main(args.augmented)
