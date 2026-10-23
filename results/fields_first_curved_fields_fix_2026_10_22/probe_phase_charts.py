"""Evaluate quadratic primitives with frozen transported fields and periods."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/takers_curved_fields_20261022/mpl')
from pathlib import Path
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
from fields_first_phase_charts import fit_phase_charts, evaluate_phase_charts
from run_fields_first_sensor import jsonable


def encode_with_charts(model, query, derivative=False):
    """Rejected quadratic extension, isolated from the production encoder."""
    if 'phase_charts' not in model:
        return core.encode(model, query, derivative=derivative)
    x = np.asarray(query); single = x.ndim == 1
    if single:
        x = x[None]
    cx = model['complex']; centers = cx['landmarks']
    edges = cx['edges']; r = model['r']; p = x.shape[1]; n = len(centers)
    C = np.zeros((n, n, r))
    C[edges[:, 0], edges[:, 1]] = cx['integer_cocycles'].T
    C[edges[:, 1], edges[:, 0]] = -cx['integer_cocycles'].T
    radius2 = cx['cover_radius']**2
    result = np.full((len(x), r), np.nan)
    jac = np.full((len(x), r, p), np.nan) if derivative else None
    covered = np.zeros(len(x), dtype=bool)
    for start in range(0, len(x), 128):
        xx = x[start:start+128]; diff = xx[:, None] - centers[None]
        d2 = np.sum(diff*diff, axis=2); active = d2 < radius2
        cover = active.any(axis=1); root = np.argmax(active, axis=1)
        logs = np.full_like(d2, -np.inf)
        logs[active] = radius2/(d2[active]-radius2)
        logs -= np.where(cover, np.max(logs, axis=1), 0.)[:, None]
        w = np.exp(logs)
        w /= np.maximum(w.sum(axis=1, keepdims=True), np.finfo(float).tiny)
        local, slopes = evaluate_phase_charts(model['phase_charts'], xx, derivative=derivative)
        local += C[root]
        value = 2*np.pi*np.einsum('nl,nlr->nr', w, local) - model.get('origin', 0.)
        value[~cover] = np.nan
        result[start:start+len(xx)] = np.angle(np.exp(1j*value))
        covered[start:start+len(xx)] = cover
        if derivative:
            factor = np.zeros_like(d2)
            factor[active] = -2*radius2/(d2[active]-radius2)**2
            loggrad = factor[:, :, None]*diff
            mean = np.einsum('nl,nlp->np', w, loggrad)
            dw = w[:, :, None]*(loggrad-mean[:, None])
            jj = 2*np.pi*(np.einsum('nlp,nlr->nrp', dw, local)
                         + np.einsum('nl,nlrp->nrp', w, slopes))
            jj[~cover] = np.nan
            jac[start:start+len(xx)] = jj
    return dict(phases=result[0] if single else result,
        covered=covered[0] if single else covered,
        jacobian=(jac[0] if single else jac) if derivative else None)


def main():
    started = time.monotonic()
    scratch = Path('/private/tmp/takers_curved_fields_20261022')
    model = pickle.loads((scratch/'augmented_full_model.pkl').read_bytes())
    model['phase_charts'] = fit_phase_charts(model)
    model.pop('origin', None)
    x = model['X']; r = model['r']
    model['origin'] = encode_with_charts(model, x[[0]])['phases'][0]
    encoded = encode_with_charts(model, x, derivative=True)
    ph = encoded['phases']; product = encoded['jacobian']@core.phase_field(model, x)
    d = model['diagnostics']
    d['encoder_field_duality_relative'] = float(linalg.norm(product-np.eye(r)[None])/np.sqrt(len(x)*r))
    d['phase_step'] = np.angle(np.mean(np.exp(1j*np.diff(ph, axis=0)), axis=0))
    d['train_phase_step_rmse'] = float(np.sqrt(np.mean(np.angle(np.exp(1j*(np.diff(ph, axis=0)-d['phase_step'])))**2)))
    model['phases'] = ph
    report = dict(duality_rms=d['encoder_field_duality_relative'],
        phase_step_rmse=d['train_phase_step_rmse'],
        charts=model['phase_charts']['diagnostics'],
        elapsed_seconds=time.monotonic()-started, fields_and_periods_refitted=False)
    pickle.dump(model, (scratch/'quadratic_full_model.pkl').open('wb'))
    Path(__file__).with_name('quadratic_probe.json').write_text(json.dumps(jsonable(report), indent=2)+'\n')
    print(json.dumps(jsonable(report)), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
