"""Separate geometry error from frame approximation on the curved example.

Synthetic phases are used ONLY for the diagnostic reference, not for the
overlap fits. Run sequentially with one numerical thread.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/takers_curved_fields_20261022/mpl')
import json
import pickle
from pathlib import Path
import sys
import time
import numpy as np
from scipy import linalg
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'code/lag_ssa'))
from fields_first_frame import dictionary, fps
from fields_first_phase import overlap_maps
from fields_first_tangent_atlas import evaluate_tangent_frame
from fields_first_preprocess import highpass_decompose
from fields_first_sensor_cases import observation_factory
from tangent_overlap import windows
from run_fields_first_sensor import jsonable

SCRATCH = Path('/private/tmp/takers_curved_fields_20261022')


def fit_coefficients(Z, q, a, x, spatial_weight):
    n, r, k = Z.shape
    residual = (Z[1:]-np.einsum('nab,nbk->nak', a, Z[:-1])).reshape(-1, k)/np.sqrt((n-1)*r)
    anchor = Z[0]
    c0 = anchor.T@linalg.solve(anchor@anchor.T, np.eye(r), assume_a='pos')
    null = linalg.null_space(anchor)
    blocks = [residual, np.sqrt(1e-6/k)*np.eye(k)]
    if spatial_weight:
        ambient = np.einsum('npa,nak->npk', q, Z)
        near = NearestNeighbors(n_neighbors=2).fit(x).kneighbors(x, return_distance=False)[:, 1]
        smooth = (ambient-ambient[near]).reshape(-1, k)/np.sqrt(n*x.shape[1])
        blocks.append(np.sqrt(spatial_weight)*linalg.qr(smooth, mode='economic')[1])
    system = np.vstack(blocks)
    coeff = c0 + null@linalg.lstsq(system@null, -system@c0, cond=1e-10, lapack_driver='gelsy')[0]
    return coeff


def main():
    began = time.monotonic()
    previous = Path('/private/tmp/takers_curved_r_fix_20261022')
    data = pickle.loads((previous/'curved_inputs.pkl').read_bytes())
    atlas = pickle.loads((previous/'curved_atlas_L4_k48.pkl').read_bytes())
    residual = data['residual']; lag = 4
    mean, scale = residual.mean(0), residual.std(0)
    full = windows((residual-mean)/scale, lag)/4
    _, singular, vh = linalg.svd(full, full_matrices=False)
    basis = vh[singular>singular[0]*1e-10].T
    x = full@basis
    assert np.max(abs(x-atlas['X'])) < 1e-12
    q = evaluate_tangent_frame(atlas, x)['Q']
    a, overlap = overlap_maps(q, 4, basis)
    print('GEOMETRY', overlap, flush=True)

    # A separate diagnostic reference, never passed to fit_coefficients.
    theta = data['truth']['theta'][:2400]
    observe = observation_factory('threephase4sensor', 31)
    reference = []
    eps = 1e-5
    for j in range(3):
        shifted = theta.copy(); shifted[:, j] += eps
        plus = highpass_decompose(data['preprocess'], observe(shifted))
        shifted[:, j] -= 2*eps
        minus = highpass_decompose(data['preprocess'], observe(shifted))
        derivative = ((plus['residual']-minus['residual'])/(2*eps))[plus['valid_mask']]
        reference.append(windows(derivative/scale, lag)/4@basis)
    jacobian = np.stack(reference, axis=2)
    constant_basis = linalg.lstsq(jacobian[0], q[0])[0]
    true_w = jacobian@constant_basis
    pickle.dump(dict(jacobian=jacobian, true_w=true_w, theta=theta[95:95+len(x)]),
        (SCRATCH/'diagnostic_reference.pkl').open('wb'))
    true_c = np.einsum('npa,npr->nar', q, true_w)
    projected_true = np.einsum('npa,nar->npr', q, true_c)
    reference_report = dict(
        synthetic_phases_for_diagnosis_only=True,
        tangent_projection_relative=float(linalg.norm(projected_true-true_w)/linalg.norm(true_w)),
        projected_reference_overlap_relative=float(linalg.norm(true_c[1:]-a@true_c[:-1])/linalg.norm(true_c[1:])))
    if '--reference-only' in sys.argv:
        print('REFERENCE', reference_report, flush=True)
        return
    ids = fps(x, 64); centers = x[ids]; cq = q[ids]
    h = float(np.median(NearestNeighbors(n_neighbors=4).fit(centers).kneighbors(x)[0][:, -1]))
    local = dictionary(x, q, centers, cq, h)
    features = np.column_stack((np.ones(len(x)), x))
    linear = np.einsum('npa,nb->napb', q, features).reshape(len(x), 3, -1)
    records = []
    for name, design, spatial_weight in (
            ('local_original', local, 1e-3), ('local_no_spatial', local, 0.),
            ('linear_original', linear, 1e-3), ('linear_no_spatial', linear, 0.)):
        start = time.monotonic()
        coeff = fit_coefficients(design, q, a, x, spatial_weight)
        c = design@coeff
        w = np.einsum('npa,nar->npr', q, c)
        sv = linalg.svdvals(w.reshape(-1, 3))
        node_singular = np.linalg.svd(w, compute_uv=False)
        row = dict(name=name, columns=design.shape[-1], spatial_weight=spatial_weight,
            transfer_relative=float(linalg.norm(c[1:]-a@c[:-1])/linalg.norm(c[1:])),
            error_to_anchored_reference=float(linalg.norm(w-true_w)/linalg.norm(true_w)),
            minimum_singular=float(node_singular[:, -1].min()),
            minimum_relative_singular=float((node_singular[:, -1]/node_singular[:, 0]).min()),
            seconds=time.monotonic()-start)
        records.append(row)
        print('FRAME', json.dumps(row), flush=True)
        pickle.dump(dict(coeff=coeff, x=x, q=q, a=a, atlas=atlas, basis=basis,
            mean=mean, scale=scale, name=name, centers=centers, cq=cq, h=h),
            (SCRATCH/(name+'.pkl')).open('wb'))
    report = dict(reference=reference_report, results=records,
        elapsed_seconds=time.monotonic()-began, numerical_threads=1)
    Path(__file__).with_name('frame_diagnosis.json').write_text(json.dumps(jsonable(report), indent=2)+'\n')
    print('REFERENCE', reference_report, flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
