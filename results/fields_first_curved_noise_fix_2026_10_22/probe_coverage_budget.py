"""TRAIN-only FPS coverage at intermediate budgets; no persistence or fields."""
import os
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'

import json
import pickle
import time
from pathlib import Path

import numpy as np
from scipy import linalg
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits


COUNTS = (1600, 1664, 1728, 1792, 1856)
FOLDER = Path(__file__).resolve().parent
CACHE = Path('/private/tmp/takers_curved_noise_fix_20261022/noise10_inputs.pkl')


def source_windows(cache, lag):
    if lag == 8:
        full, right = cache['fullx'], cache['vh']
    else:
        y = np.asarray(cache['residual'])
        scaled = (y-y.mean(axis=0))/y.std(axis=0)
        view = np.lib.stride_tricks.sliding_window_view(scaled, lag, axis=0)
        full = view.transpose(0, 2, 1).reshape(len(view), -1)/np.sqrt(lag*y.shape[1])
        _, _, right = linalg.svd(full, full_matrices=False)
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        x = full @ right[:12].T
    if not np.isfinite(x).all():
        raise FloatingPointError('nonfinite cached source coordinates')
    return x


def probe(x, prior):
    started = time.monotonic()
    metric = cdist(x, x)
    if not np.isfinite(metric).all():
        raise FloatingPointError('nonfinite source metric')
    metric_seconds = time.monotonic()-started
    np.fill_diagonal(metric, np.inf)
    direct_delta = .5*float(np.median(np.min(metric, axis=1)))
    np.fill_diagonal(metric, 0.)
    delta = float(prior['training_sampling_buffer'])
    if not np.isclose(delta, direct_delta, rtol=1e-11, atol=1e-13):
        raise ValueError('cached coordinates disagree with the previous TRAIN buffer')
    death = float(prior['selection']['common_death'])
    twice_birth = 2*float(prior['selection']['common_birth'])
    nearest = metric[0].copy(); rows = []
    fps_started = time.monotonic()
    for count in range(2, max(COUNTS)+1):
        index = int(np.argmax(nearest))
        np.minimum(nearest, metric[index], out=nearest)
        if count in COUNTS:
            eps = float(nearest.max())
            buffered_lower = 2*(eps+delta)
            lower = max(twice_birth, buffered_lower)
            rows.append(dict(landmarks=count, fill_distance=eps,
                buffered_threshold_lower=buffered_lower,
                allowed_lower_with_previous_birth=lower,
                margin_below_previous_death=death-lower,
                coverage_interval_exists_with_previous_bar=lower<death,
                farthest_train_index=int(np.argmax(nearest)),
                cumulative_fps_seconds=time.monotonic()-fps_started))
    passing = [row['landmarks'] for row in rows
               if row['coverage_interval_exists_with_previous_bar']]
    return dict(states=len(x), dimension=x.shape[1], fps_start_train_index=0,
        training_sampling_buffer=delta, direct_buffer_control=direct_delta,
        previous_common_death=death, previous_twice_common_birth=twice_birth,
        metric_seconds=metric_seconds, budgets=rows,
        smallest_probed_budget_with_coverage=min(passing) if passing else None,
        elapsed_seconds=time.monotonic()-started,
        certificate_scope='TRAIN coverage lower bound only; new landmarks require new persistence checks',
        persistence_or_fields_computed=False, phases_or_truth_used=False)


def main():
    started = time.monotonic()
    cache = pickle.load(CACHE.open('rb'))
    previous = json.loads((FOLDER/'noise10_limit1536.json').read_text())
    report = dict(maximum_numerical_threads=1, noise_fraction=.10, lags={})
    for lag in (4, 8):
        prior = next(a['diagnostics']['dimension_diagnostic']
            for a in previous['fit']['attempts'] if a.get('lag') == lag)
        report['lags'][str(lag)] = probe(source_windows(cache, lag), prior)
    report['elapsed_seconds'] = time.monotonic()-started
    (FOLDER/'coverage_budget.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
