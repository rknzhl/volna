"""Численные проверки формул interpretation.tex, seed 20261002."""
import json
from pathlib import Path

import numpy as np


def windows(y, length):
    return np.lib.stride_tricks.sliding_window_view(y, length).T


def product(first, second):
    return np.einsum('ij,jk->ik', first, second, optimize=False)


def average(matrix):
    length, columns = matrix.shape
    return np.array([np.mean([matrix[a, n - a] for a in range(length)
                             if 0 <= n - a < columns])
                     for n in range(length + columns - 1)])


def chords(vector):
    pairs = vector.reshape(2, 2)
    lengths = np.linalg.norm(pairs, axis=1)
    perpendicular = np.column_stack([-pairs[:, 1], pairs[:, 0]])
    for j in range(2):
        perpendicular[j] = (perpendicular[j] / lengths[j]
                            if lengths[j] else np.array([0.0, 1.0]))
    midpoint = perpendicular * np.sqrt(1 - lengths[:, None] ** 2 / 4)
    return (midpoint + pairs / 2).ravel(), (midpoint - pairs / 2).ravel()


rng = np.random.default_rng(20261002)
results = {'seed': 20261002}
errors = []
for _ in range(200):
    amplitude = rng.uniform(.1, 3)
    phase = rng.uniform(.1, np.pi - .1)
    alpha = rng.uniform(-np.pi, np.pi)
    y = amplitude * np.cos(phase * np.arange(15) + alpha)
    matrix = windows(y, 6)
    expected = -amplitude ** 2 * np.sin(phase) ** 2
    errors.append(abs(np.linalg.det(matrix[:2, :2]) - expected))
    assert np.linalg.matrix_rank(matrix, tol=1e-10) == 2
results['harmonic_determinant_max_error'] = max(errors)

y = rng.normal(size=(4, 31))
length = 9
columns = 31 - length + 1
matrices = [windows(row, length) for row in y]
matrix = np.hstack(matrices)
u, singular, vh = np.linalg.svd(matrix, full_matrices=False)
group = np.array([0, 2, 5])
projector = product(u[:, group], u[:, group].T)
reconstructed = product(projector, matrix)
weights = np.convolve(np.ones(length), np.ones(columns))
assert np.array_equal(np.convolve(np.ones(3), np.ones(3)), [1, 2, 3, 2, 1])
restored = np.array([average(reconstructed[:, i * columns:(i + 1) * columns])
                     for i in range(4)])
results['mssa_gram_error'] = float(np.linalg.norm(
    product(matrix, matrix.T) - sum(product(m, m.T) for m in matrices)))
results['hankel_weight_error'] = float(abs(np.sum(matrix ** 2) - np.sum(weights * y ** 2)))
matrix_error = np.sum((matrix - reconstructed) ** 2)
tail = np.sum(np.delete(singular, group) ** 2)
results['svd_tail_error'] = float(abs(matrix_error - tail))
weighted_error = np.sum(weights * (y - restored) ** 2)
results['averaging_errors'] = {'weighted_series': float(weighted_error),
                               'matrix': float(matrix_error)}
assert weighted_error <= matrix_error + 1e-10
transformed = np.fft.fft(y - restored, norm='ortho', axis=1)
results['parseval_error'] = float(abs(np.sum((y - restored) ** 2) - np.sum(abs(transformed) ** 2)))

frequencies = 2 * np.pi * np.array([6, 6 * np.sqrt(2)])
delays = [.0, .037]
amplitudes = rng.uniform(.2, 2, size=(3, 2))
phases = rng.uniform(-np.pi, np.pi, size=(3, 2))
rows = []
for delay in delays:
    shifted = phases - frequencies * delay
    rows.extend(np.stack([amplitudes * np.cos(shifted),
                          -amplitudes * np.sin(shifted)], axis=-1).reshape(3, 4))
observation = np.array(rows)
time = np.arange(201) * .01
angles = time[:, None] * frequencies
p = np.stack([np.cos(angles), np.sin(angles)], axis=-1).reshape(-1, 4)
direct = np.concatenate([np.sum(amplitudes[None, :, :] * np.cos(
    (time[:, None, None] - delay) * frequencies + phases), axis=2)
    for delay in delays], axis=1)
results['delay_matrix_error'] = float(np.max(abs(direct - product(p, observation.T))))
assert np.linalg.matrix_rank(observation) == 4

collision_errors = []
unit_errors = []
for rank in range(4):
    for _ in range(50):
        left, _ = np.linalg.qr(rng.normal(size=(6, 6)))
        right, _ = np.linalg.qr(rng.normal(size=(4, 4)))
        deficient = product(left[:, :rank], right[:, :rank].T)
        vector = right[:, rank]
        vector /= np.max(np.linalg.norm(vector.reshape(2, 2), axis=1))
        first, second = chords(vector)
        unit_errors.append(max(np.max(abs(np.linalg.norm(first.reshape(2, 2), axis=1) - 1)),
                               np.max(abs(np.linalg.norm(second.reshape(2, 2), axis=1) - 1))))
        collision_errors.append(np.linalg.norm(deficient @ (first - second)))
        assert np.linalg.norm(first - second) > .1
results['rank_deficient_collision_max_error'] = float(max(collision_errors))
results['chord_unit_circle_max_error'] = float(max(unit_errors))
for vector in [np.array([0, 0, .3, -.5]), np.array([1, 0, 0, 0]), np.zeros(4)]:
    first, second = chords(vector)
    assert np.allclose(first - second, vector)
    assert np.allclose(np.linalg.norm(first.reshape(2, 2), axis=1), 1)
    assert np.allclose(np.linalg.norm(second.reshape(2, 2), axis=1), 1)

results['first_phase_count'] = len(set((3 * np.arange(500)) % 50))
assert results['first_phase_count'] == 50
for key, value in results.items():
    if key.endswith('error'):
        assert value < 1e-10, (key, value)
path = Path(__file__).with_name('interpretation-torus-checks-20261002.json')
path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(results, ensure_ascii=False, indent=2))
