"""Численные попытки опровергнуть формулы спектрального конспекта."""
import json
from pathlib import Path

import numpy as np


def hankel(y, length):
    return np.lib.stride_tricks.sliding_window_view(y, length).T


def mssa(y, length):
    return np.hstack([hankel(row, length) for row in y])


def product(a, b):
    """Прямое умножение без платформенного BLAS."""
    return np.einsum("ij,jk->ik", a, b, optimize=False)


def average(matrix):
    length, columns = matrix.shape
    return np.array([np.mean([matrix[a, n - a] for a in range(length)
                              if 0 <= n - a < columns])
                     for n in range(length + columns - 1)])


def phase_matrix(z, frequencies, delays):
    rows = z[:, None, :] * np.exp(-1j * np.outer(delays, frequencies))[None, :, :]
    return np.stack([rows.real, -rows.imag], axis=-1).reshape(-1, 2 * len(frequencies))


rng = np.random.default_rng(7)
results = {}
y = rng.normal(size=(7, 31))
weights = rng.uniform(0.1, 2, size=7)
ft = np.fft.fft(y, axis=1, norm="ortho")
energy = np.sum(weights[:, None] * y ** 2)
results["weighted_parseval_error"] = float(abs(energy - np.sum(weights[:, None] * abs(ft) ** 2)))
results["fft2_singular_value_error"] = float(np.max(abs(
    np.linalg.svd(y, compute_uv=False) - np.linalg.svd(np.fft.fft2(y, norm="ortho"), compute_uv=False))))
length = 9
columns = y.shape[1] - length + 1
repeat = np.array([min(n + 1, length, columns, y.shape[1] - n) for n in range(y.shape[1])])
h = mssa(y, length)
u, s, vh = np.linalg.svd(h, full_matrices=False)
results["hankel_weight_identity_error"] = float(abs(np.sum(h ** 2) - np.sum(repeat * y ** 2)))
hr = product(u[:, :3] * s[:3], vh[:3])
yr = np.array([average(hr[:, i * columns:(i + 1) * columns]) for i in range(len(y))])
results["rank3_errors"] = {
    "matrix": float(np.sum((h - hr) ** 2)),
    "svd_tail": float(np.sum(s[3:] ** 2)),
    "weighted_series": float(np.sum(repeat * (y - yr) ** 2)),
    "series": float(np.sum((y - yr) ** 2)),
}
assert results["rank3_errors"]["series"] <= results["rank3_errors"]["weighted_series"] <= results["rank3_errors"]["matrix"] + 1e-10

length = columns = 60
n = np.arange(length + columns - 1)
frequencies = 2 * np.pi * np.array([5, 11]) / 60
y1 = np.cos(frequencies[0] * n)
y2 = 0.6 * np.cos(frequencies[1] * n + 0.3)
h1, h2 = hankel(y1, length), hankel(y2, length)
results["exact_separation"] = {
    "left_cross_norm": float(np.linalg.norm(product(h1.T, h2))),
    "right_cross_norm": float(np.linalg.norm(product(h1, h2.T))),
    "singular_values": np.linalg.svd(h1 + h2, compute_uv=False)[:4].tolist(),
}
u = np.linalg.svd(h1 + h2, full_matrices=False)[0][:, :2]
results["exact_separation"]["first_group_relative_error"] = float(np.linalg.norm(product(u, product(u.T, h1 + h2)) - h1) / np.linalg.norm(h1))
mixing = []
for delta in [6.0, 1.0, 0.4, 0.1]:
    y2 = 0.6 * np.cos(2 * np.pi * (5 + delta) / 60 * n + 0.3)
    h2 = hankel(y2, length)
    u = np.linalg.svd(h1 + h2, full_matrices=False)[0][:, :2]
    mixing.append({"bin_gap": delta, "relative_group_error": float(np.linalg.norm(product(u, product(u.T, h1 + h2)) - h1) / np.linalg.norm(h1))})
results["close_frequency_pair_mixing"] = mixing

phase_cases = []
for phase in [0, np.pi / 2]:
    z = np.array([[1], [2 * np.exp(1j * phase)]])
    yy = np.real(z * np.exp(1j * frequencies[0] * n))
    phase_cases.append({
        "phase": phase,
        "mssa_pair": np.linalg.svd(mssa(yy, length), compute_uv=False)[:2].tolist(),
        "instantaneous_phase_map_singular_values": np.linalg.svd(phase_matrix(z, frequencies[:1], [0]), compute_uv=False).tolist(),
    })
results["same_mssa_spectrum_different_geometry"] = phase_cases

z = rng.normal(size=(3, 2)) + 1j * rng.normal(size=(3, 2))
freq = np.array([1.0, np.sqrt(2)])
b = phase_matrix(z, freq, [0, 0.37])
sb = np.linalg.svd(b, compute_uv=False)
theta = rng.uniform(-np.pi, np.pi, size=(2000, 2))
psi = rng.uniform(-np.pi, np.pi, size=(2000, 2))
p = lambda angles: np.stack([np.cos(angles), np.sin(angles)], axis=-1).reshape(-1, 4)
d = p(theta) - p(psi)
stretch = np.linalg.norm(product(d, b.T), axis=1) / np.linalg.norm(d, axis=1)
results["embedding_chord_bound"] = {"sigma_min": float(sb[-1]), "sigma_max": float(sb[0]), "sampled_min": float(stretch.min()), "sampled_max": float(stretch.max())}
assert sb[-1] <= stretch.min() + 1e-12 and stretch.max() <= sb[0] + 1e-12

angles = 2 * np.pi * np.arange(1024) / 1024
circle = np.column_stack([np.cos(angles), np.sin(angles)])
c = np.array([[2.0, 0.7], [0.2, 0.8]])
ellipse = product(circle, c.T)
ev, axes = np.linalg.eigh(2 * product(ellipse.T, ellipse) / len(ellipse))
whitening = product(axes / np.sqrt(ev), axes.T)
results["ellipse_whitening_radius_error"] = float(np.max(abs(np.linalg.norm(product(ellipse, whitening.T), axis=1) - 1)))

signed = np.r_[freq, -freq]
coeff = np.hstack([z / 2, z.conj() / 2])
length, columns = 13, 27
v = np.exp(1j * np.outer(np.arange(length), signed))
vv = np.exp(1j * np.outer(np.arange(columns), signed))
cc = np.hstack([coeff[i, :, None] * vv.T for i in range(len(z))])
actual = product(cc, cc.conj().T)
expected = np.array([[np.sum(coeff[:, a] * coeff[:, bb].conj()) * np.sum(np.exp(1j * np.arange(columns) * (signed[a] - signed[bb]))) for bb in range(4)] for a in range(4)])
yy = np.real(product(z, np.exp(1j * np.outer(freq, np.arange(length + columns - 1)))))
results["signed_factorization_error"] = float(np.linalg.norm(product(v, cc) - mssa(yy, length)))
results["signed_gram_identity_error"] = float(np.linalg.norm(actual - expected))

delay = np.array([0, 0.37])
design = np.column_stack([np.ones(101)] + [g(f * np.arange(101) * 0.1) for f in freq for g in (np.cos, np.sin)])
dz = 0.001 * (rng.normal(size=z.shape) + 1j * rng.normal(size=z.shape))
dc = np.column_stack([np.zeros(len(z)), np.stack([dz.real, -dz.imag], axis=-1).reshape(len(z), -1)])
ee = product(dc, design.T)
db = phase_matrix(z + dz, freq, delay) - b
sd = np.linalg.svd(design, compute_uv=False)[-1]
observed = np.linalg.norm(db, ord=2)
bound = np.sqrt(len(delay)) * np.linalg.norm(ee) / sd
perturbed = np.linalg.svd(b + db, compute_uv=False)[-1]
results["fixed_frequency_certificate"] = {"coefficient_matrix_error": float(observed), "error_bound": float(bound), "perturbed_sigma_min": float(perturbed), "lower_bound": float(sb[-1] - observed)}
assert observed <= bound + 1e-12 and perturbed >= sb[-1] - observed - 1e-12

collision_error = []
for _ in range(50):
    deficient = rng.normal(size=(3, 4))
    kernel = np.linalg.svd(deficient, full_matrices=True)[2][-1].reshape(2, 2)
    diff = kernel / np.max(np.linalg.norm(kernel, axis=1))
    perp = np.column_stack([-diff[:, 1], diff[:, 0]])
    perp /= np.linalg.norm(perp, axis=1, keepdims=True)
    mid = perp * np.sqrt(1 - np.sum(diff ** 2, axis=1, keepdims=True) / 4)
    first, second = mid + diff / 2, mid - diff / 2
    assert np.allclose(np.linalg.norm(first, axis=1), 1) and np.allclose(np.linalg.norm(second, axis=1), 1)
    collision_error.append(float(np.linalg.norm(product(deficient, (first - second).reshape(4, 1)))))
results["rank_deficient_collision_error"] = max(collision_error)

zz = rng.normal(size=200) + 1j * rng.normal(size=200)
eta = rng.uniform(0, 0.99, size=200)
pert = eta * abs(zz) * np.exp(1j * rng.uniform(-np.pi, np.pi, size=200))
angles = abs(np.angle((zz + pert) * zz.conj()))
assert np.all(angles <= np.arcsin(eta) + 1e-12)
results["phase_error_bound_max_ratio"] = float(np.max(angles / np.arcsin(eta)))

for key in ["weighted_parseval_error", "fft2_singular_value_error", "hankel_weight_identity_error", "ellipse_whitening_radius_error", "signed_factorization_error", "signed_gram_identity_error"]:
    assert results[key] < 1e-10, (key, results[key])
results["seed"] = 7
results["numpy"] = np.__version__
results["matrix_products"] = "numpy.einsum, optimize=False"
destination = Path(__file__).with_name("spectral_checks.json")
destination.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
print(destination)
print(json.dumps(results, ensure_ascii=False, indent=2))
