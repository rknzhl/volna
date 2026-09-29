"""Модели прогноза от M1 до M4."""
import numpy as np
import torch
from scipy.optimize import least_squares, minimize_scalar
from scipy.special import jv, spherical_jn, sph_harm
from torch.nn import Linear, Sequential, Tanh

from scenarios import GRID, WALL
from wave import grid, lattice, medium, solve


def peaks(Y, dt, k):
    """Сильнейшие пики спектра."""
    n = Y.shape[1]
    power = np.sum(np.abs(np.fft.rfft(Y - Y.mean(1, keepdims=True), n=16 * n, axis=1)) ** 2, axis=0)
    freq = np.fft.rfftfreq(16 * n, dt)
    top = [i for i in range(1, len(power) - 1) if power[i - 1] <= power[i] >= power[i + 1]]
    chosen = []
    for i in sorted(top, key=lambda i: -power[i]):
        if power[i] < 0.01 * power.max() or len(chosen) == k:
            break
        if all(abs(freq[i] - freq[j]) > 3 / (n * dt) for j in chosen):
            chosen.append(i)
    return freq[chosen]


def basis(t, freqs):
    """Постоянная, косинусы и синусы."""
    return np.column_stack([np.ones_like(t)] + [g(2 * np.pi * f * t) for f in freqs for g in (np.cos, np.sin)])


def tones(t, Y, k):
    """Уточнённые частоты."""
    def misfit(f):
        B = basis(t, f)
        return (B @ np.linalg.lstsq(B, Y.T, rcond=None)[0] - Y.T).ravel()
    return least_squares(misfit, peaks(Y, t[1] - t[0], k), method="lm").x


def amplitudes(t, Y, freqs):
    """Комплексные амплитуды датчиков на каждой частоте."""
    c = np.linalg.lstsq(basis(t, freqs), Y.T, rcond=None)[0][1:]
    return c[0::2] - 1j * c[1::2]


def continue_tones(t, Y, Z, freqs, t_new):
    """Прогноз суммой колебаний."""
    wave = lambda tt: sum(np.real(np.outer(z, np.exp(2j * np.pi * f * tt))) for z, f in zip(Z, freqs))
    return (Y - wave(t)).mean(1, keepdims=True) + wave(t_new)


def forecast_m1(t, Y, P, t_new, sc=None):
    """M1: подбор источников и среды по волновому уравнению."""
    if sc is None or sc["d"] == 3:
        return None
    freqs = tones(t, Y, 2)
    Z = amplitudes(t, Y, freqs)
    d = sc["d"]
    R, _ = grid(GRID[d], d)
    kind = sc.get("medium", "однородная")
    wall = R[0] >= WALL if sc.get("wall") else None
    fit_alpha = kind != "однородная"
    drive = lambda tt: sum(np.cos(2 * np.pi * nu * tt) for nu in freqs)

    def response(pos, alpha):
        """Амплитуды датчиков от единичного источника."""
        tt, U, _ = solve(medium(kind, R, alpha=alpha), [(pos, drive)], P, 2.5, wall=wall)
        return amplitudes(tt[tt > 1.5], U[:, tt > 1.5], freqs)

    def fit(G, j):
        """Отклик с наилучшей амплитудой."""
        return np.vdot(G[j], Z[j]) / np.vdot(G[j], G[j]) * G[j]

    cand = lattice(np.linspace(0.22, 0.78, {1: 15, 2: 7}[d]), d)
    best = None
    for alpha in np.linspace(0.05, 0.45, 5) if fit_alpha else [0.0]:
        scan = [response(q, alpha) for q in cand]
        costs = np.array([[np.linalg.norm(fit(G, j) - Z[j]) ** 2 for j in range(2)] for G in scan])
        if best is None or costs.min(0).sum() < best[0]:
            best = (costs.min(0).sum(), alpha, [cand[i] for i in costs.argmin(0)])

    def unpack(p):
        return [np.clip(p[d * j:d * (j + 1)], 0.02, 0.98) for j in range(2)], (p[-1] if fit_alpha else 0.0)

    def residual(p):
        pos, alpha = unpack(p)
        r = np.concatenate([fit(response(pos[j], alpha), j) - Z[j] for j in range(2)])
        return np.concatenate([r.real, r.imag])

    x0 = np.concatenate(best[2] + ([[best[1]]] if fit_alpha else []))
    pos, alpha = unpack(least_squares(residual, x0, method="lm", diff_step=0.01, max_nfev=10 * (len(x0) + 1)).x)
    return continue_tones(t, Y, [fit(response(pos[j], alpha), j) for j in range(2)], freqs, t_new)


def harmonics(P, k):
    """Базис решений уравнения Гельмгольца."""
    X = P - P.mean(0)
    q, d = P.shape
    if d == 1:
        return np.column_stack([np.cos(k * X[:, 0]), np.sin(k * X[:, 0])])
    rho, phi = np.linalg.norm(X, axis=1), np.arctan2(X[:, 1], X[:, 0])
    if d == 2:
        M = (q // 2 - 1) // 2
        return np.column_stack([jv(m, k * rho) * np.exp(1j * m * phi) for m in range(-M, M + 1)])
    L = int(np.sqrt(q // 2)) - 1
    theta = np.arccos(np.clip(X[:, 2] / np.maximum(rho, 1e-12), -1, 1))
    return np.column_stack([spherical_jn(l, k * rho) * sph_harm(m, l, phi, theta)
                            for l in range(L + 1) for m in range(-l, l + 1)])


def forecast_m2(t, Y, P, t_new, sc=None):
    """M2: разложение по решениям уравнения Гельмгольца."""
    freqs = tones(t, Y, 8)
    Z = amplitudes(t, Y, freqs)
    D = np.linalg.norm(P[:, None] - P[None], axis=2)
    fitted = []
    for z, f in zip(Z, freqs):
        def fit(c):
            H = harmonics(P, 2 * np.pi * f / c)
            return H @ np.linalg.lstsq(H, z, rcond=None)[0]
        err = lambda c: np.linalg.norm(fit(c) - z)
        cs = np.geomspace(f * D[D > 0].min() / 4, 20 * f * D.max(), 301)
        i = int(np.argmin([err(c) for c in cs]))
        bounds = (cs[max(i - 1, 0)], cs[min(i + 1, len(cs) - 1)])
        fitted.append(fit(minimize_scalar(err, bounds=bounds, method="bounded", options={"xatol": 1e-12}).x))
    return continue_tones(t, Y, fitted, freqs, t_new)


def forecast_fourier(t, Y, P, t_new, sc=None):
    """M3: сумма колебаний каждого датчика."""
    out = []
    for y in Y:
        freqs = tones(t, y[None, :], 8)
        out.append(basis(t_new, freqs) @ np.linalg.lstsq(basis(t, freqs), y, rcond=None)[0])
    return np.array(out)


def forecast_autoencoder(t, Y, P, t_new, sc=None):
    """M3: автоэнкодер с линейным шагом по времени."""
    torch.manual_seed(0)
    scale = np.abs(Y).max()
    X = torch.tensor((Y / scale).T, dtype=torch.float32)
    q = X.shape[1]
    E = Sequential(Linear(q, 64), Tanh(), Linear(64, 8))
    D = Sequential(Linear(8, 64), Tanh(), Linear(64, q))
    K = Linear(8, 8, bias=False)
    opt = torch.optim.Adam([*E.parameters(), *D.parameters(), *K.parameters()], lr=1e-3)
    for _ in range(3000):
        z = E(X)
        loss = ((D(z) - X) ** 2).mean() + ((D(K(z[:-1])) - X[1:]) ** 2).mean() + ((K(z[:-1]) - z[1:]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        K.weight /= torch.clamp(torch.linalg.eigvals(K.weight).abs().max(), min=1.0)
        z, out = E(X[-1:]), []
        for _ in t_new:
            z = K(z)
            out.append(D(z)[0].numpy())
    return scale * np.array(out).T


def hankel(x, L):
    """Траекторная матрица."""
    return np.lib.stride_tricks.sliding_window_view(x, L).T


def subspace(X):
    """Главные сингулярные векторы до наибольшего скачка среди первых 20."""
    U, s, _ = np.linalg.svd(X, full_matrices=False)
    return U[:, :int(np.argmax(s[:20] / s[1:21])) + 1]


def recurrence(U):
    """Веса линейной рекурренты."""
    pi = U[-1]
    return U[:-1] @ pi / (1 - pi @ pi)


def smooth(X, U):
    """Сглаженный ряд."""
    Pr = U @ (U.T @ X)
    L, M = Pr.shape
    return np.array([np.mean(np.diagonal(Pr[::-1], offset=k - L + 1)) for k in range(L + M - 1)])


def extend(x, a, steps):
    """Продолжение ряда."""
    x = list(x)
    for _ in range(steps):
        x.append(a @ np.array(x[-len(a):]))
    return np.array(x[-steps:])


def forecast_ssa(t, Y, P, t_new, sc=None):
    """M4: SSA для каждого датчика."""
    L, out = Y.shape[1] // 2, []
    for y in Y:
        H = hankel(y, L)
        U = subspace(H)
        out.append(extend(smooth(H, U), recurrence(U), len(t_new)))
    return np.array(out)


def forecast_mssa(t, Y, P, t_new, sc=None):
    """M4: MSSA для всех датчиков вместе."""
    L = Y.shape[1] // 2
    U = subspace(np.hstack([hankel(y, L) for y in Y]))
    a = recurrence(U)
    return np.array([extend(smooth(hankel(y, L), U), a, len(t_new)) for y in Y])


MODELS = {"M1 уравнение": forecast_m1, "M2 гармоники": forecast_m2, "M3 Фурье": forecast_fourier,
          "M3 автоэнкодер": forecast_autoencoder, "M4 SSA": forecast_ssa, "M4 MSSA": forecast_mssa}
