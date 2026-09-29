"""Численное решение волнового уравнения в 1D, 2D и 3D."""
import numpy as np


def grid(N, d):
    """Узлы сетки и шаг."""
    x = np.linspace(0, 1, N)
    return np.meshgrid(*[x] * d, indexing="ij"), x[1] - x[0]


def lattice(axis, d):
    """Точки решётки с одинаковыми осями."""
    return np.stack(np.meshgrid(*[axis] * d, indexing="ij"), -1).reshape(-1, d)


def node(p, h):
    """Ближайший узел."""
    return tuple(np.rint(p / h).astype(int))


def spread(p, h):
    """Веса точечного источника на соседних узлах."""
    q = p / h
    base, frac = np.floor(q).astype(int), q - np.floor(q)
    return [(tuple(base + c), np.prod([f if ci else 1 - f for f, ci in zip(frac, c)])) for c in np.ndindex((2,) * len(q))]


def medium(kind, R, alpha=0.3):
    """Показатель преломления среды."""
    z = R[-1]
    if kind == "однородная":
        return np.ones_like(z)
    if kind == "слоистая":
        return np.select([z < 1 / 3, z < 2 / 3], [1.0, 1 + alpha], 1 + alpha / 2)
    if kind == "линза":
        return 1 + alpha * np.exp(-sum((x - 0.5) ** 2 for x in R) / (2 * 0.1 ** 2))
    if kind == "волновод":
        return 1 + alpha * np.exp(-(z - 0.5) ** 2 / (2 * 0.05 ** 2))
    raise ValueError(kind)


def laplacian(u, h):
    """Разностный лапласиан."""
    return sum(np.roll(u, 1, a) + np.roll(u, -1, a) - 2 * u for a in range(u.ndim)) / h ** 2


def sponge(R, width):
    """Затухание в слое у края."""
    dist = np.minimum.reduce([np.minimum(x, 1 - x) for x in R])
    return 3 * np.log(1000) / (2 * width) * np.clip(1 - dist / width, 0, None) ** 2


def absorb(new, old, courant):
    """Поглощающее условие на границе."""
    for a in range(new.ndim):
        N, O, C = (np.moveaxis(v, a, 0) for v in (new, old, courant))
        N[0] = O[0] + C[0] * (O[1] - O[0])
        N[-1] = O[-1] + C[-1] * (O[-2] - O[-1])


def solve(n, sources, points, t_end, dt_out=0.01, wall=None, snap_times=(), width=0.2):
    """Расчёт поля: времена, записи датчиков, снимки."""
    d, N = n.ndim, n.shape[0]
    R, h = grid(N, d)
    c = 1 / n
    c2 = c ** 2
    dt = 0.5 * h / (c.max() * np.sqrt(d))
    every = int(np.ceil(dt_out / dt))
    dt = dt_out / every
    damp = sponge(R, width) * dt / 2 if d > 1 else 0.0
    courant = c * dt / h
    src = [(spread(p, h), f) for p, f in sources]
    rec = [node(p, h) for p in points]
    snap_steps = {int(round(ts / dt)): ts for ts in snap_times}

    u_old, u = np.zeros(n.shape), np.zeros(n.shape)
    records, snaps = [], {}
    for m in range(1, int(round(t_end / dt)) + 1):
        force = c2 * laplacian(u, h)
        for weights, f in src:
            value = f((m - 1) * dt) / h ** d
            for i, w in weights:
                force[i] += w * value
        u_new = (2 * u - (1 - damp) * u_old + dt ** 2 * force) / (1 + damp)
        absorb(u_new, u, courant)
        if wall is not None:
            u_new[wall] = 0
        u_old, u = u, u_new
        if m % every == 0:
            records.append([u[i] for i in rec])
        if m in snap_steps:
            snaps[snap_steps[m]] = u.copy()
    return dt_out * np.arange(1, len(records) + 1), np.array(records).T, snaps


def exact(r, t, s, d):
    """Точное решение для точечного источника."""
    if d == 3:
        return s(t - r) * (t > r) / (4 * np.pi * r)
    x = np.linspace(0, 1, 4000)
    if d == 1:
        tau = x * np.maximum(t - r, 0)[:, None]
        return np.trapezoid(s(tau), tau, axis=1) / 2
    top = np.arccosh(np.maximum(t / r, 1.0))
    return top * np.trapezoid(s(t[:, None] - r * np.cosh(x * top[:, None])), x, axis=1) / (2 * np.pi)
