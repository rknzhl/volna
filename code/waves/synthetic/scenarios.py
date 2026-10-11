"""Сценарии синтетических данных."""
import numpy as np

from wave import grid, lattice, medium, solve

NU = (6.0, 6.0 * 2 ** 0.5)  # Гц
AMP = (1.0, 0.7)
GRID = {1: 801, 2: 201, 3: 101}
DURATION = {1: 20.0, 2: 10.0, 3: 6.0}  # с
START = 1.5  # с
RAMP = 0.5  # с
WALL = 0.8

SCENARIOS = {
    "1D, бегущая волна": {"d": 1},
    "1D, стоячая волна": {"d": 1, "wall": True},
    "1D, встречные волны": {"d": 1, "sources": "both"},
    "1D, слоистая среда": {"d": 1, "medium": "слоистая"},
    "1D, шум источника": {"d": 1, "noise": 0.3},
    "2D, однородная": {"d": 2},
    "2D, слоистая": {"d": 2, "medium": "слоистая"},
    "2D, линза": {"d": 2, "medium": "линза"},
    "2D, волновод": {"d": 2, "medium": "волновод"},
    "3D, однородная": {"d": 3},
    "3D, линза": {"d": 3, "medium": "линза"},
}


def point(x, d):
    """Точка на средней линии."""
    return np.array([x] + [0.5] * (d - 1))


def ramp(t):
    """Множитель плавного включения источника."""
    s = np.clip(t / RAMP, 0, 1)
    return s ** 3 * (10 - 15 * s + 6 * s ** 2)


def band_noise(t, rng):
    """Случайный сигнал в полосе от 0.5 до 20 Гц."""
    freq = np.fft.rfftfreq(t.size, t[1] - t[0])
    x = np.fft.irfft(np.fft.rfft(rng.standard_normal(t.size)) * ((freq >= 0.5) & (freq <= 20.0)), t.size)
    return x / x.std()


def sources(kind, noise, d, rng, t):
    """Источники двух колебаний."""
    F = [np.gradient(ramp(t) * a * (np.sin(2 * np.pi * nu * t) + noise * band_noise(t, rng)), t) for a, nu in zip(AMP, NU)]
    f = [lambda tt, Fj=Fj: np.interp(tt, t, Fj) for Fj in F]
    if kind == "both":
        return [(point(0.25, d), f[0]), (point(0.8, d), f[1])]
    return [(point(0.25, d), lambda tt: f[0](tt) + f[1](tt))]


def simulate(sc, seed=0, snap_times=(START,), t_end=None):
    """Расчёт сценария."""
    d = sc["d"]
    R, _ = grid(GRID[d], d)
    n = medium(sc.get("medium", "однородная"), R)
    wall = R[0] >= WALL if sc.get("wall") else None
    rng = np.random.default_rng(seed)
    src = sources(sc.get("sources"), sc.get("noise", 0.0), d, rng, np.arange(0, START + DURATION[d] + 0.02, 0.001))
    P = lattice(np.linspace(0.4, 0.75, {1: 8, 2: 4, 3: 3}[d]), d)
    t, Y, snaps = solve(n, src, P, t_end or START + DURATION[d], wall=wall, snap_times=snap_times)
    keep = t > START
    return {"t": t[keep] - START, "Y": Y[:, keep], "n": n, "snaps": snaps,
            "sensors": P, "sources": np.array([p for p, _ in src])}
