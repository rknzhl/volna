"""Three explanatory figures for sections 3--5 of phase_ssa_method.tex.

Run from the repository root with `.venv/bin/python
code/lag_ssa/illustrations/method_figures.py`.  No manuscript is edited.
The pictured torus is a geometric explanatory model, not a PCA projection
or the raw block-window embedding in R^(mL).  Block selectors are used only
in the separate exact window-overlap inset of figure 3.
"""

from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "tmp" / "method_figures_mpl"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle
from matplotlib.backends.backend_pdf import PdfPages

OUT = ROOT / "results" / "lag_ssa" / "method_illustrations"
PDF_OUT = ROOT / "output" / "pdf"
OUT.mkdir(parents=True, exist_ok=True)
PDF_OUT.mkdir(parents=True, exist_ok=True)

# Only two accent colors; all other marks are neutral.
BLUE = "#236F9A"
ORANGE = "#C7782F"
INK = "#26333F"
GREY = "#65717B"
PALE = "#DCE3E7"
LIGHT = "#F3F5F6"
WHITE = "#FFFFFF"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 13,
    "mathtext.fontset": "stix",
    "text.color": INK,
    "axes.labelcolor": INK,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "path",
    "savefig.facecolor": WHITE,
    "lines.solid_capstyle": "round",
    "lines.solid_joinstyle": "round",
})

ALPHA = np.array([0.45, 0.60 * np.sqrt(2)])
R45 = np.array([[1., -1.], [1., 1.]]) / np.sqrt(2)
B = np.diag([1 / 3, 1.]) @ R45
PI = 2 * np.pi * np.linalg.inv(B)
THETA_STAR = np.zeros(2)


def psi(theta):
    """Torus embedding, vectorized over the final coordinate dimension."""
    theta = np.asarray(theta)
    u, w = theta[..., 0], theta[..., 1]
    r = 2 + np.cos(w)
    return np.stack([r * np.cos(u), r * np.sin(u), np.sin(w)], axis=-1)


def dpsi(theta):
    u, w = np.asarray(theta)
    return np.array([
        [-(2 + np.cos(w)) * np.sin(u), -np.sin(w) * np.cos(u)],
        [(2 + np.cos(w)) * np.cos(u), -np.sin(w) * np.sin(u)],
        [0., np.cos(w)],
    ])


def frame(theta):
    """Invariant frame; no pointwise normalization is applied."""
    return dpsi(theta) @ B


def orthoframe(theta):
    u, w = np.asarray(theta)
    e = np.array([
        [-np.sin(u), -np.sin(w) * np.cos(u)],
        [np.cos(u), -np.sin(w) * np.sin(u)],
        [0., np.cos(w)],
    ])
    return e @ R45


def transfer(theta):
    """Derivative of theta -> theta+alpha, restricted to tangent spaces."""
    return dpsi(np.asarray(theta) + ALPHA) @ np.linalg.pinv(dpsi(theta))


def coefficient_transfer(theta):
    ratio = (2 + np.cos(theta[1] + ALPHA[1])) / (2 + np.cos(theta[1]))
    return R45.T @ np.diag([ratio, 1.]) @ R45


# A fixed orthographic camera.  All torus arrows are exact 3D tangents
# projected by this same linear map, rather than aesthetically normalized.
AZ, EL = np.deg2rad([-65., 30.])
CAMERA = np.array([
    [-np.sin(AZ), np.cos(AZ), 0.],
    [-np.sin(EL) * np.cos(AZ), -np.sin(EL) * np.sin(AZ), np.cos(EL)],
])


def project(xyz):
    return np.asarray(xyz) @ CAMERA.T


def new_figure(number, title, subtitle):
    fig = plt.figure(figsize=(11.7, 8.3), facecolor=WHITE)
    fig.text(.042, .952, str(number), color=BLUE, fontsize=23, weight="bold")
    fig.text(.092, .952, title, fontsize=22, weight="bold")
    fig.text(.042, .910, subtitle, fontsize=13.5, color=GREY)
    fig.add_artist(plt.Line2D([.042, .958], [.891, .891], color=PALE, lw=1))
    return fig


def blank_axes(fig, rect, xlim=None, ylim=None, equal=True):
    ax = fig.add_axes(rect)
    ax.set_axis_off()
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if equal:
        ax.set_aspect("equal")
    return ax


def arrow(ax, start, end, color=INK, lw=1.9, ms=13, style="-|>", **kwargs):
    patch = FancyArrowPatch(start, end, arrowstyle=style, color=color,
                            mutation_scale=ms, lw=lw, shrinkA=0, shrinkB=0,
                            **kwargs)
    ax.add_patch(patch)
    return patch


def fig_arrow(fig, start, end, label=None, color=GREY, label_offset=.025):
    patch = FancyArrowPatch(start, end, transform=fig.transFigure,
                            arrowstyle="-|>", color=color, lw=1.4,
                            mutation_scale=12, shrinkA=0, shrinkB=0)
    fig.add_artist(patch)
    if label:
        fig.text((start[0] + end[0]) / 2,
                 (start[1] + end[1]) / 2 + label_offset,
                 label, ha="center", fontsize=14, color=INK)


def label(ax, point, text, offset=(0, 0), color=INK, fontsize=15,
          ha="left", va="center", leader=False):
    xy = np.asarray(point)
    pos = xy + np.asarray(offset)
    if leader:
        ax.plot([xy[0], pos[0]], [xy[1], pos[1]], color=GREY, lw=.8, zorder=7)
    ax.text(*pos, text, color=color, fontsize=fontsize, ha=ha, va=va,
            zorder=10, bbox=dict(facecolor=WHITE, edgecolor="none", pad=1.4, alpha=.92))


def torus_wire(ax, alpha=1., lw=.65):
    q = np.linspace(0, 2 * np.pi, 241)
    for v in np.linspace(0, 2 * np.pi, 9, endpoint=False):
        curve = project(psi(np.column_stack([q, np.full_like(q, v)])))
        ax.plot(curve[:, 0], curve[:, 1], color=PALE, lw=lw, alpha=alpha, zorder=1)
    for u in np.linspace(0, 2 * np.pi, 16, endpoint=False):
        curve = project(psi(np.column_stack([np.full_like(q, u), q])))
        ax.plot(curve[:, 0], curve[:, 1], color=PALE, lw=lw, alpha=alpha, zorder=1)


def curve_arrow(ax, xy, at=.7, color=INK, lw=1.8, ms=11):
    n = len(xy)
    i = min(n - 5, max(1, int(at * n)))
    j = min(n - 1, i + max(3, n // 28))
    arrow(ax, xy[i], xy[j], color=color, lw=lw, ms=ms, zorder=8)


def model_footer(fig, extra=""):
    fig.add_artist(plt.Line2D([.042, .958], [.087, .087], color=PALE, lw=1))
    fig.text(.042, .060, "Геометрическая модель многообразия окон, $r=2$:",
             fontsize=11.5, color=GREY)
    fig.text(.042, .032,
             r"$\psi(\theta)=((2+\cos\theta_2)\cos\theta_1,\ "
             r"(2+\cos\theta_2)\sin\theta_1,\ \sin\theta_2)$",
             fontsize=13, color=INK)
    if extra:
        fig.text(.958, .049, extra, fontsize=11.5, color=GREY, ha="right")


def make_transport():
    fig = new_figure(3, "Перенос касательных векторов",
                     "Производная сдвига переносит малое изменение окна за один шаг времени.")
    fig.text(.045, .856, "1. Сдвиг на многообразии", fontsize=14, weight="bold")
    fig.text(.386, .856, "2. Касательная плоскость", fontsize=14, weight="bold")
    fig.text(.691, .856, "3. Образ при переносе", fontsize=14, weight="bold")

    ax = blank_axes(fig, [.036, .470, .314, .351], (-3.35, 3.5), (-2.4, 2.35))
    torus_wire(ax)
    s = np.linspace(0, 1, 100)
    time_curve = project(psi(s[:, None] * ALPHA))
    ax.plot(time_curve[:, 0], time_curve[:, 1], color=INK, lw=2, zorder=4)
    curve_arrow(ax, time_curve, at=.5, ms=12)
    z0, z1 = project(psi([0., 0.])), project(psi(ALPHA))
    ax.scatter([z0[0], z1[0]], [z0[1], z1[1]], s=36, c=INK, zorder=8)
    label(ax, z0, r"$z$", offset=(-.10, -.48), ha="center")
    label(ax, z1, r"$\mathcal{T}z$", offset=(.10, .42), ha="center")
    label(ax, (time_curve[45] + time_curve[50]) / 2, r"$\Delta t$",
          offset=(-.88, .06), ha="center", leader=True, fontsize=15)
    ax.text(-2.65, 1.9, "$M$", fontsize=20, color=GREY)
    fig.text(.042, .442, r"$z=\psi(\theta),\quad\mathcal{T}z=\psi(\theta+\alpha)$", fontsize=14)

    # These two planes are displayed in their own orthonormal coordinates,
    # so circle/ellipse shapes show the metric, without perspective distortion.
    ang = np.linspace(0, 2 * np.pi, 360)
    circle = np.column_stack([np.cos(ang), np.sin(ang)])
    A = coefficient_transfer(np.zeros(2))
    ellipse = circle @ A.T
    b = np.array([.62, np.sqrt(1 - .62**2)])
    ab = A @ b
    ratio = (2 + np.cos(ALPHA[1])) / 3

    for rect, center_text, plane_text, basis_text, vertices, vec, col in [
        ([.382, .469, .282, .350], "$z$", r"$T_zM$", "$Q(z)$", circle, b, BLUE),
        ([.688, .469, .282, .350], "$\mathcal{T}z$", r"$T_{\mathcal{T}z}M$", "$Q(\mathcal{T}z)$", ellipse, ab, ORANGE),
    ]:
        aa = blank_axes(fig, rect, (-1.34, 1.34), (-1.36, 1.36))
        aa.add_patch(Rectangle((-1.19, -1.19), 2.38, 2.38,
                               facecolor=LIGHT, edgecolor=PALE, lw=.8, zorder=0))
        aa.fill(vertices[:, 0], vertices[:, 1], color=col, alpha=.075, zorder=1)
        aa.plot(vertices[:, 0], vertices[:, 1], color=col, lw=2.1, zorder=2)
        aa.text(-1.10, 1.02, plane_text, fontsize=16, color=INK)
        if col == ORANGE:
            # In Q=E R45 coordinates, principal axes really are diagonal.
            for radius, direction in [(1., np.array([1., -1.])/np.sqrt(2)),
                                      (ratio, np.array([1., 1.])/np.sqrt(2))]:
                ends = radius * np.array([-direction, direction])
                aa.plot(ends[:, 0], ends[:, 1], color=GREY, lw=.7,
                        ls=(0, (3, 3)), alpha=.5, zorder=2)
        for j in range(2):
            end = np.zeros(2)
            end[j] = 1.18
            arrow(aa, (0, 0), end, color=GREY, lw=1, ms=10, zorder=3)
        aa.text(1.25, -.05, "$q_1$", fontsize=15, ha="center")
        aa.text(-.05, 1.25, "$q_2$", fontsize=15, ha="right")
        aa.scatter([0], [0], s=26, c=INK, zorder=5)
        aa.text(-.09, -.15, center_text, fontsize=16, ha="right")
        arrow(aa, (0, 0), vec, color=col, lw=2.7, ms=15, zorder=6)
        if col == BLUE:
            label(aa, vec, "$v$", offset=(.14, -.10), color=col, fontsize=18)
        else:
            label(aa, vec, r"$D\mathcal{T}_zv$", offset=(.09, -.11), color=col, fontsize=17)
        aa.text(-1.12, -1.08, "ортонормированный базис " + basis_text,
                fontsize=10.5, color=GREY)

    fig_arrow(fig, (.657, .650), (.692, .650), r"$D\mathcal{T}_z$", label_offset=.057)
    fig.text(.395, .445, r"$v=Q(z)b$", color=BLUE, fontsize=17)
    fig.text(.701, .445, r"$D\mathcal{T}_zv=Q(\mathcal{T}z)A(z)b$", color=ORANGE, fontsize=17)
    fig.text(.545, .414, r"$A(z)$ не обязана быть ортогональной", fontsize=12.5, color=GREY, ha="center")

    fig.add_artist(plt.Line2D([.042, .958], [.385, .385], color=PALE, lw=1))
    fig.text(.042, .357, "Перекрытие окон", fontsize=14, weight="bold")
    fig.text(.958, .357, r"$P_1x_n=P_0x_{n+1}$", fontsize=16, ha="right")
    wa = blank_axes(fig, [.045, .196, .907, .140], (0, 13.1), (-.05, 2.35), equal=False)
    start, bw, bh = 2.4, 1.64, .73
    for row, shift, prefix in [(1.37, 0, "$x_n$"), (.26, 1, "$x_{n+1}$")]:
        wa.text(.0, row + bh / 2, prefix + " =", fontsize=17, va="center")
        for j in range(4):
            idx = j + shift
            x = start + idx * bw
            shared = idx in (1, 2, 3)
            wa.add_patch(Rectangle((x, row), bw - .05, bh,
                                   facecolor=BLUE if shared else LIGHT,
                                   alpha=.10 if shared else 1,
                                   edgecolor=BLUE if shared else PALE, lw=1.0))
            txt = "$y_n$" if idx == 0 else rf"$y_{{n+{idx}}}$"
            wa.text(x + (bw - .05) / 2, row + bh / 2, txt,
                    fontsize=16, ha="center", va="center")
    for k in (1, 2, 3):
        xx = start + k * bw + (bw - .05) / 2
        wa.plot([xx, xx], [1.37, .99], color=BLUE, lw=1, ls=(0, (2, 2)))
    wa.text(10.92, 1.37 + bh / 2, "$P_1$", color=BLUE, fontsize=17, va="center")
    wa.text(10.92, .26 + bh / 2, "$P_0$", color=BLUE, fontsize=17, va="center")
    wa.text(12.05, 1.25, "общие\nснимки", fontsize=12.5, color=GREY,
            ha="center", va="center")
    fig.text(.045, .157,
             r"$P_0D\mathcal{T}_zv=P_1v,\qquad J(z)A(z)=P_1Q(z),\qquad J(z)=P_0Q(\mathcal{T}z)$",
             fontsize=15)
    return fig


def make_fields():
    fig = new_figure(4, "Независимые фазовые направления",
                     "Постоянные направления в фазах дают меняющиеся касательные поля на многообразии.")
    fig.text(.045, .856, "Поля на $M$", fontsize=15, weight="bold")
    fig.text(.727, .856, "Направления в фазах", fontsize=15, weight="bold")
    ax = blank_axes(fig, [.040, .252, .631, .578], (-3.5, 3.75), (-2.4, 2.5))
    torus_wire(ax, lw=.7)
    arrow_scale = .73
    theta_samples = [np.array(x) for x in [
        (0., 0.), tuple(ALPHA), (1.65, .2), (2.75, 1.18),
        (3.65, -.45), (4.60, .3), (5.6, -1.4),
    ]]
    for th in theta_samples:
        p = project(psi(th))
        w = frame(th)
        ax.scatter([p[0]], [p[1]], s=15, c=GREY, zorder=4)
        for j, col in enumerate([BLUE, ORANGE]):
            end = p + arrow_scale * project(w[:, j])
            arrow(ax, p, end, color=col, lw=2.1, ms=12, zorder=6)
    z0, z1 = project(psi([0., 0.])), project(psi(ALPHA))
    ax.scatter([z0[0], z1[0]], [z0[1], z1[1]], s=32, c=INK, zorder=8)
    label(ax, z0, "$z$", offset=(.20, -.29), fontsize=17)
    label(ax, z1, "$\mathcal{T}z$", offset=(.20, .32), fontsize=17)
    # A time trajectory and its velocity are neutral, distinct from the basis fields.
    time_start = np.array([3.90, .70])
    ss = np.linspace(0, 1.8, 180)
    time_curve = project(psi(time_start + ss[:, None] * ALPHA))
    ax.plot(time_curve[:, 0], time_curve[:, 1], color=GREY, lw=1.15,
            ls=(0, (3, 2)), zorder=4)
    curve_arrow(ax, time_curve, at=.66, color=INK, lw=1.4, ms=10)
    th = time_start + .3 * ALPHA
    pp = project(psi(th))
    vel = project(dpsi(th) @ ALPHA)
    arrow(ax, pp, pp + .46 * vel, color=INK, lw=1.9, ms=12, zorder=6)
    label(ax, pp + .46 * vel, "$V$", offset=(.15, .15), fontsize=17)
    label(ax, time_curve[100], "ход времени", offset=(-.8, .50),
          fontsize=12.5, leader=True)
    ax.text(-2.8, 1.95, "$M$", fontsize=22, color=GREY)

    pa = blank_axes(fig, [.722, .466, .239, .345], (-1.25, 2.0), (-.6, 2.4))
    arrow(pa, (-1.02, -.05), (1.80, -.05), color=GREY, lw=1, ms=9)
    arrow(pa, (.0, -.3), (.0, 2.15), color=GREY, lw=1, ms=9)
    pa.text(1.86, -.14, r"$\theta_1$", fontsize=16, ha="center")
    pa.text(-.1, 2.28, r"$\theta_2$", fontsize=16, ha="right")
    pa.scatter([0.], [0.], s=20, c=INK, zorder=5)
    for j, col in enumerate([BLUE, ORANGE]):
        end = 2.28 * B[:, j]
        arrow(pa, (0., 0.), end, color=col, lw=2.4, ms=14, zorder=6)
        off = (.15, .08) if j == 0 else (-.17, .08)
        label(pa, end, rf"$B_{{\cdot {j+1}}}$", offset=off, color=col,
              fontsize=17, ha="left" if j == 0 else "right")
    cstart = np.array([1.04, .24])
    cend = cstart + 1.24 * ALPHA
    arrow(pa, cstart, cend, color=INK, lw=1.8, ms=12, zorder=6)
    label(pa, cend, "$c$", offset=(.13, .05), fontsize=17)
    pa.text(.65, -.47, r"$c=\mathrm{const}$", ha="center", color=GREY, fontsize=12)
    fig_arrow(fig, (.727, .658), (.659, .658), r"$D\psi_\theta$", label_offset=.025)

    fig.text(.722, .375, r"$W(h_L(\theta))=Dh_L(\theta)B$", fontsize=16)
    fig.text(.722, .339, r"$V_c(h_L(\theta))=Dh_L(\theta)c$", fontsize=16)
    fig.text(.722, .260, r"$V_c(\mathcal{T}z)=D\mathcal{T}_zV_c(z)$", fontsize=16)

    # Legend uses the same colors as all arrows, without adding extra accents.
    la = blank_axes(fig, [.045, .161, .59, .051], (0, 12), (0, 1), equal=False)
    arrow(la, (0, .53), (.75, .53), color=BLUE, lw=2.3, ms=11)
    la.text(.95, .53, "$V_1$", fontsize=16, va="center")
    arrow(la, (2.1, .53), (2.85, .53), color=ORANGE, lw=2.3, ms=11)
    la.text(3.05, .53, "$V_2$", fontsize=16, va="center")
    la.plot([4.7, 5.45], [.53, .53], color=GREY, lw=1.2, ls=(0, (3, 2)))
    la.text(5.7, .53, "временная траектория", fontsize=12.5, va="center")
    fig.text(.045, .130, r"$V_1,V_2$ — переносимые поля, не произвольные главные компоненты.",
             fontsize=11.8, color=GREY)
    return fig


def make_periods():
    fig = new_figure(5, "От полных оборотов к углам",
                     "Периоды образуют решётку; её нормировка задаёт углы с полным оборотом $2\pi$.")
    fig.text(.044, .856, "1. Полные обороты", fontsize=14.5, weight="bold")
    fig.text(.382, .856, "2. Накопленное перемещение", fontsize=14.5, weight="bold")
    fig.text(.756, .856, "3. Углы на торе", fontsize=14.5, weight="bold")

    ax = blank_axes(fig, [.037, .405, .306, .407], (-3.4, 3.55), (-2.4, 2.65))
    torus_wire(ax, lw=.65)
    ss = np.linspace(0, 2 * np.pi, 400)
    loop1 = project(psi(np.column_stack([ss, np.zeros_like(ss)])))
    loop2 = project(psi(np.column_stack([np.zeros_like(ss), ss])))
    ax.plot(loop1[:, 0], loop1[:, 1], color=BLUE, lw=2.05, zorder=3)
    ax.plot(loop2[:, 0], loop2[:, 1], color=ORANGE, lw=2.05, zorder=4)
    curve_arrow(ax, loop1, at=.37, color=BLUE, ms=12)
    curve_arrow(ax, loop2, at=.17, color=ORANGE, ms=12)
    label(ax, loop1[170], "$\gamma_1$", offset=(-.45, -.35), color=BLUE, fontsize=18)
    label(ax, loop2[90], "$\gamma_2$", offset=(-.45, .4), color=ORANGE, fontsize=18)
    fraction = np.array([.34, .41])
    theta_end = 2 * np.pi * fraction
    uu = np.linspace(0, 1, 170)
    path = project(psi(uu[:, None] * theta_end))
    ax.plot(path[:, 0], path[:, 1], color=INK, lw=1.55, ls=(0, (3, 2)), zorder=5)
    curve_arrow(ax, path, at=.70, color=INK, lw=1.4, ms=10)
    label(ax, path[90], r"$\gamma$", offset=(.06, .34), fontsize=17)
    zs, ze = path[0], path[-1]
    ax.scatter([zs[0], ze[0]], [zs[1], ze[1]], s=34, c=INK, zorder=7)
    label(ax, zs, "$z_*$", offset=(.16, -.4), fontsize=17)
    label(ax, ze, "$z$", offset=(-.27, -.28), fontsize=17, ha="right")
    ax.text(-2.9, 2.17, "$M$", fontsize=20, color=GREY)
    fig.text(.045, .350, r"$\gamma_1$: вокруг отверстия", fontsize=12.5, color=BLUE)
    fig.text(.045, .317, r"$\gamma_2$: вокруг трубки", fontsize=12.5, color=ORANGE)
    fig.text(.045, .275, r"$[\gamma_1],[\gamma_2]$: полный базис $H_1(M;\mathbb{Z})$", fontsize=12.5)
    fig.text(.045, .233, "Путь от $z_*$ к $z$ показан пунктиром.", fontsize=11.8, color=GREY)

    p1, p2 = PI[:, 0], PI[:, 1]
    acc = PI @ fraction
    acc_shift = acc + p2
    limx, limy = (-2.25, 19.25), (-15.0, 9.0)
    aa = blank_axes(fig, [.382, .340, .324, .470], limx, limy)
    arrow(aa, (-1.45, 0.), (18.65, 0.), color=PALE, lw=1, ms=9, zorder=0)
    arrow(aa, (0., -14.2), (0., 8.2), color=PALE, lw=1, ms=9, zorder=0)
    vertices = np.array([np.zeros(2), p1, p1 + p2, p2])
    aa.add_patch(Polygon(vertices, closed=True, facecolor=LIGHT, edgecolor=GREY, lw=1.15, zorder=1))
    # One adjacent cell visualizes the same point after an added full loop.
    aa.add_patch(Polygon(vertices + p2, closed=True, facecolor="none", edgecolor=PALE,
                         ls=(0, (3, 3)), lw=1.0, zorder=0))
    for frac in (.25, .5, .75):
        x = frac * p1
        y = frac * p2
        aa.plot([x[0], (x+p2)[0]], [x[1], (x+p2)[1]], color=PALE, lw=.8, zorder=2)
        aa.plot([y[0], (y+p1)[0]], [y[1], (y+p1)[1]], color=PALE, lw=.8, zorder=2)
    arrow(aa, (0., 0.), p1, color=BLUE, lw=2.3, ms=14, zorder=4)
    arrow(aa, (0., 0.), p2, color=ORANGE, lw=2.3, ms=14, zorder=4)
    label(aa, .66 * p1, r"$\Pi_{\cdot1}$", offset=(-1.1, -.45), color=BLUE, fontsize=18, ha="right")
    label(aa, .55 * p2, r"$\Pi_{\cdot2}$", offset=(-1.05, .38), color=ORANGE, fontsize=18, ha="right")
    aa.scatter([0], [0], s=23, c=INK, zorder=5)
    aa.text(-.55, -.65, "$0$", fontsize=15, ha="right")
    arrow(aa, (0., 0.), acc, color=INK, lw=1.4, ms=11, zorder=5)
    aa.scatter([acc[0]], [acc[1]], s=36, c=INK, zorder=6)
    label(aa, acc, r"$\int_\gamma\beta$", offset=(.35, -.9), fontsize=18)
    arrow(aa, acc, acc_shift, color=ORANGE, lw=1.1, ms=11,
          linestyle=(0, (3, 3)), zorder=5)
    aa.scatter([acc_shift[0]], [acc_shift[1]], s=36, facecolor=WHITE,
               edgecolor=INK, linewidth=1.4, zorder=7)
    label(aa, acc_shift, r"$\int_\gamma\beta+\Pi_{\cdot2}$", offset=(-2.9, 1.5), fontsize=15)
    aa.text(7.9, 5.8, "тот же $z$", fontsize=12, color=GREY)
    fig.text(.392, .301, r"$\Pi_{\cdot j}=\oint_{\gamma_j}\beta$", fontsize=16)
    fig.text(.392, .253, r"Все периоды: $\Pi\mathbb{Z}^2$", fontsize=14)
    fig.text(.392, .213, r"Базис $W$: $\ v=W(z)a$, $\ \beta_z(v)=a$", fontsize=13.2)

    # A fundamental angular square, with matching opposite-edge markings.
    sa = blank_axes(fig, [.753, .414, .220, .345], (-.40, 6.85), (-.45, 6.85))
    twopi = 2 * np.pi
    sa.add_patch(Rectangle((0, 0), twopi, twopi, facecolor=LIGHT, edgecolor="none"))
    for xx in (0., twopi):
        sa.plot([xx, xx], [0, twopi], color=BLUE, lw=1.9)
        arrow(sa, (xx, 3.15), (xx, 3.95), color=BLUE, lw=1.6, ms=11)
    for yy in (0., twopi):
        sa.plot([0, twopi], [yy, yy], color=ORANGE, lw=1.9)
        arrow(sa, (3.25, yy), (4.10, yy), color=ORANGE, lw=1.6, ms=11)
    phi = theta_end
    sa.plot([phi[0], phi[0]], [0., phi[1]], color=PALE, lw=1.1, ls=(0, (2, 2)))
    sa.plot([0., phi[0]], [phi[1], phi[1]], color=PALE, lw=1.1, ls=(0, (2, 2)))
    sa.scatter([phi[0]], [phi[1]], s=39, c=INK, zorder=5)
    label(sa, phi, "$\Phi(z)$", offset=(.32, .43), fontsize=18)
    for xx in (0., twopi):
        sa.scatter([xx], [1.15], s=25, facecolor=WHITE, edgecolor=BLUE, zorder=5)
    for yy in (0., twopi):
        sa.scatter([1.30], [yy], s=25, facecolor=WHITE, edgecolor=ORANGE, zorder=5)
    sa.text(-.22, -.25, "$0$", fontsize=13, ha="right")
    sa.text(twopi, -.30, "$2\pi$", fontsize=13, ha="center", va="top")
    sa.text(-.25, twopi, "$2\pi$", fontsize=13, ha="right", va="center")
    sa.text(3.4, -.73, r"$\varphi_1$", fontsize=17, ha="center")
    sa.text(.45, 5.45, r"$\varphi_2$", fontsize=17, ha="left")
    fig.text(.756, .371, r"$[0,2\pi)^2$", fontsize=16)
    fig.text(.756, .312, "склейка сторон", fontsize=11.8, color=GREY)
    fig.text(.748, .242, r"$\Phi(z)=2\pi\Pi^{-1}\!\int_\gamma\!\beta\;\mathrm{mod}\;2\pi$", fontsize=17)
    fig_arrow(fig, (.702, .596), (.748, .596), r"$2\pi\Pi^{-1}$", label_offset=.043)

    fig.add_artist(plt.Line2D([.042, .958], [.186, .186], color=PALE, lw=1))
    fig.text(.045, .148, r"$H_1(M;\mathbb{Z})\simeq\mathbb{Z}^2$: полные обороты $(n_1,n_2)$.",
             fontsize=12.6)
    fig.text(.558, .148, r"Смена полного базиса: $\Gamma\in GL(2,\mathbb{Z})$, $|\det\Gamma|=1$.",
             fontsize=12.6)
    return fig


CAPTIONS = r"""% Independent captions for the three explanatory figures.
% The manuscript phase_ssa_method.tex has not been edited.
\paragraph{Рисунок 3. Перенос касательных векторов.}
Геометрический пример при $r=2$: $M=\psi(\mathbb T^2)$,
$\psi(\theta)=((2+\cos\theta_2)\cos\theta_1,(2+\cos\theta_2)\sin\theta_1,\sin\theta_2)$,
$\mathcal{T}\psi(\theta)=\psi(\theta+\alpha)$ и
$\alpha=(0.45,0.60\sqrt2)$. Два касательных пространства показаны в собственных
ортонормированных координатах $Q(z)$ и $Q(\mathcal{T}z)$; их столбцы обозначены
$q_1,q_2$. Вектор $v=Q(z)b$ переходит в
$D\mathcal{T}_zv=Q(\mathcal{T}z)A(z)b$.
При $z=\psi(0,0)$ и $Q=ER_{45}$, где $E$ состоит из нормированных
координатных касательных и $R_{45}=2^{-1/2}\bigl(\begin{smallmatrix}1&-1\\1&1\end{smallmatrix}\bigr)$,
$A=R_{45}^{\mathsf T}\operatorname{diag}((2+\cos\alpha_2)/3,1)R_{45}$.
Единичный круг переходит в эллипс с полуосями $1$ и $0.887$:
это производная временного сдвига, которая не обязана сохранять длины.
Нижняя вставка отдельно показывает настоящие окна при $L=4$;
каждый блок $y_n$ содержит один снимок всех датчиков. $P_1$ удаляет первый,
а $P_0$ последний блок, поэтому $P_1x_n=P_0x_{n+1}$ и
$P_0D\mathcal{T}_zv=P_1v$; отсюда $JA=P_1Q$, $J=P_0Q(\mathcal{T}z)$.
Изображённый трёхмерный тор является геометрической моделью многообразия окон,
а не проекцией PCA и не самим вложением $h_L$ в $\mathbb R^{mL}$;
матрицы $P_0,P_1$ не применяются к его трём координатам.

\paragraph{Рисунок 4. Независимые фазовые направления.}
Для геометрической модели $M=\psi(\mathbb T^2)$, показанной на рисунке,
выбрана постоянная обратимая матрица
$B=2^{-1/2}\bigl(\begin{smallmatrix}1/3&-1/3\\1&1\end{smallmatrix}\bigr)$.
Её столбцы $B_{\cdot1},B_{\cdot2}$ задают постоянные направления на фазовой плоскости,
а $V_j(\psi(\theta))=D\psi_\theta B_{\cdot j}$ задают два независимых касательных поля
на $M$; $W=[V_1,V_2]$. В начальной точке $W(\psi(0,0))=Q(\psi(0,0))$.
Их направления и длины в пространстве меняются, но
$W(\mathcal{T}z)=D\mathcal{T}_zW(z)$.
Стрелки обоих полей имеют один общий множитель длины $0.73$ и не нормируются
отдельно в каждой точке. Нейтральный пунктир показывает короткую временную
траекторию; её скорость $V(\psi(\theta))=D\psi_\theta c$ соответствует
постоянному $c=\alpha/\Delta t$. Направления $V_1,V_2$ образуют базис
инвариантных полей и не являются двумя временными траекториями или
сингулярными компонентами.

\paragraph{Рисунок 5. От полных оборотов к углам.}
Базис инвариантных полей $W=[V_1,V_2]$ задаёт представление касательных векторов.
Форма $\beta_z$ возвращает коэффициенты касательного вектора:
если $v=W(z)a$, то $\beta_z(v)=a$. В показанной геометрической модели
$W(\psi(\theta))=D\psi_\theta B$,
$B=2^{-1/2}\bigl(\begin{smallmatrix}1/3&-1/3\\1&1\end{smallmatrix}\bigr)$,
поэтому $\beta=B^{-1}d\theta$. От опорного окна $z_*=\psi(0,0)$ вдоль пути к $z$
накапливается перемещение $\int_\gamma\beta$, где $\gamma$ идёт от $z_*$ к $z$.
Примитивные обходы $\gamma_1$ вокруг отверстия и $\gamma_2$ вокруг трубки
составляют полный целочисленный базис $H_1(M;\mathbb{Z})\simeq\mathbb{Z}^2$.
Их периоды $\Pi_{\cdot j}=\oint_{\gamma_j}\beta$ являются столбцами
$\Pi=2\pi B^{-1}=(2\pi/\sqrt2)\bigl(\begin{smallmatrix}3&1\\-3&1\end{smallmatrix}\bigr)$;
здесь матрица целых оборотов $\Gamma=I$.
На средней панели показан фундаментальный параллелограмм решётки
$\Pi\mathbb{Z}^2$ и два накопленных вектора $\int_\gamma\beta$ и
$\int_\gamma\beta+\Pi_{\cdot2}$, соответствующих
одному окну. Преобразование $2\pi\Pi^{-1}$ переводит параллелограмм в
квадрат углов $[0,2\pi)^2$ с отождествлёнными противоположными сторонами:
$\Phi(z)=2\pi\Pi^{-1}\int_\gamma\beta\pmod{2\pi}$.
Точка показана при $\theta=2\pi(0.34,0.41)$.
Полный базис гарантирует $\Gamma\in GL(2,\mathbb{Z})$, $|\det\Gamma|=1$;
другой полный базис меняет углы обратимой целочисленной заменой координат.
"""


def verify_math():
    """Checks of the illustrative model, not tests of research conclusions."""
    theta0 = np.zeros(2)
    q0 = orthoframe(theta0)
    checks = {"Q0=W0": np.max(np.abs(q0 - frame(theta0)))}
    sample = [theta0, ALPHA, np.array([1.2, -.7]), np.array([2.8, 2.3])]
    dual_errors, transport_errors, a_errors = [], [], []
    for th in sample:
        w = frame(th)
        beta = np.linalg.solve(w.T @ w, w.T)
        dual_errors.append(np.max(np.abs(beta @ w - np.eye(2))))
        transport_errors.append(np.max(np.abs(transfer(th) @ w - frame(th + ALPHA))))
        a_errors.append(np.max(np.abs(transfer(th) @ orthoframe(th) -
                                         orthoframe(th + ALPHA) @ coefficient_transfer(th))))
    checks["beta W=I"] = max(dual_errors)
    checks["D T W(theta)=W(theta+alpha)"] = max(transport_errors)
    checks["D T Q(theta)=Q(theta+alpha) A(theta)"] = max(a_errors)
    checks["Pi=2pi B^-1"] = np.max(np.abs(PI -
                           2*np.pi/np.sqrt(2)*np.array([[3., 1.], [-3., 1.]])))
    checks["2pi Pi^-1 Pi=2pi I"] = np.max(np.abs(2*np.pi*np.linalg.solve(PI, PI) - 2*np.pi*np.eye(2)))
    th = 2*np.pi*np.array([.34, .41])
    acc = np.linalg.solve(B, th)
    checks["2pi Pi^-1 accumulated=theta"] = np.max(np.abs(2*np.pi*np.linalg.solve(PI, acc) - th))
    checks["adding Pi2 adds exactly one normalized full turn"] = np.max(np.abs(
        2*np.pi*np.linalg.solve(PI, acc+PI[:,1]) - th - np.array([0., 2*np.pi])))
    assert max(checks.values()) < 1e-12, checks
    return checks


def main():
    checks = verify_math()
    names = ["figure_03_transport", "figure_04_fields", "figure_05_periods"]
    figures = [make_transport(), make_fields(), make_periods()]
    with PdfPages(PDF_OUT / "phase_ssa_illustrations.pdf") as book:
        info = book.infodict()
        info["Title"] = "Фазовая реконструкция: три геометрические иллюстрации"
        info["Author"] = ""
        info["Subject"] = "Касательный перенос, инвариантные поля и нормировка периодов"
        for fig, name in zip(figures, names):
            for ext in ("svg", "pdf", "png"):
                fig.savefig(OUT / f"{name}.{ext}", dpi=240)
            book.savefig(fig)
            plt.close(fig)
    (OUT / "captions.tex").write_text(CAPTIONS, encoding="utf-8")
    report = "Illustrative-model algebra checks (absolute residuals):\n"
    report += "\n".join(f"{name}: {error:.3e}" for name, error in checks.items())
    report += f"\nalpha = {ALPHA.tolist()}\nellipse minor semiaxis = {(2+np.cos(ALPHA[1]))/3:.9f}\n"
    (OUT / "verification.txt").write_text(report, encoding="utf-8")
    print(report)
    print("Created 3 independent vector PDFs/SVGs and PNGs, captions, and a 3-page combined PDF.")


if __name__ == "__main__":
    main()
