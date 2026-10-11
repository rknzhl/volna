"""Интерактивная трёхмерная анимация поля."""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from scenarios import SCENARIOS, START, simulate


def animation3d():
    """Анимация трёхмерных сценариев с линзой, датчиками и источником."""
    times = np.round(np.arange(0.05, START + 1e-9, 0.05), 2)
    names = [n for n in SCENARIOS if n.startswith("3D")]
    fig = make_subplots(1, 2, specs=[[{"type": "scene"}] * 2], subplot_titles=names, horizontal_spacing=0.02)
    values, iso = [], []
    a, b = np.meshgrid(np.linspace(0, 2 * np.pi, 40), np.linspace(0, np.pi, 30))
    for col, name in enumerate(names, 1):
        run = simulate(SCENARIOS[name], snap_times=times, t_end=START)
        n, N = run["n"], len(run["n"])
        g = np.linspace(0, 1, N)
        box = (slice(int(0.1 * N), None, 2), slice(N // 2, None, 2), slice(int(0.15 * N), int(0.85 * N) + 1, 2))
        X, Y, Z = (v.astype(np.float32).ravel() for v in np.meshgrid(*(g[s] for s in box), indexing="ij"))
        scale = np.abs(run["snaps"][START][int(0.4 * N):]).max()
        frames = [np.clip(100 * run["snaps"][tk][box] / scale, -127, 127).astype(np.int8).ravel() for tk in times]
        values.append(frames)
        iso.append(len(fig.data))
        fig.add_trace(go.Isosurface(x=X, y=Y, z=Z, value=frames[-1], isomin=-35, isomax=35, surface_count=2,
                                    colorscale=[[0, "#2166ac"], [1, "#b2182b"]], cmin=-35, cmax=35, showscale=False,
                                    opacity=0.55, caps=dict(x_show=False, y_show=False, z_show=False), hoverinfo="skip",
                                    lighting=dict(ambient=0.55, diffuse=0.8, specular=0.25, roughness=0.6)), 1, col)
        if np.ptp(n) > 0:
            r = np.abs(g[n[:, N // 2, N // 2] >= n.min() + 0.55 * np.ptp(n)] - 0.5).max()
            fig.add_trace(go.Surface(x=0.5 + r * np.cos(a) * np.sin(b), y=0.5 + r * np.sin(a) * np.sin(b), z=0.5 + r * np.cos(b),
                                     colorscale=[[0, "#4d4d4d"], [1, "#4d4d4d"]], showscale=False, opacity=0.45,
                                     hoverinfo="skip"), 1, col)
        for pts, size, color in zip((run["sensors"], run["sources"]), (2.5, 7), ("black", "gold")):
            fig.add_trace(go.Scatter3d(x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="markers", hoverinfo="skip",
                                       marker=dict(size=size, color=color, line=dict(color="black", width=1))), 1, col)
    axis = dict(showgrid=False, showticklabels=False, showbackground=True, backgroundcolor="rgb(240,240,240)", zeroline=False)
    fig.update_scenes(xaxis=dict(title="x", range=[0, 1], **axis), yaxis=dict(title="y", range=[0, 1], **axis),
                      zaxis=dict(title="z", range=[0, 1], **axis), aspectmode="cube", camera_eye=dict(x=-1.0, y=-1.6, z=0.9))
    labels = [f"{tk:.2f}" for tk in times]
    fig.frames = [go.Frame(name=label, traces=iso, data=[go.Isosurface(value=v[k]) for v in values])
                  for k, label in enumerate(labels)]
    play = dict(frame=dict(duration=120, redraw=True), fromcurrent=True, transition=dict(duration=0))
    stop = dict(mode="immediate", frame=dict(duration=0, redraw=False))
    jump = dict(mode="immediate", frame=dict(duration=0, redraw=True))
    buttons = [dict(label="пуск", method="animate", args=[None, play]),
               dict(label="стоп", method="animate", args=[[None], stop])]
    steps = [dict(method="animate", label=label, args=[[label], jump]) for label in labels]
    fig.update_layout(template="plotly_white", height=560, showlegend=False, margin=dict(l=0, r=0, t=40, b=0),
                      updatemenus=[dict(type="buttons", direction="left", x=0.02, y=0.02, xanchor="left", yanchor="bottom",
                                        showactive=False, buttons=buttons)],
                      sliders=[dict(active=len(times) - 1, x=0.15, len=0.83, y=0.02,
                                    currentvalue=dict(prefix="t = ", suffix=" с"), steps=steps)])
    return fig
