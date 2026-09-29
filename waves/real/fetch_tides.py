"""Уровень воды и гармонические постоянные станций NOAA."""
import json
import urllib.request
from pathlib import Path

import numpy as np

DATA = Path(__file__).parent / "data"
STATIONS = {
    "8557380": "Lewes", "8555889": "Brandywine Shoal", "8537121": "Ship John Shoal", "8551910": "Reedy Point",
    "8540433": "Marcus Hook", "8545240": "Philadelphia", "8539094": "Burlington", "8548989": "Newbold",
}
MONTHS = [("20240101", "20240131"), ("20240201", "20240229")]
API = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?product=water_level&application=takers_research" \
      "&begin_date={}&end_date={}&datum=MSL&station={}&time_zone=gmt&units=metric&format=json"
META = "https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations/{}{}.json?units=metric"
CONSTITUENTS = ["M2", "S2", "N2", "K1", "O1", "M4"]
R_EARTH = 6371.0


def get(url, path):
    """Скачать один раз."""
    if not path.exists():
        urllib.request.urlretrieve(url, path)
    return json.loads(path.read_text())


def distance(a, b):
    """Расстояние между станциями, км."""
    s = np.sin((b[0] - a[0]) / 2) ** 2 + np.cos(a[0]) * np.cos(b[0]) * np.sin((b[1] - a[1]) / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(s))


if __name__ == "__main__":
    DATA.mkdir(exist_ok=True)
    t = np.arange(0, 60 * 24 * 10) * 0.1  # часы от начала 1 января 2024 года по GMT
    start = np.datetime64("2024-01-01T00:00")
    eta, pos, harcon = [], [], []
    for sid in STATIONS:
        level = np.full(t.size, np.nan)
        for begin, end in MONTHS:
            for row in get(API.format(begin, end, sid), DATA / f"tide_{sid}_{begin}.json").get("data", []):
                i = int((np.datetime64(row["t"].replace(" ", "T")) - start) / np.timedelta64(6, "m"))
                if row["v"] and i < t.size:
                    level[i] = float(row["v"])
        eta.append(level)
        info = get(META.format(sid, ""), DATA / f"tide_{sid}_meta.json")["stations"][0]
        pos.append((info["lat"], info["lng"]))
        hc = {h["name"]: h for h in get(META.format(sid, "/harcon"), DATA / f"tide_{sid}_harcon.json")["HarmonicConstituents"]}
        harcon.append([[hc[c]["amplitude"], hc[c]["phase_GMT"], hc[c]["speed"]] for c in CONSTITUENTS])
    lat, lng = np.array(pos).T
    rad = np.radians(pos)
    km = np.concatenate([[0], np.cumsum([distance(a, b) for a, b in zip(rad, rad[1:])])])
    np.savez(DATA / "tides.npz", t=t, eta=np.array(eta), names=np.array(list(STATIONS.values())), ids=np.array(list(STATIONS)),
             distance_km=km, harcon=np.array(harcon), constituents=np.array(CONSTITUENTS), lat=lat, lng=lng)
    print("пропусков по станциям:", [int(np.isnan(e).sum()) for e in eta])
    print("расстояние вдоль цепочки станций, км:", np.round(km, 1))
