"""Данные The Well."""
from pathlib import Path

import h5py
import numpy as np
from huggingface_hub import HfFileSystem

DATA = Path(__file__).parent / "data"
TASKS = {
    "discontinuous": "acoustic_scattering_discontinuous/data/test/acoustic_scattering_discontinuous_chunk_18.hdf5",
    "inclusions": "acoustic_scattering_inclusions/data/test/acoustic_scattering_inclusions_chunk_36.hdf5",
    "maze": "acoustic_scattering_maze/data/test/acoustic_scattering_maze_chunk_18.hdf5",
}


def load(task):
    """Два расчёта задачи."""
    path = DATA / f"well_{task}.npz"
    if not path.exists():
        DATA.mkdir(exist_ok=True)
        url = "datasets/polymathic-ai/" + TASKS[task]
        with HfFileSystem().open(url, "rb", block_size=2 ** 22) as f, h5py.File(f, "r") as h:
            np.savez(path, time=h["dimensions/time"][:], pressure=h["t0_fields/pressure"][:2],
                     speed=h["t0_fields/speed_of_sound"][:2])
    return dict(np.load(path))


def sensor_nodes():
    """Узлы датчиков."""
    axis = np.rint(np.linspace(0.2, 0.8, 4) * 255).astype(int)
    return np.stack(np.meshgrid(axis, axis, indexing="ij"), -1).reshape(-1, 2)


def records(w, k):
    """Записи датчиков и их координаты."""
    idx = sensor_nodes()
    return w["pressure"][k][:, idx[:, 0], idx[:, 1]].T, 2 * idx / 255 - 1
