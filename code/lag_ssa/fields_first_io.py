"""Strict numeric sensor loading without time guessing, row removal, or interpolation."""
from __future__ import annotations

import csv
from pathlib import Path
import numpy as np


def _table(path, delimiter):
    with open(path, "r", newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.reader(stream, delimiter=delimiter))
    if not rows or not rows[0]:
        raise ValueError("empty sensor table")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("sensor table contains empty or ragged rows")
    try:
        [float(value) for value in rows[0]]
        numeric_first = True
    except ValueError:
        numeric_first = False
    header = None
    if not numeric_first:
        header = [value.strip() for value in rows.pop(0)]
        if any(not name for name in header) or len(set(header)) != len(header):
            raise ValueError("header names must be nonempty and unique")
    if not rows:
        raise ValueError("sensor table has no numeric rows")
    try:
        values = np.asarray(rows, dtype=float)
    except (ValueError, TypeError) as exc:
        raise ValueError("sensor table must contain numeric values only") from exc
    return values, header


def _time_index(time_column, header, width):
    if isinstance(time_column, str):
        if header is not None and time_column in header:
            return header.index(time_column)
        try:
            index = int(time_column)
        except ValueError as exc:
            raise ValueError("time_column must be a zero-based index or an existing header name") from exc
    elif isinstance(time_column, (int, np.integer)) and not isinstance(time_column, (bool, np.bool_)):
        index = int(time_column)
    else:
        raise ValueError("time_column must be a zero-based index or an existing header name")
    if not 0 <= index < width:
        raise ValueError("time_column index outside the sensor table")
    return index


def read_sensor_data(path, time_column=None, key="Y", delimiter=","):
    """Return (Y, time), with Y[N,m] and optional strictly uniform numeric time.

    Supported extensions: .npy, .npz, .csv, .tsv. NPZ loads exactly `key`, with
    allow_pickle=False. A CSV/TSV first row is treated as a header iff it cannot
    be parsed entirely as numbers. TSV uses tab when delimiter has its default
    comma value; any explicitly different one-character delimiter is honored.

    All entries must be finite and there must be at least three time rows and
    one sensor after removing the explicitly designated time column. Numeric
    timestamps must increase and have constant spacing within relative 1e-6
    plus eight floating-point ulps at their absolute magnitude.
    """
    path = Path(path)
    extension = path.suffix.lower()
    header = None
    if extension == ".npy":
        values = np.load(path, allow_pickle=False)
    elif extension == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if key not in archive:
                raise ValueError("NPZ does not contain the explicitly selected key: " + str(key))
            values = np.array(archive[key], copy=True)
    elif extension in (".csv", ".tsv"):
        effective = "\t" if extension == ".tsv" and delimiter == "," else delimiter
        if not isinstance(effective, str) or len(effective) != 1:
            raise ValueError("delimiter must be one character")
        values, header = _table(path, effective)
    else:
        raise ValueError("supported sensor formats are .npy, .npz, .csv, and .tsv")
    if np.iscomplexobj(values):
        raise ValueError("sensor readings must be real-valued")
    try:
        values = np.asarray(values, dtype=float)
    except (ValueError, TypeError) as exc:
        raise ValueError("sensor readings must be numeric") from exc
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or values.shape[0] < 3 or values.shape[1] < 1:
        raise ValueError("sensor data require at least three rows and one channel")
    if not np.isfinite(values).all():
        raise ValueError("sensor data contain NaN or infinite values")
    if time_column is None:
        return values.copy(), None
    index = _time_index(time_column, header, values.shape[1])
    time = values[:, index].copy()
    Y = np.delete(values, index, axis=1)
    if Y.shape[1] == 0:
        raise ValueError("no sensor columns remain after removing time_column")
    spacing = np.diff(time)
    if not np.all(spacing > 0):
        raise ValueError("timestamps must be strictly increasing")
    tolerance = 8 * abs(float(np.spacing(np.max(np.abs(time)))))
    if not np.allclose(spacing, np.median(spacing), rtol=1e-6, atol=tolerance):
        raise ValueError("timestamps must be uniformly spaced; interpolation is not performed")
    return Y.copy(), time
