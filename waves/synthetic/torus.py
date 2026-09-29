"""Фазовое пространство: точки, главные оси и спектр."""
import numpy as np


def with_delay(Y, lag):
    """Точки фазового пространства."""
    return np.hstack([Y[:, lag:].T, Y[:, :-lag].T])


def pca(X):
    """Координаты в главных осях."""
    Xc = X - X.mean(0)
    return Xc @ np.linalg.svd(Xc, full_matrices=False)[2].T


def spectrum(X):
    """Сингулярные числа облака точек, делённые на наибольшее."""
    s = np.linalg.svd(X - X.mean(0), compute_uv=False)
    return s / s[0]
