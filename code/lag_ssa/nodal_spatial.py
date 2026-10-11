"""Geometry-only P1 interpolation of one scalar field sampled at sensor nodes.

All nodal values are coefficients. The physical metric is the exact L2 mass
of piecewise affine interpolation on the sensors' convex hull. No PDE,
spatial profile, polynomial law of the true field, or temporal frequency is
supplied. This is a spatial discretization for ordinary weighted MSSA.
"""

from __future__ import annotations

from math import factorial

import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay, QhullError

class NodalSpatialModel:
    """Nodal P1 field on a 1D/2D/3D convex hull; outside evaluation is rejected."""

    def __init__(self, positions: np.ndarray):
        points = np.asarray(positions, dtype=float)
        if points.ndim == 1:
            points = points[:, None]
        if (points.ndim != 2 or points.shape[1] not in (1, 2, 3)
                or not np.isfinite(points).all()):
            raise ValueError("finite sensor positions with dimension 1, 2 or 3 required")
        count, dimension = points.shape
        if count < dimension+1 or len(np.unique(points, axis=0)) != count:
            raise ValueError("distinct sensor nodes spanning at least one simplex required")
        if np.linalg.matrix_rank(points-points.mean(axis=0)) != dimension:
            raise ValueError("sensor nodes must span their full spatial dimension")
        self.positions = points.copy()
        self.dimension = dimension
        self.triangulation = None
        self.dropped_degenerate_cell_count = 0
        self.cell_volume_tolerance = 0.
        if dimension == 1:
            self.node_order = np.argsort(points[:, 0])
            cells = np.column_stack((self.node_order[:-1], self.node_order[1:]))
            volumes = np.diff(points[self.node_order, 0])
        else:
            try:
                self.triangulation = Delaunay(points)
            except QhullError as error:
                raise ValueError("sensor geometry cannot form a nondegenerate simplicial mesh") from error
            cells = self.triangulation.simplices.copy()
            edges = points[cells[:, 1:]]-points[cells[:, :1]]
            volumes = abs(np.linalg.det(edges))/factorial(dimension)
            # Co-spherical Cartesian grids can produce zero-volume Qhull
            # simplices. They have no L2 mass; retain the original points and
            # triangulation for interpolation, and omit only their integral.
            volume_scale = float(np.prod(np.ptp(points, axis=0)))/factorial(dimension)
            self.cell_volume_tolerance = 64*np.finfo(float).eps*volume_scale
            valid = volumes > self.cell_volume_tolerance
            self.dropped_degenerate_cell_count = int(np.sum(~valid))
            cells, volumes = cells[valid], volumes[valid]
        if (not len(cells) or not np.isfinite(volumes).all()
                or np.any(volumes <= 0) or float(volumes.sum()) <= 0):
            raise ValueError("positive-volume interpolation cells required")
        if len(np.unique(cells)) != count:
            raise ValueError("every sensor node must belong to a positive-volume interpolation cell")
        mass = np.zeros((count, count))
        template = np.eye(dimension+1)+np.ones((dimension+1, dimension+1))
        for nodes, volume in zip(cells, volumes):
            mass[np.ix_(nodes, nodes)] += volume*template/((dimension+1)*(dimension+2))
        self.cells = cells
        self.cell_volumes = volumes
        self.domain_volume = float(volumes.sum())
        self.physical_mass = mass
        self.mass = mass
        eigenvalues, eigenvectors = np.linalg.eigh(mass)
        if eigenvalues[0] <= 0:
            raise ValueError("nodal interpolation mass must be positive definite")
        self.mass_sqrt = np.einsum("ij,kj->ik", eigenvectors*np.sqrt(eigenvalues),
                                   eigenvectors, optimize=False)
        self.design = np.eye(count)
        self.decoder = np.eye(count)
        self.weights = mass.sum(axis=1)
        self.sampling_gram = np.diag(self.weights)

    @classmethod
    def from_positions(cls, positions: np.ndarray) -> "NodalSpatialModel":
        return cls(positions)

    def _values(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        if (array.ndim not in (1, 2) or array.shape[0] != len(self.positions)
                or not np.isfinite(array).all()):
            raise ValueError("finite nodal values with sensor index as the first axis required")
        return array

    def recover(self, signals: np.ndarray) -> np.ndarray:
        return self._values(signals).copy()

    def to_mass_coordinates(self, coefficients: np.ndarray) -> np.ndarray:
        values = self._values(coefficients)
        if values.ndim == 1:
            return np.einsum("ij,j->i", self.mass_sqrt, values, optimize=False)
        return np.einsum("ij,jt->it", self.mass_sqrt, values, optimize=False)

    def coordinates(self, signals: np.ndarray) -> np.ndarray:
        return self.to_mass_coordinates(self.recover(signals))

    def evaluate(self, coefficients: np.ndarray, positions: np.ndarray) -> np.ndarray:
        values = self._values(coefficients)
        points = np.asarray(positions, dtype=float)
        if points.ndim == 1:
            points = (points[:, None] if self.dimension == 1 else points[None, :])
        if (points.ndim != 2 or points.shape[1] != self.dimension
                or not np.isfinite(points).all()):
            raise ValueError("finite query positions matching the spatial dimension required")
        if self.dimension == 1:
            nodes = self.positions[self.node_order, 0]
            tolerance = 1e-12*max(1., nodes[-1]-nodes[0])
            if np.any((points[:, 0] < nodes[0]-tolerance)
                      | (points[:, 0] > nodes[-1]+tolerance)):
                raise ValueError("evaluation points must lie inside the sensor convex hull")
            query = np.clip(points[:, 0], nodes[0], nodes[-1])
            right = np.clip(np.searchsorted(nodes, query, side="right"), 1, len(nodes)-1)
            left = right-1
            fraction = (query-nodes[left])/(nodes[right]-nodes[left])
            ordered = values[self.node_order]
            if values.ndim == 2:
                fraction = fraction[:, None]
            return (1-fraction)*ordered[left]+fraction*ordered[right]
        if np.any(self.triangulation.find_simplex(points, tol=1e-12) < 0):
            raise ValueError("evaluation points must lie inside the sensor convex hull")
        evaluated = np.asarray(LinearNDInterpolator(self.triangulation, values)(points))
        if not np.isfinite(evaluated).all():
            raise ValueError("query lies outside the interpolation hull or on an unresolved numerical boundary")
        return evaluated

    def reconstruct(self, mass_coordinates: np.ndarray, positions: np.ndarray) -> np.ndarray:
        coefficients = np.linalg.solve(self.mass_sqrt, self._values(mass_coordinates))
        return self.evaluate(coefficients, positions)

    def field_normcoeff(self, coefficients: np.ndarray) -> float | np.ndarray:
        values = self.to_mass_coordinates(coefficients)
        if values.ndim == 1:
            return float(np.linalg.norm(values))
        return np.linalg.norm(values, axis=0)

    def noise_covcoeff(self, sigma: float) -> np.ndarray:
        if not np.isfinite(sigma) or sigma < 0:
            raise ValueError("nonnegative finite sensor noise standard deviation required")
        return sigma**2*np.eye(len(self.positions))

    def noise_covmass(self, sigma: float) -> np.ndarray:
        if not np.isfinite(sigma) or sigma < 0:
            raise ValueError("nonnegative finite sensor noise standard deviation required")
        return sigma**2*self.physical_mass

    def metadata(self) -> dict:
        return {"model": "geometry-only nodal P1 interpolation",
                "sensor_count": len(self.positions), "spatial_dimension": self.dimension,
                "domain": "sensor convex hull", "domain_volume": self.domain_volume,
                "cell_count": len(self.cells), "known_field_law_supplied": False,
                "dropped_degenerate_cell_count": self.dropped_degenerate_cell_count,
                "cell_volume_tolerance": self.cell_volume_tolerance,
                "outside_hull_evaluation": "rejected",
                "noise_estimation": "external estimate required: all nodal values are coefficients"}
