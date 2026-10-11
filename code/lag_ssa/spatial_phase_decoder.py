"""Train-node-CV interpolation of learned complex spatial profiles.

This uses sensor geometry, fitted cosine/sine coefficients and learned
frequencies only. Polar interpolation assumes smoothly liftable phases and
nonzero envelopes. Integer windings inferred from short edges are a prior,
not an identification certificate. Complex P1/RBF remain comparison methods.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.spatial import distance_matrix

from nodal_spatial import NodalSpatialModel


def _mm(first, second):
    return np.einsum("ij,jk->ik", first, second, optimize=False)


def complex_profiles(coefficients: np.ndarray, line_count: int) -> np.ndarray:
    rows = np.asarray(coefficients, dtype=float)
    if rows.ndim != 2 or rows.shape[0] != 1+2*line_count:
        raise ValueError("coefficient rows must be constant, all cosines, all sines")
    return rows[1:1+line_count].T-1j*rows[1+line_count:].T


def _edges(model: NodalSpatialModel) -> np.ndarray:
    return np.asarray(sorted({tuple(sorted(edge)) for cell in model.cells
                              for edge in combinations(cell.tolist(), 2)}), dtype=int)


def unwrap_geometric_phase(model: NodalSpatialModel, profiles: np.ndarray) -> tuple[np.ndarray, dict]:
    positions = model.positions
    edges = _edges(model)
    vectors = positions[edges[:, 1]]-positions[edges[:, 0]]
    lengths = np.linalg.norm(vectors, axis=1)
    short = lengths <= 1.05*lengths.min()
    for multiplier in (1.5, 2., 3., 5.):
        if np.linalg.matrix_rank(vectors[short]) == model.dimension:
            break
        short = lengths <= multiplier*lengths.min()
    phase, amplitude = np.angle(profiles), abs(profiles)
    unwrapped = np.empty_like(phase)
    audits = []
    for coordinate in range(profiles.shape[1]):
        wrapped = np.angle(profiles[edges[:, 1], coordinate]
                           *np.conj(profiles[edges[:, 0], coordinate]))
        node_amplitude = amplitude[:, coordinate]
        reliability = np.maximum(np.minimum(node_amplitude[edges[:, 0]],
                                           node_amplitude[edges[:, 1]])**2,
                                 1e-6*max(node_amplitude.max()**2, 1e-30))
        weights = reliability/lengths**2
        local_weights = np.sqrt(weights[short])
        gradient = np.linalg.lstsq(vectors[short]*local_weights[:, None],
                                   wrapped[short]*local_weights, rcond=None)[0]
        gradient_prediction = np.einsum("ij,j->i", vectors, gradient, optimize=False)
        lifts = np.rint((gradient_prediction-wrapped)/(2*np.pi)).astype(int)
        lifted = wrapped+2*np.pi*lifts
        if model.dimension == 1:
            order = np.argsort(positions[:, 0])
            gap = np.diff(positions[order, 0])
            increments = np.angle(profiles[order[1:], coordinate]
                                  *np.conj(profiles[order[:-1], coordinate]))
            turns = np.rint((gradient[0]*gap-increments)/(2*np.pi)).astype(int)
            values = phase[order[0], coordinate]+np.r_[0., np.cumsum(increments+2*np.pi*turns)]
            unwrapped[order, coordinate] = values
            lifted_edge_error = np.zeros(len(edges))
        else:
            incidence = np.zeros((len(edges), len(positions)))
            incidence[np.arange(len(edges)), edges[:, 0]] = -1
            incidence[np.arange(len(edges)), edges[:, 1]] = 1
            anchor = int(np.argmax(node_amplitude))
            other = np.delete(np.arange(len(positions)), anchor)
            rhs = lifted-incidence[:, anchor]*phase[anchor, coordinate]
            solution = np.linalg.lstsq(incidence[:, other]*np.sqrt(weights)[:, None],
                                       rhs*np.sqrt(weights), rcond=None)[0]
            approximate = np.empty(len(positions))
            approximate[anchor], approximate[other] = phase[anchor, coordinate], solution
            # Restore exact observed wrapped values. Graph inconsistencies are
            # retained as diagnostics instead of silently changing the data.
            unwrapped[:, coordinate] = phase[:, coordinate]+2*np.pi*np.rint(
                (approximate-phase[:, coordinate])/(2*np.pi))
            lifted_edge_error = np.einsum("ij,j->i", incidence,
                                          unwrapped[:, coordinate], optimize=False)-lifted
        audits.append({"short_edge_gradient_rad_per_distance": gradient.tolist(),
            "corrected_integer_edge_count": int(np.sum(lifts != 0)),
            "lifted_edge_residual_rms_rad": float(np.sqrt(np.average(lifted_edge_error**2, weights=weights))),
            "max_wrapped_short_edge_increment_rad": float(np.max(abs(wrapped[short]))),
            "weak_amplitude_nodes_below_5pct_max": int(np.sum(node_amplitude < .05*node_amplitude.max())),
            "integer_winding_identified": False})
    return unwrapped, {"method": "short-edge gradient plus length-aware integer lifts; graph least squares in d>1",
        "edge_count": len(edges), "short_edge_count": int(short.sum()),
        "minimum_edge_length": float(lengths.min()), "maximum_edge_length": float(lengths.max()),
        "observed_phase_agreement_max_error": float(np.max(abs(np.exp(1j*unwrapped)-np.exp(1j*phase)))),
        "coordinates": audits,
        "assumption": "short edges resolve a locally smooth phase gradient; long-gap winding is inferred from this prior"}


@dataclass
class PreparedProfiles:
    positions: np.ndarray
    coefficients: np.ndarray
    frequencies_hz: np.ndarray

    def __post_init__(self):
        self.spatial = NodalSpatialModel.from_positions(self.positions)
        self.profiles = complex_profiles(self.coefficients, len(self.frequencies_hz))
        if not len(self.frequencies_hz):
            raise ValueError("phase interpolation requires at least one learned oscillation")
        self.amplitudes = abs(self.profiles)
        self.phases, self.unwrap_audit = unwrap_geometric_phase(self.spatial, self.profiles)
        distances = distance_matrix(self.positions, self.positions)
        np.fill_diagonal(distances, np.inf)
        self.spacing = float(np.median(distances.min(axis=1)))
        self.origin = self.positions.mean(axis=0)
        affine_design = np.column_stack((np.ones(len(self.positions)), self.positions-self.origin))
        self.affine_phase = np.column_stack([np.linalg.lstsq(
            affine_design*np.maximum(self.amplitudes[:, j], 1e-12)[:, None],
            self.phases[:, j]*np.maximum(self.amplitudes[:, j], 1e-12), rcond=None)[0]
            for j in range(len(self.frequencies_hz))])
        self.delay, self.delay_intercepts, self.delay_audit = self._shared_delay()

    def _shared_delay(self):
        q, p = self.profiles.shape
        if p != 2:
            return None, None, {"eligible": False, "reason": "requires exactly two learned frequencies"}
        omega = 2*np.pi*self.frequencies_hz
        design = np.zeros((q*p, p+q-1))
        for node in range(q):
            for coordinate in range(p):
                row = node*p+coordinate
                design[row, coordinate] = 1.
                if node:
                    design[row, p+node-1] = -omega[coordinate]
        weights = np.maximum(self.amplitudes.ravel(), 1e-8)
        fitted = np.linalg.lstsq(design*weights[:, None], self.phases.ravel()*weights, rcond=None)[0]
        residual = (np.einsum("ij,j->i", design, fitted, optimize=False)
                    -self.phases.ravel()).reshape(q, p)
        rms = float(np.sqrt(np.average(residual**2, weights=self.amplitudes**2)))
        return np.r_[0., fitted[p:]], fitted[:p], {"eligible": True,
            "phase_fit_weighted_rms_rad": rms,
            "conditional_on_selected_phase_lifts": True,
            "absolute_delay_identified": False,
            "hypothesis": "two independent envelopes with shared learned delay; no wave equation supplied"}


def _rbf(data: PreparedProfiles, values: np.ndarray, query: np.ndarray, spec: dict) -> np.ndarray:
    kwargs = {"kernel": spec.get("kernel", "thin_plate_spline"),
              "degree": 0 if spec.get("kernel") == "gaussian" else 1,
              "smoothing": spec.get("smoothing", 0.)}
    if kwargs["kernel"] == "gaussian":
        kwargs["epsilon"] = 1/spec["lengthscale_multiplier"]
    return RBFInterpolator((data.positions-data.origin)/data.spacing, values, **kwargs)(
        (query-data.origin)/data.spacing)


def predict_profiles(data: PreparedProfiles, query: np.ndarray, spec: dict) -> np.ndarray:
    query = np.asarray(query, dtype=float).reshape(-1, data.positions.shape[1])
    method = spec["method"]
    if method == "complex_p1":
        return data.spatial.evaluate(data.profiles.real, query)+1j*data.spatial.evaluate(data.profiles.imag, query)
    if method == "complex_rbf":
        p = data.profiles.shape[1]
        prediction = _rbf(data, np.column_stack((data.profiles.real, data.profiles.imag)), query, spec)
        return prediction[:, :p]+1j*prediction[:, p:]
    if method == "polar_rbf":
        amplitude = np.exp(_rbf(data, np.log(np.maximum(data.amplitudes, 1e-12)), query, spec))
        phase = _rbf(data, data.phases, query, spec)
    else:
        amplitude = data.spatial.evaluate(data.amplitudes, query)
        if method == "polar_p1":
            phase = data.spatial.evaluate(data.phases, query)
        elif method == "polar_affine":
            phase = _mm(np.column_stack((np.ones(len(query)), query-data.origin)), data.affine_phase)
        elif method == "shared_delay_p1":
            if data.delay is None:
                raise ValueError("shared delay is not available")
            delay = data.spatial.evaluate(data.delay, query)
            phase = data.delay_intercepts[None, :]-delay[:, None]*(2*np.pi*data.frequencies_hz)[None, :]
        else:
            raise ValueError("unknown spatial profile decoder")
    return amplitude*np.exp(1j*phase)


def coefficient_rows(data: PreparedProfiles, query: np.ndarray, spec: dict) -> np.ndarray:
    profile = predict_profiles(data, query, spec)
    mean = data.spatial.evaluate(data.coefficients[0], query)
    return np.vstack((mean[None, :], profile.real.T, -profile.imag.T))


def candidates() -> list[dict]:
    result = [{"method": method} for method in ("complex_p1", "polar_p1", "polar_affine")]
    result += [{"method": "polar_rbf", "kernel": "thin_plate_spline", "smoothing": smoothing}
               for smoothing in (0., .001, .1)]
    result += [{"method": "complex_rbf", "kernel": "gaussian", "smoothing": smoothing,
                "lengthscale_multiplier": lengthscale}
               for lengthscale in (.5, 1., 2., 4.) for smoothing in (1e-6, .001, .1)]
    result.append({"method": "shared_delay_p1"})
    return result


def fit_spatial_phase_decoder(positions: np.ndarray, coefficients: np.ndarray,
                              frequencies_hz: np.ndarray) -> dict:
    positions, coefficients = np.asarray(positions), np.asarray(coefficients)
    prepared = PreparedProfiles(positions, coefficients, np.asarray(frequencies_hz))
    folds, skipped = [], []
    for held in range(len(positions)):
        keep = np.delete(np.arange(len(positions)), held)
        try:
            fold = PreparedProfiles(positions[keep], coefficients[:, keep], prepared.frequencies_hz)
            fold.spatial.evaluate(coefficients[0, keep], positions[[held]])
            folds.append((held, fold))
        except (ValueError, np.linalg.LinAlgError):
            skipped.append(held)
    if not folds:
        raise ValueError("no retained-node CV folds inside their remaining sensor hull")
    grid = []
    for spec in candidates():
        errors, targets, fold_scores = [], [], []
        try:
            for held, fold in folds:
                predicted = predict_profiles(fold, positions[[held]], spec)[0]
                target = prepared.profiles[held]
                errors.append(float(np.sum(abs(predicted-target)**2)))
                targets.append(float(np.sum(abs(target)**2)))
                fold_scores.append({"held_retained_node": held,
                    "squared_complex_profile_error": errors[-1],
                    "target_profile_energy": targets[-1]})
            score = float(sum(errors)/max(sum(targets), 1e-30))
            grid.append({"spec": spec, "cv_relative_profile_mse": score, "folds": fold_scores})
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
            grid.append({"spec": spec, "failure": f"{type(error).__name__}: {error}"})
    independent = [row for row in grid if "cv_relative_profile_mse" in row
                   and row["spec"]["method"] != "shared_delay_p1"]
    best = min(independent, key=lambda row: row["cv_relative_profile_mse"])
    shared = next((row for row in grid if row["spec"]["method"] == "shared_delay_p1"
                   and "cv_relative_profile_mse" in row), None)
    shared_accepted = bool(shared and prepared.delay_audit.get("phase_fit_weighted_rms_rad", np.inf) <= .3
                           and shared["cv_relative_profile_mse"] <= .8*best["cv_relative_profile_mse"])
    selected = shared if shared_accepted else best
    audit = {"selected_spec": selected["spec"],
        "selected_cv_relative_profile_mse": selected["cv_relative_profile_mse"],
        "cv_kind": "leave-one-retained-sensor profile out; frequencies trained on retained sensors remain fixed",
        "eligible_cv_nodes": [held for held, _ in folds], "skipped_hull_vertex_cv_nodes": skipped,
        "cv_grid": grid, "phase_unwrap": prepared.unwrap_audit,
        "shared_delay": prepared.delay_audit | {"cv_gate_accepted": shared_accepted,
            "predeclared_gate": "weighted phase RMS <= .3 rad and CV MSE <= .8 of best independent-profile model"},
        "known_field_law_supplied": False, "true_frequency_supplied": False,
        "spatial_winding_certified": False,
        "limits": ["CV selects a spatial smoothness hypothesis; it does not identify missing windings",
                   "phase is ill-defined near zero amplitude",
                   "conditional CV does not include uncertainty of the learned temporal frequencies",
                   "kernel and polar decoders interpolate supplied geometry, not the PDE domain"]}
    return {"prepared": prepared, "selected_spec": selected["spec"], "audit": audit}
