"""TRAIN-only linear span selection before tangent and phase estimation."""
import numpy as np
from scipy import linalg

from fields_first_periods import _local_dimension_hint


NUMERICAL_RTOL = 1e-10


def select_span(fullx, span_rtol='auto'):
    """Return (basis, coordinates, diagnostic) using source windows only.

    Auto truncates at the largest internal singular-value ratio at least 10.
    It retains the numerical span if the discarded relative norm exceeds .05
    or a resolved local-dimension hint is lost. These are finite-data checks,
    not a guarantee that arbitrarily weak phase directions are preserved.
    """
    source = np.asarray(fullx, dtype=float)
    if (source.ndim != 2 or min(source.shape) < 1 or len(source) < 4
            or not np.isfinite(source).all()):
        raise ValueError('finite source windows with at least four rows required')
    automatic = isinstance(span_rtol, str) and span_rtol == 'auto'
    if not automatic:
        if isinstance(span_rtol, (str, bool, np.bool_)):
            raise ValueError('span_rtol must be auto or a positive finite number')
        requested = float(span_rtol)
        if not np.isfinite(requested) or requested <= 0:
            raise ValueError('span_rtol must be auto or a positive finite number')
    else:
        requested = 'auto'
    _, singular, right = linalg.svd(source, full_matrices=False)
    if singular[0] <= 0:
        raise ValueError('source windows have no numerical span')
    full_dimension = int(np.sum(singular > singular[0]*NUMERICAL_RTOL))
    full_basis = right[:full_dimension].T
    full_coordinates = source @ full_basis
    full_hint, full_hint_report = _local_dimension_hint(full_coordinates)
    ratios = singular[:full_dimension-1] / singular[1:full_dimension]
    gap_index = None; gap_ratio = None; fallback = None
    if automatic:
        eligible = np.flatnonzero(ratios >= 10.)
        if len(eligible):
            index = int(eligible[np.argmax(ratios[eligible])])
            proposed_dimension = index+1
            gap_index = proposed_dimension
            gap_ratio = float(ratios[index])
            cutoff = float(np.sqrt(singular[index]/singular[0]
                                   * (singular[index+1]/singular[0])))
        else:
            proposed_dimension = full_dimension
            cutoff = NUMERICAL_RTOL
    else:
        cutoff = max(requested, NUMERICAL_RTOL)
        proposed_dimension = int(np.sum(singular[:full_dimension] > singular[0]*cutoff))
    total = float(linalg.norm(singular))
    proposed_residual = float(linalg.norm(singular[proposed_dimension:])/total)
    if proposed_dimension == full_dimension:
        reduced_hint, reduced_hint_report = full_hint, full_hint_report
    elif proposed_dimension:
        reduced_hint, reduced_hint_report = _local_dimension_hint(
            full_coordinates[:, :proposed_dimension])
    else:
        reduced_hint, reduced_hint_report = None, dict(reason='empty_proposed_span')
        fallback = 'cutoff_removes_all_numerical_directions'
    if (fallback is None and full_hint is not None and reduced_hint != full_hint):
        fallback = 'resolved_local_dimension_hint_not_preserved'
    if fallback is None and proposed_residual > .05:
        fallback = 'discarded_window_error_exceeds_inverse_budget'
    dimension = full_dimension if fallback is not None else proposed_dimension
    selected_cutoff = NUMERICAL_RTOL if fallback is not None else cutoff
    basis = full_basis[:, :dimension]
    coordinates = full_coordinates[:, :dimension]
    diagnostic = dict(rule='largest internal singular gap at least 10' if automatic
        else 'supplied relative singular cutoff, bounded by numerical cutoff',
        requested_span_rtol=requested, numerical_cutoff=NUMERICAL_RTOL,
        selected_cutoff=float(selected_cutoff), proposed_cutoff=float(cutoff),
        numerical_span_dimension=full_dimension, proposed_dimension=proposed_dimension,
        retained_dimension=dimension, gap_index=gap_index, gap_ratio=gap_ratio,
        relative_singular_values=singular/singular[0],
        consecutive_singular_ratios=ratios,
        projection_relative=float(linalg.norm(singular[dimension:])/total),
        proposed_projection_relative=proposed_residual,
        full_dimension_hint=full_hint, reduced_dimension_hint=reduced_hint,
        selected_dimension_hint=full_hint if fallback is not None else reduced_hint,
        full_dimension_diagnostic=full_hint_report,
        reduced_dimension_diagnostic=reduced_hint_report,
        fallback_reason=fallback, automatic_span_is_heuristic=automatic,
        arbitrary_weak_phase_preservation_certified=False,
        train_only=True, independent_phase_coordinates_used=False)
    return basis, coordinates, diagnostic
