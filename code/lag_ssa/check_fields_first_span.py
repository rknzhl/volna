"""Small checks of source-only span truncation and phase-loss safeguards."""
import unittest
from unittest.mock import patch

import numpy as np

from fields_first_span import select_span


class SpanSelectionChecks(unittest.TestCase):
    def matrix(self, singular):
        singular = np.asarray(singular)
        rng = np.random.default_rng(92)
        left = np.linalg.qr(rng.normal(size=(max(8, 2*len(singular)), len(singular))))[0]
        right = np.linalg.qr(rng.normal(size=(len(singular), len(singular))))[0]
        return (left*singular) @ right.T

    def test_auto_removes_small_noise_directions_from_a_curved_signal(self):
        angle = 2*np.pi*np.arange(128)/128
        rng = np.random.default_rng(19)
        source = np.column_stack((np.cos(angle), np.sin(angle), .3*np.cos(2*angle),
                                  .002*rng.normal(size=(128, 2))))
        basis, coordinates, report = select_span(source)
        self.assertEqual(report['numerical_span_dimension'], 5)
        self.assertEqual(report['retained_dimension'], 3)
        self.assertIsNone(report['fallback_reason'])
        self.assertEqual(report['full_dimension_hint'], 1)
        self.assertEqual(report['reduced_dimension_hint'], 1)
        self.assertEqual(report['gap_index'], 3)
        self.assertGreaterEqual(report['gap_ratio'], 10.)
        np.testing.assert_allclose(basis.T@basis, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(coordinates, source@basis, atol=1e-12)
        error = np.linalg.norm(source-coordinates@basis.T)/np.linalg.norm(source)
        self.assertAlmostEqual(error, report['projection_relative'], places=12)

    def test_resolved_phase_dimension_hint_cannot_be_discarded(self):
        source = self.matrix([10., 8., .02, .015])
        with patch('fields_first_span._local_dimension_hint', side_effect=[(2, {}), (1, {})]):
            basis, _, report = select_span(source)
        self.assertEqual(report['proposed_dimension'], 2)
        self.assertEqual(basis.shape[1], 4)
        self.assertEqual(report['fallback_reason'], 'resolved_local_dimension_hint_not_preserved')
        self.assertEqual(report['selected_dimension_hint'], 2)
        self.assertEqual(report['selected_cutoff'], 1e-10)

    def test_large_gap_does_not_override_discarded_error_budget(self):
        with patch('fields_first_span._local_dimension_hint', return_value=(None, {})):
            basis, _, report = select_span(self.matrix([1., .09, .09]))
        self.assertEqual(report['proposed_dimension'], 1)
        self.assertGreater(report['proposed_projection_relative'], .05)
        self.assertEqual(basis.shape[1], 3)
        self.assertEqual(report['fallback_reason'], 'discarded_window_error_exceeds_inverse_budget')

    def test_no_large_internal_gap_keeps_numerical_span(self):
        with patch('fields_first_span._local_dimension_hint', return_value=(None, {})) as hint:
            basis, _, report = select_span(self.matrix([10., 8., 6., 4.]))
        self.assertEqual(basis.shape[1], 4)
        self.assertIsNone(report['gap_index'])
        self.assertIsNone(report['fallback_reason'])
        self.assertEqual(hint.call_count, 1)

    def test_numeric_cutoff_supports_legacy_and_explicit_truncation(self):
        source = self.matrix([10., 8., .03, 1e-12])
        with patch('fields_first_span._local_dimension_hint', return_value=(1, {})):
            basis, _, report = select_span(source, span_rtol=1e-10)
            smaller, _, truncated = select_span(source, span_rtol=.01)
            bounded, _, floor = select_span(source, span_rtol=1e-12)
        self.assertEqual(basis.shape[1], 3)
        self.assertEqual(report['selected_cutoff'], 1e-10)
        self.assertEqual(smaller.shape[1], 2)
        self.assertEqual(truncated['selected_cutoff'], .01)
        self.assertEqual(bounded.shape[1], 3)
        self.assertEqual(floor['selected_cutoff'], 1e-10)

    def test_invalid_or_empty_cutoff_is_reported(self):
        source = self.matrix([10., 8., 6.])
        for value in (0., -1., np.nan, np.inf, True, 'unknown'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                select_span(source, span_rtol=value)
        with patch('fields_first_span._local_dimension_hint', return_value=(None, {})):
            basis, _, report = select_span(source, span_rtol=1.)
        self.assertEqual(basis.shape[1], 3)
        self.assertEqual(report['fallback_reason'], 'cutoff_removes_all_numerical_directions')


if __name__ == '__main__':
    unittest.main()
