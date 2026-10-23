"""Small checks of dimension/scale decisions; no vector-field fits."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from fields_first_periods import (
    SETTINGS, _infer_torus_dimension_once, _select_prefix, _select_scale, infer_torus_dimension,
)


class DimensionSelectionChecks(unittest.TestCase):
    def diagram(self):
        # Lifetimes: 2, 1.8, .5, .3. The largest eligible gap follows class 2;
        # class 3 nevertheless has a valid common standard range.
        return np.array([[.1, 2.1], [.1, 1.9], [.1, .6], [.41, .71]])

    def topology(self, diagram=None, uncovered=False):
        distance = np.zeros((4, 4))
        if uncovered:
            distance[:, 0] = 1.
        return SimpleNamespace(_dgms=[None, self.diagram() if diagram is None else diagram],
            _dist_land_data=distance, get_cover_radius=Mock(side_effect=lambda p, b, d, standard:
                (((1-p)*2*b+p*d)/2, (1-p)*2*b+p*d)))

    def test_geometry_count_can_include_a_shorter_resolved_class(self):
        legacy, _ = _select_prefix(self.diagram(), SETTINGS)
        selected, report = _select_prefix(self.diagram(), SETTINGS, preferred_count=3)
        self.assertEqual(legacy, 2)
        self.assertEqual(selected, 3)
        self.assertLess(report['lifetime_gap'], SETTINGS.lifetime_gap)
        self.assertFalse(report['lifetime_gap_required'])
        self.assertLess(2*report['common_birth'], report['common_death'])

    def test_geometry_hint_does_not_bypass_standard_range(self):
        diagram = self.diagram(); diagram[2] = [.31, .6]
        selected, report = _select_prefix(diagram, SETTINGS, preferred_count=3)
        self.assertIsNone(selected)
        self.assertEqual(report['reason'], 'preferred_dimension_has_no_common_standard_range')

    def test_scale_avoids_extra_classes_active_at_the_old_midpoint(self):
        diagram = self.diagram(); diagram[3] = [.3, .6]
        with patch('fields_first_periods.ToroidalCoords', return_value=self.topology(diagram)):
            selected, report = _infer_torus_dimension_once(.01*np.eye(4), nland=4, preferred_count=3)
        self.assertEqual(selected,3)
        self.assertEqual(report['active_b1_mod_prime'],3)
        self.assertLess(report['rips_threshold'],.3)

    def test_extra_class_active_throughout_the_standard_range_is_refused(self):
        diagram = self.diagram(); diagram[3] = [.19, .61]
        with patch('fields_first_periods.ToroidalCoords', return_value=self.topology(diagram)):
            selected, report = _infer_torus_dimension_once(.01*np.eye(4), nland=4, preferred_count=3)
        self.assertIsNone(selected)
        self.assertEqual(report['failure_reason'],
            'no_standard_scale_with_exact_rank_and_training_cover_buffer')

    def test_incomplete_training_cover_refuses_the_dimension(self):
        with patch('fields_first_periods.ToroidalCoords', return_value=self.topology(uncovered=True)):
            selected, report = _infer_torus_dimension_once(.01*np.eye(4), nland=4, preferred_count=3)
        self.assertIsNone(selected)
        self.assertEqual(report['failure_reason'],
            'no_standard_scale_with_exact_rank_and_training_cover_buffer')
        self.assertEqual(report['train_fill_distance'],1.)

    def test_bare_training_coverage_without_sampling_buffer_is_refused(self):
        diagram=np.array([[.1,.6]])
        metric=np.full((4,4),.02);np.fill_diagonal(metric,0)
        threshold,report=_select_scale(diagram,1,
            dict(common_birth=.1,common_death=.6),np.full((2,4),.295),metric)
        self.assertIsNone(threshold)
        self.assertLess(report['train_fill_distance'],.6/2)
        self.assertAlmostEqual(report['training_sampling_buffer'],.01)

    def test_widest_clean_interval_is_selected_after_short_extra_bars_die(self):
        diagram=np.array([[.1,.7],[.25,.4]])
        metric=np.full((4,4),.02);np.fill_diagonal(metric,0)
        threshold,report=_select_scale(diagram,1,
            dict(common_birth=.1,common_death=.7),np.full((2,4),.1),metric)
        self.assertAlmostEqual(threshold,.55)
        self.assertEqual(report['selected_scale_interval'],[.4,.7])
        self.assertGreater(report['train_cover_margin'],report['training_sampling_buffer'])

    def test_equal_width_intervals_prefer_the_larger_scale(self):
        diagram=np.array([[.125,1.25],[.5,1.]])
        metric=np.full((4,4),.02);np.fill_diagonal(metric,0)
        threshold,_=_select_scale(diagram,1,
            dict(common_birth=.125,common_death=1.25),np.zeros((2,4)),metric)
        self.assertEqual(threshold,1.125)

    def test_geometry_count_stays_fixed_when_resolution_is_refined(self):
        def infer(_x, nland, **kwargs):
            self.assertEqual(kwargs['preferred_count'], 3)
            selected = 3 if nland == 384 else None
            return selected, dict(landmarks=nland, selected_dimension=selected, selection={},
                **({'failure_reason': 'landmark_cover_incomplete_on_training_data',
                    'train_uncovered_count': 5} if selected is None else {}))

        with patch('fields_first_periods._local_dimension_hint', return_value=(3, {'median_singular_gaps': [1., 1., 3.]})), \
             patch('fields_first_periods._infer_torus_dimension_once', side_effect=infer) as probe:
            selected, report = infer_torus_dimension(np.zeros((1200, 4)))
        self.assertEqual(selected, 3)
        self.assertEqual([call.args[1] for call in probe.call_args_list], [192, 384])
        self.assertEqual(report['geometry_dimension_hint'], 3)
        self.assertEqual([row['landmarks'] for row in report['resolution_attempts']], [192, 384])
        self.assertEqual(report['resolution_attempts'][0]['failure_reason'],
                         'landmark_cover_incomplete_on_training_data')
        self.assertEqual(report['resolution_attempts'][0]['train_uncovered_count'], 5)

    def test_unresolved_resolution_search_is_bounded_without_a_geometry_hint(self):
        def infer(_x, nland, **kwargs):
            self.assertIsNone(kwargs['preferred_count'])
            return None, dict(landmarks=nland, selected_dimension=None, selection={})

        with patch('fields_first_periods._local_dimension_hint', return_value=(None, {})), \
             patch('fields_first_periods._infer_torus_dimension_once', side_effect=infer) as probe:
            selected, report = infer_torus_dimension(np.zeros((1800, 4)))
        self.assertIsNone(selected)
        self.assertEqual([call.args[1] for call in probe.call_args_list], [192, 384, 768, 1024, 1536, 1800])
        self.assertEqual(report['maximum_topology_landmarks'], 2048)


if __name__ == '__main__':
    unittest.main()
