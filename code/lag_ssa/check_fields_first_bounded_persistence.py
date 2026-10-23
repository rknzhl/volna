"""Tiny real persistence checks; no sensor pipeline or phase fitting."""
import unittest

import numpy as np
from scipy.spatial.distance import cdist
from ripser import ripser
from dreimac import ToroidalCoords

from fields_first_bounded_persistence import bounded_persistence
from fields_first_periods import SETTINGS, _select_prefix, _select_scale


class BoundedPersistenceChecks(unittest.TestCase):
    def circle(self, count=18, radius=1., center=(0., 0.)):
        angle = 2*np.pi*np.arange(count)/count
        return radius*np.column_stack((np.cos(angle), np.sin(angle))) + center

    def test_cyclic_source_relabel_preserves_landmark_local_index_mapping(self):
        source = self.circle()
        permutation = np.roll(np.arange(len(source)), 5)
        metric = cdist(source[permutation], source[permutation])
        raw = ripser(metric, distance_matrix=True, n_perm=10, maxdim=1,
                     coeff=47, thresh=1.9, do_cocycles=True)
        model = bounded_persistence(metric, 10, 47, 1.9)
        np.testing.assert_array_equal(model._idx_land, raw['idx_perm'])
        np.testing.assert_allclose(model._dist_land_data, metric[model._idx_land], atol=1e-12)
        np.testing.assert_allclose(model._dist_land_land,
            metric[np.ix_(model._idx_land, model._idx_land)], atol=1e-12)
        inverse = np.full(len(source), -1, dtype=int)
        inverse[model._idx_land] = np.arange(10)
        self.assertGreater(len(model._dgms[1]), 0)
        for sorted_index, raw_index in enumerate(model._h1_sort_order):
            expected = raw['cocycles'][1][raw_index].copy()
            expected[:, :-1] = inverse[expected[:, :-1]]
            representative = model.get_representative_cocycle(sorted_index, 1)[2]
            np.testing.assert_array_equal(representative, expected)
            self.assertTrue(np.all((representative[:, :-1] >= 0)
                                   & (representative[:, :-1] < 10)))

    def test_lifetime_sort_keeps_two_real_representatives_aligned(self):
        source = np.vstack((self.circle(12, 1., (-3., 0.)),
                            self.circle(12, .4, (3., 0.))))
        metric = cdist(source, source)
        raw = ripser(metric, distance_matrix=True, maxdim=1, coeff=47,
                     thresh=2., do_cocycles=True)
        model = bounded_persistence(metric, len(source), 47, 2.)
        self.assertGreaterEqual(len(model._dgms[1]), 2)
        lifetime = model._dgms[1][:, 1]-model._dgms[1][:, 0]
        self.assertTrue(np.all(np.diff(lifetime) <= 0))
        for index, raw_index in enumerate(model._h1_sort_order):
            birth, death, cocycle = model.get_representative_cocycle(index, 1)
            np.testing.assert_allclose([birth, death], raw['dgms'][1][raw_index])
            np.testing.assert_array_equal(cocycle, raw['cocycles'][1][raw_index])

    def test_completed_bars_match_full_rips_and_lookup(self):
        metric = cdist(self.circle(), self.circle())
        bounded = bounded_persistence(metric, len(metric), 47, 1.9)
        full = ToroidalCoords(metric, n_landmarks=len(metric), distance_matrix=True,
                             prime=47, maxdim=1, verbose=False)
        self.assertFalse(bounded._h1_right_censored.any())
        np.testing.assert_allclose(bounded._dgms[1], full._dgms[1])
        np.testing.assert_array_equal(bounded._cns_lookup_table, full._cns_lookup_table)
        for bounded_cocycle, full_cocycle in zip(bounded._cocycles[1], full._cocycles[1]):
            np.testing.assert_array_equal(bounded_cocycle, full_cocycle)

    def test_censored_death_is_preserved_as_a_lower_bound_and_scale_stays_below_cap(self):
        metric = cdist(self.circle(24), self.circle(24))
        model = bounded_persistence(metric, 18, 47, 1.7)
        self.assertTrue(model._h1_right_censored.any())
        np.testing.assert_array_equal(np.isposinf(model._raw_h1_diagram[:, 1]),
                                      model._h1_right_censored)
        np.testing.assert_allclose(model._dgms[1][model._h1_right_censored, 1], 1.7)
        selected, selection = _select_prefix(model._dgms[1], SETTINGS, preferred_count=1)
        self.assertEqual(selected, 1)
        threshold, _ = _select_scale(model._dgms[1], selected, selection,
                                     model._dist_land_data, metric)
        self.assertIsNotNone(threshold)
        self.assertLess(threshold, model._filtration_cap)


if __name__ == '__main__':
    unittest.main()
