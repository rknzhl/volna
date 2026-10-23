"""Mocked capped-filtration decisions and one exact finite-cycle certificate."""
import unittest
from itertools import combinations
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from fields_first_periods import _infer_torus_dimension_once,prepare_period_complex
from dreimac.combinatorial import combinatorial_number_system_table


class FiltrationCapChecks(unittest.TestCase):
    def topology(self,diagram,censored,count=4):
        return SimpleNamespace(_dgms=[None,np.asarray(diagram,dtype=float)],
            _h1_right_censored=np.asarray(censored,dtype=bool),
            _dist_land_data=np.zeros((count,count)))

    def infer(self,topology,preferred_count=1,source=None,cap=1.):
        source=.01*np.eye(4) if source is None else source
        with patch('fields_first_bounded_persistence.bounded_persistence',return_value=topology):
            return _infer_torus_dimension_once(source,nland=len(source),
                preferred_count=preferred_count,filtration_cap=cap)

    def test_censored_death_is_a_conservative_bound_for_the_selected_scale(self):
        topology=self.topology([[.1,1.],[.41,.6]],[True,False])
        selected,report=self.infer(topology)
        self.assertEqual(selected,1)
        self.assertLess(report['rips_threshold'],report['filtration_cap'])
        self.assertEqual(report['selection']['common_death_lower_bound'],1.)
        self.assertTrue(report['selection']['selected_classes_include_censored_deaths'])
        self.assertFalse(report['selection']['full_lifetime_gap_available'])
        self.assertNotIn('lifetime_gap',report['selection'])
        for later_death in (1.,1.01,10.):
            self.assertLess(report['rips_threshold'],later_death)

    def test_censored_lifetimes_cannot_supply_the_dimension_without_a_local_hint(self):
        topology=self.topology([[.1,1.],[.41,.6]],[True,False])
        selected,report=self.infer(topology,preferred_count=None)
        self.assertIsNone(selected)
        self.assertEqual(report['selection']['reason'],
            'censored_lifetime_gap_requires_local_dimension_hint')

    def test_cap_too_low_for_the_standard_range_remains_unresolved(self):
        topology=self.topology([[.55,1.]],[True])
        selected,report=self.infer(topology)
        self.assertIsNone(selected)
        self.assertNotIn('rips_threshold',report)
        self.assertEqual(report['selection']['reason'],
                         'preferred_dimension_has_no_common_standard_range')

    def test_an_extra_censored_class_is_not_removed_from_the_active_count(self):
        topology=self.topology([[.1,1.],[.15,1.]],[True,True])
        selected,report=self.infer(topology)
        self.assertIsNone(selected)
        self.assertEqual(report['failure_reason'],
            'no_standard_scale_with_exact_rank_and_training_cover_buffer')

    def test_finite_death_below_the_cap_is_still_exact(self):
        topology=self.topology([[.1,.7],[.5,1.]],[False,True])
        selected,report=self.infer(topology)
        self.assertEqual(selected,1)
        self.assertEqual(report['selection']['common_death_lower_bound'],.7)
        self.assertFalse(report['selection']['selected_classes_include_censored_deaths'])
        self.assertLess(report['rips_threshold'],.7)

    def test_period_preparation_uses_the_same_scale_and_exact_integer_certificate(self):
        angle=2*np.pi*np.arange(8)/8
        source=np.column_stack((np.cos(angle),np.sin(angle)))
        distance=np.sqrt(np.sum((source[:,None]-source[None])**2,axis=2))
        birth=float(distance[0,1]);cap=1.7
        representative=np.array([[i,j,46] for i,j in combinations(range(8),2)
                                 if j-i>=6],dtype=np.int64)
        topology=self.topology([[birth,cap]],[True],count=8)
        topology._dist_land_data=distance
        topology._dist_land_land=distance
        topology._idx_land=np.arange(8)
        topology._cns_lookup_table=combinatorial_number_system_table(8,2)
        topology.get_representative_cocycle=lambda *_:(birth,cap,representative.copy())
        selected,report=self.infer(topology,source=source,cap=cap)
        self.assertEqual(selected,1)
        with patch('fields_first_bounded_persistence.bounded_persistence',return_value=topology):
            complex_=prepare_period_complex(source,1,nland=8,filtration_cap=cap)
        self.assertEqual(complex_['status'],'prepared_full_primitive_finite_period_complex')
        self.assertAlmostEqual(complex_['rips_threshold'],report['rips_threshold'])
        self.assertLess(complex_['rips_threshold'],cap)
        self.assertTrue(complex_['certificate']['full_primitive_free_cohomology_basis_certified'])
        self.assertEqual(complex_['diagnostics']['integer_closure']['maximum_integer_closure_defect'],0)


if __name__=='__main__':unittest.main()
