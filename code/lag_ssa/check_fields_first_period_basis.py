"""Exact basis-change and period-unit checks, without geometry fitting."""
import unittest

import numpy as np
from sympy import Matrix

from cohomology_primitivity import _torus_complex, certify_primitivity
from fields_first_periods import reduce_period_basis


class PeriodBasisChecks(unittest.TestCase):
    def problem(self):
        edges,cocycles,n=_torus_complex()
        certificate=certify_primitivity(edges,cocycles,2,vertex_count=n,
            integer_closure_verified=True,same_complex_verified=True,
            return_cycle_matrix=True)
        cycle_periods=certificate.pop('graph_cycle_period_matrix_integer')
        pi=np.array([[1.,18.],[0.,1.]])
        potential=np.random.default_rng(7).normal(size=(n,2));potential-=potential[0]
        normalized_potential=np.linalg.solve(pi,potential.T)
        normalized_edges=np.asarray(cocycles,dtype=float)+(
            normalized_potential[:,edges[:,1]]-normalized_potential[:,edges[:,0]])
        pm=dict(status='fitted_fields_first_finite_periods',Pi=pi,periods=pi,
            potential=potential,normalized_potential=normalized_potential,
            normalized_edge_integrals=normalized_edges,
            closed_edge_integrals=(pi@normalized_edges).T,diagnostics={})
        cx=dict(edges=edges,vertex_count=n,r=2,integer_cocycles=cocycles,
            graph_cycle_period_matrix_integer=cycle_periods,
            certificate=certificate,diagnostics={})
        return pm,cx

    def test_change_is_exact_unimodular_and_preserves_primitive_basis(self):
        pm,cx=self.problem();old_cocycles=cx['integer_cocycles'].copy()
        new_pm,new_cx=reduce_period_basis(pm,cx,np.eye(2))
        report=new_pm['diagnostics']['period_basis_reduction']
        U=Matrix(report['U']);Ui=Matrix(report['Ui'])
        self.assertEqual(abs(U.det()),1)
        self.assertEqual(U*Ui,Matrix.eye(2))
        np.testing.assert_array_equal(new_cx['integer_cocycles'],
            np.asarray(Ui.tolist(),dtype=object)@old_cocycles)
        np.testing.assert_array_equal(new_cx['graph_cycle_period_matrix_integer'],
            np.asarray(Ui.tolist(),dtype=object)@cx['graph_cycle_period_matrix_integer'])
        fresh=certify_primitivity(cx['edges'],new_cx['integer_cocycles'],2,
            vertex_count=cx['vertex_count'],integer_closure_verified=True,
            same_complex_verified=True)
        self.assertTrue(fresh['full_primitive_free_cohomology_basis_certified'])
        self.assertTrue(new_cx['certificate']['full_primitive_free_cohomology_basis_certified'])
        np.testing.assert_array_equal(cx['integer_cocycles'],old_cocycles)
        self.assertNotIn('period_basis_reduction',pm['diagnostics'])

    def test_period_decomposition_and_raw_frame_units_are_preserved(self):
        pm,cx=self.problem();new_pm,new_cx=reduce_period_basis(pm,cx,np.eye(2))
        np.testing.assert_array_equal(new_pm['potential'],pm['potential'])
        np.testing.assert_array_equal(new_pm['closed_edge_integrals'],pm['closed_edge_integrals'])
        np.testing.assert_allclose((new_pm['Pi']@new_pm['normalized_edge_integrals']).T,
            pm['closed_edge_integrals'],atol=1e-12)
        edge=cx['edges'];potential=new_pm['normalized_potential']
        np.testing.assert_allclose(new_pm['normalized_edge_integrals'],
            new_cx['integer_cocycles']+potential[:,edge[:,1]]-potential[:,edge[:,0]],atol=1e-12)
        np.testing.assert_allclose((new_pm['Pi']@potential).T,pm['potential'],atol=1e-12)

    def test_reduction_shortens_an_artificially_sheared_basis(self):
        pm,cx=self.problem();new_pm,_=reduce_period_basis(pm,cx,np.eye(2))
        before=pm['Pi'].T@pm['Pi'];after=new_pm['Pi'].T@new_pm['Pi']
        self.assertLess(np.trace(after),np.trace(before)/100)
        self.assertLess(np.linalg.cond(after),np.linalg.cond(before)/100)

    def test_degenerate_gram_is_refused_before_lll(self):
        pm,cx=self.problem()
        with self.assertRaisesRegex(ValueError,'not numerically positive'):
            reduce_period_basis(pm,cx,np.diag([1.,0.]))


if __name__=='__main__':
    unittest.main()
