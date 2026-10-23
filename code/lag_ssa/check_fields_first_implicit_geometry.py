"""Implicit geometry controls; no phase estimation or field fitting."""
import unittest

import numpy as np

from fields_first_implicit_geometry import fit_implicit_geometry,evaluate_implicit_geometry


class ImplicitGeometryChecks(unittest.TestCase):
    def source(self):
        a,b=np.meshgrid(np.linspace(0,2*np.pi,12,endpoint=False),
                        np.linspace(0,2*np.pi,10,endpoint=False),indexing='ij')
        a=a.ravel();b=b.ravel()
        x=np.column_stack((np.cos(a),np.sin(a),.6*np.cos(b),.6*np.sin(b)))
        tangent=np.zeros((len(x),4,2))
        tangent[:,0,0]=-np.sin(a);tangent[:,1,0]=np.cos(a)
        tangent[:,2,1]=-np.sin(b);tangent[:,3,1]=np.cos(b)
        return x,tangent

    def test_quadratic_relations_recover_the_common_tangent_kernel(self):
        x,tangent=self.source();model,diagnostic=fit_implicit_geometry(x,2)
        self.assertIsNotNone(model,diagnostic)
        result=evaluate_implicit_geometry(model,x)
        expected=tangent@tangent.transpose(0,2,1)
        np.testing.assert_allclose(result['P'],expected,atol=2e-11)
        relations=np.einsum('np,spq,nq->ns',x,model['C'],x)+x@model['A'].T+model['b']
        np.testing.assert_allclose(relations,0,atol=1e-12)
        self.assertGreater(result['gap'].min(),1-1e-10)

    def test_projector_derivative_matches_a_fixed_model_finite_difference(self):
        x,_=self.source();model,diagnostic=fit_implicit_geometry(x,2)
        self.assertIsNotNone(model,diagnostic)
        query=x[[3,17,35]];direction=np.random.default_rng(12).normal(size=query.shape)
        result=evaluate_implicit_geometry(model,query,directions=direction)
        epsilon=1e-6
        plus=evaluate_implicit_geometry(model,query+epsilon*direction)['P']
        minus=evaluate_implicit_geometry(model,query-epsilon*direction)['P']
        np.testing.assert_allclose(result['dP'],(plus-minus)/(2*epsilon),rtol=2e-6,atol=2e-8)

    def test_quadratic_relations_must_hold_on_the_excluded_rows(self):
        angle=np.linspace(0,2*np.pi,100,endpoint=False)
        x=np.column_stack((np.cos(angle),np.sin(angle)));x[::5]*=1.1
        model,diagnostic=fit_implicit_geometry(x,1)
        self.assertIsNone(model)
        self.assertEqual(diagnostic['reason'],'quadratic_relations_fail_validation')

    def test_nonquadratic_geometry_and_wrong_kernel_dimension_are_refused(self):
        t=np.linspace(-1,1,80)
        model,diagnostic=fit_implicit_geometry(np.column_stack((t,t**3)),1)
        self.assertIsNone(model)
        self.assertEqual(diagnostic['reason'],'no_exact_quadratic_relations')
        x,_=self.source();model,diagnostic=fit_implicit_geometry(x,1)
        self.assertIsNone(model)
        self.assertEqual(diagnostic['reason'],'quadratic_gradient_kernel_has_wrong_dimension')

    def test_feature_budget_and_support_are_bounded(self):
        model,diagnostic=fit_implicit_geometry(np.zeros((30,20)),1)
        self.assertIsNone(model)
        self.assertEqual(diagnostic['reason'],'quadratic_feature_budget_exceeded')
        x,_=self.source();model,diagnostic=fit_implicit_geometry(x,2)
        self.assertIsNotNone(model,diagnostic)
        with self.assertRaisesRegex(ValueError,'compact tangent-atlas support'):
            evaluate_implicit_geometry(model,np.full(4,100.))

    def test_implicit_geometry_refuses_local_graph_retraction_before_mutation(self):
        from fields_first_phase import enable_geometry_constrained_flows
        model=dict(atlas=dict(kind='implicit_quadratic'),
                   flow_domain_diagnostics=dict(maximum_nearest_distance=.2))
        with self.assertRaisesRegex(ValueError,'requires_local_graph_geometry'):
            enable_geometry_constrained_flows(model)
        self.assertEqual(model['flow_domain_diagnostics'],dict(maximum_nearest_distance=.2))
        self.assertNotIn('ambient_flow_domain_diagnostics',model)

    def test_auto_keeps_the_exact_branch_when_it_validates(self):
        x,_=self.source();model,diagnostic=fit_implicit_geometry(x,2,relation_tolerance='auto')
        self.assertIsNotNone(model,diagnostic)
        self.assertFalse(diagnostic['approximate_relations'])
        self.assertEqual(diagnostic['relation_tolerance'],1e-10)
        self.assertEqual(len(diagnostic['candidates']),1)

    def test_auto_near_relations_recover_a_noisy_torus_tangent(self):
        clean,tangent=self.source()
        x=clean+2e-4*np.random.default_rng(19).normal(size=clean.shape)
        exact,_=fit_implicit_geometry(x,2)
        self.assertIsNone(exact)
        model,diagnostic=fit_implicit_geometry(x,2,relation_tolerance='auto')
        self.assertIsNotNone(model,diagnostic)
        self.assertTrue(diagnostic['approximate_relations'])
        self.assertFalse(diagnostic['true_manifold_geometry_certified'])
        self.assertEqual(len(diagnostic['candidates']),8)
        self.assertGreaterEqual(diagnostic['minimum_source_gap'],.1)
        result=evaluate_implicit_geometry(model,clean)
        expected=tangent@tangent.transpose(0,2,1)
        self.assertLess(float(np.max(abs(result['P']-expected))),.005)

    def test_auto_does_not_invent_quadratic_geometry_for_a_gaussian_cloud(self):
        x=np.random.default_rng(20).normal(size=(240,4))
        model,diagnostic=fit_implicit_geometry(x,2,relation_tolerance='auto')
        self.assertIsNone(model)
        self.assertEqual(diagnostic['reason'],'no_validated_quadratic_geometry')

    def test_auto_refuses_too_few_independent_normal_directions(self):
        x=np.random.default_rng(22).normal(size=(180,3))
        x/=np.linalg.norm(x,axis=1,keepdims=True)
        model,diagnostic=fit_implicit_geometry(x,1,relation_tolerance='auto')
        self.assertIsNone(model)
        approximate=[candidate for candidate in diagnostic['candidates']
                     if candidate['approximate_relations']]
        self.assertTrue(all(candidate.get('minimum_normal_relative_eigenvalue',1)<1e-8
                            for candidate in approximate))

    def test_auto_relations_still_have_to_validate_on_excluded_rows(self):
        angle=np.linspace(0,2*np.pi,100,endpoint=False)
        x=np.column_stack((np.cos(angle),np.sin(angle)));x[::5]*=1.1
        model,diagnostic=fit_implicit_geometry(x,1,relation_tolerance='auto')
        self.assertIsNone(model)
        self.assertTrue(all(candidate['reason']=='quadratic_relations_fail_validation'
                            for candidate in diagnostic['candidates']))

    def test_numeric_tolerance_can_request_validated_near_relations(self):
        clean,_=self.source()
        x=clean+2e-4*np.random.default_rng(19).normal(size=clean.shape)
        model,diagnostic=fit_implicit_geometry(x,2,relation_tolerance=.0032)
        self.assertIsNotNone(model,diagnostic)
        self.assertTrue(diagnostic['approximate_relations'])
        self.assertEqual(diagnostic['relation_tolerance'],.0032)
        self.assertEqual(len(diagnostic['candidates']),1)


if __name__=='__main__':unittest.main()
