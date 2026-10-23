"""Check the actual smooth phase interpolant, including its landmark limits."""
import unittest
import numpy as np
from fields_first_phase import encode


class PhaseInterpolationChecks(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(14)
        self.centers=np.array([[-.2,0.],[.2,.1],[0.,-.3],[.1,.3]])
        edges=np.array([(i,j) for i in range(4) for j in range(i+1,4)])
        self.q=rng.normal(scale=.03,size=(4,2))
        self.slopes=rng.normal(scale=.1,size=(4,2,2))
        self.model=dict(r=2,cardinal_phase_weights=True,
            complex=dict(landmarks=self.centers,edges=edges,
                         integer_cocycles=np.zeros((2,len(edges)),dtype=int),cover_radius=1.),
            period_model=dict(normalized_potential=self.q.T),
            center_normalized_beta=self.slopes)

    def test_landmark_values_and_derivatives_equal_the_fitted_jets(self):
        result=encode(self.model,self.centers,derivative=True)
        np.testing.assert_allclose(result['phases'],2*np.pi*self.q,atol=1e-13)
        np.testing.assert_allclose(result['jacobian'],2*np.pi*self.slopes,atol=1e-13)

    def test_interpolant_derivative_matches_finite_difference(self):
        query=np.vstack((self.centers[[0]],[[.03,.04],[-.13,.12]]))
        result=encode(self.model,query,derivative=True)
        epsilon=1e-6
        for j in range(2):
            step=np.eye(2)[j]*epsilon
            plus=encode(self.model,query+step)['phases']
            minus=encode(self.model,query-step)['phases']
            derivative=np.angle(np.exp(1j*(plus-minus)))/(2*epsilon)
            np.testing.assert_allclose(result['jacobian'][:,:,j],derivative,rtol=2e-6,atol=2e-8)


if __name__=='__main__':unittest.main()
