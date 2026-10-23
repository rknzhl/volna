"""Small regression checks for optional ambient affine frame features."""
import unittest

import numpy as np

from fields_first_frame import evaluate_W, global_affine_dictionary


class FrameFeatureChecks(unittest.TestCase):
    def test_global_and_local_derivative_matches_finite_difference(self):
        rng=np.random.default_rng(21);p,r,count=4,2,3
        centers=rng.normal(size=(count,p))
        frames=np.stack([np.linalg.qr(rng.normal(size=(p,r)))[0]
                         for _ in range(count)])
        model=dict(centers=centers,centerq=frames,h=1.3,affine=True,
            C=rng.normal(size=(count*r*(r+1),r)),r=r,p=p,
            global_affine_C=rng.normal(size=(p,p+1,r)))
        x=rng.normal(size=(5,p));_,jacobian=evaluate_W(model,x,derivative=True)
        epsilon=1e-6
        for j in range(p):
            direction=np.eye(p)[j]*epsilon
            difference=(evaluate_W(model,x+direction)-evaluate_W(model,x-direction))/(2*epsilon)
            np.testing.assert_allclose(jacobian[:,:,j,:],difference,rtol=2e-7,atol=2e-8)

    def test_global_block_represents_phase_velocity_on_curved_coordinates(self):
        theta=np.linspace(0,2*np.pi,17,endpoint=False)
        x=np.column_stack((np.cos(theta),np.sin(theta),
                           .3*np.cos(2*theta),.3*np.sin(2*theta)))
        velocity=np.column_stack((-np.sin(theta),np.cos(theta),
                                  -.6*np.sin(2*theta),.6*np.cos(2*theta)))
        q=(velocity/np.linalg.norm(velocity,axis=1,keepdims=True))[:,:,None]
        generator=np.array([[0,-1,0,0],[1,0,0,0],
                            [0,0,0,-2],[0,0,2,0]],dtype=float)
        coefficients=np.zeros((4,5,1));coefficients[:,1:,0]=generator
        model=dict(centers=x[[0]],centerq=q[[0]],h=1.,affine=True,
                   C=np.zeros((2,1)),r=1,p=4,global_affine_C=coefficients)
        np.testing.assert_allclose(evaluate_W(model,x)[:,:,0],velocity,atol=1e-14)
        projected=global_affine_dictionary(x,q)@coefficients.reshape(-1,1)
        expected=np.einsum('npa,np->na',q,velocity)
        np.testing.assert_allclose(projected[:,:,0],expected,atol=1e-14)

    def test_legacy_model_without_global_block_keeps_local_affine_field(self):
        frame=np.array([[1.,0.],[0.,1.],[0.,0.]])
        coefficient=np.array([[[[1.,2.],[.5,-.3],[.2,.7]],
                               [[3.,4.],[-.1,.6],[.8,.4]]]])
        model=dict(centers=np.zeros((1,3)),centerq=frame[None],h=2.,
                   affine=True,C=coefficient.reshape(-1,2),r=2,p=3)
        x=np.array([[.2,-.4,1.],[1.,.8,-2.]])
        expected=np.array([[[1.01,1.83],[2.83,3.98],[0.,0.]],
                           [[1.33,2.13],[3.27,4.46],[0.,0.]]])
        np.testing.assert_allclose(evaluate_W(model,x),expected,atol=1e-14)
        self.assertNotIn('global_affine_C',model)


if __name__=='__main__':
    unittest.main()
