"""Small regression checks for reconstruction and finite-data refusal gates."""
import unittest
from unittest.mock import patch
import numpy as np
from fields_first_sensor_pipeline import SensorPipeline
from fields_first_phase import overlap_maps


class RefusalChecks(unittest.TestCase):
    def pipeline(self):
        model=dict(lag=2,channels=1,r=1,mean=np.array([0.]),scale=np.array([1.]),
                   normal=1.,basis=np.eye(2),atlas={},diagnostics=dict(phase_step=np.array([.1])))
        obj=SensorPipeline();obj.model=model;obj.channels=1;obj.preprocess={}
        obj.status='accepted_conditionally_by_training_gates'
        return obj

    def transform(self,pipeline,n=4,inverse_factor=1.):
        residual=np.arange(1,n+1,dtype=float)[:,None]
        split=dict(residual=residual,slow=np.zeros_like(residual),denoised=residual,
                   valid_mask=np.ones(n,dtype=bool),time_index=np.arange(n))
        count=n-1
        encoded=dict(phases=(.1*np.arange(count))[:,None],covered=np.ones(count,dtype=bool),
                     jacobian=np.tile([[[1.,0.]]],(count,1,1)))
        x=np.column_stack((residual[:-1,0],residual[1:,0]))
        with patch('fields_first_sensor_pipeline.highpass_decompose',return_value=split), \
             patch('fields_first_sensor_pipeline.core.encode',return_value=encoded), \
             patch('fields_first_sensor_pipeline.core.evaluate_projector',return_value=dict(gap=np.ones(count))), \
             patch('fields_first_sensor_pipeline.core.inverse',return_value=inverse_factor*x), \
             patch('fields_first_sensor_pipeline.core.phase_field',return_value=np.tile([[[1.],[0.]]],(count,1,1))):
            return pipeline.transform(residual)

    def test_inverse_disagreement_is_rejected(self):
        result=self.transform(self.pipeline(),inverse_factor=4.)
        self.assertEqual(result['status'],'unresolved_query_inverse')

    def test_single_window_has_no_dynamic_validation(self):
        result=self.transform(self.pipeline(),n=2)
        self.assertEqual(result['status'],'unresolved_query_block_too_short')

    def test_nonfinite_dynamics_is_rejected(self):
        obj=self.pipeline();obj.model['diagnostics']['phase_step'][:]=np.nan
        result=self.transform(obj)
        self.assertEqual(result['status'],'unresolved_nonfinite_query_diagnostic')

    def test_consistent_windows_fold_to_the_same_readings(self):
        result=self.transform(self.pipeline())
        self.assertEqual(result['status'],'transformed_conditionally')
        np.testing.assert_allclose(result['reconstruction'][:,0],np.arange(1,5))

    def test_prefix_with_fewer_rows_than_tangent_rank_is_rejected(self):
        q=np.tile(np.eye(2),(2,1,1))
        with self.assertRaisesRegex(ValueError,'not_injective'):
            overlap_maps(q,channels=1)

    def test_short_and_constant_records_are_explicit_refusals(self):
        self.assertEqual(SensorPipeline().fit(np.ones((10,1))).status,'unresolved_training_block_too_short')
        self.assertEqual(SensorPipeline().fit(np.ones((240,1))).status,'unresolved_constant_readings')


if __name__=='__main__':
    unittest.main()
