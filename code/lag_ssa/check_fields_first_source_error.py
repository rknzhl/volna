"""The inverse gate includes source directions discarded before field fitting."""
import unittest
from unittest.mock import patch

import numpy as np

from fields_first_sensor_pipeline import GATES,_training_gates_once


class FullSourceErrorChecks(unittest.TestCase):
    def gate(self,discarded,reduced_error=0.):
        count=16;coordinates=np.ones((count,1))
        full_source=np.column_stack((coordinates[:,0],np.full(count,discarded)))
        full_prediction=np.column_stack((coordinates[:,0]+reduced_error,np.zeros(count)))
        full_rms=float(np.sqrt(np.mean(np.sum(full_source**2,axis=1))))
        model=dict(status='conditional_fields_first_model',X=coordinates,r=1,
            phases=np.zeros((count,1)),source_full_rms=full_rms,
            source_projection_residual_squared=np.full(count,discarded**2),
            diagnostics=dict(periods=dict(projection_relative=0.),
                encoder_field_duality_relative=0.,train_phase_step_rmse=0.,
                atlas_min_gap=1.,train_encoder_coverage=1.))
        with patch('fields_first_sensor_pipeline.core.flow',
                   side_effect=lambda model,starts,*args,**kwargs:starts.copy()), \
             patch('fields_first_sensor_pipeline.core.inverse',
                   return_value=coordinates+reduced_error):
            result=_training_gates_once(model)
        expected=float(np.linalg.norm(full_prediction-full_source)/
                       (np.sqrt(count)*full_rms))
        return result,expected

    def test_perfect_reduced_inverse_cannot_hide_discarded_source_error(self):
        result,expected=self.gate(discarded=.08)
        self.assertFalse(result['passed'])
        self.assertFalse(result['checks']['inverse'])
        self.assertGreater(result['inverse_window_relative'],GATES['inverse_window_relative'])
        self.assertAlmostEqual(result['inverse_window_relative'],expected)
        self.assertAlmostEqual(result['inverse_discarded_span_relative'],expected)

    def test_orthogonal_errors_are_counted_together(self):
        result,expected=self.gate(discarded=.04,reduced_error=.04)
        self.assertFalse(result['passed'])
        self.assertAlmostEqual(result['inverse_window_relative'],expected)
        self.assertGreater(expected,GATES['inverse_window_relative'])

    def test_small_full_window_error_still_passes_the_unchanged_gate(self):
        result,expected=self.gate(discarded=.02)
        self.assertTrue(result['passed'])
        self.assertAlmostEqual(result['inverse_window_relative'],expected)


if __name__=='__main__':unittest.main()
