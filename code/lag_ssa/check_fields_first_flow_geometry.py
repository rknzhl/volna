"""Small checks of constrained-flow numerics, without fitting a phase model.

The 32-point atlas is deliberately coarse. Its fitted normal constraint is an
approximation to the source circle, so these checks assert convergence and
flow consistency, not exact recovery of the unit radius. The analytic rotation
field isolates the integrator from field-estimation errors.

Run with OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1.
"""
import unittest
from unittest.mock import patch

import numpy as np

import fields_first_phase as phase
from fields_first_sensor_pipeline import _training_gates, _training_gates_once
from fields_first_tangent_atlas import (
    fit_atlas, normal_graph_residual, retract_to_atlas,
)


class ConstrainedGeometryChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        angle = np.linspace(0., 2 * np.pi, 32, endpoint=False)
        cls.source = np.column_stack((np.cos(angle), np.sin(angle)))
        cls.atlas = fit_atlas(cls.source, r=1, k=8)

    def test_normal_constraint_converges_and_is_idempotent(self):
        query = np.array([[1.05, 0.], [0., .95], [-1.03, 0.]])
        before = np.linalg.norm(normal_graph_residual(self.atlas, query)[0], axis=1)
        projected, report = retract_to_atlas(self.atlas, query)
        after = np.linalg.norm(normal_graph_residual(self.atlas, projected)[0], axis=1)
        repeated, _ = retract_to_atlas(self.atlas, projected)
        self.assertLess(float(after.max()), 1e-9)
        self.assertLess(float(after.max()), float(before.max()) / 1000)
        self.assertLessEqual(report['iterations'], 20)
        np.testing.assert_allclose(repeated, projected, rtol=0., atol=1e-9)

    def test_retraction_preserves_distinct_tangential_locations(self):
        # No period, phase coordinate, reference state or end-point target is
        # supplied: different locations must remain different after projection.
        angle = np.array([0., .7, 2.])
        query = np.array([1.01, .99, 1.02])[:, None] * np.column_stack(
            (np.cos(angle), np.sin(angle)))
        projected, _ = retract_to_atlas(self.atlas, query)
        change = np.angle(np.exp(1j * (np.arctan2(projected[:, 1], projected[:, 0]) - angle)))
        self.assertLess(float(np.max(np.abs(change))), .01)
        self.assertGreater(float(np.linalg.norm(projected[0] - projected[2])), 1.)

    def test_unsupported_query_is_refused(self):
        with self.assertRaisesRegex(ValueError, 'support'):
            retract_to_atlas(self.atlas, np.array([[10., 0.]]))

    def test_unconverged_projection_is_refused(self):
        with self.assertRaisesRegex(ValueError, 'did_not_converge'):
            retract_to_atlas(self.atlas, np.array([[1.05, 0.]]), max_iterations=1)

    def test_projected_rotation_closes_and_improves_with_smaller_step(self):
        reference, _ = retract_to_atlas(self.atlas, self.source[[0]])
        model = dict(atlas=self.atlas, flow_geometry='normal_graph_constraint')

        def rotation(_model, points):
            x = np.atleast_2d(points)
            return np.column_stack((-x[:, 1], x[:, 0]))[:, :, None]

        with patch('fields_first_phase.phase_field', side_effect=rotation):
            coarse = phase.flow(model, reference, 2 * np.pi, axis=0, step=.24)
            fine = phase.flow(model, reference, 2 * np.pi, axis=0, step=.12)
        coarse_error = float(np.linalg.norm(coarse - reference))
        fine_error = float(np.linalg.norm(fine - reference))
        self.assertLess(fine_error, 2e-5)
        self.assertLess(fine_error, coarse_error / 4)
        self.assertLessEqual(
            model['flow_retraction_diagnostics']['maximum_normal_residual_relative'], 1e-9)

    def test_retry_does_not_swallow_unrelated_programming_error(self):
        with patch('fields_first_sensor_pipeline._training_gates_once',
                   side_effect=ValueError('incompatible array shape')), \
             patch('fields_first_sensor_pipeline.core.enable_geometry_constrained_flows') as retry:
            with self.assertRaisesRegex(ValueError, 'incompatible array shape'):
                _training_gates({})
            retry.assert_not_called()

    def _run_constrained_gate_with_flow(self, flow):
        angle = np.arctan2(self.source[:, 1], self.source[:, 0])[:, None]
        model = dict(X=self.source, phases=angle, r=1,
            status='conditional_fields_first_model',
            flow_geometry='normal_graph_constraint', flow_step=.12,
            flow_retraction_diagnostics={}, diagnostics=dict(
                periods=dict(projection_relative=0.), encoder_field_duality_relative=0.,
                train_phase_step_rmse=0., atlas_min_gap=1., train_encoder_coverage=1.,
                flow_reference_displacement_relative=0.))

        def inverse(_model, phases):
            return np.column_stack((np.cos(phases[:, 0]), np.sin(phases[:, 0])))

        with patch('fields_first_sensor_pipeline.core.constrained_flow_duality', return_value=(0., 16)), \
             patch('fields_first_sensor_pipeline.core.retract_flow_states', side_effect=lambda _m, x: x.copy()), \
             patch('fields_first_sensor_pipeline.core.flow', side_effect=flow), \
             patch('fields_first_sensor_pipeline.core.inverse', side_effect=inverse):
            return _training_gates_once(model)

    def test_coarse_support_failure_retries_a_finer_step(self):
        def rotation(model, points, angle, axis, step=None):
            if model['flow_step'] == .12 and step is None:
                raise ValueError('flow_outside_geometric_support')
            c, s = np.cos(angle), np.sin(angle)
            return points @ np.array([[c, s], [-s, c]])

        result = self._run_constrained_gate_with_flow(rotation)
        self.assertTrue(result['passed'])
        self.assertEqual(result['flow_step'], .06)
        self.assertEqual([trial['step'] for trial in result['flow_step_trials']], [.12, .06])
        self.assertEqual(result['flow_step_trials'][0]['exception'], 'flow_outside_geometric_support')

    def test_trial_loop_preserves_unrelated_programming_error(self):
        def broken_flow(*_args, **_kwargs):
            raise ValueError('incompatible array shape')

        with self.assertRaisesRegex(ValueError, 'incompatible array shape'):
            self._run_constrained_gate_with_flow(broken_flow)


class TopologyResolutionChecks(unittest.TestCase):
    def test_three_phase_readings_refine_unresolved_landmark_resolution(self):
        # This deterministic fixture reproduced the saved topology-resolution
        # probe: 192 landmarks were unresolved, while 384 resolved three classes.
        # Only measured coordinates enter inference; no phase count is supplied.
        from fields_first_periods import infer_torus_dimension
        from fields_first_preprocess import fit_preprocess, highpass_decompose
        from fields_first_sensor_cases import generate_variant

        readings, _ = generate_variant('threephase6sensor', 2400, seed=31,
            noise=0., trend=None, train_count=2400)
        preprocessing = fit_preprocess(readings, lag=32, maxrank=24, trend_degrees=(0,))
        split = highpass_decompose(preprocessing, readings, window=129, polyorder=2)
        interior = split['residual'][split['valid_mask']]
        centered = (interior - interior.mean(axis=0)) / interior.std(axis=0)
        joint = phase.windows(centered, 4) / np.sqrt(4 * readings.shape[1])
        _, singular, vh = phase.linalg.svd(joint, full_matrices=False)
        states = joint @ vh[singular > singular[0] * 1e-10].T

        dimension, diagnostic = infer_torus_dimension(states, nland=192)
        self.assertEqual(dimension, 3)
        attempts = diagnostic['resolution_attempts']
        self.assertEqual([attempt['landmarks'] for attempt in attempts], [192, 384])
        self.assertEqual([attempt['selected_dimension'] for attempt in attempts], [None, 3])
        self.assertFalse(diagnostic['initial_phase_coordinates_computed'])
        self.assertFalse(diagnostic['integer_period_basis_computed'])


if __name__ == '__main__':
    unittest.main()
