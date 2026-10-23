"""Algebraic checks of the rejected experimental local primitive extension."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'code/lag_ssa'))
from fields_first_phase_charts import fit_phase_charts, evaluate_phase_charts


def primitive(x):
    a, b = x[..., 0], x[..., 1]
    return np.stack((a + .2*a*a + .3*a*b, b - .1*a*b + .4*b*b), axis=-1)


def derivative(x):
    a, b = x[..., 0], x[..., 1]
    out = np.zeros(x.shape[:-1] + (2, 3))
    out[..., 0, 0] = 1 + .4*a + .3*b; out[..., 0, 1] = .3*a
    out[..., 1, 0] = -.1*b; out[..., 1, 1] = 1 - .1*a + .8*b
    return out


class PhaseChartChecks(unittest.TestCase):
    def model(self, centers):
        n = len(centers)
        frames = np.tile(np.eye(3)[:, :2], (n, 1, 1))
        # Integer changes of local lifts must not change the represented map.
        lift = np.column_stack((np.arange(n) % 3, -(np.arange(n) % 2)))
        edges = np.asarray([(i, j) for i in range(n) for j in range(i+1, n)])
        labels = lift[edges[:, 1]] - lift[edges[:, 0]]
        model = dict(r=2, atlas={}, center_normalized_beta=derivative(centers),
            complex=dict(landmarks=centers, edges=edges, integer_cocycles=labels.T,
                         cover_radius=1.),
            period_model=dict(normalized_potential=(primitive(centers)-lift).T))
        return model, frames, lift

    def fit(self, model, frames):
        with patch('fields_first_phase_charts.evaluate_tangent_frame', return_value={'Q': frames}):
            return fit_phase_charts(model)

    def test_quadratic_primitive_and_derivative_survive_integer_lifts(self):
        centers = np.asarray([(a, b, 0.) for a in (-.8, 0., .9) for b in (-.7, .1, .8)])
        model, frames, lift = self.model(centers)
        charts = self.fit(model, frames)
        self.assertEqual(charts['diagnostics']['linear_fallback_count'], 0)
        query = np.array([[.2, -.1, .3], [-.3, .4, -.2]])
        value, jacobian = evaluate_phase_charts(charts, query, derivative=True)
        np.testing.assert_allclose(value, primitive(query)[:, None]-lift[None], atol=1e-12)
        np.testing.assert_allclose(jacobian, np.broadcast_to(derivative(query)[:, None], jacobian.shape), atol=1e-12)
        root = 2
        lifted = value + lift[None] - lift[None, [root]]
        np.testing.assert_allclose(lifted, np.broadcast_to(
            primitive(query)[:, None]-lift[None, [root]], lifted.shape), atol=1e-12)

    def test_rank_deficient_design_keeps_original_linear_beta_jet(self):
        centers = np.column_stack((np.linspace(-1., 1., 7), np.zeros((7, 2))))
        model, frames, _ = self.model(centers)
        charts = self.fit(model, frames)
        self.assertTrue(charts['linear_fallback'].all())
        query = np.array([[.13, .24, -.2]])
        value, jacobian = evaluate_phase_charts(charts, query, derivative=True)
        expected = charts['potential'][None] + np.einsum(
            'lrp,nlp->nlr', model['center_normalized_beta'], query[:, None]-centers[None])
        np.testing.assert_allclose(value, expected, atol=1e-14)
        np.testing.assert_allclose(jacobian[0], model['center_normalized_beta'], atol=1e-14)

    def test_analytic_chart_jacobian_matches_central_difference(self):
        centers = np.asarray([(a, b, 0.) for a in (-.8, 0., .9) for b in (-.7, .1, .8)])
        model, frames, _ = self.model(centers)
        charts = self.fit(model, frames)
        query = np.array([[.2, -.1, .15]])
        _, jacobian = evaluate_phase_charts(charts, query, derivative=True)
        for a in range(3):
            delta = np.eye(3)[a] * 1e-6
            plus = evaluate_phase_charts(charts, query+delta)[0]
            minus = evaluate_phase_charts(charts, query-delta)[0]
            np.testing.assert_allclose((plus-minus)/(2e-6), jacobian[..., a], atol=1e-9)


if __name__ == '__main__':
    unittest.main()
