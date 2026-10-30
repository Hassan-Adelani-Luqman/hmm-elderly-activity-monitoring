"""Exact numerical checks and edge cases using synthetic observations only."""

import unittest
import numpy as np
from scripts.hmm import GaussianHMM, logsumexp


def toy_model():
    return GaussianHMM(2, 1).set_parameters([.6, .4], [[.75, .25], [.3, .7]], [[-1.], [1.5]], [[.5], [.8]])


class InferenceTests(unittest.TestCase):
    def test_gaussian_constants_and_positive_log_density(self):
        m = GaussianHMM(1, 2).set_parameters([1], [[1]], [[0, 1]], [[.001, .004]])
        expected = -np.log(2*np.pi) - .5*np.log(.001*.004)
        self.assertAlmostEqual(m._log_emission([[0, 1]])[0, 0], expected)
        self.assertGreater(m.score([np.array([[0, 1]])]), 0)

    def test_single_sample_posteriors_and_sequence_scores(self):
        m = toy_model(); X = np.array([[.2]])
        gamma, xi, ll = m._posteriors(X)
        expected = m.pi * np.exp(m._log_emission(X)[0])
        np.testing.assert_allclose(gamma[0], expected/expected.sum())
        self.assertEqual(xi.shape, (0, 2, 2))
        self.assertAlmostEqual(m.score([X, X]), 2*ll)

    def test_scaling_and_long_extreme_sequence(self):
        m = toy_model(); X = np.full((1200, 1), 100.)
        a, c, ll = m._forward(m._log_emission(X))
        self.assertTrue(np.isfinite(ll))
        np.testing.assert_allclose(np.exp(a).sum(axis=1), 1, atol=1e-12)
        np.testing.assert_allclose(m.predict_proba(X).sum(axis=1), 1, atol=1e-12)
        self.assertEqual(ll, c.sum())

    def test_zero_probabilities_and_impossible_paths(self):
        m = GaussianHMM(2, 1).set_parameters([1, 0], np.eye(2), [[0], [1]], [[1], [1]])
        np.testing.assert_array_equal(m.predict_proba([[0], [2]]), [[1, 0], [1, 0]])
        with self.assertRaisesRegex(ValueError, 'no finite-probability path'):
            m._forward([[-np.inf, 0]])
        self.assertEqual(logsumexp([-np.inf, -np.inf]), -np.inf)

    def test_validation_and_parameter_copy(self):
        m = toy_model(); A = m.A.copy()
        with self.assertRaises(ValueError):
            m.set_parameters([1, 0], [[1, 1], [0, 1]], [[0], [1]], [[1], [1]])
        np.testing.assert_array_equal(m.A, A)
        for X in [[], [[np.nan]], [[1, 2]]]:
            with self.assertRaises(ValueError): m.score([X])
        with self.assertRaises(ValueError): m.score(np.zeros((3, 1)))
        with self.assertRaises(ValueError): m.score([])
        with self.assertRaises(ValueError): GaussianHMM(0, 1)


if __name__ == '__main__':
    unittest.main()
