"""Exact numerical checks and edge cases using synthetic observations only."""

import unittest
import numpy as np
from scripts.hmm import GaussianHMM, logsumexp
from scripts.verify_hmm import exact_verification, synthetic_verification


def toy_model():
    return GaussianHMM(2, 1).set_parameters([.6, .4], [[.75, .25], [.3, .7]], [[-1.], [1.5]], [[.5], [.8]])


class InferenceTests(unittest.TestCase):
    def test_all_paths_likelihood_viterbi_posteriors_and_exact_em_step(self):
        result = exact_verification()
        self.assertEqual(result['paths_enumerated'], 64)
        self.assertTrue(result['exact_em_step_passed'])

    def test_viterbi_persistence_backtracking_and_ties(self):
        m = GaussianHMM(2, 1).set_parameters([.99, .01], [[.999, .001], [.001, .999]], [[0], [2]], [[1], [1]])
        X = np.array([[0.], [2.], [0.]])
        np.testing.assert_array_equal(m.predict_viterbi(X), [0, 0, 0])
        self.assertEqual(np.argmax(m._log_emission(X), axis=1)[1], 1)
        np.testing.assert_array_equal(m.predict_viterbi([[0.]]), [0])
        m.set_parameters([.5, .5], np.full((2, 2), .5), [[0], [0]], [[1], [1]])
        np.testing.assert_array_equal(m.predict_viterbi(X), [0, 0, 0])

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


class SupervisedTests(unittest.TestCase):
    def test_exact_counts_without_cross_recording_transitions(self):
        m = GaussianHMM(2, 1).fit_supervised([np.array([[0.], [2.]]), np.array([[10.], [10.]])],
                                            [np.array([0, 0]), np.array([1, 1])])
        np.testing.assert_allclose(m.pi, [.5, .5])
        np.testing.assert_allclose(m.A, [[2/3, 1/3], [1/3, 2/3]])
        np.testing.assert_array_equal(m.supervised_counts_['transitions'], np.eye(2))
        np.testing.assert_allclose(m.means, [[1], [10]])
        np.testing.assert_allclose(m.variances, [[1], [m.var_floor]])

    def test_custom_prior_and_real_within_sequence_transition(self):
        m = GaussianHMM(2, 1).fit_supervised([np.array([[0.], [2.], [2.]])], [np.array([0, 1, 1])],
                                            transition_prior=[[9, 1], [1, 9]], initial_prior=[2, 1])
        np.testing.assert_allclose(m.pi, [.75, .25])
        np.testing.assert_allclose(m.A, [[9/11, 2/11], [1/11, 10/11]])

    def test_missing_states_invalid_labels_and_priors_fail(self):
        m = GaussianHMM(2, 1); X = [np.zeros((2, 1))]
        for y in [[], [np.array([0, 0])], [np.array([0., 1.])], [np.array([0, 2])], [np.array([0])]]:
            with self.assertRaises(ValueError): m.fit_supervised(X, y)
        with self.assertRaises(ValueError):
            m.fit_supervised(X, [np.array([0, 1])], transition_prior=0)


class EMTests(unittest.TestCase):
    def test_seeded_random_start_recovers_synthetic_parameters(self):
        result = synthetic_verification()
        self.assertTrue(result['converged'])
        self.assertLess(result['maximum_A_error'], result['A_error_tolerance'])
        self.assertLess(result['maximum_mean_error'], result['mean_error_tolerance'])

    def test_em_monotonicity_convergence_and_all_step_invariants(self):
        rng = np.random.default_rng(81)
        X = [np.r_[rng.normal(-2, .3, (50, 1)), rng.normal(2, .4, (50, 1))] for _ in range(3)]
        m = toy_model().fit_baum_welch(X)
        self.assertTrue(m.converged_)
        self.assertEqual(m.stop_reason_, 'likelihood_tolerance')
        self.assertLess(m.n_iter_, m.max_iter)
        self.assertTrue((np.diff(m.history_) >= 0).all())
        self.assertLess(abs(m.history_[-1]-m.history_[-2]), m.tol)
        self.assertAlmostEqual(m.score(X), m.history_[-1])
        self.assertEqual(len(m.invariant_history_), m.n_iter_)
        for row in m.invariant_history_:
            self.assertLessEqual(row['A_row_sum_error'], 1e-12)
            self.assertLessEqual(row['pi_sum_error'], 1e-12)
            self.assertGreaterEqual(row['minimum_variance'], m.var_floor)

    def test_singletons_do_not_create_transitions_and_zero_occupancy(self):
        m = GaussianHMM(2, 1).set_parameters([1, 0], np.eye(2), [[0], [99]], [[1], [2]])
        m.fit_baum_welch([np.array([[2.]]), np.array([[2.]])])
        np.testing.assert_array_equal(m.A, np.eye(2))
        np.testing.assert_array_equal(m.means, [[2], [99]])
        np.testing.assert_array_equal(m.variances, [[m.var_floor], [2]])

    def test_one_step_matches_independent_single_state_mle(self):
        m = GaussianHMM(1, 1, max_iter=1).set_parameters([1], [[1]], [[0]], [[5]])
        m.fit_baum_welch([np.array([[2.], [4.]]), np.array([[6.]])])
        np.testing.assert_allclose(m.means, [[4.]])
        np.testing.assert_allclose(m.variances, [[8/3]])
        self.assertFalse(m.converged_)
        self.assertEqual(m.stop_reason_, 'max_iter')

    def test_likelihood_decrease_rolls_back(self):
        class BrokenUpdate(GaussianHMM):
            def _m_step(self, sequences, posteriors):
                self.means[:] = 1e5
        m = BrokenUpdate(1, 1).set_parameters([1], [[1]], [[0]], [[1]])
        with self.assertRaisesRegex(FloatingPointError, 'decreased'):
            m.fit_baum_welch([np.zeros((3, 1))])
        np.testing.assert_array_equal(m.means, [[0]])
        self.assertEqual(len(m.history_), 1)


if __name__ == '__main__':
    unittest.main()
