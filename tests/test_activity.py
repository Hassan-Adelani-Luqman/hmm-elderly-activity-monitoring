"""Hand-computed mapping/evaluation checks and train/test separation contracts."""

import unittest
import numpy as np
from scripts.train_activity import state_mapping
from scripts.evaluate_activity import predict_gaussian
from scripts.hmm import GaussianHMM


class MappingTests(unittest.TestCase):
    def test_nonidentity_permutation_has_correct_direction(self):
        mapping, counts = state_mapping([np.array([0, 0, 1, 1, 2])], [np.array([2, 2, 0, 0, 1])], 3)
        np.testing.assert_array_equal(mapping, [1, 2, 0])
        np.testing.assert_array_equal(mapping[[2, 0, 1]], [0, 1, 2])
        self.assertEqual(counts[mapping, np.arange(3)].sum(), 5)

    def test_missing_training_classes_and_misalignment_fail(self):
        for labels, predicted in [([np.array([0])], [np.array([0])]), ([np.array([0, 1])], [np.array([0])])]:
            with self.assertRaises(ValueError): state_mapping(labels, predicted, 2)


class BaselineTests(unittest.TestCase):
    def test_baseline_ignores_transitions_and_sequence_neighbours(self):
        model = GaussianHMM(2, 1).set_parameters([.5, .5], [[.99, .01], [.01, .99]], [[0], [2]], [[1], [1]])
        X = np.array([[0.], [2.], [0.]])
        np.testing.assert_array_equal(predict_gaussian(model, X), [0, 1, 0])
        model.A[:] = [[0, 1], [1, 0]]
        np.testing.assert_array_equal(predict_gaussian(model, X), [0, 1, 0])
        np.testing.assert_array_equal(predict_gaussian(model, X[1:2]), [1])

    def test_baseline_uses_fixed_initial_prior_not_test_class_counts(self):
        model = GaussianHMM(2, 1).set_parameters([.9, .1], np.eye(2), [[0], [0]], [[1], [1]])
        np.testing.assert_array_equal(predict_gaussian(model, [[2], [9]]), [0, 0])


if __name__ == '__main__':
    unittest.main()
