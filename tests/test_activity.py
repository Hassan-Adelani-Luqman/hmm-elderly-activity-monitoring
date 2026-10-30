"""Hand-computed mapping/evaluation checks and train/test separation contracts."""

import unittest
import numpy as np
from scripts.train_activity import state_mapping
from scripts.evaluate_activity import predict_gaussian, confusion_matrix, metrics_from_confusion
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


class MetricTests(unittest.TestCase):
    def test_hand_computed_multiclass_counts_and_ovr_vs_overall(self):
        C = np.array([[3, 1, 0], [0, 2, 1], [1, 0, 2]])
        rows = metrics_from_confusion(C, ['a', 'b', 'c'])
        self.assertEqual([rows[0][k] for k in ['TP', 'FN', 'FP', 'TN']], [3, 1, 1, 5])
        self.assertAlmostEqual(rows[0]['sensitivity'], .75)
        self.assertAlmostEqual(rows[0]['specificity'], 5/6)
        self.assertAlmostEqual(rows[0]['accuracy'], .8)
        self.assertAlmostEqual(rows[-1]['accuracy'], .7)
        self.assertAlmostEqual(rows[-1]['sensitivity'], (.75+2/3+2/3)/3)
        self.assertEqual(sum(r['n_windows'] for r in rows[:-1]), rows[-1]['n_windows'])

    def test_confusion_orientation_and_validation(self):
        C = confusion_matrix(np.array([0, 0, 1]), np.array([1, 1, 1]), 2)
        np.testing.assert_array_equal(C, [[0, 2], [0, 1]])
        for true,pred in [([],[]),([0],[2]),([0.],[0.]),([0,1],[0])]:
            with self.assertRaises(ValueError): confusion_matrix(true,pred,2)
        with self.assertRaises(ValueError): metrics_from_confusion(np.zeros((2,2),dtype=int),['a','b'])

    def test_absent_class_denominators_are_undefined(self):
        rows = metrics_from_confusion(np.array([[2, 0], [0, 0]]), ['a','b'])
        self.assertTrue(np.isnan(rows[0]['specificity']))
        self.assertTrue(np.isnan(rows[1]['sensitivity']))
        self.assertEqual(rows[-1]['accuracy'],1)


if __name__ == '__main__':
    unittest.main()
