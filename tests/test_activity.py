"""Hand-computed mapping/evaluation checks and train/test separation contracts."""

import unittest
from pathlib import Path
import copy
import shutil
import tempfile
from unittest.mock import patch
import numpy as np
from scripts.train_activity import state_mapping, train_models, load_models, file_hash
from scripts.features import load_feature_sequences
from scripts.evaluate_activity import predict_gaussian, confusion_matrix, metrics_from_confusion, evaluate_models
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


ROOT = Path(__file__).resolve().parents[1]


class SeparationTests(unittest.TestCase):
    def copy_features(self, root, include_test=False):
        """Create an isolated artifact fixture; raw data stay in the original root."""
        folder = root/'data/features'
        folder.mkdir(parents=True)
        shutil.copy2(ROOT/'data/features/transform.json', folder/'transform.json')
        shutil.copy2(ROOT/'data/split_manifest.csv', root/'data/split_manifest.csv')
        shutil.copytree(ROOT/'data/features/train', folder/'train')
        if include_test:
            shutil.copytree(ROOT/'data/features/test', folder/'test')
            shutil.copytree(ROOT/'data/models', root/'data/models')

    def test_training_loader_never_requests_test_or_unlabelled_features(self):
        calls = []
        def loader(root, split):
            calls.append(split)
            self.assertEqual(split, 'train')
            return load_feature_sequences(ROOT, split)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); self.copy_features(root)
            with patch('scripts.train_activity.load_feature_sequences', side_effect=loader):
                result = train_models(root)
            self.assertEqual(calls, ['train'])
            self.assertEqual(result['fit_windows'], 338)
            self.assertEqual(len(result['fit_sequences']), 44)
            self.assertFalse(result['protocol']['include_unlabelled'])
            self.assertTrue(result['models']['baum_welch']['converged'])
            np.testing.assert_array_equal(np.array(result['supervised_counts']['transitions']), np.diag([88,77,69,60]))
            # Single-activity clips carry no transition evidence: EM must keep the prior A.
            models = result['models']
            np.testing.assert_array_equal(models['baum_welch']['parameters']['A'],
                                          models['supervised']['parameters']['A'])
            self.assertTrue((np.array(models['baum_welch']['parameters']['A']) > 0).all())

    def test_evaluation_cannot_fit_or_remap_and_test_labels_do_not_change_predictions(self):
        frozen = load_models(ROOT)
        sequences = {split: load_feature_sequences(ROOT, split) for split in ['train','test']}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); self.copy_features(root, include_test=True)
            digest = file_hash(root/'data/models/activity_models.json')
            def loader(root, split): return copy.deepcopy(sequences[split])
            with patch('scripts.evaluate_activity.load_models', return_value=frozen), \
                 patch('scripts.evaluate_activity.load_feature_sequences', side_effect=loader), \
                 patch.object(GaussianHMM,'fit_supervised',side_effect=AssertionError('No evaluation fitting')), \
                 patch.object(GaussianHMM,'fit_baum_welch',side_effect=AssertionError('No evaluation fitting')), \
                 patch('scripts.train_activity.state_mapping',side_effect=AssertionError('No test mapping')):
                first = evaluate_models(root)
                for item in sequences['test']:
                    item['y'] = (item['y']+1)%4
                changed_labels = evaluate_models(root)
            self.assertEqual(file_hash(root/'data/models/activity_models.json'),digest)
            for model in frozen['protocol']['models']:
                np.testing.assert_array_equal(first['predictions'][model]['test'].predicted_state,
                                              changed_labels['predictions'][model]['test'].predicted_state)
            self.assertFalse(np.array_equal(first['confusion']['baum_welch']['test'],changed_labels['confusion']['baum_welch']['test']))


if __name__ == '__main__':
    unittest.main()
