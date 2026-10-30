"""Analytical feature checks, independent-window boundaries and fit leakage checks."""

from pathlib import Path
import copy
import json
import unittest

import numpy as np
import pandas as pd

from scripts.features import (FEATURE_NAMES, fft_features, extract_features,
                             window_signal, feature_sequence, select_features,
                             zscore_fit, zscore_transform, load_feature_sequences)
from scripts.preprocessing import SIGNALS, load_labelled_sequences

ROOT = Path(__file__).resolve().parents[1]
EPOCH = 1_792_885_238_606_000_000


def sequence(n=126, activity="walking"):
    t = np.arange(n) / 25
    values = np.zeros((n, 6))
    values[:, 2] = 9.81 + np.sin(2 * np.pi * 2 * t)
    df = pd.DataFrame(values, columns=SIGNALS)
    df["timestamp_ns"] = EPOCH + np.arange(n, dtype=np.int64) * 40_000_000
    df["seconds_elapsed"] = 7 + t
    if activity is not None:
        df["activity"] = activity
    return df


class WindowTests(unittest.TestCase):
    def test_exact_counts_and_exclusive_bounds(self):
        windows = window_signal(sequence(126))
        self.assertEqual(len(windows), 4)
        self.assertEqual([w["start_sample"] for w in windows], [0, 25, 50, 75])
        self.assertEqual(windows[-1]["stop_sample"], 125)
        self.assertEqual(windows[0]["stop_timestamp_ns"] - windows[0]["start_timestamp_ns"], 2_000_000_000)
        self.assertEqual(windows[0]["start_s"], 7)
        self.assertEqual(windows[0]["stop_s"], 9)

    def test_incomplete_tail_is_not_padded_or_joined(self):
        self.assertEqual(len(window_signal(sequence(49))), 0)
        self.assertEqual(len(window_signal(sequence(50))), 1)
        self.assertEqual(len(window_signal(sequence(74))), 1)
        self.assertEqual(len(window_signal(sequence(75))), 2)
        with self.assertRaisesRegex(ValueError, "no complete windows"):
            feature_sequence("short", sequence(49))
        for name in ["first", "second"]:
            frame = feature_sequence(name, sequence(60))
            self.assertEqual(len(frame), 1)
            self.assertEqual(frame.sequence_id.iloc[0], name)

    def test_majority_vote_and_tie_break_are_explicit(self):
        df = sequence(50)
        df.loc[:24, "activity"] = "standing"
        window = window_signal(df)[0]
        self.assertEqual(window["activity"], "standing")
        self.assertEqual(window["label_purity"], .5)
        df.loc[25, "activity"] = "standing"
        self.assertEqual(window_signal(df)[0]["label_purity"], .52)

    def test_unknown_labels_and_nonuniform_data_fail(self):
        df = sequence()
        df.loc[3, "activity"] = "unknown"
        with self.assertRaisesRegex(ValueError, "Unknown"):
            window_signal(df)
        with self.assertRaisesRegex(ValueError, "gaps"):
            window_signal(sequence().drop(index=25))
        with self.assertRaisesRegex(ValueError, "integer number"):
            window_signal(sequence(), window_s=.03)

    def test_unlabelled_sequences_remain_unlabelled(self):
        frame = feature_sequence("auxiliary", sequence(activity=None))
        self.assertNotIn("activity", frame)
        self.assertNotIn("label_purity", frame)


class FeatureMathTests(unittest.TestCase):
    def test_stationary_vector_has_correct_units_and_no_nan(self):
        samples = np.tile([3., 4., 0., .1, .2, .2], (50, 1))
        features = extract_features(samples)
        self.assertEqual(set(features), set(FEATURE_NAMES))
        self.assertAlmostEqual(features["sma_acc"], 7)
        self.assertAlmostEqual(features["sma_gyro"], .5)
        self.assertEqual(features["std_acc_mag"], 0)
        self.assertEqual(features["rms_centered_acc_mag"], 0)
        self.assertEqual(features["corr_acc_yz"], 0)
        self.assertEqual(features["dominant_freq_acc"], 0)
        self.assertEqual(features["spectral_entropy_acc"], 0)
        self.assertEqual(features["band_energy_ratio_acc"], 0)
        self.assertTrue(np.isfinite(list(features.values())).all())

    def test_sinusoid_peak_energy_and_rms_have_known_values(self):
        t = np.arange(50) / 25
        samples = np.zeros((50, 6))
        samples[:, 2] = 9.81 + 2 * np.sin(2 * np.pi * 2 * t)
        f = extract_features(samples)
        self.assertEqual(f["dominant_freq_acc"], 2)
        self.assertAlmostEqual(f["std_acc_mag"], np.sqrt(2), places=12)
        self.assertAlmostEqual(f["rms_centered_acc_mag"], np.sqrt(2), places=12)
        self.assertAlmostEqual(f["spectral_energy_acc"], 2, places=12)

    def test_parseval_power_for_even_and_odd_windows(self):
        rng = np.random.default_rng(17)
        for n in [50, 51]:
            signal = 10 + rng.normal(size=n)
            hann = np.hanning(n + 1)[:-1]
            expected = np.sum(((signal - signal.mean()) * hann) ** 2) / np.sum(hann ** 2)
            self.assertAlmostEqual(fft_features(signal)["spectral_energy_acc"], expected, places=12)

    def test_nyquist_bin_is_not_doubled(self):
        signal = 10 + (-1.) ** np.arange(50)
        f = fft_features(signal)
        self.assertEqual(f["dominant_freq_acc"], 12.5)
        self.assertAlmostEqual(f["spectral_energy_acc"], 1, places=12)

    def test_band_ratio_counts_three_hz_only_once(self):
        t = np.arange(50) / 25
        # A 3 Hz sinusoid puts 5/6 of Hann power in [0.5,3] and 1/6 above 3.
        f = fft_features(10 + np.sin(2 * np.pi * 3 * t))
        self.assertAlmostEqual(f["band_energy_ratio_acc"], 5, places=10)
        self.assertTrue(0 <= f["spectral_entropy_acc"] <= 1)

    def test_axis_correlation_handles_anticorrelation(self):
        data = np.zeros((50, 6))
        data[:, 1] = np.arange(50)
        data[:, 2] = -2 * np.arange(50)
        self.assertAlmostEqual(extract_features(data)["corr_acc_yz"], -1)


class TransformTests(unittest.TestCase):
    def test_population_scaling_and_test_extremes_do_not_refit(self):
        train = np.array([[1., 2.], [3., 6.], [5., 10.]])
        stats = zscore_fit(train)
        snapshot = copy.deepcopy(stats)
        scaled = zscore_transform(train, stats)
        np.testing.assert_allclose(scaled.mean(axis=0), 0, atol=1e-12)
        np.testing.assert_allclose(scaled.std(axis=0), 1, atol=1e-12)
        test = np.array([[1000., -2000.]])
        actual = zscore_transform(test, stats)
        np.testing.assert_allclose(actual, (test - train.mean(axis=0)) / train.std(axis=0))
        self.assertEqual(stats, snapshot)
        self.assertGreater(abs(actual[0, 0]), 100)  # No clipping to training bounds.

    def test_constant_features_cannot_create_division_by_zero(self):
        with self.assertRaisesRegex(ValueError, "constant"):
            zscore_fit(np.ones((5, 2)))
        with self.assertRaisesRegex(ValueError, "Invalid"):
            zscore_transform([[1., np.nan]], {"mean": [0., 0.], "std": [1., 1.]})

    def test_correlated_and_constant_candidates_removed_deterministically(self):
        rng = np.random.default_rng(13)
        frame = pd.DataFrame(rng.normal(size=(1000, 14)), columns=FEATURE_NAMES)
        frame["rms_centered_acc_mag"] = frame.std_acc_mag
        frame["mean_acc_z"] = -3 * frame.mean_acc_y
        frame["corr_acc_yz"] = 0
        kept, audit, _ = select_features(frame)
        self.assertNotIn("rms_centered_acc_mag", kept)
        self.assertNotIn("mean_acc_z", kept)
        self.assertNotIn("corr_acc_yz", kept)
        self.assertEqual(audit.set_index("feature").loc["rms_centered_acc_mag", "paired_feature"], "std_acc_mag")
        corr = frame[kept].corr().to_numpy()
        self.assertLessEqual(np.max(np.abs(corr[np.triu_indices(len(kept), 1)])), .95)

    def test_exported_fit_uses_only_training_recordings(self):
        config = json.loads((ROOT / "data/features/transform.json").read_text())
        train_names = {name for name, _ in load_labelled_sequences(ROOT, "train")}
        test_names = {name for name, _ in load_labelled_sequences(ROOT, "test")}
        self.assertEqual(set(config["fit_sequences"]), train_names)
        self.assertFalse(set(config["fit_sequences"]) & test_names)
        candidates = pd.read_csv(ROOT / "data/features/training_candidates.csv")
        self.assertEqual(set(candidates.sequence_id), train_names)
        selected = config["feature_order"]
        expected = zscore_fit(candidates[selected])
        np.testing.assert_allclose(config["statistics"]["mean"], expected["mean"], atol=1e-12)
        np.testing.assert_allclose(config["statistics"]["std"], expected["std"], atol=1e-12)
        self.assertEqual(config["fit_windows"], len(candidates))
        for name, df in load_labelled_sequences(ROOT, "test"):
            saved = pd.read_csv(ROOT / "data/features/test" / f"{name}.csv")
            raw = feature_sequence(name, df)
            np.testing.assert_allclose(saved[selected], zscore_transform(raw[selected], config["statistics"]), atol=1e-12)

    def test_model_loader_excludes_metadata_and_keeps_auxiliary_labels_absent(self):
        config = json.loads((ROOT / "data/features/transform.json").read_text())
        for split, count in [("train", 44), ("test", 8), ("unlabelled", 4)]:
            sequences = load_feature_sequences(ROOT, split)
            self.assertEqual(len(sequences), count)
            for item in sequences:
                self.assertEqual(item["X"].shape[1], len(config["feature_order"]))
                self.assertTrue(np.isfinite(item["X"]).all())
                if split == "unlabelled":
                    self.assertIsNone(item["y"])
                else:
                    self.assertEqual(len(item["y"]), len(item["X"]))
                    self.assertTrue(np.isin(item["y"], [0, 1, 2, 3]).all())


if __name__ == "__main__":
    unittest.main()
