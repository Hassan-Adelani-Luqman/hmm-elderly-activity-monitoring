"""Regression checks for silent data corruption identified in the phase review."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

from scripts.preprocessing import (
    ACTIVITIES, SIGNALS, STEP_NS, build_dataset, harmonise_recording,
    label_session, load_recording, load_sensor, processed_frame,
    require_training_ready, sampling_stats, save_processed, trim_clip,
    recording_fingerprint, unlabelled_session_blocks,
)

EPOCH = 1_792_885_238_606_000_000


def recording(fs=60, seconds=16, offset_ns=8_333_333, frequency=8):
    step = round(1e9 / fs)
    rec = {"name": "synthetic", "meta": {"epoch_ns": EPOCH}}
    for sensor, offset in [("acc", 0), ("gyro", offset_ns)]:
        relative = np.arange(0, int(seconds * 1e9), step, dtype=np.int64) + offset
        t = relative / 1e9
        rec[sensor] = pd.DataFrame({"timestamp_ns": EPOCH + relative, "t": t,
                                    "x": np.sin(2 * np.pi * frequency * t),
                                    "y": np.zeros(len(t)), "z": np.full(len(t), 9.81)})
    return rec


class PreprocessingTests(unittest.TestCase):
    def test_one_common_grid_matches_direct_native_interpolation(self):
        rec = recording()
        out = harmonise_recording(rec)
        start = int(out.timestamp_ns.iloc[0])
        for sensor in ("acc", "gyro"):
            df = rec[sensor]
            sos = butter(4, 10, fs=sampling_stats(df)["measured_hz"], output="sos")
            expected = np.interp((out.timestamp_ns.to_numpy() - start) / 1e9,
                                 (df.timestamp_ns.to_numpy() - start) / 1e9,
                                 sosfiltfilt(sos, df.x.to_numpy()))
            np.testing.assert_allclose(out[f"{sensor}_x"], expected, atol=1e-12)
        middle = out[out.t.between(2, 14)]
        # Identical continuous signals should not acquire the old ~28% amplitude mismatch.
        self.assertLess(abs(middle.acc_x.std() / middle.gyro_x.std() - 1), .02)
        self.assertTrue((np.diff(out.timestamp_ns) == STEP_NS).all())

    def test_lowpass_suppresses_out_of_band_signal(self):
        low = harmonise_recording(recording(frequency=2))
        high = harmonise_recording(recording(frequency=20))
        self.assertLess(high.loc[high.t.between(2, 14), "acc_x"].std(),
                        .01 * low.loc[low.t.between(2, 14), "acc_x"].std())

    def test_lower_native_rate_is_supported(self):
        out = harmonise_recording(recording(fs=29.2, frequency=2))
        self.assertTrue((np.diff(out.timestamp_ns) == STEP_NS).all())

    def test_gaps_are_rejected_before_interpolation(self):
        rec = recording()
        rec["acc"] = rec["acc"][~rec["acc"].t.between(4, 10)].reset_index(drop=True)
        with self.assertRaisesRegex(ValueError, "large gaps"):
            harmonise_recording(rec)

    def test_recovered_blocks_respect_gaps_in_either_sensor(self):
        rec = recording(seconds=30)
        rec["acc"] = rec["acc"][~rec["acc"].t.between(8, 12)].reset_index(drop=True)
        rec["gyro"] = rec["gyro"][~rec["gyro"].t.between(10, 14)].reset_index(drop=True)
        blocks = unlabelled_session_blocks(rec)
        self.assertEqual(len(blocks), 2)
        self.assertLess(blocks[0]["df"].t.iloc[-1], 8)
        self.assertGreater(blocks[1]["df"].t.iloc[0], 14)
        for block in blocks:
            self.assertNotIn("activity", block["df"])
            self.assertTrue((np.diff(block["df"].timestamp_ns) == STEP_NS).all())
            np.testing.assert_allclose((block["df"].timestamp_ns.to_numpy() - EPOCH) / 1e9,
                                       block["df"].t, atol=1e-9, rtol=0)

    def test_existing_sessions_are_rejected(self):
        root = Path(__file__).resolve().parents[1]
        for path in sorted((root / "data/raw/sessions").iterdir()):
            if not path.is_dir():
                continue
            rec = load_recording(path)
            # This test remains valid when the user replaces either recording.
            if any(sampling_stats(rec[s])["large_gaps"] for s in ("acc", "gyro")):
                with self.assertRaisesRegex(ValueError, "large gaps"):
                    harmonise_recording(rec)

    def test_upsampling_and_invalid_cutoff_fail(self):
        with self.assertRaisesRegex(ValueError, "Upsampling"):
            harmonise_recording(recording(fs=20))
        with self.assertRaisesRegex(ValueError, "Nyquist"):
            harmonise_recording(recording(), cutoff_hz=12.5)

    def test_crop_limits_duration_and_preserves_origin(self):
        rec = recording(seconds=22)
        out = trim_clip(harmonise_recording(rec))
        self.assertEqual(len(out), 251)
        self.assertGreater(out.t.iloc[0], 1)
        out["activity"] = "standing"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "out.csv"
            save_processed(out, path)
            saved = pd.read_csv(path)
        self.assertGreater(saved.timestamp_ns.iloc[0], EPOCH)
        np.testing.assert_allclose((saved.timestamp_ns.to_numpy() - EPOCH) / 1e9,
                                   saved.seconds_elapsed, atol=1e-9, rtol=0)
        self.assertTrue((np.diff(saved.timestamp_ns) == STEP_NS).all())

    def test_too_short_crop_fails(self):
        with self.assertRaisesRegex(ValueError, "5–10"):
            trim_clip(harmonise_recording(recording(seconds=6)))

    def labels(self, rows, df=None):
        df = harmonise_recording(recording()) if df is None else df
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.csv"
            pd.DataFrame(rows, columns=["start_s", "end_s", "activity"]).to_csv(path, index=False)
            return label_session(df, path)

    def test_labels_exclude_sync_prefix_and_preserve_real_transitions(self):
        rows = [(2, 5, "still"), (5, 8, "standing"), (8, 11, "walking"), (11, 14, "jumping")]
        out = self.labels(rows)
        self.assertTrue((np.diff(out.timestamp_ns) == STEP_NS).all())
        self.assertEqual(set(out.activity), set(ACTIVITIES))
        self.assertTrue((out.loc[out.t < 5, "activity"] == "still").all())
        self.assertTrue((out.loc[(out.t >= 5) & (out.t < 8), "activity"] == "standing").all())

    def test_overlapping_gapped_unknown_and_outside_labels_fail(self):
        cases = [([(2, 5, "still"), (4, 8, "standing")], "overlap"),
                 ([(2, 5, "still"), (6, 8, "standing")], "internal gaps"),
                 ([(2, 5, "flying")], "Unknown"),
                 ([(2, 30, "still")], "outside"),
                 ([(4, 3, "still")], "duration")]
        for rows, message in cases:
            with self.subTest(rows=rows), self.assertRaisesRegex(ValueError, message):
                self.labels(rows)

    def test_nonfinite_and_discontinuous_exports_fail(self):
        out = trim_clip(harmonise_recording(recording()))
        out["activity"] = "standing"
        broken = out.copy()
        broken.loc[3, "acc_x"] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            processed_frame(broken)
        with self.assertRaisesRegex(ValueError, "25 Hz"):
            processed_frame(out.drop(index=3))

    def test_exact_duplicates_removed_and_conflicting_duplicates_rejected(self):
        raw = recording()["acc"].rename(columns={"timestamp_ns": "time"})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "TotalAcceleration.csv"
            pd.concat([raw, raw.iloc[[3]]]).to_csv(path, index=False)
            loaded = load_sensor(directory, "acc", EPOCH)
            self.assertEqual(len(loaded), len(raw))
            self.assertEqual(sampling_stats(loaded)["duplicates_removed"], 1)
            conflicting = raw.iloc[[3]].copy()
            conflicting["x"] += 1
            pd.concat([raw, conflicting]).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "Conflicting"):
                load_sensor(directory, "acc", EPOCH)

    def test_incomplete_dataset_blocks_training(self):
        with self.assertRaisesRegex(ValueError, "Training blocked"):
            require_training_ready({"training_ready": False, "blockers": ["missing verified sessions"]})
        require_training_ready({"training_ready": True, "blockers": []})

    def test_pipeline_removes_stale_labels_and_exports_only_unlabelled_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / "data/raw/clips/standing/standing_01",
                     root / "data/raw/sessions/train_session_01"]
            for path in paths:
                path.mkdir(parents=True)
                rec = recording(seconds=30)
                if "sessions" in path.parts:
                    for sensor in ("acc", "gyro"):
                        rec[sensor] = rec[sensor][~rec[sensor].t.between(8, 14)].reset_index(drop=True)
                for sensor, filename in [("acc", "TotalAcceleration.csv"), ("gyro", "Gyroscope.csv")]:
                    rec[sensor].rename(columns={"timestamp_ns": "time"}).to_csv(path / filename, index=False)
                pd.DataFrame([{"device name": "synthetic", "platform": "test", "platform version": "1",
                               "appVersion": "1", "recording time": "2026-10-29_12-00-00",
                               "recording epoch time": EPOCH // 1_000_000,
                               "sensors": "TotalAcceleration|Gyroscope", "sampleRateMs": "10|10"}]).to_csv(path / "Metadata.csv", index=False)
            pd.DataFrame([{"file": "standing_01", "start_s": 2, "end_s": 12,
                           "source_sha256": recording_fingerprint(paths[0])}]).to_csv(root / "data/clip_intervals.csv", index=False)
            pd.DataFrame([{"file": "train_session_01", "status": "pending", "source_sha256": "",
                           "evidence": ""}]).to_csv(root / "data/session_review.csv", index=False)
            stale = root / "data/processed/train/sessions/train_session_01.csv"
            stale.parent.mkdir(parents=True)
            stale.write_text("activity\nwalking\n")
            result = build_dataset(root)
            self.assertFalse(stale.exists())
            self.assertFalse(result["status"]["training_ready"])
            self.assertEqual(result["status"]["accepted_clips"], 1)
            self.assertEqual(result["status"]["accepted_sessions"], 0)
            self.assertEqual(result["status"]["unlabelled_session_sequences"], 2)
            files = list((root / "data/processed/unlabelled/sessions").glob("*.csv"))
            self.assertEqual(len(files), 2)
            self.assertTrue(all("activity" not in pd.read_csv(path).columns for path in files))
            # Changing a crop's source invalidates its approval instead of reusing it.
            path = paths[0] / "TotalAcceleration.csv"
            changed = pd.read_csv(path)
            changed.loc[0, "x"] += .01
            changed.to_csv(path, index=False)
            revised = build_dataset(root)
            self.assertEqual(revised["status"]["accepted_clips"], 0)
            self.assertIn("crop interval missing or stale", revised["rejected"].reason.iloc[0])
            import json
            self.assertFalse(json.loads((root / "data/quality/status.json").read_text())["training_ready"])


if __name__ == "__main__":
    unittest.main()
