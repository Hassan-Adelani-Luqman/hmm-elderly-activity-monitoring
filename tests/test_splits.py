"""Prevent recording-level leakage, stale manifests and accidental train/test mixing."""

from pathlib import Path
import shutil
import tempfile
import unittest

import pandas as pd

from scripts.splits import load_split_manifest, source_fingerprint, verify_split_outputs
from scripts.preprocessing import load_labelled_sequences

ROOT = Path(__file__).resolve().parents[1]


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.paths = []
        rows = []
        for name, split in [("standing_01", "train"), ("standing_02", "test")]:
            source = ROOT / "data/raw/clips/standing" / name
            dest = self.root / "data/raw/clips/standing" / name
            dest.mkdir(parents=True)
            for filename in ["Metadata.csv", "TotalAcceleration.csv", "Gyroscope.csv"]:
                shutil.copyfile(source / filename, dest / filename)
            self.paths.append(dest)
            rows.append({"file": name, "activity": "standing", "split": split,
                         "source_sha256": source_fingerprint(dest)})
        self.manifest = pd.DataFrame(rows)
        self.path = self.root / "data/split_manifest.csv"
        self.manifest.to_csv(self.path, index=False)

    def test_whole_recordings_have_one_fixed_role(self):
        manifest = load_split_manifest(self.root)
        self.assertEqual(manifest.loc["standing_01", "split"], "train")
        self.assertEqual(manifest.loc["standing_02", "split"], "test")

    def test_missing_manifest_fails_instead_of_training_on_everything(self):
        self.path.unlink()
        with self.assertRaisesRegex(ValueError, "Missing fixed split"):
            load_split_manifest(self.root)

    def test_duplicate_membership_fails(self):
        pd.concat([self.manifest, self.manifest.iloc[[0]]]).to_csv(self.path, index=False)
        with self.assertRaisesRegex(ValueError, "more than once"):
            load_split_manifest(self.root)

    def test_unlisted_recordings_fail(self):
        self.manifest.iloc[[0]].to_csv(self.path, index=False)
        with self.assertRaisesRegex(ValueError, "every raw clip"):
            load_split_manifest(self.root)

    def test_renaming_a_training_recording_does_not_make_it_unseen(self):
        for filename in ["Metadata.csv", "TotalAcceleration.csv", "Gyroscope.csv"]:
            shutil.copyfile(self.paths[0] / filename, self.paths[1] / filename)
        self.manifest.loc[1, "source_sha256"] = source_fingerprint(self.paths[1])
        self.manifest.to_csv(self.path, index=False)
        with self.assertRaisesRegex(ValueError, "leakage"):
            load_split_manifest(self.root)

    def test_stale_source_fails(self):
        path = self.paths[1] / "Gyroscope.csv"
        path.write_bytes(path.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "stale"):
            load_split_manifest(self.root)

    def test_test_file_left_in_training_is_detected(self):
        manifest = load_split_manifest(self.root)
        output = self.root / "data/processed"
        for folder, name in [("train/clips", "standing_01"), ("test", "standing_02")]:
            p = output / folder / f"{name}.csv"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("placeholder")
        verify_split_outputs(output, manifest)
        (output / "train/clips/standing_02.csv").write_text("placeholder")
        with self.assertRaisesRegex(ValueError, "train: processed"):
            verify_split_outputs(output, manifest)

    def test_project_holdout_is_balanced_and_loads_separately(self):
        manifest = load_split_manifest(ROOT)
        counts = manifest.groupby(["activity", "split"]).size().unstack()
        self.assertTrue((counts.train == 11).all())
        self.assertTrue((counts.test == 2).all())
        train = load_labelled_sequences(ROOT, "train")
        test = load_labelled_sequences(ROOT, "test")
        self.assertEqual(len(train), 44)
        self.assertEqual(len(test), 8)
        self.assertFalse(set(n for n, _ in train) & set(n for n, _ in test))
        self.assertTrue(all(df.activity.nunique() == 1 for _, df in test))


if __name__ == "__main__":
    unittest.main()
