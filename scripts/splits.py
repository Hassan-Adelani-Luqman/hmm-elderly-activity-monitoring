"""Fixed recording-level membership; no window-level or automatic re-splitting."""

from pathlib import Path
import hashlib

import pandas as pd

ACTIVITIES = {"still", "standing", "walking", "jumping"}


def source_fingerprint(rec_dir):
    """Use the same raw-plus-metadata fingerprint as the crop provenance."""
    digest = hashlib.sha256()
    for name in ["Metadata.csv", "TotalAcceleration.csv", "Gyroscope.csv"]:
        path = Path(rec_dir) / name
        digest.update(name.encode() + b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def load_split_manifest(root):
    """Validate frozen membership and raw sources; fail closed if it is missing.

    Every original single-activity clip must appear exactly once. Identical sensor
    exports under different names cannot cross the train/test boundary, nor can
    copies with the same recording origin. Sessions/pilots are not test clips.
    """
    root = Path(root)
    path = root / "data/split_manifest.csv"
    if not path.exists():
        raise ValueError("Missing fixed split manifest; refusing to treat test clips as training")
    manifest = pd.read_csv(path, keep_default_na=False)
    required = {"file", "activity", "split", "source_sha256"}
    if not required <= set(manifest.columns) or manifest.empty:
        raise ValueError("Invalid split manifest schema")
    if not manifest.file.is_unique:
        raise ValueError("A recording appears more than once in the split manifest")
    if not manifest.split.isin(["train", "test"]).all() or not manifest.activity.isin(ACTIVITIES).all():
        raise ValueError("Invalid split or activity")
    paths = {p.parent.name: p.parent for p in (root / "data/raw/clips").glob("*/*/Metadata.csv")}
    if set(manifest.file) != set(paths):
        raise ValueError("Split manifest must cover every raw clip exactly once")
    seen_signals, seen_origins = {}, {}
    for row in manifest.itertuples():
        rec_dir = paths[row.file]
        if rec_dir.parent.name != row.activity:
            raise ValueError(f"{row.file}: manifest activity differs from original recording")
        if source_fingerprint(rec_dir) != row.source_sha256:
            raise ValueError(f"{row.file}: raw recording changed; split provenance is stale")
        digest = hashlib.sha256()
        for name in ["TotalAcceleration.csv", "Gyroscope.csv"]:
            digest.update(name.encode() + b"\0" + (rec_dir / name).read_bytes())
        identity = digest.hexdigest()
        epoch = str(pd.read_csv(rec_dir / "Metadata.csv").iloc[0]["recording epoch time"])
        for seen, key in [(seen_signals, identity), (seen_origins, epoch)]:
            if key in seen and seen[key] != row.split:
                raise ValueError("Train/test leakage: duplicate original recording across splits")
            seen[key] = row.split
    return manifest.set_index("file")


def verify_split_outputs(root, manifest):
    """Check each output belongs to its fixed split, with no missing/stale files."""
    root = Path(root)
    for split, directory in [("train", root / "train/clips"), ("test", root / "test")]:
        expected = set(manifest.index[manifest.split == split])
        actual = {p.stem for p in directory.glob("*.csv")}
        if actual != expected:
            raise ValueError(f"{split}: processed files do not match the fixed split")
