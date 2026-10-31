"""Auditable phase 1–3 preprocessing; no gaps or ground-truth labels are invented.

Run from any directory: python scripts/preprocessing.py [--require-complete]
The strict option exits nonzero until all 52 clips pass their fixed 44/8 split. Unverified session
blocks are exported separately without labels; they are never supervised truth.
"""

from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

if __package__:
    from .splits import load_split_manifest, verify_split_outputs
else:
    from splits import load_split_manifest, verify_split_outputs

ACTIVITIES = ["still", "standing", "walking", "jumping"]
SENSOR_FILES = {"acc": "TotalAcceleration.csv", "gyro": "Gyroscope.csv"}
AXES = ["x", "y", "z"]
SIGNALS = [f"{sensor}_{axis}" for sensor in SENSOR_FILES for axis in AXES]
SCHEMA = ["timestamp_ns", "seconds_elapsed", *SIGNALS, "activity"]
TARGET_FS = 25
STEP_NS = 40_000_000


def load_metadata(rec_dir):
    """Return device, configured rates, and integer Unix-epoch recording origin."""
    meta = pd.read_csv(Path(rec_dir) / "Metadata.csv").iloc[0]
    rates = dict(zip(meta["sensors"].split("|"), str(meta["sampleRateMs"]).split("|")))
    return {
        "device": meta["device name"],
        "platform": f'{meta["platform"]} {meta["platform version"]}',
        "app_version": str(meta["appVersion"]),
        "recorded_at": meta["recording time"],
        "epoch_ns": int(meta["recording epoch time"]) * 1_000_000,
        "configured_hz": {s: 1000 / float(rates[Path(f).stem]) for s, f in SENSOR_FILES.items()},
    }


def load_sensor(rec_dir, sensor, epoch_ns):
    """Load finite axes and integer timestamps; sort and remove exact duplicates.

    Conflicting readings at an identical timestamp are rejected. Relative seconds
    always use the recording origin, including after cropping and CSV export.
    """
    df = pd.read_csv(Path(rec_dir) / SENSOR_FILES[sensor])
    if not pd.api.types.is_integer_dtype(df["time"]):
        raise ValueError("Sensor timestamps must be integer Unix nanoseconds")
    if not np.isfinite(df[["time", *AXES]].to_numpy()).all():
        raise ValueError("Non-finite sensor values")
    duplicates = df.duplicated("time").sum()
    repeated = df[df.duplicated("time", keep=False)]
    if not repeated.empty and (repeated.groupby("time")[AXES].nunique() > 1).any().any():
        raise ValueError("Conflicting duplicate timestamps")
    df = df.sort_values("time").drop_duplicates("time").reset_index(drop=True)
    if len(df) < 20:
        raise ValueError("Insufficient samples for filtering")
    out = df[["time", *AXES]].rename(columns={"time": "timestamp_ns"}).copy()
    out["t"] = (out.timestamp_ns.to_numpy() - epoch_ns) / 1e9
    out.attrs["duplicates_removed"] = int(duplicates)
    return out


def recording_fingerprint(rec_dir, label_file=None):
    """Bind crop/review decisions to the precise raw streams, metadata and labels."""
    paths = [Path(rec_dir) / f for f in ["Metadata.csv", *SENSOR_FILES.values()]]
    if label_file is not None:
        paths.append(Path(label_file))
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode() + b"\0")
        digest.update(path.read_bytes() if path.exists() else b"MISSING")
        digest.update(b"\0")
    return digest.hexdigest()


def load_recording(rec_dir):
    """Load both sensors without modifying the raw export."""
    rec_dir = Path(rec_dir)
    meta = load_metadata(rec_dir)
    return {"name": rec_dir.name, "path": rec_dir, "meta": meta,
            **{s: load_sensor(rec_dir, s, meta["epoch_ns"]) for s in SENSOR_FILES}}


def sampling_stats(df):
    """Report median rate, full timing jitter, and gaps (>3 median intervals).

    'Dropped' counts unusually long intervals, not the number of missing samples.
    Normal-interval jitter is reported separately; outages remain visible.
    """
    dt = np.diff(df.timestamp_ns.to_numpy()) / 1e9
    med = np.median(dt)
    normal = dt <= 3 * med
    return {
        "n_samples": len(df), "duration_s": float(dt.sum()),
        "measured_hz": float(1 / med), "median_dt_ms": float(med * 1e3),
        "jitter_ms": float(dt.std() * 1e3),
        "normal_interval_jitter_ms": float(dt[normal].std() * 1e3),
        "max_dt_ms": float(dt.max() * 1e3),
        "dropped": int((dt > 1.5 * med).sum()), "large_gaps": int((~normal).sum()),
        "duplicates_removed": df.attrs.get("duplicates_removed", 0),
    }


def gap_intervals(df):
    """List missing-data spans in seconds since recording start."""
    dt = np.diff(df.timestamp_ns.to_numpy()) / 1e9
    return [{"start_s": float(df.t.iloc[i]), "end_s": float(df.t.iloc[i + 1]),
             "gap_s": float(dt[i])} for i in np.flatnonzero(dt > 3 * np.median(dt))]


def continuous_blocks(rec, min_duration_s=7.0):
    """Split at the UNION of both sensors' gaps before any filtering.

    A >=7 s native block leaves >=5 s after excluding one second at each
    boundary. Each returned block is an independent sequence with its original
    Unix timestamps and recording origin. No gap is shortened or concatenated.
    """
    start = max(float(rec[s].t.iloc[0]) for s in SENSOR_FILES)
    end = min(float(rec[s].t.iloc[-1]) for s in SENSOR_FILES)
    spans = [(start, end)]
    for left, right in sorted((g["start_s"], g["end_s"])
                              for s in SENSOR_FILES for g in gap_intervals(rec[s])):
        remaining = []
        for lo, hi in spans:
            if right <= lo or left >= hi:
                remaining.append((lo, hi))
            else:
                if left > lo:
                    remaining.append((lo, left))
                if right < hi:
                    remaining.append((right, hi))
        spans = remaining
    blocks = []
    for lo, hi in spans:
        if hi - lo < min_duration_s:
            continue
        block = {"name": f"{rec['name']}_part{len(blocks)+1:02d}", "meta": rec["meta"]}
        for sensor in SENSOR_FILES:
            block[sensor] = rec[sensor][rec[sensor].t.between(lo, hi)].reset_index(drop=True)
        if min(b.t.iloc[-1] - b.t.iloc[0] for b in (block[s] for s in SENSOR_FILES)) >= min_duration_s:
            blocks.append(block)
    return blocks


def unlabelled_session_blocks(rec):
    """Return finite, uniformly sampled, independently filtered unlabelled blocks.

    The output intentionally has NO activity column. Original approximate label
    tables are diagnostic evidence only. Exclude one second at each boundary to
    reduce startup/shutdown and filter-edge effects, without capping long sessions.
    """
    output = []
    for block in continuous_blocks(rec):
        df = harmonise_recording(block)
        df = df[(df.t >= df.t.iloc[0] + 1) & (df.t <= df.t.iloc[-1] - 1)].copy()
        if len(df) < 2 or (int(df.timestamp_ns.iloc[-1]) - int(df.timestamp_ns.iloc[0])) / 1e9 < 5:
            continue
        if not np.isfinite(df[SIGNALS]).all().all() or not np.all(np.diff(df.timestamp_ns) == STEP_NS):
            raise ValueError("Invalid continuous session block")
        output.append({"name": block["name"], "parent": rec["name"], "df": df.reset_index(drop=True)})
    return output


def harmonise_recording(rec, target_fs=TARGET_FS, cutoff_hz=10.0):
    """Filter native streams, then interpolate ONCE onto one integer time grid.

    Entire recordings containing large gaps are rejected before filtering. This
    conservative policy avoids manufacturing activity or transitions across an
    outage. Tiny timing jitter is approximated by the median native rate.
    """
    if target_fs <= 0 or not 0 < cutoff_hz < target_fs / 2:
        raise ValueError("Require 0 < cutoff < target Nyquist")
    step_ns = int(round(1e9 / target_fs))
    if not np.isclose(step_ns * target_fs, 1e9, rtol=0, atol=1e-4):
        raise ValueError("Target rate must have an integer nanosecond period")
    for sensor in SENSOR_FILES:
        stats = sampling_stats(rec[sensor])
        if stats["large_gaps"]:
            raise ValueError(f"{rec['name']} {sensor}: large gaps; re-record or explicitly split/review")
        if stats["measured_hz"] < target_fs:
            raise ValueError("Upsampling is not allowed")
    start = max(int(rec[s].timestamp_ns.iloc[0]) for s in SENSOR_FILES)
    end = min(int(rec[s].timestamp_ns.iloc[-1]) for s in SENSOR_FILES)
    if end <= start:
        raise ValueError("Sensor streams do not overlap")
    grid = np.arange(start, end + 1, step_ns, dtype=np.int64)
    epoch = rec["meta"]["epoch_ns"]
    out = pd.DataFrame({"timestamp_ns": grid, "t": (grid - epoch) / 1e9})
    # Subtract an integer origin BEFORE conversion to float, preserving ns precision.
    grid_s = (grid - start) / 1e9
    for sensor in SENSOR_FILES:
        df = rec[sensor]
        sos = butter(4, cutoff_hz, fs=sampling_stats(df)["measured_hz"], output="sos")
        filtered = sosfiltfilt(sos, df[AXES].to_numpy(), axis=0)
        source_s = (df.timestamp_ns.to_numpy() - start) / 1e9
        for i, axis in enumerate(AXES):
            out[f"{sensor}_{axis}"] = np.interp(grid_s, source_s, filtered[:, i])
    return out


def trim_clip(df, start_s=None, end_s=None, trim_s=1.0):
    """Keep a reviewed stable interval, or a central <=10 s interval after trim.

    Duration is last timestamp minus first (conservative relative to N/fs).
    Explicit crop bounds must also leave at least one second at both raw ends.
    """
    lo, hi = float(df.t.iloc[0] + trim_s), float(df.t.iloc[-1] - trim_s)
    if start_s is not None:
        lo = max(lo, float(start_s))
    if end_s is not None:
        hi = min(hi, float(end_s))
    out = df[(df.t >= lo) & (df.t <= hi)].copy()
    max_samples = 10 * TARGET_FS + 1
    if len(out) > max_samples:
        offset = (len(out) - max_samples) // 2
        out = out.iloc[offset:offset + max_samples].copy()
    if len(out) < 2 or not 5 <= (int(out.timestamp_ns.iloc[-1]) - int(out.timestamp_ns.iloc[0])) / 1e9 <= 10:
        raise ValueError("Cleaned clip must span 5–10 seconds")
    return out.reset_index(drop=True)


def label_session(df, label_file):
    """Apply validated, contiguous [start,end) segments; no silent row dropping.

    A leading sync marker and trailing shutdown may lie outside the label range.
    Internal gaps, overlap, invalid activities and out-of-recording bounds fail.
    Independent evidence approval is enforced by build_dataset().
    """
    seg = pd.read_csv(label_file)
    if list(seg.columns) != ["start_s", "end_s", "activity"] or seg.empty:
        raise ValueError("Labels require start_s,end_s,activity and at least one row")
    if not np.isfinite(seg[["start_s", "end_s"]].to_numpy()).all():
        raise ValueError("Non-finite label boundaries")
    if not seg.activity.isin(ACTIVITIES).all() or (seg.end_s <= seg.start_s).any():
        raise ValueError("Unknown activity or invalid label duration")
    if not np.allclose(seg.start_s.to_numpy()[1:], seg.end_s.to_numpy()[:-1], rtol=0, atol=1e-9):
        raise ValueError("Label intervals overlap, are unsorted, or have internal gaps")
    if seg.start_s.iloc[0] < 0 or seg.end_s.iloc[-1] > df.t.iloc[-1] + 1 / TARGET_FS:
        raise ValueError("Labels extend outside the recording")
    if seg.start_s.iloc[0] < df.t.iloc[0] - 1 / TARGET_FS:
        raise ValueError("Labels begin before common sensor coverage")
    out = df[(df.t >= seg.start_s.iloc[0]) & (df.t < seg.end_s.iloc[-1])].copy()
    out["activity"] = ""
    for row in seg.itertuples():
        mask = (out.t >= row.start_s) & (out.t < row.end_s)
        if not mask.any():
            raise ValueError("Label interval has no sampled observations")
        out.loc[mask, "activity"] = row.activity
    if out.empty or (out.activity == "").any():
        raise ValueError("Incomplete label coverage")
    return out.reset_index(drop=True)


def processed_frame(df):
    """Create and validate the export schema; both times keep the recording origin."""
    out = df.rename(columns={"t": "seconds_elapsed"})[SCHEMA].copy()
    if out.empty or out.isna().any().any() or not np.isfinite(out[SIGNALS + ["seconds_elapsed"]]).all().all():
        raise ValueError("Processed data is empty or non-finite")
    if not out.activity.isin(ACTIVITIES).all():
        raise ValueError("Invalid processed labels")
    if not pd.api.types.is_integer_dtype(out.timestamp_ns) or not np.all(np.diff(out.timestamp_ns) == STEP_NS):
        raise ValueError("Processed timestamps must be integer Unix ns, exactly 25 Hz")
    if not np.allclose(np.diff(out.seconds_elapsed), 1 / TARGET_FS, atol=1e-9, rtol=0):
        raise ValueError("Elapsed time disagrees with the sampling grid")
    # Integer subtraction avoids floating-point loss at Unix-epoch scale.
    relative_ns = out.timestamp_ns.to_numpy() - int(out.timestamp_ns.iloc[0])
    if not np.allclose(relative_ns / 1e9, out.seconds_elapsed - out.seconds_elapsed.iloc[0], atol=1e-9, rtol=0):
        raise ValueError("Timestamp columns disagree")
    return out


def save_processed(df, out_path):
    """Validate before saving; retain Unix timestamps and recording-relative seconds."""
    out = processed_frame(df)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)


def require_training_ready(status):
    """Require the fixed split and valid collection; never fit on test clips."""
    if not status["training_ready"]:
        raise ValueError("Training blocked: " + "; ".join(status["blockers"]))


def load_labelled_sequences(root, split):
    """Load only one fixed split as separate (name, DataFrame) sequences.

    Future feature extraction must fit on split='train' only. A test sequence
    is never concatenated with another clip, and no state transition is counted
    across files. Recheck raw provenance and processed membership on every load.
    """
    if split not in ("train", "test"):
        raise ValueError("Choose train or test explicitly")
    root = Path(root)
    manifest = load_split_manifest(root)
    verify_split_outputs(root / "data/processed", manifest)
    folder = root / "data/processed" / ("train/clips" if split == "train" else "test")
    sequences = []
    for name in sorted(manifest.index[manifest.split == split]):
        df = pd.read_csv(folder / f"{name}.csv")
        processed_frame(df.rename(columns={"seconds_elapsed": "t"}))
        if not (df.activity == manifest.loc[name, "activity"]).all():
            raise ValueError(f"{name}: processed activity differs from the original label")
        sequences.append((name, df))
    return sequences


def build_dataset(root, output_dir=None):
    """Regenerate validated artifacts and reports; leave every raw recording intact.

    Rejected or stale derived files are removed from the pipeline-owned output
    folders, so subsequent code cannot accidentally consume an obsolete session.
    """
    root = Path(root)
    raw = root / "data/raw"
    output_dir = Path(output_dir) if output_dir else root / "data/processed"
    quality_dir = root / "data/quality"
    quality_dir.mkdir(parents=True, exist_ok=True)
    (quality_dir / "status.json").write_text(json.dumps({
        "training_ready": False, "blockers": ["Preprocessing has not completed successfully"]
    }, indent=2) + "\n")
    split_manifest = load_split_manifest(root)
    crop_path = root / "data/clip_intervals.csv"
    crops = pd.read_csv(crop_path).set_index("file")
    if not crops.index.is_unique:
        raise ValueError("Duplicate crop entries")
    prior_path = root / "data/metadata.csv"
    prior = pd.read_csv(prior_path).set_index("file") if prior_path.exists() else pd.DataFrame()
    records, quality, gaps, catalog, accepted, rejected, crop_log = [], [], [], [], [], [], []
    unlabelled = []
    paths = sorted(p.parent for p in raw.rglob("Metadata.csv"))
    for path in paths:
        relative = path.relative_to(raw)
        rec_type = {"pilot": "pilot", "clips": "clip", "sessions": "session", "test": "test"}.get(relative.parts[0])
        if rec_type is None:
            continue
        name = path.name
        activity = relative.parts[1] if rec_type == "clip" else "mixed" if rec_type in ("session", "test") else "pilot"
        split = split_manifest.loc[name, "split"] if rec_type == "clip" else "pilot" if rec_type == "pilot" else "test" if rec_type == "test" else "train"
        # Fail loudly on malformed files; do not leave them silently usable.
        rec = load_recording(path)
        rec.update(activity=activity, rec_type=rec_type, split=split)
        records.append(rec)
        stats = {s: sampling_stats(rec[s]) for s in SENSOR_FILES}
        for sensor in SENSOR_FILES:
            quality.append({"recording": name, "type": rec_type, "activity": activity,
                            "sensor": sensor, "configured_hz": rec["meta"]["configured_hz"][sensor], **stats[sensor]})
            gaps.extend({"recording": name, "sensor": sensor, **gap} for gap in gap_intervals(rec[sensor]))
        catalog.append({"file": name, "split": split,
                        "type": rec_type, "activity": activity, "configured_hz": rec["meta"]["configured_hz"]["acc"],
                        "measured_acc_hz": stats["acc"]["measured_hz"], "measured_gyro_hz": stats["gyro"]["measured_hz"],
                        "duration_s": stats["acc"]["duration_s"], "date": str(rec["meta"]["recorded_at"])[:10],
                        "location_type": prior.loc[name, "location_type"] if name in prior.index else "not_recorded"})
        if rec_type not in ("clip", "session"):
            continue
        if rec_type == "session":
            unlabelled.extend(unlabelled_session_blocks(rec))
        reasons = [f"{s}: {stats[s]['large_gaps']} large gaps (max {stats[s]['max_dt_ms']/1000:.3f}s)"
                   for s in SENSOR_FILES if stats[s]["large_gaps"]]
        label_file = root / "data/labels" / f"{name}.csv"
        fingerprint = recording_fingerprint(path, label_file if rec_type == "session" else None)
        if rec_type == "session":
            if not label_file.exists():
                reasons.append("missing segment labels")
            # Only an approximate timer schedule exists, so sessions never become labelled data.
            reasons.append("approximate session labels; exported only as unlabelled blocks")
        elif name not in crops.index or crops.loc[name, "source_sha256"] != fingerprint:
            reasons.append("crop interval missing or stale; inspect new recording")
        if not reasons:
            try:
                df = harmonise_recording(rec)
                if rec_type == "clip":
                    crop = crops.loc[name]
                    df = trim_clip(df, crop.start_s, crop.end_s)
                    df["activity"] = activity
                    crop_log.append({"file": name, "activity": activity, "split": split,
                                     "start_s": df.t.iloc[0], "end_s": df.t.iloc[-1],
                                     "duration_s": (int(df.timestamp_ns.iloc[-1]) - int(df.timestamp_ns.iloc[0])) / 1e9,
                                     "n_samples": len(df), "source_sha256": fingerprint})
                else:
                    df = label_session(df, label_file)
                    if set(df.activity) != set(ACTIVITIES):
                        raise ValueError("Session must contain all four activities")
                processed_frame(df)
                accepted.append({"name": name, "activity": activity, "rec_type": rec_type, "split": split, "df": df})
            except ValueError as error:
                reasons.append(str(error))
        if reasons:
            rejected.append({"recording": name, "type": rec_type, "reason": "; ".join(reasons), "source_sha256": fingerprint})
    # This is a single-phone collection. Reject any optional training block whose
    # acquisition time overlaps a held-out clip, including renamed session copies.
    test_records = [rec for rec in records if rec["rec_type"] == "clip" and rec["split"] == "test"]
    for block in unlabelled:
        lo, hi = block["df"].timestamp_ns.iloc[[0, -1]]
        for rec in test_records:
            test_lo = min(rec[s].timestamp_ns.iloc[0] for s in SENSOR_FILES)
            test_hi = max(rec[s].timestamp_ns.iloc[-1] for s in SENSOR_FILES)
            if lo <= test_hi and hi >= test_lo:
                raise ValueError("Unlabelled training block overlaps a held-out recording")
    for kind, split, folder in [("clip", "train", "train/clips"), ("clip", "test", "test"),
                                 ("session", "train", "train/sessions")]:
        target = output_dir / folder
        target.mkdir(parents=True, exist_ok=True)
        selected = [rec for rec in accepted if rec["rec_type"] == kind and rec["split"] == split]
        expected = {rec["name"] + ".csv" for rec in selected}
        for stale in target.glob("*.csv"):
            if stale.name not in expected:
                stale.unlink()
        for rec in selected:
            save_processed(rec["df"], target / f"{rec['name']}.csv")
    crop_df = pd.DataFrame(crop_log, columns=["file", "activity", "split", "start_s", "end_s", "duration_s", "n_samples", "source_sha256"])
    totals = {a: round(float(crop_df.loc[crop_df.activity == a, "duration_s"].sum()), 2) for a in ACTIVITIES}
    counts = {a: int((crop_df.activity == a).sum()) for a in ACTIVITIES}
    n_sessions = sum(r["rec_type"] == "session" for r in accepted)
    blockers = [f"{a}: {counts[a]}/13 valid clips" for a in ACTIVITIES if counts[a] != 13]
    blockers += [f"{a}: {totals[a]:.2f}s < 90s" for a in ACTIVITIES if totals[a] < 90]
    for split, count in [("train", 11), ("test", 2)]:
        for activity in ACTIVITIES:
            actual = int(((crop_df.activity == activity) & (crop_df.split == split)).sum())
            if actual != count:
                blockers.append(f"{activity}: {actual}/{count} valid {split} clips")
    try:
        verify_split_outputs(output_dir, split_manifest)
    except ValueError as error:
        blockers.append(str(error))
    # Updated no-re-recording strategy: the rubric requires unseen TEST files,
    # not two labelled TRAINING sessions. Never certify approximate boundaries.
    target = output_dir / "unlabelled/sessions"
    target.mkdir(parents=True, exist_ok=True)
    expected = {block["name"] + ".csv" for block in unlabelled}
    for stale in target.glob("*.csv"):
        if stale.name not in expected:
            stale.unlink()
    block_log = []
    for block in unlabelled:
        df = block["df"]
        df.rename(columns={"t": "seconds_elapsed"})[["timestamp_ns", "seconds_elapsed", *SIGNALS]].to_csv(
            target / f"{block['name']}.csv", index=False)
        parent = next(rec for rec in records if rec["name"] == block["parent"])
        block_log.append({"sequence": block["name"], "recording": block["parent"],
                          "start_s": df.t.iloc[0], "end_s": df.t.iloc[-1],
                          "duration_s": (int(df.timestamp_ns.iloc[-1]) - int(df.timestamp_ns.iloc[0])) / 1e9,
                          "n_samples": len(df), "source_sha256": recording_fingerprint(parent["path"]),
                          "label_status": "unverified; no labels exported"})
    pd.DataFrame(block_log, columns=["sequence", "recording", "start_s", "end_s", "duration_s", "n_samples", "source_sha256", "label_status"]).to_csv(
        quality_dir / "unlabelled_sequences.csv", index=False)
    status = {"training_ready": not blockers, "accepted_clips": len(crop_df), "accepted_sessions": n_sessions,
              "training_clips": int((crop_df.split == "train").sum()),
              "test_clips": int((crop_df.split == "test").sum()),
              "test_ready": not blockers,
              "clip_seconds_per_activity": totals, "blockers": blockers,
              "unlabelled_session_sequences": len(unlabelled),
              "verified_session_labels_available": n_sessions > 0,
              "training_strategy": "44 labelled training clips; 8 held-out clips excluded from fitting/tuning; explicit transition prior; optional independent unlabelled training blocks for EM",
              "evaluation_scope": "Previously inspected same-collection clips; no claim of unseen-day/environment or cross-activity transition generalisation"}
    pd.DataFrame(catalog).to_csv(prior_path, index=False, float_format="%.6f")
    pd.DataFrame(quality).to_csv(quality_dir / "sampling.csv", index=False)
    pd.DataFrame(gaps, columns=["recording", "sensor", "start_s", "end_s", "gap_s"]).to_csv(quality_dir / "gaps.csv", index=False)
    pd.DataFrame(rejected, columns=["recording", "type", "reason", "source_sha256"]).to_csv(quality_dir / "rejected.csv", index=False)
    crop_df.to_csv(quality_dir / "clip_intervals.csv", index=False)
    (quality_dir / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    return {"records": records, "accepted": accepted, "quality": pd.DataFrame(quality),
            "rejected": pd.DataFrame(rejected), "crops": crop_df, "status": status,
            "unlabelled": unlabelled, "block_manifest": pd.DataFrame(block_log),
            "split_manifest": split_manifest}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    result = build_dataset(Path(__file__).resolve().parents[1])
    print(json.dumps(result["status"], indent=2))
    if args.require_complete:
        require_training_ready(result["status"])


if __name__ == "__main__":
    main()
