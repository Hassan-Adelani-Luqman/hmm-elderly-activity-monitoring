"""Phase 5: independent windows, justified features, training-only fit/selection.

Run: python scripts/features.py
This generates feature sequences, never fits an HMM or evaluates test performance.
"""

from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

if __package__:
    from .preprocessing import (ACTIVITIES, SIGNALS, TARGET_FS, build_dataset,
                                load_labelled_sequences, require_training_ready)
else:
    from preprocessing import (ACTIVITIES, SIGNALS, TARGET_FS, build_dataset,
                               load_labelled_sequences, require_training_ready)

WINDOW_S = 2.0
HOP_S = 1.0
CORRELATION_LIMIT = 0.95
EPS = 1e-12
FEATURE_SPECS = [
    ("mean_acc_x", "time", "m/s²", "Gravity orientation under the fixed phone-grip protocol."),
    ("mean_acc_y", "time", "m/s²", "Upright-grip gravity cue distinguishing standing from phone-flat still."),
    ("mean_acc_z", "time", "m/s²", "Flat-phone gravity cue; potentially redundant with the y-axis mean."),
    ("std_acc_mag", "time", "m/s²", "Dynamic motion intensity around the mean acceleration magnitude."),
    ("var_gyro_mag", "time", "(rad/s)²", "Variation in angular motion during walking and jumping."),
    ("rms_centered_acc_mag", "time", "m/s²", "Dynamic RMS; mathematically equal to population std and audited for removal."),
    ("sma_acc", "time", "m/s²", "Mean sum of absolute acceleration axes; contains gravity/orientation as well as motion."),
    ("sma_gyro", "time", "rad/s", "Mean sum of absolute gyro axes measures overall rotational movement."),
    ("ptp_acc_mag", "time", "m/s²", "Range of acceleration magnitude captures large impact excursions."),
    ("corr_acc_yz", "time", "unitless", "Axis coordination; defined as zero when either axis is constant."),
    ("dominant_freq_acc", "frequency", "Hz", "Strongest positive-frequency component of detrended magnitude; periodicity cue."),
    ("spectral_energy_acc", "frequency", "(m/s²)²", "Hann-corrected FFT power measures dynamic energy; may duplicate time-domain intensity."),
    ("band_energy_ratio_acc", "frequency", "unitless", "Power at 0.5–3 Hz divided by power above 3 through 10 Hz; low/high spectral balance."),
    ("spectral_entropy_acc", "frequency", "unitless", "Normalised positive-frequency entropy distinguishes concentrated periodic power from spread power."),
]
FEATURE_NAMES = [s[0] for s in FEATURE_SPECS]
WINDOW_COLUMNS = ["sequence_id", "window_index", "start_sample", "stop_sample",
                  "start_timestamp_ns", "stop_timestamp_ns", "start_s", "stop_s"]


def sample_count(seconds, fs):
    """Convert seconds to a positive integer count without silent truncation."""
    count = seconds * fs
    if not np.isfinite(count) or count <= 0 or not np.isclose(count, round(count), rtol=0, atol=1e-9):
        raise ValueError("Window/hop must be a positive integer number of samples")
    return int(round(count))


def window_signal(df, fs=TARGET_FS, window_s=WINDOW_S, hop_s=HOP_S):
    """Segment ONE recording into complete windows; incomplete tails are omitted.

    A window of 50 samples at 25 Hz occupies [start,start+2s); its last
    measurement is at start+1.96s. Timestamp end and sample stop are exclusive.
    Labels use majority vote; ties follow ACTIVITIES order and purity is recorded.
    Unlabelled sequences stay unlabelled. No filtering, fitting or concatenation
    takes place here. Nonuniform timestamps, unknown labels and NaNs are errors.
    """
    n, hop = sample_count(window_s, fs), sample_count(hop_s, fs)
    if df.empty or not np.isfinite(df[SIGNALS + ["seconds_elapsed"]]).all().all():
        raise ValueError("Empty or non-finite sequence")
    if not pd.api.types.is_integer_dtype(df.timestamp_ns):
        raise ValueError("Expected integer nanosecond timestamps")
    step_ns = int(round(1e9 / fs))
    if not np.isclose(step_ns * fs, 1e9, atol=1e-4, rtol=0):
        raise ValueError("Sampling period must be representable in integer ns")
    if not np.all(np.diff(df.timestamp_ns.to_numpy()) == step_ns):
        raise ValueError("Sequence has gaps or nonuniform timestamps; never window across gaps")
    relative = (df.timestamp_ns.to_numpy() - int(df.timestamp_ns.iloc[0])) / 1e9
    if not np.allclose(relative, df.seconds_elapsed - df.seconds_elapsed.iloc[0], rtol=0, atol=1e-9):
        raise ValueError("Timestamp columns disagree")
    labelled = "activity" in df.columns
    if labelled and not df.activity.isin(ACTIVITIES).all():
        raise ValueError("Unknown or missing activity label")
    windows = []
    for index, start in enumerate(range(0, len(df) - n + 1, hop)):
        part = df.iloc[start:start + n]
        row = {"window_index": index, "start_sample": start, "stop_sample": start + n,
               "start_timestamp_ns": int(part.timestamp_ns.iloc[0]),
               "stop_timestamp_ns": int(part.timestamp_ns.iloc[-1]) + step_ns,
               "start_s": float(part.seconds_elapsed.iloc[0]),
               "stop_s": float(part.seconds_elapsed.iloc[0] + n / fs),
               "samples": part[SIGNALS].to_numpy(dtype=float)}
        if labelled:
            counts = np.array([(part.activity == activity).sum() for activity in ACTIVITIES])
            row["activity"] = ACTIVITIES[int(np.argmax(counts))]
            row["label_purity"] = float(counts.max() / n)
        windows.append(row)
    return windows


def fft_features(magnitude, fs=TARGET_FS):
    """Compute four FFT features with Parseval-consistent one-sided power.

    Subtract the mean, apply a periodic Hann window, and use rfft. Interior
    positive bins are doubled; DC and an even-N Nyquist bin are not. Dividing
    by N*sum(Hann²) gives a window-energy-corrected mean-square spectrum.
    Energy sums all bins; peak and entropy exclude DC. Band endpoints are
    disjoint: [0.5,3] and (3,10]. Signals with positive-frequency power <=1e-12
    have zero peak/ratio/entropy. The high-band denominator has a 1e-12 floor.
    """
    magnitude = np.asarray(magnitude, dtype=float)
    if magnitude.ndim != 1 or len(magnitude) < 4 or not np.isfinite(magnitude).all() or not np.isfinite(fs) or fs <= 0:
        raise ValueError("FFT requires a finite 1-D signal of length >=4 and positive fs")
    n = len(magnitude)
    hann = np.hanning(n + 1)[:-1]
    spectrum = np.fft.rfft((magnitude - magnitude.mean()) * hann)
    power = np.abs(spectrum) ** 2 / (n * np.sum(hann ** 2))
    power[1:-1 if n % 2 == 0 else None] *= 2
    frequencies = np.fft.rfftfreq(n, d=1 / fs)
    positive = power[1:]
    total = float(positive.sum())
    energy = float(power.sum())
    if total <= EPS:
        return dict(dominant_freq_acc=0.0, spectral_energy_acc=energy,
                    band_energy_ratio_acc=0.0, spectral_entropy_acc=0.0)
    peak = float(frequencies[1 + np.argmax(positive)])
    low = float(power[(frequencies >= .5) & (frequencies <= 3)].sum())
    high = float(power[(frequencies > 3) & (frequencies <= 10)].sum())
    probabilities = positive / total
    nonzero = probabilities[probabilities > 0]
    entropy = float(-np.sum(nonzero * np.log(nonzero)) / np.log(len(positive)))
    return dict(dominant_freq_acc=peak, spectral_energy_acc=energy,
                band_energy_ratio_acc=low / max(high, EPS),
                spectral_entropy_acc=float(np.clip(entropy, 0, 1)))


def extract_features(samples, fs=TARGET_FS):
    """Return 14 candidates from an (N,6) array in SIGNALS column order.

    Variance/std use ddof=0. SMA is mean(sum(abs(axes))), a per-time-unit average
    rather than a sample-count-dependent sum. Centred RMS intentionally duplicates
    population std so the training-only redundancy audit can demonstrate removal.
    """
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 2 or samples.shape[1] != 6 or len(samples) < 4 or not np.isfinite(samples).all():
        raise ValueError("Expected finite samples of shape (N>=4,6)")
    acc, gyro = samples[:, :3], samples[:, 3:]
    a, w = np.linalg.norm(acc, axis=1), np.linalg.norm(gyro, axis=1)
    y, z = acc[:, 1] - acc[:, 1].mean(), acc[:, 2] - acc[:, 2].mean()
    denominator = np.linalg.norm(y) * np.linalg.norm(z)
    correlation = 0.0 if denominator <= EPS else float(np.clip(np.dot(y, z) / denominator, -1, 1))
    row = {f"mean_acc_{axis}": float(acc[:, i].mean()) for i, axis in enumerate("xyz")}
    row.update(std_acc_mag=float(a.std()), var_gyro_mag=float(w.var()),
               rms_centered_acc_mag=float(np.sqrt(np.mean((a - a.mean()) ** 2))),
               sma_acc=float(np.abs(acc).sum(axis=1).mean()),
               sma_gyro=float(np.abs(gyro).sum(axis=1).mean()),
               ptp_acc_mag=float(np.ptp(a)), corr_acc_yz=correlation)
    row.update(fft_features(a, fs))
    return row


def feature_sequence(name, df):
    """Extract features from one recording, preserving its window provenance."""
    rows = []
    for window in window_signal(df):
        samples = window.pop("samples")
        rows.append({"sequence_id": name, **window, **extract_features(samples)})
    if not rows:
        raise ValueError(f"{name}: no complete windows")
    return pd.DataFrame(rows)


def select_features(training, correlation_limit=CORRELATION_LIMIT):
    """Greedy training-only redundancy removal in the declared feature order.

    Earlier interpretable features take priority. Constant features are removed
    first. A later feature is dropped only if |r| > limit against a retained
    feature. Every decision and its paired feature/correlation are returned.
    """
    x = training[FEATURE_NAMES]
    if len(x) < 2 or not np.isfinite(x).all().all() or not 0 < correlation_limit < 1:
        raise ValueError("Invalid training features or correlation threshold")
    std = x.std(ddof=0)
    variable = [name for name in FEATURE_NAMES if std[name] > EPS]
    correlations = x[variable].corr()
    kept, audit = [], []
    for name in FEATURE_NAMES:
        if name not in variable:
            audit.append({"feature": name, "retained": False, "reason": "constant on training windows",
                          "paired_feature": "", "correlation": None})
            continue
        match = next((other for other in kept if abs(correlations.loc[name, other]) > correlation_limit), None)
        if match is None:
            kept.append(name)
            audit.append({"feature": name, "retained": True, "reason": "retained",
                          "paired_feature": "", "correlation": None})
        else:
            audit.append({"feature": name, "retained": False, "reason": "training correlation exceeds limit",
                          "paired_feature": match, "correlation": float(correlations.loc[name, match])})
    if not kept:
        raise ValueError("No variable features remain")
    return kept, pd.DataFrame(audit), correlations


def zscore_fit(training):
    """Fit population mean/std on finite training rows only; reject constants."""
    values = np.asarray(training, dtype=float)
    if values.ndim != 2 or min(values.shape) < 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Expected finite 2-D training features with at least two rows")
    mean, std = values.mean(axis=0), values.std(axis=0, ddof=0)
    if (std <= EPS).any():
        raise ValueError("Remove constant training features before normalising")
    return {"mean": mean.tolist(), "std": std.tolist()}


def zscore_transform(values, statistics):
    """Apply fitted statistics without updating, clipping, or fitting on new data."""
    x = np.asarray(values, dtype=float)
    mean, std = np.asarray(statistics["mean"]), np.asarray(statistics["std"])
    if (x.ndim != 2 or mean.ndim != 1 or std.shape != mean.shape or x.shape[1] != len(mean)
            or not np.isfinite(x).all() or not np.isfinite(mean).all()
            or not np.isfinite(std).all() or (std <= EPS).any()):
        raise ValueError("Invalid features or fitted statistics")
    out = (x - mean) / std
    if not np.isfinite(out).all():
        raise ValueError("Non-finite normalised feature values")
    return out


def build_features(root, dataset=None):
    """Fit on the 44 labelled training clips; transform held-out/auxiliary data.

    Optional unlabelled sequences do not influence selection or fitted scaling.
    Each output CSV is one sequence and contains only retained, z-scored features
    plus provenance/labels. Candidate values and correlations are exported for
    labelled training only. Raw sensor data and split membership remain unchanged.
    """
    root = Path(root)
    output = root / "data/features"
    output.mkdir(parents=True, exist_ok=True)
    (output / "status.json").write_text(json.dumps({"ready": False, "reason": "Feature build has not finished"}) + "\n")
    dataset = build_dataset(root) if dataset is None else dataset
    require_training_ready(dataset["status"])
    train_sequences = load_labelled_sequences(root, "train")
    train_frames = {name: feature_sequence(name, df) for name, df in train_sequences}
    training = pd.concat(train_frames.values(), ignore_index=True)
    kept, selection, correlations = select_features(training)
    domains = {name: domain for name, domain, _, _ in FEATURE_SPECS}
    if sum(domains[name] == "time" for name in kept) < 3 or sum(domains[name] == "frequency" for name in kept) < 2:
        raise ValueError("Retained features must include >=3 time-domain and >=2 FFT features")
    statistics = zscore_fit(training[kept])
    # All learned parameters are fixed BEFORE test or auxiliary feature extraction.
    test_frames = {name: feature_sequence(name, df) for name, df in load_labelled_sequences(root, "test")}
    auxiliary = {item["name"]: feature_sequence(item["name"], item["df"].rename(columns={"t": "seconds_elapsed"}))
                 for item in dataset["unlabelled"]}
    raw = {"train": train_frames, "test": test_frames, "unlabelled": auxiliary}
    normalized, counts = {}, []
    for split, frames in raw.items():
        folder = output / split
        folder.mkdir(exist_ok=True)
        normalized[split] = {}
        for stale in folder.glob("*.csv"):
            if stale.stem not in frames:
                stale.unlink()
        for name, frame in frames.items():
            meta_columns = [c for c in frame if c not in FEATURE_NAMES]
            transformed = frame[meta_columns].copy()
            transformed[kept] = zscore_transform(frame[kept], statistics)
            transformed.to_csv(folder / f"{name}.csv", index=False)
            normalized[split][name] = transformed
            counts.append({"sequence_id": name, "split": split, "windows": len(frame),
                           "activity": frame.activity.iloc[0] if "activity" in frame else "unlabelled"})
    fit_names = sorted(train_frames)
    transformation = {
        "fs": TARGET_FS, "window_s": WINDOW_S, "hop_s": HOP_S,
        "window_samples": sample_count(WINDOW_S, TARGET_FS), "hop_samples": sample_count(HOP_S, TARGET_FS),
        "feature_order": kept, "candidate_features": FEATURE_NAMES, "state_order": ACTIVITIES,
        "correlation_limit": CORRELATION_LIMIT, "variance_ddof": 0, "spectral_floor": EPS,
        "fit_split": "train", "fit_sequences": fit_names, "fit_windows": len(training),
        "statistics": statistics,
        "split_manifest_sha256": hashlib.sha256((root / "data/split_manifest.csv").read_bytes()).hexdigest(),
        "fit_input_sha256": {name: hashlib.sha256((root / "data/processed/train/clips" / f"{name}.csv").read_bytes()).hexdigest() for name in fit_names},
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "selection_rule": "Declared candidate order; drop constants then later features with |training r| > 0.95 against retained features",
        "unlabelled_fit_policy": "Transform with labelled-training statistics; no selection or scaler refit",
        "input_sha256": {split: {name: hashlib.sha256((root / "data/processed" /
                           {"train": "train/clips", "test": "test", "unlabelled": "unlabelled/sessions"}[split] /
                           f"{name}.csv").read_bytes()).hexdigest() for name in frames}
                         for split, frames in raw.items()},
        "output_sha256": {split: {name: hashlib.sha256((output / split / f"{name}.csv").read_bytes()).hexdigest()
                                  for name in frames} for split, frames in normalized.items()},
    }
    (output / "transform.json").write_text(json.dumps(transformation, indent=2, allow_nan=False) + "\n")
    training.to_csv(output / "training_candidates.csv", index=False)
    selection.to_csv(output / "selection.csv", index=False)
    correlations.to_csv(output / "training_correlations.csv")
    counts = pd.DataFrame(counts)
    counts.to_csv(output / "sequence_counts.csv", index=False)
    status = {"ready": True, "retained_features": len(kept), "feature_order": kept,
              "sequences": {s: len(f) for s, f in raw.items()},
              "windows": {s: sum(len(f) for f in frames.values()) for s, frames in raw.items()},
              "fit_policy": "Feature selection and z-score fitted on the 44 labelled training clips only"}
    (output / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    return {"raw": raw, "normalized": normalized, "training": training,
            "selection": selection, "correlations": correlations, "counts": counts,
            "transform": transformation, "status": status}


def load_feature_sequences(root, split):
    """Load model-ready (T,D) matrices without metadata/label columns as features.

    Return dicts with name, X, y and windows. y is encoded in ACTIVITIES order
    for labelled files, or None for optional unlabelled training blocks. Every
    file stays separate. Refuse stale code/split provenance or modified artifacts.
    """
    if split not in ("train", "test", "unlabelled"):
        raise ValueError("Choose train, test or unlabelled explicitly")
    root = Path(root)
    folder = root / "data/features"
    if not json.loads((folder / "status.json").read_text())["ready"]:
        raise ValueError("Feature generation has not completed")
    config = json.loads((folder / "transform.json").read_text())
    if config["implementation_sha256"] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise ValueError("Feature code changed; regenerate features")
    if config["split_manifest_sha256"] != hashlib.sha256((root / "data/split_manifest.csv").read_bytes()).hexdigest():
        raise ValueError("Split changed; regenerate features")
    training_names = {name for name, _ in load_labelled_sequences(root, "train")}
    if training_names != set(config["fit_sequences"]):
        raise ValueError("Fitted feature transform does not match training membership")
    for name, digest in config["fit_input_sha256"].items():
        if digest != hashlib.sha256((root / "data/processed/train/clips" / f"{name}.csv").read_bytes()).hexdigest():
            raise ValueError("Training inputs changed; regenerate features")
    expected = config["output_sha256"][split]
    input_dir = root / "data/processed" / {"train": "train/clips", "test": "test", "unlabelled": "unlabelled/sessions"}[split]
    for name, digest in config["input_sha256"][split].items():
        if hashlib.sha256((input_dir / f"{name}.csv").read_bytes()).hexdigest() != digest:
            raise ValueError("Feature inputs changed; regenerate features")
    if {p.stem for p in (folder / split).glob("*.csv")} != set(expected):
        raise ValueError("Feature sequence membership changed")
    output = []
    for name, digest in expected.items():
        path = folder / split / f"{name}.csv"
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("Feature file changed; regenerate features")
        frame = pd.read_csv(path)
        X = frame[config["feature_order"]].to_numpy(dtype=float)
        if not np.isfinite(X).all():
            raise ValueError("Non-finite model features")
        y = None if split == "unlabelled" else np.array([config["state_order"].index(a) for a in frame.activity], dtype=np.int64)
        output.append({"name": name, "X": X, "y": y, "windows": frame[WINDOW_COLUMNS].copy()})
    return output


if __name__ == "__main__":
    result = build_features(Path(__file__).resolve().parents[1])
    print(json.dumps(result["status"], indent=2))
