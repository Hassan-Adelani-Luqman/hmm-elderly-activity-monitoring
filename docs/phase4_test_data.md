# Phase 4 — fixed recording-level holdout

Completed 29 October 2026 under the revised existing-clip design. The participant
clarified that part of the 52 recorded clips was intended for testing. No new
recordings, labels, dates or locations have been fabricated.

## The fixed split

- 44 training clips: 11 per activity.
- 8 test clips: 2 per activity, all four activities represented.
- The four recovered unlabelled session blocks remain optional training data.
- The four pilots remain calibration/preprocessing examples, not test data.

`data/split_manifest.csv` is the authoritative recording-level partition. It binds
each original clip to its activity, split and raw-file SHA-256 fingerprint.
Selection used NumPy `default_rng(42)`, two names without replacement per activity,
from sorted clip names in the order still, standing, walking, jumping. This was a
single selection before fitting. The pipeline reads the frozen manifest and
never rerandomises it to obtain better results.

| Activity | Held-out recordings |
|---|---|
| Still | still_02, still_11 |
| Standing | standing_06, standing_13 |
| Walking | walking_02, walking_10 |
| Jumping | jumping_02, jumping_07 |

The processed files are in `data/processed/train/clips/` and
`data/processed/test/`. All originals remain under `data/raw/clips/`; membership
comes from the manifest, not a renamed raw directory. `data/metadata.csv` records
the same split. There are 52 collected clips and ≥90 seconds per activity across
both subsets; these collection totals must not be described as training-only.

## Why split whole recordings

Overlapping windows from the same recording are strongly related. Randomly
splitting windows could place nearly the same sensor samples in training and
measurement of test performance. Holding out entire recordings before windowing
avoids that form of leakage. Each clip remains an independent sequence: never
concatenate clips to manufacture transitions or count transitions across files.

The reusable loader is `load_labelled_sequences(ROOT, "train")` or
`load_labelled_sequences(ROOT, "test")` from `scripts.preprocessing`. It verifies
source fingerprints, expected file membership, finite data, exact sample spacing
and activity labels. It returns separate `(recording_name, DataFrame)` pairs.
All model fitting, normalisation fitting, feature selection, state mapping and
hyperparameter choices must use the training subset only. Use recording-level
training-only validation for model decisions. Test metrics are for final use.

## What this evaluation can establish

It assesses activity recognition on held-out single-activity recordings from the
same participant, device and collection conditions. Training and test dates
are shared within the collection. Every selected clip was recorded indoors at
the configured 100 Hz setting (approximately 58.4–58.5 Hz delivered), and both
partitions use the unchanged 10 Hz low-pass and 25 Hz resampling code.

The original plan's two new mixed-activity sessions, different day/place and
lower-native-rate test are replaced by eight independent single-activity test
files. This provides more than two held-out files, but not the originally planned
continuous-transition experiment. Do not claim generalisation to a new day,
participant or environment, or validated real cross-activity transitions.
The lower-rate pilot demonstrates preprocessing compatibility only.

All 52 clips had already been inspected and cropped for data quality before this
split. That must be disclosed: they were not untouched by all exploratory work.
No HMM, learned feature selector, normalisation statistics or model parameters
had been fitted before partitioning. The crop intervals and processing settings
remain frozen. Further choices must not be guided by test results.

The approximate session labels are not used as evaluation truth. The recovered
unlabelled session blocks may support EM training, but not sensitivity,
specificity or accuracy calculations.

## Verification

```bash
.venv/bin/python scripts/preprocessing.py --require-complete
.venv/bin/python -m unittest discover -s tests -v
```

The pipeline fails on a missing manifest, stale source fingerprints, duplicate
membership or copied recordings spanning both splits. It removes held-out clips
from generated training outputs, checks the fixed 11/2 per-class counts and
ensures optional unlabelled training blocks do not overlap held-out acquisition
times. Raw files remain unchanged. The notebook demonstrates the partition and
actual collection conditions without fitting or scoring any test data.

Phase 4 is complete for this revised design. The next phase is feature extraction,
windowing and training-only normalisation.
