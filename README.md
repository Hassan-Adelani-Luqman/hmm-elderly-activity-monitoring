# HMM Elderly Activity Monitoring

Human activity recognition from smartphone accelerometer and gyroscope data, with a Hidden Markov Model implemented from scratch in NumPy. The use case is elderly fall-risk and inactivity monitoring. The four activities are **still, standing, walking, jumping**. Phases 1–7 are implemented: collection inspection, preprocessing, a fixed split, features, training-only normalisation, HMM verification, activity-model training and held-out evaluation. Report assembly remains Phase 8. Jumping is an impact activity, not validated fall data.

The revised collection uses 52 labelled clips (13 per activity), partitioned into **44 training and 8 test clips** (11/2 per activity), plus independent **unlabelled** blocks recovered from incomplete sessions. Approximate session labels are retained only for diagnostics. All four video links have been received and their metadata/previews are accessible without login. Preview inspection found full-body framing issues; video compliance remains open. See [video review](docs/video_review.md). See [collection status and remaining evidence](docs/collection_completion.md).

## Repository Structure

```
scripts/        preprocessing.py; splits.py; features.py; hmm.py; verify_hmm.py; train_activity.py;
                evaluate_activity.py; activity_plots.py; redact_metadata.py
data/
  raw/          Sensor Logger exports (pilot, clips, sessions, test); only the device id is redacted
  processed/    44 train/clips, 8 test clips; optional unlabelled/sessions
  labels/       Original session annotations (currently approximate/unverified)
  metadata.csv  One row per recording (activity, sampling rates, duration, date)
  clip_intervals.csv  Reviewed signal-based crops bound to raw-file hashes
  split_manifest.csv  Frozen train/test membership and source hashes (seed 42)
  quality/      Sampling, gaps, crop/sequence manifests and current readiness
  features/     Normalised per-sequence features, training fit parameters and audit tables
  verification/ Synthetic HMM verification, source hashes and likelihood histories
  models/       Frozen activity-model parameters, mappings and training provenance
  evaluation/   Model comparison, per-window/clip/class results and confusion matrices
notebooks/      hmm_activity_recognition.ipynb
figures/        Figures exported from the notebook
report/         Final report (.docx and .pdf)
tests/          Numerical and data-integrity regression checks
docs/           Collection completion notes and video verification checklist
```

## How to Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/hmm_activity_recognition.ipynb
```

Run all cells from top to bottom. All paths are relative to the repository root.

To regenerate processed files and run the checks without Jupyter:

```bash
python scripts/preprocessing.py --require-complete
python scripts/features.py
python scripts/verify_hmm.py
python scripts/train_activity.py
python scripts/evaluate_activity.py
python scripts/activity_plots.py
python -m unittest discover -s tests -v
```

The pipeline rejects gaps for whole-recording processing. Its recovery path splits sessions before filtering and exports each valid block as a separate sequence without activity labels. Never join these files into one sequence or use approximate session labels for evaluation. Generated output folders are managed by the pipeline; obsolete derived CSVs are removed on rerun. Raw exports remain unchanged.

Processed `timestamp_ns` values are Unix epoch nanoseconds. `seconds_elapsed` is measured from the original recording start, including after cropping; it does not reset to zero. All consecutive timestamps differ by exactly 40,000,000 ns (25 Hz). The labelled schema is `timestamp_ns, seconds_elapsed, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, activity`; unlabelled blocks omit `activity`.

The strict command verifies all 52 collected clips, ≥90 s per activity across the collection, and the fixed 44/8 partition. It does not certify videos or session ground truth. Test clips come from the same collection and were already quality-inspected; no model or learned features were fitted before splitting. This evaluates held-out single-activity recognition, not a different day/place or real activity transitions. See [Phase 4 details](docs/phase4_test_data.md).

For subsequent features/models, load `load_labelled_sequences(ROOT, "train")` or `load_labelled_sequences(ROOT, "test")` from `scripts.preprocessing`. Keep every file as a separate sequence. Fit scaling, select features and tune models using training data only.

## Feature sequences

Phase 5 uses 2 s windows with a 1 s hop. Training-only screening retains **10 features** (7 time-domain, 3 FFT). Population z-score statistics are fitted on the **338 windows from 44 labelled training clips** and applied unchanged to the **65 test windows** and **137 optional unlabelled windows**.

Use `load_feature_sequences(ROOT, "train")`, `"test"`, or `"unlabelled"` from `scripts.features` for model-ready `X` matrices, labels and window metadata. Each recording remains a separate sequence. Feature order and fitted statistics are stored in `data/features/transform.json`. See [Phase 5 definitions, checks and limitations](docs/phase5_features.md).

## HMM verification

`GaussianHMM` in `scripts/hmm.py` implements Gaussian emissions, scaled log-space forward/backward, Viterbi, supervised initialisation and multi-sequence Baum–Welch. The notebook displays the implementation and mathematical verification in Sections 5–6.

All six Phase 6 checks pass using exact enumeration and generated synthetic sequences. The recovery example converges in 7 EM iterations on the log-likelihood tolerance, with maximum transition/mean errors of 0.01421/0.03212. That verification phase uses synthetic data only. See [Phase 6 methods and verification](docs/phase6_hmm.md) and `data/verification/hmm.json`.

## Activity evaluation

The preselected Baum–Welch HMM scores **63/65 test windows (96.92%)**; the supervised-only HMM and independent Gaussian baseline each score **64/65 (98.46%)**. EM improves training likelihood but slightly reduces classification accuracy. All primary errors are standing windows predicted as walking in one clip.

Training and state mapping use only 44 labelled clips. Testing uses eight independent held-out clips from the same collection; their 65 overlapping windows are not independent trials. No new-person/environment or real cross-activity transition performance is established. See [Phase 7 results and limitations](docs/phase7_results.md). Report Figures 2–4 and the executed notebook show the fixed model, comparisons and per-clip outcomes.

## Report

_Link to be added._
