# HMM Elderly Activity Monitoring

Human activity recognition from smartphone accelerometer and gyroscope data, with a planned Hidden Markov Model implemented from scratch in NumPy. The use case is elderly fall-risk and inactivity monitoring. The four activities are **still, standing, walking, jumping**. Phases 1–3 (collection inspection and preprocessing) are implemented; feature extraction, Baum–Welch, Viterbi and unseen-data evaluation remain later phases. Jumping is an impact activity, not validated fall data.

The revised collection uses 52 labelled clips (13 per activity), plus independent **unlabelled** blocks recovered from incomplete sessions. Approximate session labels are retained only for diagnostics. All four video links have been received and their metadata/previews are accessible without login. Preview inspection found full-body framing issues; video compliance remains open. See [video review](docs/video_review.md). See [collection status and remaining evidence](docs/collection_completion.md).

## Repository Structure

```
scripts/        preprocessing.py; splits.py; redact_metadata.py
data/
  raw/          Sensor Logger exports (pilot, clips, sessions, test); only the device id is redacted
  processed/    Validated train/clips; separate unlabelled/sessions; future test data
  labels/       Original session annotations (currently approximate/unverified)
  metadata.csv  One row per recording (activity, sampling rates, duration, date)
  clip_intervals.csv  Reviewed signal-based crops bound to raw-file hashes
  quality/      Sampling, gaps, crop/sequence manifests and current readiness
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
python -m unittest discover -s tests -v
```

The pipeline rejects gaps for whole-recording processing. Its recovery path splits sessions before filtering and exports each valid block as a separate sequence without activity labels. Never join these files into one sequence or use approximate session labels for evaluation. Generated output folders are managed by the pipeline; obsolete derived CSVs are removed on rerun. Raw exports remain unchanged.

Processed `timestamp_ns` values are Unix epoch nanoseconds. `seconds_elapsed` is measured from the original recording start, including after cropping; it does not reset to zero. All consecutive timestamps differ by exactly 40,000,000 ns (25 Hz). The labelled schema is `timestamp_ns, seconds_elapsed, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, activity`; unlabelled blocks omit `activity`.

The strict command verifies all 52 clips and ≥90 s per activity. It does not certify videos or session ground truth. Two genuinely unseen test recordings are still required in the later evaluation phase.

## Report

_Link to be added._
